"""Per-token recurrence ceilings, the cache depth policy, and the router-update fix.

The agent loop wants to read a tool observation at depth 1 and think at depth 8 on the
same cache. Without ``recurrent.token_depth`` that is undefined -- a core state at
iteration 3 would be read by a token whose predecessors never ran iteration 3 -- and the
model refuses it. With it, each iteration runs the compacted subsequence of tokens still
active, in training and at inference alike, so the two are the same computation. These
tests pin that equivalence down numerically rather than by argument.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from prophet.config import ProphetConfig
from prophet.data.tokenizer import N_BYTES, SPECIAL_TOKENS, ProphetTokenizer
from prophet.modeling.model import ProphetCache, ProphetModel
from prophet.modeling.moe import MoERouter, apply_router_updates
from prophet.train.loop import TOOL_ID, Trainer

torch.manual_seed(0)


def _cfg(**recurrent) -> ProphetConfig:
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg = dataclasses.replace(cfg, recurrent=dataclasses.replace(cfg.recurrent, **recurrent))
    cfg.validate()
    return cfg


def _model(**recurrent) -> ProphetModel:
    torch.manual_seed(1)
    return ProphetModel(_cfg(**recurrent)).eval()


# --------------------------------------------------------------------------------------
# Compaction is the same computation as running the token shallow
# --------------------------------------------------------------------------------------


def test_uniform_ceilings_are_bit_identical_to_a_plain_depth():
    model = _model(token_depth=True)
    ids = torch.randint(0, 2048, (2, 12))
    with torch.no_grad():
        plain = model(ids, loop_k=3).logits
        ceil = model(ids, token_depth=torch.full((2, 12), 3)).logits
    assert torch.equal(plain, ceil)


def test_mixed_ceilings_match_an_incremental_decode_at_each_tokens_depth():
    """The full pass with a ceiling vector must equal feeding the tokens one at a time,
    each at its own depth -- which is what the agent loop does."""
    model = _model(token_depth=True)
    ids = torch.randint(0, 2048, (1, 9))
    depth = torch.tensor([[3, 3, 1, 1, 2, 3, 1, 3, 2]])
    with torch.no_grad():
        full = model(ids, token_depth=depth).logits
        cache = ProphetCache()
        steps = [
            model(ids[:, t : t + 1], cache=cache, loop_k=int(depth[0, t])).logits
            for t in range(ids.shape[1])
        ]
    inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()
    assert cache.loop_k == 3  # deepest slot in use, not a pin


def test_segments_of_differing_depth_share_one_cache():
    model = _model(token_depth=True)
    ids = torch.randint(0, 2048, (1, 10))
    depth = torch.tensor([[1, 1, 1, 4, 4, 2, 2, 2, 3, 3]])
    with torch.no_grad():
        full = model(ids, token_depth=depth).logits
        cache = ProphetCache()
        parts = []
        for lo, hi, k in ((0, 3, 1), (3, 5, 4), (5, 8, 2), (8, 10, 3)):
            parts.append(model(ids[:, lo:hi], cache=cache, loop_k=k).logits)
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-4)


def test_end_padding_of_a_compacted_batch_is_exact():
    """Rows with different active counts are padded at the end; a causal core and a
    discarded final state make that exact, and it had better be."""
    model = _model(token_depth=True)
    ids = torch.randint(0, 2048, (2, 8))
    depth = torch.tensor([[3, 1, 1, 3, 2, 1, 1, 1], [1, 3, 3, 3, 3, 3, 2, 3]])
    with torch.no_grad():
        batched = model(ids, token_depth=depth).logits
        rows = [model(ids[i : i + 1], token_depth=depth[i : i + 1]).logits for i in range(2)]
    assert torch.allclose(batched, torch.cat(rows, 0), atol=1e-4)


def test_cached_batch_with_unequal_active_counts_is_refused():
    model = _model(token_depth=True)
    ids = torch.randint(0, 2048, (2, 4))
    depth = torch.tensor([[2, 2, 2, 2], [2, 1, 1, 2]])
    with pytest.raises(ValueError, match="same number of active tokens"), torch.no_grad():
        model(ids, cache=ProphetCache(), token_depth=depth)


def test_ceilings_are_refused_on_a_model_trained_at_one_depth():
    model = _model(token_depth=False)
    with pytest.raises(ValueError, match="recurrent.token_depth is off"), torch.no_grad():
        model(torch.randint(0, 2048, (1, 4)), token_depth=torch.full((1, 4), 2))


def test_ponder_mass_is_forced_at_the_ceiling():
    model = _model(token_depth=True, halting="ponder")
    ids = torch.randint(0, 2048, (1, 6))
    depth = torch.tensor([[1, 2, 3, 3, 1, 2]])
    with torch.no_grad():
        out = model(ids, token_depth=depth)
    p = out.halt_probs[0]  # (seq, k)
    for t in range(6):
        d = int(depth[0, t])
        assert p[t, :d].sum().item() == pytest.approx(1.0, abs=1e-6)
        assert p[t, d:].sum().item() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Without the switch, a cache's depth may only shrink
# --------------------------------------------------------------------------------------


def test_fixed_depth_cache_shrinks_but_never_grows():
    model = _model(token_depth=False)
    cache = ProphetCache()

    def tok():
        return torch.randint(0, 2048, (1, 3))

    with torch.no_grad():
        model(tok(), cache=cache, loop_k=3)
        with pytest.raises(ValueError, match="only shrink"):
            model(tok(), cache=cache, loop_k=4)
        model(tok(), cache=cache, loop_k=2)  # shallower: the deeper slot retires
        assert cache.loop_k == 2
        out = model(tok(), cache=cache, loop_k=None)  # follows the pin, no raise
        assert out.loop_k == 2
        with pytest.raises(ValueError):
            model(tok(), cache=cache, loop_k=3)


def test_shrinking_is_exact():
    """Going shallower on a cache equals a sequence whose tail always ran shallow: the
    tail never reads the retired slots."""
    model = _model(token_depth=False)
    ids = torch.randint(0, 2048, (1, 8))
    with torch.no_grad():
        cache = ProphetCache()
        model(ids[:, :5], cache=cache, loop_k=3)
        tail = model(ids[:, 5:], cache=cache, loop_k=1).logits
        ref = ProphetCache()
        model(ids[:, :5], cache=ref, loop_k=3)
        # Reference: the same first 5 tokens at depth 3, then the tail token by token
        # at depth 1 -- what the pin promises.
        ref_tail = torch.cat(
            [model(ids[:, t : t + 1], cache=ref, loop_k=1).logits for t in range(5, 8)], 1
        )
    assert torch.allclose(tail, ref_tail, atol=1e-4)


# --------------------------------------------------------------------------------------
# Trainer ceilings
# --------------------------------------------------------------------------------------


def _fake_trainer(**recurrent):
    return SimpleNamespace(model_config=_cfg(token_depth=True, **recurrent))


def test_trainer_ceilings_mark_tool_spans_and_nothing_else():
    assistant = N_BYTES + SPECIAL_TOKENS.index("<|assistant|>")
    batch = torch.tensor([[400, 401, TOOL_ID, 402, 403, assistant, 404, 405]])
    depth = Trainer.token_depth(_fake_trainer(ingest_depth=1, token_depth_random_spans=0.0), batch, 4)
    assert depth.tolist() == [[4, 4, 1, 1, 1, 4, 4, 4]]


def test_trainer_ceilings_never_exceed_the_sampled_depth():
    batch = torch.tensor([[TOOL_ID, 400, 401]])
    depth = Trainer.token_depth(_fake_trainer(ingest_depth=3, token_depth_random_spans=0.0), batch, 2)
    assert depth.max().item() == 2


def test_trainer_random_spans_are_shallow_and_reproducible():
    batch = torch.randint(300, 2048, (4, 32))
    trainer = _fake_trainer(ingest_depth=1, token_depth_random_spans=1.0)
    torch.manual_seed(3)
    a = Trainer.token_depth(trainer, batch, 5)
    torch.manual_seed(3)
    b = Trainer.token_depth(trainer, batch, 5)
    assert torch.equal(a, b)
    assert bool((a == 1).any(dim=1).all())  # every row got a span
    assert bool((a == 5).any())


# --------------------------------------------------------------------------------------
# Router balancing under activation checkpointing
# --------------------------------------------------------------------------------------


def _moe_model() -> ProphetModel:
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg = dataclasses.replace(
        cfg,
        ffn=dataclasses.replace(
            cfg.ffn, kind="moe", n_experts=4, n_experts_per_token=2, n_shared_experts=1,
            moe_first_dense_layers=0,
        ),
    )
    torch.manual_seed(2)
    return ProphetModel(cfg).train()


def test_moe_backward_survives_activation_checkpointing():
    """The forward used to move the router bias in place; the checkpointed recompute
    then routed differently and backward raised CheckpointError on every MoE config."""
    model = _moe_model()
    model.gradient_checkpointing = True
    router = next(m for m in model.modules() if isinstance(m, MoERouter))
    before = router.expert_bias.clone()
    out = model(torch.randint(0, 2048, (2, 16)), loop_k=2)
    assert torch.equal(router.expert_bias, before)  # nothing moves inside the forward
    out.logits.float().mean().backward()
    assert model.apply_router_updates(out) > 0
    assert not torch.equal(router.expert_bias, before)


def test_router_update_is_applied_once_per_forward_call():
    model = _moe_model()
    out = model(torch.randint(0, 2048, (1, 8)), loop_k=3)
    # A looped core block is called once per iteration; pick one and check that every
    # call recorded its own step.
    routers = [m for m in model.modules() if isinstance(m, MoERouter)]
    counts = {r: sum(1 for s in out.router_stats if s.router is r) for r in routers}
    router = max(routers, key=counts.get)
    mine = [s for s in out.router_stats if s.router is router]
    assert len(mine) == 3
    before = router.expert_bias.clone()
    apply_router_updates(mine)
    step = sum(s.bias_step for s in mine) * router.bias_update_rate
    assert torch.allclose(router.expert_bias, before - step)


# --------------------------------------------------------------------------------------
# Tokenizer: control strings are inert unless parsed
# --------------------------------------------------------------------------------------


def test_special_strings_are_inert_unless_parsed():
    tok = ProphetTokenizer(merges=[])
    sid = tok.special_id("<|assistant|>")
    plain = tok.encode("ok <|assistant|> ok")
    assert sid not in plain and tok.decode(plain) == "ok <|assistant|> ok"
    parsed = tok.encode("ok <|assistant|> ok", parse_special=True)
    assert parsed.count(sid) == 1
    assert tok.decode(parsed, skip_special=False) == "ok <|assistant|> ok"
