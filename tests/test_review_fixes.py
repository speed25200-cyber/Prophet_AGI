"""Regression tests for the defects found by the independent codebase review (A1).

Each test names the bug it guards and the way it used to manifest. Almost all were
silent: the model trained normally and was wrong. That is the category of failure this
project has decided to treat as the most dangerous, so each one gets an assertion that
would have failed on the code as it was.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from prophet.budget import (
    block_passes_per_token,
    expected_train_loop_k,
    tokens_affordable,
    training_memory,
)
from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MemoryConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.data.decontaminate import Decontaminator
from prophet.modeling.layers import AttentionCache, CausalSelfAttention, RotaryEmbedding
from prophet.modeling.model import ProphetCache, ProphetModel
from prophet.train.loss import compute_loss
from prophet.train.optim import build_param_groups

VOCAB = 128


def small(**kw) -> ProphetConfig:
    base = dict(
        d_model=64, max_seq_len=64,
        frontend=FrontendConfig(vocab_size=VOCAB),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=4, n_kv_heads=2, head_dim=16,
            sliding_window=8, attention_sink_tokens=2, linear_heads=2, linear_head_dim=16,
            nope_layers=(1,),
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=2, coda_layers=2,
            core_pattern=["gdn"], default_loop_k=3, train_loop_max=6,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    base.update(kw)
    return ProphetConfig(**base)


# --------------------------------------------------------------------------------------
# B1 -- the forget gate the model init used to zero
# --------------------------------------------------------------------------------------


def test_b1_forget_gate_starts_near_one_inside_a_built_model():
    """A standalone GatedDeltaNet kept its bias at 3.0; inside ProphetModel, _init_weights
    zeroed it, so every recurrent layer started with a one-token half-life."""
    model = ProphetModel(small())
    for block in model.sections["core"]:
        alpha = torch.sigmoid(block.mixer.a_proj.bias)
        assert (alpha > 0.9).all(), f"forget gate alpha at init: {alpha.tolist()}"


# --------------------------------------------------------------------------------------
# B2 -- the ponder gradient that was the same at every position
# --------------------------------------------------------------------------------------


def _halting_cfg() -> ProphetConfig:
    return small(recurrent=RecurrentCoreConfig(
        enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
        core_pattern=["gdn"], default_loop_k=4, halting="ponder",
        halting_loss_weight=0.05, halting_target_steps=2.0,
    ))


def test_b2_ponder_gradient_differs_across_positions():
    """A product of two batch means gives d/dp the same value everywhere; the head could
    only learn one constant distribution. Per-token weighting must give per-token
    gradients."""
    torch.manual_seed(0)
    model = ProphetModel(_halting_cfg())
    batch = torch.randint(0, VOCAB, (2, 10))
    out = model(batch, loop_k=4)
    out.halt_probs.retain_grad()
    terms = compute_loss(out, batch, ponder_weight=1.0, z_loss_weight=0.0,
                         ponder_target_steps=2.0, project=model._project)
    terms.ponder.backward()
    g = out.halt_probs.grad
    assert g is not None
    # Distinct gradient values across positions, for at least one stopping step.
    assert max(g[..., i].unique().numel() for i in range(g.shape[-1])) > 1


def test_b2_hidden_per_step_is_in_the_normalised_space():
    """Pre-norm coda output has rms ~0.1; the LM head expects ~1. Scoring stopping points
    on the wrong scale gave near-uniform logits for every candidate."""
    torch.manual_seed(0)
    model = ProphetModel(_halting_cfg()).eval()
    with torch.no_grad():
        out = model(torch.randint(0, VOCAB, (1, 8)), loop_k=3)
    rms_final = out.hidden.pow(2).mean().sqrt().item()
    for h in out.hidden_per_step:
        rms = h.pow(2).mean().sqrt().item()
        assert 0.5 < rms / rms_final < 2.0


# --------------------------------------------------------------------------------------
# B3 -- residual scaling applied at forward time
# --------------------------------------------------------------------------------------


def test_b3_residual_branches_are_added_unscaled():
    """With scaling at init the *function* of a block is x + f(x). A converted donor
    block must therefore reproduce the donor's arithmetic, not a damped version."""
    torch.manual_seed(0)
    model = ProphetModel(small()).eval()
    block = model.sections["prelude"][0]
    x = torch.randn(1, 5, 64)
    with torch.no_grad():
        h = block.norm1(x)
        mixed = block.mixer(h)
        expected = x + mixed
        expected = expected + block.ffn(block.norm2(expected))
        got = block(x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_b3_init_scaling_lands_on_output_projections_only():
    torch.manual_seed(0)
    scaled = ProphetModel(small(residual_scaling=True))
    torch.manual_seed(0)
    plain = ProphetModel(small(residual_scaling=False))
    o_s = scaled.sections["prelude"][0].mixer.o_proj.weight.norm()
    o_p = plain.sections["prelude"][0].mixer.o_proj.weight.norm()
    q_s = scaled.sections["prelude"][0].mixer.q_proj.weight.norm()
    q_p = plain.sections["prelude"][0].mixer.q_proj.weight.norm()
    assert o_s < 0.5 * o_p          # output projection is damped at init
    assert torch.allclose(q_s, q_p)  # everything else untouched


# --------------------------------------------------------------------------------------
# B4 / B5 -- attention caches that were only right for one access pattern
# --------------------------------------------------------------------------------------


def test_b4_chunked_continuation_matches_full_forward_in_full_attention():
    """is_causal on a chunk after a cache is top-left aligned: query i of the chunk
    attended to keys 0..i, not 0..offset+i. Measured 1.30 max error before the fix."""
    torch.manual_seed(1)
    attn = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16).eval()
    rope = RotaryEmbedding(16, theta=1e4)
    x = torch.randn(1, 12, 64)
    cos, sin = rope(torch.arange(12)[None])
    with torch.no_grad():
        full = attn(x, cos=cos, sin=sin)
        cache = AttentionCache()
        attn(x[:, :8], cos=cos[:, :8], sin=sin[:, :8], cache=cache)
        chunk = attn(x[:, 8:], cos=cos[:, 8:], sin=sin[:, 8:], cache=cache)
    assert torch.allclose(full[:, 8:], chunk, atol=1e-5)


def test_b5_prefill_longer_than_the_window_matches_no_cache():
    """Eviction before attention plus a mask on buffer indices: every prompt longer than
    the window was wrong in every windowed layer. Measured 0.80 max error before."""
    torch.manual_seed(2)
    attn = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16,
                               window=8, sink_tokens=2).eval()
    rope = RotaryEmbedding(16, theta=1e4)
    x = torch.randn(1, 24, 64)
    cos, sin = rope(torch.arange(24)[None])
    with torch.no_grad():
        a = attn(x, cos=cos, sin=sin)
        b = attn(x, cos=cos, sin=sin, cache=AttentionCache())
    assert torch.allclose(a, b, atol=1e-5)


def test_b5_chunked_prefill_across_an_eviction_boundary():
    torch.manual_seed(3)
    attn = CausalSelfAttention(64, n_heads=4, n_kv_heads=2, head_dim=16,
                               window=6, sink_tokens=1).eval()
    rope = RotaryEmbedding(16, theta=1e4)
    x = torch.randn(1, 20, 64)
    cos, sin = rope(torch.arange(20)[None])
    with torch.no_grad():
        full = attn(x, cos=cos, sin=sin)
        cache = AttentionCache()
        parts = []
        for lo, hi in ((0, 7), (7, 11), (11, 12), (12, 20)):
            parts.append(attn(x[:, lo:hi], cos=cos[:, lo:hi], sin=sin[:, lo:hi], cache=cache))
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)
    assert cache.keys.shape[2] <= 6 + 1


def test_b5_model_level_prefill_beyond_window_matches():
    torch.manual_seed(4)
    model = ProphetModel(small()).eval()
    ids = torch.randint(0, VOCAB, (1, 30))  # window is 8
    with torch.no_grad():
        a = model(ids, loop_k=2, return_mtp=False).logits
        b = model(ids, loop_k=2, cache=ProphetCache(), return_mtp=False).logits
    assert torch.allclose(a, b, atol=1e-4)


# --------------------------------------------------------------------------------------
# B6 -- halting with a cache
# --------------------------------------------------------------------------------------


def test_b6_halting_probes_see_context_at_decode_time():
    """Cache-free probes decided halting from the current token alone. With disposable
    slot copies, the halting distribution at decode must match the full pass."""
    torch.manual_seed(5)
    model = ProphetModel(_halting_cfg()).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        full = model(ids, loop_k=4, return_mtp=False)
        cache = ProphetCache()
        steps = [model(ids[:, t:t + 1], cache=cache, loop_k=4, return_mtp=False)
                 for t in range(8)]
    inc_probs = torch.cat([s.halt_probs for s in steps], dim=1)
    # The real, cached coda must match exactly.
    assert torch.allclose(full.logits, torch.cat([s.logits for s in steps], 1), atol=1e-4)
    # The probes cannot match exactly, and should not be asked to. In a full pass the
    # iteration-i probe attends to earlier positions' *iteration-i* states; at decode
    # time it attends to their *final* states, which is what the cache holds. Making
    # those equal would need a coda cache per iteration -- the KV-times-k cost decision
    # D1 refuses. What the fix guarantees is that the probe sees its context at all:
    # measured 3e-4 apart here, against a cache-free probe that saw one token.
    assert torch.allclose(full.halt_probs, inc_probs, atol=1e-2)
    assert (full.halt_probs - inc_probs).abs().max() < 5e-3


def test_b6_early_exit_reports_the_iterations_run_and_pins_the_cache():
    torch.manual_seed(6)
    model = ProphetModel(_halting_cfg()).eval()
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        out = model(ids, loop_k=12, halt_threshold=0.05, cache=(cache := ProphetCache()))
    assert out.loop_k == out.halt_probs.shape[-1] <= 12
    assert cache.loop_k == out.loop_k


def test_b6_a_cache_refuses_a_deeper_call_than_it_was_built_at():
    """Core slots for iterations that never ran do not exist for the earlier tokens; a
    deeper call would read fresh states as though they had context."""
    torch.manual_seed(7)
    model = ProphetModel(small()).eval()
    cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 4)), cache=cache, loop_k=2)
        with pytest.raises(ValueError, match="only shrink"):
            model(torch.randint(0, VOCAB, (1, 1)), cache=cache, loop_k=5)


def test_b6_halt_threshold_is_per_sequence():
    """One confident sequence in a batch must not cut off another's thinking."""
    torch.manual_seed(8)
    model = ProphetModel(_halting_cfg()).eval()
    ids = torch.randint(0, VOCAB, (3, 6))
    with torch.no_grad():
        out = model(ids, loop_k=8, halt_threshold=0.5)
    cum = 1 - torch.cumprod(1 - torch.sigmoid(model.halt_head(out.hidden_per_step[-1]).squeeze(-1)), 0)
    # Every (sequence, position) crossed, or the loop ran out -- never a batch mean.
    survived = torch.stack([1 - torch.sigmoid(model.halt_head(h).squeeze(-1))
                            for h in out.hidden_per_step]).prod(0)
    assert out.loop_k == 8 or bool(((1 - survived) >= 0.5).all())


# --------------------------------------------------------------------------------------
# B7 / B8 -- a ledger that was never read
# --------------------------------------------------------------------------------------


def test_b7_output_mounted_ledger_is_actually_read():
    torch.manual_seed(9)
    cfg = small(memory=MemoryConfig(enabled=True, kind="product_key", mount="output",
                                    memory_dim=32, n_slots=1024))
    model = ProphetModel(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        before = model(ids).logits
        model.ledgers["output"].write(torch.randn(1, 6, 64), 50 * torch.randn(1, 6, 64))
        after = model(ids).logits
    assert not torch.allclose(before, after, atol=1e-3)


def test_b7_hidden_stays_ledger_free_so_consolidation_addresses_the_right_space():
    torch.manual_seed(10)
    cfg = small(memory=MemoryConfig(enabled=True, kind="product_key", mount="output",
                                    memory_dim=32, n_slots=1024))
    model = ProphetModel(cfg).eval()
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        h0 = model(ids).hidden
        model.ledgers["output"].write(torch.randn(1, 6, 64), 50 * torch.randn(1, 6, 64))
        h1 = model(ids).hidden
    assert torch.allclose(h0, h1)


def test_b7_coda_mount_indices_are_coda_local_and_validated():
    with pytest.raises(ValueError, match="coda-local"):
        small(memory=MemoryConfig(enabled=True, kind="product_key", mount="coda",
                                  layers=(5,), memory_dim=32, n_slots=1024)).validate()
    cfg = small(memory=MemoryConfig(enabled=True, kind="product_key", mount="coda",
                                    layers=(1,), memory_dim=32, n_slots=1024))
    model = ProphetModel(cfg).eval()
    assert "coda_1" in model.ledgers
    ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        before = model(ids).logits
        model.ledgers["coda_1"].write(torch.randn(1, 6, 64), 50 * torch.randn(1, 6, 64))
        after = model(ids).logits
    assert not torch.allclose(before, after, atol=1e-3)


def test_b8_fast_weight_is_refused_as_unimplemented():
    with pytest.raises(ValueError, match="not implemented"):
        small(memory=MemoryConfig(enabled=True, kind="fast_weight")).validate()


def test_b8_ledger_addressing_has_no_trainable_parameters():
    """The class promises frozen keys. A trainable query projection moved every stored
    association's address whenever the optimiser ran."""
    from prophet.memory.ledger import LedgerConfig, ProductKeyMemory
    mem = ProductKeyMemory(LedgerConfig(dim=32, memory_dim=32, n_slots=256))
    assert list(mem.parameters()) == []


def test_b8_memory_config_defaults_match_ledger_config():
    from prophet.memory.ledger import LedgerConfig
    assert MemoryConfig().write_lr == LedgerConfig(dim=8).write_lr
    assert MemoryConfig().decay == LedgerConfig(dim=8).decay


# --------------------------------------------------------------------------------------
# B9 -- decontaminator
# --------------------------------------------------------------------------------------


def test_b9_one_repeated_ngram_does_not_cross_the_threshold():
    d = Decontaminator(n=3, threshold=0.5)
    d.add_benchmark("b", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"])
    hits = d.check("alpha beta gamma. " * 5 + "unrelated weather")
    assert hits == [] or all(h.containment <= 0.2 for h in hits)


def test_b9_containment_never_exceeds_one():
    d = Decontaminator(n=3, threshold=0.1)
    d.add_benchmark("b", ["one two three four five six"])
    hits = d.check("one two three four five six " * 4)
    assert hits and all(h.containment <= 1.0 for h in hits)


def test_b9_trivially_short_items_are_not_indexed():
    d = Decontaminator(n=13)
    assert d.add_benchmark("b", ["Yes", "42", "The answer is 4"]) == 0
    assert not d.is_contaminated("Yes, the answer is 42 and that is that.")


def test_b9_short_items_match_on_word_boundaries():
    d = Decontaminator(n=13, min_short_words=3)
    d.add_benchmark("b", ["the answer is 4"])
    assert d.is_contaminated("as we know the answer is 4 here")
    assert not d.is_contaminated("as we know the answer is 42 here")


# --------------------------------------------------------------------------------------
# B12 -- the token budget
# --------------------------------------------------------------------------------------


def test_b12_expected_depth_follows_the_configured_distribution():
    cfg = small(recurrent=RecurrentCoreConfig(
        enabled=True, prelude_layers=1, core_layers=1, coda_layers=1, core_pattern=["gdn"],
        train_loop_min=1, train_loop_max=8, train_loop_dist="log_uniform"))
    e = expected_train_loop_k(cfg)
    assert 3.0 < e < 3.8, e            # not (1+8)/2 = 4.5
    cfg.recurrent.train_loop_dist = "uniform"
    assert expected_train_loop_k(cfg) == pytest.approx(4.5)


def test_b12_halting_probe_passes_are_counted():
    base = small(heads=HeadsConfig())
    halting = _halting_cfg()
    halting.heads = HeadsConfig()
    k = 3.0
    r_b, r_h = base.recurrent, halting.recurrent
    assert block_passes_per_token(base, k) == r_b.prelude_layers + r_b.core_layers * k + r_b.coda_layers
    assert block_passes_per_token(halting, k) == (
        r_h.prelude_layers + r_h.core_layers * k + r_h.coda_layers + r_h.coda_layers * k
    )


def test_b12_tokens_affordable_reports_the_passes_it_assumed():
    tb = tokens_affordable(small())
    assert "block_passes_per_token" in tb and tb["block_passes_per_token"] > 0


def test_b10_training_memory_defaults_describe_the_trainer_as_built():
    """fp32 params and grads, Muon/AdamW state from the real split -- not an 8-bit
    optimiser that nothing implements."""
    tm = training_memory(small())
    n = tm.detail["total_params"]
    assert tm.weights_gb == pytest.approx(n * 4 / 1024**3)
    assert tm.gradients_gb == pytest.approx(n * 4 / 1024**3)
    assert 4.0 <= tm.optimizer_gb / (n / 1024**3) <= 8.0


# --------------------------------------------------------------------------------------
# B20 / B14 / B11
# --------------------------------------------------------------------------------------


def test_b20_vector_shaped_heads_go_to_adamw():
    model = ProphetModel(_halting_cfg())
    groups = build_param_groups(model)
    muon_ids = {id(p) for p in groups["muon"]}
    for name, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) == 1:
            assert id(p) not in muon_ids, name


def test_b14_trainer_honours_a_stop_request(tmp_path):
    from prophet.data.streaming import StreamingLoader, sources_from_iterables
    from prophet.train.loop import TrainConfig, Trainer
    cfg = small()
    trainer = Trainer(
        ProphetModel(cfg),
        StreamingLoader(sources_from_iterables({"a": (1.0, [[1, 2, 3] * 8] * 4)}), seq_len=8),
        TrainConfig(total_steps=50, log_every=1000, checkpoint_every=0,
                    checkpoint_dir=str(tmp_path)),
        model_config=cfg, on_log=lambda m: None,
    )
    trainer.stop_requested = True
    trainer.train()
    assert trainer.step == 1


def test_b11_checkpoint_carries_cuda_rng_slot():
    from prophet.data.streaming import StreamingLoader, sources_from_iterables
    from prophet.train.loop import TrainConfig, Trainer
    cfg = small()
    trainer = Trainer(
        ProphetModel(cfg),
        StreamingLoader(sources_from_iterables({"a": (1.0, [[1, 2, 3] * 8] * 4)}), seq_len=8),
        TrainConfig(total_steps=1, checkpoint_every=0, checkpoint_dir="/tmp/x"),
        model_config=cfg, on_log=lambda m: None,
    )
    assert "cuda_rng" in trainer.state_dict()


def test_b22_explicit_generator_makes_training_forward_reproducible():
    torch.manual_seed(0)
    model = ProphetModel(small())
    model.train()
    ids = torch.randint(0, VOCAB, (1, 6))
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = model(ids, generator=g1).logits
    b = model(ids, generator=g2).logits
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------------------
# B13 -- head-consistent donor seeding
# --------------------------------------------------------------------------------------


def test_b13_gdn_seed_keeps_each_head_with_its_own_values():
    from prophet.convert.weights import _seed_gdn_from_attention, DONOR_KEYS
    from prophet.convert.donors import get_donor
    from dataclasses import replace
    from prophet.convert.plan import prophet_config_for_donor, plan_conversion

    d = replace(get_donor("qwen3-0.6b"), n_layers=4, d_model=64, n_heads=4, n_kv_heads=2,
                head_dim=16, ffn_hidden=128, vocab_size=100)
    cfg = prophet_config_for_donor(d, prelude_layers=1, core_layers=1, coda_layers=1)
    assert cfg.residual_scaling is False
    assert cfg.mixer.rope_theta == d.rope_theta
    model = ProphetModel(cfg)
    target = {k: v.clone() for k, v in model.state_dict().items()}

    # Donor o_proj with a distinct constant per head, so scrambling is visible.
    o = torch.zeros(64, 4 * 16)
    for h in range(4):
        o[:, h * 16:(h + 1) * 16] = float(h + 1)
    donor = {DONOR_KEYS["o_proj"].format(i=0): o}
    from prophet.convert.weights import TransferReport
    rep = TransferReport()
    _seed_gdn_from_attention(target, "sections.core.0", donor, (0,),
                             n_heads=4, n_kv_heads=2, head_dim=16, report=rep)
    w = target["sections.core.0.mixer.o_proj.weight"]  # (64, 4 heads * 32)
    for h in range(4):
        block = w[:, h * 32:(h + 1) * 32]
        assert torch.all(block[:, :16] == float(h + 1)), f"head {h} got another head's values"
        assert torch.all(block[:, 16:] == 0)
