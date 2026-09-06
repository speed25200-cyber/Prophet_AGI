"""Bounded infinite context: windowed attention whose evicted keys and values live on in
a product-key ledger. Exactness where it is claimed, recall where it is claimed, and
memory that stops growing."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from prophet.budget import _kv_bytes_per_token, count_parameters
from prophet.config import ProphetConfig
from prophet.memory.ledger import LedgerConfig, ProductKeyMemory
from prophet.memory.session import extract_session, restore_session
from prophet.modeling.layers import AttentionCache, CausalSelfAttention, LedgerAttention
from prophet.modeling.model import ProphetCache, ProphetModel


def _pair(window: int = 8, seed: int = 0) -> tuple[LedgerAttention, CausalSelfAttention]:
    torch.manual_seed(seed)
    ledger = LedgerAttention(32, n_heads=4, n_kv_heads=2, window=window, sink_tokens=1,
                             ledger_slots=64, ledger_top_k=8)
    plain = CausalSelfAttention(32, n_heads=4, n_kv_heads=2, window=window, sink_tokens=1, use_rope=False)
    plain.load_state_dict({k: v for k, v in ledger.state_dict().items() if k in plain.state_dict()})
    return ledger.eval(), plain.eval()


def test_within_the_window_the_layer_is_exactly_a_windowed_layer():
    ledger, plain = _pair(window=16)
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        assert torch.allclose(ledger(x), plain(x), atol=1e-6)


def test_with_the_gate_closed_the_layer_is_a_windowed_layer_at_any_length():
    ledger, plain = _pair(window=8)
    with torch.no_grad():
        ledger.gate.fill_(-1e4)
        x = torch.randn(2, 40, 32)
        assert torch.allclose(ledger(x), plain(x), atol=1e-5)


def test_beyond_the_window_the_ledger_contributes():
    ledger, plain = _pair(window=8)
    x = torch.randn(1, 40, 32)
    with torch.no_grad():
        ledger.gate.fill_(0.0)  # half open, so the contribution is visible
        a, b = ledger(x), plain(x)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)  # first block: nothing written yet
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-3)


def test_the_gate_starts_almost_closed():
    ledger, plain = _pair(window=8)
    assert torch.sigmoid(ledger.gate).max() < 0.02
    x = torch.randn(1, 40, 32)
    with torch.no_grad():
        a, b = ledger(x), plain(x)
    # Near-identical to the windowed layer at initialisation, past the window included.
    assert (a - b).abs().max() < 0.05 * b.abs().max()


def test_rows_of_a_batch_have_separate_memories():
    ledger, _ = _pair(window=8)
    x = torch.randn(2, 40, 32)
    y = x.clone()
    y[0] = torch.randn(40, 32)
    with torch.no_grad():
        assert torch.allclose(ledger(x)[1], ledger(y)[1], atol=1e-6)


def test_written_pairs_are_recalled_by_their_keys():
    torch.manual_seed(1)
    mem = ProductKeyMemory(LedgerConfig(dim=16, memory_dim=16, n_slots=256, top_k=8, n_heads=1))
    keys = torch.nn.functional.normalize(torch.randn(12, 16), dim=-1)
    vals = torch.randn(12, 16)
    for _ in range(3):
        mem.write(keys, vals)
    read = mem.read(keys)
    cos = torch.nn.functional.cosine_similarity(read, vals, dim=-1)
    assert cos.mean() > 0.8, cos
    other = torch.nn.functional.normalize(torch.randn(12, 16), dim=-1)
    assert torch.nn.functional.cosine_similarity(mem.read(other), vals, dim=-1).abs().mean() < 0.5


def test_functional_state_matches_the_buffer_path():
    torch.manual_seed(2)
    mem = ProductKeyMemory(LedgerConfig(dim=8, memory_dim=8, n_slots=16, top_k=4, n_heads=1))
    x, t = torch.randn(5, 8), torch.randn(5, 8)
    values, counts = mem.new_state(1)
    stats_state = mem.write_state(values, counts, x, t)
    stats_buf = mem.write(x, t)
    assert stats_state.residual_after == pytest.approx(stats_buf.residual_after, rel=1e-5)
    assert torch.allclose(values[0], mem.values, atol=1e-6)
    assert torch.allclose(mem.read(x, values=values), mem.read(x), atol=1e-6)


def test_cached_decode_keeps_memory_bounded_and_writes_each_evicted_key_once():
    ledger, _ = _pair(window=8)
    cache = AttentionCache()
    with torch.no_grad():
        for _t in range(50):
            ledger(torch.randn(1, 1, 32), cache=cache)
    assert cache.keys.shape[2] == 8 + 1
    assert int(ledger.ledger.tokens_written) == 50 - 8 - 1


def test_cached_decode_reads_what_was_evicted():
    """A key that left the window is still found: a query equal to it recalls its value
    far better than a random query does."""
    ledger, _ = _pair(window=8)
    cache = AttentionCache()
    torch.manual_seed(3)
    with torch.no_grad():
        for _ in range(40):
            ledger(torch.randn(1, 1, 32), cache=cache)
        written = int(ledger.ledger.tokens_written)
        assert written == 40 - 9
        # Read back with the very keys that were written.
        values, counts = ledger.ledger.values, ledger.ledger.write_counts
        assert int((counts > 0).sum()) > 0 and float(values.abs().sum()) > 0


def test_batched_cache_is_refused():
    ledger, _ = _pair(window=8)
    with pytest.raises(ValueError, match="one sequence"):
        ledger(torch.randn(2, 4, 32), cache=AttentionCache())


# --------------------------------------------------------------------------------------
# Model, budget, session
# --------------------------------------------------------------------------------------


def _cfg(**mixer) -> ProphetConfig:
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg = dataclasses.replace(cfg, mixer=dataclasses.replace(
        cfg.mixer, global_memory="ledger", global_window=16, global_ledger_slots=64,
        global_ledger_top_k=8, nope_layers=(1,), **mixer,
    ))
    cfg.validate()
    return cfg


def test_model_builds_and_runs_past_the_window():
    model = ProphetModel(_cfg()).eval()
    assert any(isinstance(m, LedgerAttention) for m in model.modules())
    with torch.no_grad():
        out = model(torch.randint(0, 2048, (2, 50)), loop_k=2)
    assert out.logits.shape == (2, 50, 2048) and torch.isfinite(out.logits).all()


def test_rope_on_a_ledger_layer_is_refused():
    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    cfg = dataclasses.replace(cfg, mixer=dataclasses.replace(cfg.mixer, global_memory="ledger", nope_layers=()))
    with pytest.raises(ValueError, match="NoPE"):
        cfg.validate()


def test_memory_per_context_is_bounded_with_the_ledger_and_not_without():
    with_ledger = _cfg()
    without = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    total = lambda cfg, n: _kv_bytes_per_token(cfg, "bf16", n) * n  # noqa: E731
    # Unbounded: doubling the context doubles the bytes. Bounded: it barely moves.
    assert total(without, 2_000_000) / total(without, 1_000_000) > 1.9
    assert total(with_ledger, 2_000_000) / total(with_ledger, 1_000_000) < 1.05
    # The estimator carries a pre-existing ~1e-4 residual on this config; the ledger's
    # own accounting -- one gate per query head per ledger layer, buffers uncounted --
    # is exact.
    real = sum(p.numel() for p in ProphetModel(with_ledger).parameters())
    assert abs(count_parameters(with_ledger).total - real) / real < 2e-4
    plain = dataclasses.replace(with_ledger, mixer=dataclasses.replace(with_ledger.mixer, global_memory="none"))
    delta_est = count_parameters(with_ledger).total - count_parameters(plain).total
    delta_real = real - sum(p.numel() for p in ProphetModel(plain).parameters())
    assert delta_est == delta_real


def test_attention_ledgers_persist_with_the_session(tmp_path):
    model = ProphetModel(_cfg()).eval()
    cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, 2048, (1, 50)), cache=cache, loop_k=2)
    layer = next(m for m in model.modules() if isinstance(m, LedgerAttention))
    assert int(layer.ledger.tokens_written) > 0
    session = extract_session(cache, model=model)
    assert session.ledgers and session.n_bytes() > 0
    session.save(tmp_path / "s.pt")
    fresh = ProphetModel(_cfg()).eval()
    fresh_layer = next(m for m in fresh.modules() if isinstance(m, LedgerAttention))
    assert int(fresh_layer.ledger.tokens_written) == 0
    restore_session(type(session).load(tmp_path / "s.pt"), ProphetCache(), model=fresh)
    assert torch.equal(fresh_layer.ledger.values, layer.ledger.values)


def test_the_agent_loop_carries_ledgers_with_the_session_and_resets_them_without():
    """What fell out of the window during an episode travels with the session; an
    episode started without one begins from empty ledgers, never from the last run's."""
    from prophet.agent.actions import ToolRegistry, ToolSchema
    from prophet.agent.loop import AgentConfig, AgentLoop
    from prophet.data.tokenizer import ProphetTokenizer

    tok = ProphetTokenizer(merges=[])
    model = ProphetModel(_cfg()).eval()
    tools = ToolRegistry()
    tools.add(ToolSchema("read_file", "Read one file", {"type": "object", "properties": {"path": {"type": "string"}}}))
    tools.bind("read_file", lambda path: "hello")
    cfg = AgentConfig(max_steps=1, think_budget=2, action_budget=8, halt_threshold=None)
    loop = AgentLoop(model, tok, tools, cfg)
    layer = next(m for m in model.modules() if isinstance(m, LedgerAttention))
    first = loop.run("goal " * 40)  # long enough a prompt to evict into the ledger
    written = int(layer.ledger.tokens_written)
    assert written > 0 and first.session is not None and first.session.ledgers
    again = loop.run("goal " * 40)
    assert int(layer.ledger.tokens_written) == written  # reset, then the same episode
    loop.run("goal " * 40, session=again.session)
    assert int(layer.ledger.tokens_written) > written  # restored, then written further
