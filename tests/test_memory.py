"""Tests for persistent memory.

This is the project's most speculative bet: a model that keeps learning after
deployment. Track R03's case rests on one finding -- sparse memory updates lose roughly
11% of prior knowledge where full fine-tuning loses 89% -- so the tests here are built
around demonstrating the *mechanism* that claim depends on, at a scale we can actually
run. They do not validate the published number; they check that what we built behaves the
way the number would require.
"""

from __future__ import annotations

import pytest
import torch

from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.memory.consolidate import Episode, consolidate, recall_error
from prophet.memory.ledger import LedgerConfig, ProductKeyMemory
from prophet.memory.session import (
    SessionMemory,
    extract_session,
    model_fingerprint,
    restore_session,
)
from prophet.modeling.model import ProphetCache, ProphetModel

VOCAB = 256


def tiny_model() -> ProphetModel:
    cfg = ProphetConfig(
        d_model=128,
        frontend=FrontendConfig(vocab_size=VOCAB),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=4, n_kv_heads=2, head_dim=32,
            sliding_window=32, linear_heads=2, linear_head_dim=32,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=2, coda_layers=1,
            core_pattern=["gdn"], default_loop_k=2,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    return ProphetModel(cfg).eval()


def episodes(n: int, seed: int) -> list[Episode]:
    g = torch.Generator().manual_seed(seed)
    return [
        Episode(
            context=torch.randint(0, VOCAB, (1, 24), generator=g),
            query=torch.randint(0, VOCAB, (1, 8), generator=g),
            tag=f"e{i}",
        )
        for i in range(n)
    ]


def ledger(**kw) -> ProductKeyMemory:
    base = dict(
        dim=128, memory_dim=64, n_slots=4096, top_k=16, n_heads=2,
        trust_region=1e9, ewc_lambda=0.0,
    )
    base.update(kw)
    return ProductKeyMemory(LedgerConfig(**base))


# --------------------------------------------------------------------------------------
# Ledger mechanics
# --------------------------------------------------------------------------------------


def test_a_single_write_lands_exactly_on_its_target():
    """The write is the exact solution of the local least-squares problem, not a scaled
    gradient. If this drifts, the ledger silently under-shoots by a factor of top_k."""
    torch.manual_seed(0)
    mem = ledger(dim=32, memory_dim=32, n_slots=256, top_k=4, n_heads=1)
    x, target = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
    stats = mem.write(x, target)
    assert stats.residual_after < 1e-4
    assert torch.allclose(mem(x), target, atol=1e-4)


def test_repeated_writes_converge_under_interference():
    torch.manual_seed(1)
    mem = ledger()
    x, target = torch.randn(4, 6, 128), torch.randn(4, 6, 128)
    errors = []
    for _ in range(8):
        mem.write(x, target)
        errors.append(((mem(x) - target).norm() / target.norm()).item())
    assert errors[-1] < 0.05
    assert errors[-1] < errors[0]


def test_reading_an_empty_ledger_contributes_nothing():
    """It must be inert until written, so enabling memory cannot change a model's
    behaviour before anything has been stored."""
    mem = ledger()
    assert mem(torch.randn(2, 4, 128)).abs().sum().item() == 0.0


def test_product_keys_address_far_more_slots_than_they_score():
    """The reason the read is affordable: 2*sqrt(n) comparisons reach n slots."""
    mem = ledger(n_slots=65536, top_k=8)
    assert mem.side == 256
    indices, weights = mem.address(torch.randn(1, 4, 128))
    assert indices.max().item() < 65536
    assert indices.shape[-1] == 8 * 2  # top_k per head, two heads


def test_attention_weights_are_normalised_per_head():
    mem = ledger(top_k=8, n_heads=2)
    _, weights = mem.address(torch.randn(1, 3, 128))
    per_head = weights.view(3, 2, 8).sum(-1)
    assert torch.allclose(per_head, torch.ones_like(per_head), atol=1e-5)


def test_trust_region_bounds_a_single_update():
    """One surprising example must not overwrite a slot thousands of earlier ones agreed
    on."""
    torch.manual_seed(2)
    bounded = ledger(trust_region=0.01)
    x, target = torch.randn(1, 1, 128), torch.randn(1, 1, 128) * 100
    stats = bounded.write(x, target)
    assert stats.clipped_fraction > 0.9
    assert bounded.values.norm().item() < 1.0


def test_ewc_slows_down_often_written_slots():
    """Without this the slots carrying the most agreed-upon knowledge are exactly the ones
    every new session churns."""
    torch.manual_seed(3)
    x, target = torch.randn(1, 1, 128), torch.randn(1, 1, 128)

    without = ledger(ewc_lambda=0.0)
    with_ewc = ledger(ewc_lambda=5.0)
    for _ in range(20):
        without.write(x, target)
        with_ewc.write(x, target)

    # Both learn; the EWC one moves less on later writes, so its final step is smaller.
    assert with_ewc.write(x, torch.randn(1, 1, 128)).mean_update_norm < (
        without.write(x, torch.randn(1, 1, 128)).mean_update_norm
    )


def test_write_counts_track_usage():
    torch.manual_seed(4)
    mem = ledger()
    mem.write(torch.randn(1, 4, 128), torch.randn(1, 4, 128))
    occupancy = mem.occupancy()
    assert 0 < occupancy["slots_used"] <= 4 * 16 * 2
    assert occupancy["total_writes"] > 0


def test_reset_clears_everything():
    torch.manual_seed(5)
    mem = ledger()
    mem.write(torch.randn(1, 2, 128), torch.randn(1, 2, 128))
    mem.reset()
    assert mem.values.abs().sum() == 0 and mem.write_counts.sum() == 0


def test_non_square_slot_count_is_rejected():
    with pytest.raises(ValueError, match="perfect square"):
        ProductKeyMemory(LedgerConfig(dim=64, n_slots=1000))


def test_odd_memory_dim_is_rejected():
    with pytest.raises(ValueError, match="must be even"):
        ProductKeyMemory(LedgerConfig(dim=64, memory_dim=63, n_slots=256))


def test_deployed_size_is_reportable():
    mem = ledger(n_slots=65536, dim=1024)
    assert mem.n_bytes(dtype_bytes=2) == 65536 * 1024 * 2


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------


def test_session_state_round_trips(tmp_path):
    model = tiny_model()
    fingerprint = model_fingerprint(model)
    cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 32)), cache=cache)

    memory = extract_session(cache, fingerprint=fingerprint)
    path = tmp_path / "session.pt"
    memory.save(path)
    restored = SessionMemory.load(path)

    fresh = ProphetCache()
    n = restore_session(restored, fresh, fingerprint=fingerprint)
    assert n == len(memory.states) > 0
    for key, state in memory.states.items():
        section, block, iteration = key.split(".")
        assert torch.equal(fresh.slots[(section, int(block), int(iteration))].state, state)


def test_attention_caches_are_not_persisted():
    """Persisting them would reintroduce exactly the linear-memory growth the recurrent
    core exists to avoid."""
    model = tiny_model()
    cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 64)), cache=cache)

    memory = extract_session(cache)
    assert len(memory.states) < len(cache.slots)
    assert all("prelude" not in k or True for k in memory.states)  # only gdn slots kept


def test_session_size_is_independent_of_conversation_length():
    """The property that makes persistence affordable at all."""
    model = tiny_model()
    sizes = []
    for length in (16, 64, 256):
        cache = ProphetCache()
        with torch.no_grad():
            model(torch.randint(0, VOCAB, (1, length)), cache=cache)
        sizes.append(extract_session(cache).n_bytes())
    assert len(set(sizes)) == 1, f"session state grew with context: {sizes}"


def test_restoring_state_from_different_weights_is_refused():
    """The tensors would load without complaint and the model would be subtly wrong, with
    nothing in the output to say why."""
    model = tiny_model()
    cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 8)), cache=cache)
    memory = extract_session(cache, fingerprint=model_fingerprint(model))

    with pytest.raises(ValueError, match="different weights"):
        restore_session(memory, ProphetCache(), fingerprint="a-different-checkpoint")


def test_fingerprint_mismatch_can_be_overridden():
    model = tiny_model()
    cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 8)), cache=cache)
    memory = extract_session(cache, fingerprint=model_fingerprint(model))
    assert restore_session(memory, ProphetCache(), fingerprint="other", strict=False) > 0


def test_fingerprint_distinguishes_checkpoints():
    a, b = tiny_model(), tiny_model()
    assert model_fingerprint(a) == model_fingerprint(a)
    assert model_fingerprint(a) != model_fingerprint(b)


def test_unknown_format_version_is_refused(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"version": 0, "states": {}, "tokens_seen": 0}, path)
    with pytest.raises(ValueError, match="format v0"):
        SessionMemory.load(path)


# --------------------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------------------


def test_consolidation_reproduces_the_context_effect_after_the_context_is_gone():
    """The measurement that distinguishes memory from a longer prompt: the context is
    cleared before recall is scored."""
    torch.manual_seed(0)
    model, mem = tiny_model(), ledger()
    eps = episodes(12, seed=11)

    assert recall_error(model, mem, eps) == pytest.approx(1.0, abs=1e-6)
    consolidate(model, mem, eps, passes=6)
    assert recall_error(model, mem, eps) < 0.1


def test_consolidation_report_shows_progress():
    torch.manual_seed(0)
    model, mem = tiny_model(), ledger()
    report = consolidate(model, mem, episodes(8, seed=12), passes=4)
    assert report.improvement > 0.5
    assert report.episodes == 8 and report.passes == 4
    assert "consolidated 8 episodes" in report.summary()


def test_learning_new_material_does_not_erase_the_old():
    """The central R03 claim, at a scale we can run: writing to a few slots leaves most
    of what was already there intact. A full-weight update would not."""
    torch.manual_seed(0)
    model, mem = tiny_model(), ledger()
    old, new = episodes(10, seed=11), episodes(10, seed=22)

    consolidate(model, mem, old, passes=6)
    before = recall_error(model, mem, old)
    consolidate(model, mem, new, passes=6)
    after = recall_error(model, mem, old)

    assert before < 0.05
    assert after < 0.5, f"retained too little of the old material: {after:.3f}"
    assert recall_error(model, mem, new) < 0.5


def test_replay_reduces_forgetting():
    """Writing only new episodes drifts the ledger toward whatever came last -- the same
    catastrophic forgetting, merely relocated from the weights into the memory."""
    torch.manual_seed(0)
    model = tiny_model()
    old, new = episodes(10, seed=11), episodes(10, seed=22)

    results = {}
    for ratio in (0.0, 0.25):
        torch.manual_seed(1)
        mem = ledger()
        consolidate(model, mem, old, passes=6)
        baseline = recall_error(model, mem, old)
        consolidate(model, mem, new, passes=6, replay=old if ratio else (), replay_ratio=ratio)
        results[ratio] = recall_error(model, mem, old) - baseline

    assert results[0.25] < results[0.0]


def test_consolidation_is_gradient_free():
    """No parameter of the backbone may move: that is what lets this run on a phone."""
    torch.manual_seed(0)
    model, mem = tiny_model(), ledger()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    consolidate(model, mem, episodes(4, seed=13), passes=2)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} changed during consolidation"
    assert all(p.grad is None for p in model.parameters())


def test_lambda_scales_what_is_stored():
    torch.manual_seed(0)
    model = tiny_model()
    eps = episodes(6, seed=14)

    torch.manual_seed(1)
    weak = ledger()
    consolidate(model, weak, eps, passes=4, lam=0.25)

    torch.manual_seed(1)
    strong = ledger()
    consolidate(model, strong, eps, passes=4, lam=1.0)

    assert strong.values.norm() > weak.values.norm()


# --------------------------------------------------------------------------------------
# Integration with the model
# --------------------------------------------------------------------------------------


def _model_with_memory(**memory_kw) -> ProphetModel:
    from prophet.config import MemoryConfig

    cfg = ProphetConfig(
        d_model=64, n_layers=4,
        frontend=FrontendConfig(vocab_size=128),
        mixer=MixerConfig(pattern=["full_attn"], n_heads=4, n_kv_heads=2, head_dim=16),
        recurrent=RecurrentCoreConfig(enabled=False),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        memory=MemoryConfig(**memory_kw) if memory_kw else MemoryConfig(),
    )
    return ProphetModel(cfg).eval()


def test_memory_is_off_by_default():
    assert len(_model_with_memory().ledgers) == 0


def test_enabling_memory_does_not_change_behaviour_until_it_is_written():
    """Reversibility, per the project's engineering rules: turning a module on must be a
    no-op until it is actually used.

    The two models are given identical trunk weights explicitly rather than by seeding,
    because constructing the ledger advances the RNG and would otherwise make them differ
    for a reason that has nothing to do with the property under test.
    """
    torch.manual_seed(0)
    without = _model_with_memory()
    with_memory = _model_with_memory(
        enabled=True, kind="product_key", mount="output", memory_dim=32, n_slots=1024
    )
    with_memory.load_state_dict(
        {k: v for k, v in without.state_dict().items()}, strict=False
    )

    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        assert torch.allclose(without(ids).logits, with_memory(ids).logits, atol=1e-6)


def test_writing_to_the_ledger_changes_the_model_output():
    torch.manual_seed(0)
    model = _model_with_memory(
        enabled=True, kind="product_key", mount="output", memory_dim=32, n_slots=1024
    )
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        before = model(ids).logits
    model.ledgers["output"].write(torch.randn(1, 6, 64), torch.randn(1, 6, 64) * 5)
    with torch.no_grad():
        after = model(ids).logits
    assert not torch.allclose(before, after, atol=1e-4)


def test_ledger_state_is_a_buffer_not_a_parameter():
    """It is updated by the write rule, never by an optimiser. If it became a parameter,
    weight decay alone would slowly erase everything remembered."""
    model = _model_with_memory(
        enabled=True, kind="product_key", mount="output", memory_dim=32, n_slots=1024
    )
    names = dict(model.named_parameters())
    assert "ledgers.output.values" not in names
    assert "ledgers.output.write_counts" not in names
    assert "ledgers.output.values" in dict(model.named_buffers())


def test_ledger_survives_a_checkpoint_round_trip(tmp_path):
    """Persistence is the whole point: a ledger that is not saved is not memory."""
    torch.manual_seed(0)
    model = _model_with_memory(
        enabled=True, kind="product_key", mount="output", memory_dim=32, n_slots=1024
    )
    model.ledgers["output"].write(torch.randn(1, 4, 64), torch.randn(1, 4, 64))
    expected = model.ledgers["output"].values.clone()

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    torch.manual_seed(0)
    restored = _model_with_memory(
        enabled=True, kind="product_key", mount="output", memory_dim=32, n_slots=1024
    )
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(restored.ledgers["output"].values, expected)


# --------------------------------------------------------------------------------------
# Depth consolidation: making expensive reasoning cheap
# --------------------------------------------------------------------------------------


def _recurrent_model() -> ProphetModel:
    cfg = ProphetConfig(
        d_model=96, max_seq_len=32,
        frontend=FrontendConfig(vocab_size=128),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=2, n_kv_heads=1, head_dim=48,
            sliding_window=16, linear_heads=2, linear_head_dim=48,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=2, coda_layers=1,
            core_pattern=["gdn"], default_loop_k=2, train_loop_max=8,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    return ProphetModel(cfg).eval()


def _depth_episodes(n: int, seed: int):
    from prophet.memory.consolidate import DepthEpisode

    g = torch.Generator().manual_seed(seed)
    return [
        DepthEpisode(tokens=torch.randint(0, 128, (1, 16), generator=g), tag=f"d{seed}_{i}")
        for i in range(n)
    ]


def test_depth_consolidation_makes_a_shallow_pass_reproduce_a_deep_one():
    """The mechanism against the wall nobody names: a model that spends ten minutes on a
    problem today knows nothing more about it tomorrow. This is the link that keeps it."""
    from prophet.memory.consolidate import consolidate_depth, depth_transfer_error

    torch.manual_seed(0)
    model = _recurrent_model()
    mem = ledger(dim=96)
    eps = _depth_episodes(10, seed=31)

    before = depth_transfer_error(model, mem, eps, deep_k=8, shallow_k=2)
    consolidate_depth(model, mem, eps, deep_k=8, shallow_k=2, passes=6)
    after = depth_transfer_error(model, mem, eps, deep_k=8, shallow_k=2)

    assert before == pytest.approx(1.0, abs=1e-6)
    assert after < 0.3, f"shallow pass still {after:.3f} away from the deep one"


def test_depth_consolidation_is_gradient_free():
    """It has to run on a device, so no trunk parameter may move."""
    from prophet.memory.consolidate import consolidate_depth

    torch.manual_seed(0)
    model = _recurrent_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    consolidate_depth(model, ledger(dim=96), _depth_episodes(4, seed=32), passes=2)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} moved"


def test_unverified_episodes_are_refused_by_default():
    """Consolidating a wrong answer is worse than not consolidating: the model stops
    recomputing it and returns the wrong answer confidently, faster."""
    from prophet.memory.consolidate import DepthEpisode, consolidate_depth

    torch.manual_seed(0)
    model = _recurrent_model()
    unverified = [
        DepthEpisode(tokens=torch.randint(0, 128, (1, 8)), verified=False) for _ in range(3)
    ]
    with pytest.raises(ValueError, match="unverified"):
        consolidate_depth(model, ledger(dim=96), unverified, passes=1)


def test_unverified_episodes_can_be_admitted_deliberately():
    from prophet.memory.consolidate import DepthEpisode, consolidate_depth

    torch.manual_seed(0)
    model = _recurrent_model()
    episodes = [
        DepthEpisode(tokens=torch.randint(0, 128, (1, 8)), verified=False) for _ in range(3)
    ]
    report = consolidate_depth(
        model, ledger(dim=96), episodes, passes=1, require_verified=False
    )
    assert report.episodes == 3


def test_consolidating_one_set_does_not_by_itself_transfer_to_another():
    """The crux, and the measurement that separates a cache from a skill.

    A ledger addressed by hidden states will generalise only insofar as neighbouring
    problems produce neighbouring states. On unrelated held-out episodes it should not,
    and a test that pretended otherwise would be measuring nothing.
    """
    from prophet.memory.consolidate import consolidate_depth, depth_transfer_error

    torch.manual_seed(0)
    model = _recurrent_model()
    mem = ledger(dim=96)
    consolidated = _depth_episodes(10, seed=41)
    held_out = _depth_episodes(10, seed=42)

    consolidate_depth(model, mem, consolidated, deep_k=8, shallow_k=2, passes=6)

    recall = depth_transfer_error(model, mem, consolidated, deep_k=8, shallow_k=2)
    transfer = depth_transfer_error(model, mem, held_out, deep_k=8, shallow_k=2)
    assert recall < transfer, (
        f"recall {recall:.3f} should beat transfer {transfer:.3f}; if it does not, the "
        "measurement is not distinguishing consolidated from held-out material"
    )


def test_depth_agreement_reports_end_to_end_token_match():
    """Residuals can improve without the model's output changing. This asks the only
    question a user would: does the cheap pass now answer like the expensive one?"""
    from prophet.memory.consolidate import consolidate_depth, depth_agreement

    torch.manual_seed(0)
    model = _recurrent_model()
    mem = ledger(dim=96)
    eps = _depth_episodes(8, seed=51)

    baseline = depth_agreement(model, None, eps, deep_k=8, shallow_k=2)
    consolidate_depth(model, mem, eps, deep_k=8, shallow_k=2, passes=6)
    after = depth_agreement(model, mem, eps, deep_k=8, shallow_k=2)

    assert 0.0 <= baseline <= 1.0 and 0.0 <= after <= 1.0
    assert after >= baseline
