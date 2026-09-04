"""Tests for the configuration schema and the budget calculator.

These deliberately avoid PyTorch: they must run in any environment, because their job
is to catch a broken configuration *before* it consumes an A100-hour.
"""

from __future__ import annotations

import pytest

from prophet.budget import (
    DEVICES,
    allocation_warnings,
    count_parameters,
    inference_profile,
    report,
    tokens_affordable,
    training_memory,
)
from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    MemoryConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)


def _hybrid(**kw) -> ProphetConfig:
    base = dict(
        name="test-hybrid",
        d_model=1024,
        frontend=FrontendConfig(mode="bpe", vocab_size=32768),
        mixer=MixerConfig(pattern=["gdn", "gdn", "gdn", "full_attn"], n_heads=16, n_kv_heads=4),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=4, coda_layers=2, default_loop_k=4
        ),
    )
    base.update(kw)
    return ProphetConfig(**base)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def test_baseline_config_validates():
    cfg = ProphetConfig()
    cfg.validate()
    assert cfg.effective_depth() == cfg.n_layers
    assert cfg.parameterised_depth() == cfg.n_layers


def test_recurrence_multiplies_depth_without_multiplying_weights():
    """The central architectural bet, asserted numerically."""
    cfg = _hybrid()
    assert cfg.parameterised_depth() == 8
    assert cfg.effective_depth(loop_k=1) == 8
    assert cfg.effective_depth(loop_k=4) == 20
    assert cfg.effective_depth(loop_k=16) == 68

    # Parameter count must be independent of the loop depth.
    assert count_parameters(cfg, loop_k=1).total == count_parameters(cfg, loop_k=16).total


def test_mixer_pattern_cycles():
    cfg = _hybrid()
    kinds = [cfg.layer_mixer(i) for i in range(8)]
    assert kinds == ["gdn", "gdn", "gdn", "full_attn"] * 2


def test_moe_layer_placement_skips_leading_dense_blocks():
    cfg = _hybrid(
        ffn=FeedForwardConfig(kind="moe", n_experts=32, n_experts_per_token=4,
                              moe_first_dense_layers=2, moe_layer_stride=2)
    )
    assert [cfg.layer_is_moe(i) for i in range(8)] == [
        False, False, True, False, True, False, True, False,
    ]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"mixer": MixerConfig(n_heads=16, n_kv_heads=5)}, "divisible"),
        ({"mixer": MixerConfig(pattern=[])}, "at least one"),
        (
            {"ffn": FeedForwardConfig(kind="moe", n_experts=8, n_experts_per_token=16)},
            "exceeds",
        ),
        (
            {"memory": MemoryConfig(enabled=True, kind="fast_weight", layers=(999,))},
            "outside the trunk depth",
        ),
        ({"memory": MemoryConfig(enabled=True, kind="none")}, "memory.kind"),
        (
            {"recurrent": RecurrentCoreConfig(enabled=True, train_loop_min=8, train_loop_max=2)},
            "train_loop_min",
        ),
        (
            {"recurrent": RecurrentCoreConfig(enabled=True, halting_loss_weight=-0.1)},
            "halting_loss_weight",
        ),
        (
            {"recurrent": RecurrentCoreConfig(enabled=True, halting_target_steps=1.0)},
            "halting_target_steps",
        ),
        (
            {"recurrent": RecurrentCoreConfig(enabled=True, halting_target_steps=float("nan"))},
            "halting_target_steps",
        ),
        (
            {"recurrent": RecurrentCoreConfig(enabled=True, halting_loss_weight=float("inf"))},
            "halting_loss_weight",
        ),
        (
            {"recurrent": RecurrentCoreConfig(enabled=True, default_loop_k=0)},
            "default_loop_k",
        ),
    ],
)
def test_validate_rejects_impossible_configs(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        ProphetConfig(**kwargs).validate()


def test_json_roundtrip_preserves_tuples(tmp_path):
    cfg = _hybrid(memory=MemoryConfig(enabled=True, kind="fast_weight", layers=(2, 5)))
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    restored = ProphetConfig.from_json(path)
    assert restored == cfg
    assert isinstance(restored.memory.layers, tuple)


def test_from_dict_tolerates_missing_keys():
    cfg = ProphetConfig.from_dict({"d_model": 512, "mixer": {"n_heads": 8}})
    assert cfg.d_model == 512
    assert cfg.mixer.n_heads == 8
    assert cfg.mixer.n_kv_heads == ProphetConfig().mixer.n_kv_heads


# --------------------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------------------


def test_moe_decouples_capacity_from_per_token_cost():
    """The reason for sparsity: total params grow, active params do not."""
    dense = _hybrid(ffn=FeedForwardConfig(kind="dense", hidden_mult=4.0))
    moe = _hybrid(
        ffn=FeedForwardConfig(
            kind="moe", n_experts=64, n_experts_per_token=6, n_shared_experts=2,
            expert_hidden_mult=0.5,
        )
    )
    p_dense = count_parameters(dense)
    p_moe = count_parameters(moe)

    assert p_moe.total > 3 * p_dense.total
    assert p_moe.active_per_token < 1.5 * p_dense.active_per_token
    assert p_moe.total / p_moe.active_per_token > 3.0


def test_byte_frontend_removes_the_vocabulary_tax():
    bpe = ProphetConfig(d_model=1024, frontend=FrontendConfig(mode="bpe", vocab_size=131072))
    byte = ProphetConfig(
        d_model=1024,
        frontend=FrontendConfig(
            mode="byte_patch", local_dim=256, hash_ngram_sizes=(3, 4, 5),
            hash_ngram_buckets=16384,
        ),
    )
    assert count_parameters(byte).embedding < count_parameters(bpe).embedding


def test_mla_shrinks_the_kv_cache():
    plain = _hybrid()
    mla = _hybrid(
        mixer=MixerConfig(
            pattern=["gdn", "gdn", "gdn", "full_attn"], n_heads=16, n_kv_heads=4,
            kv_compression="mla", kv_lora_rank=256,
        )
    )
    a = inference_profile(plain, context_len=131072)
    b = inference_profile(mla, context_len=131072)
    assert b.kv_state_gb < a.kv_state_gb


def test_allocation_warning_fires_on_oversized_hash_tables():
    cfg = ProphetConfig(
        d_model=512,
        frontend=FrontendConfig(
            mode="byte_patch", local_dim=1024, hash_ngram_sizes=(3, 4, 5, 6, 7, 8),
            hash_ngram_buckets=262144,
        ),
    )
    warnings = allocation_warnings(cfg)
    assert any("hash n-gram" in w for w in warnings)


def test_no_allocation_warnings_on_a_sane_config():
    """A well-formed configuration should be silent -- allocation *and* design checks.

    ``_hybrid`` deliberately omits ``core_pattern``, which puts attention inside the loop,
    so this test builds its stack explicitly rather than reusing the helper.
    """
    cfg = ProphetConfig(
        d_model=1536,
        frontend=FrontendConfig(mode="bpe", vocab_size=32768),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=12, n_kv_heads=3, head_dim=128,
            nope_layers=(1,), linear_beta_max=2.0,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=4, core_layers=4, coda_layers=4,
            core_pattern=["gdn"], default_loop_k=4,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=4.0),
    )
    assert allocation_warnings(cfg) == []


# --------------------------------------------------------------------------------------
# Memory and throughput
# --------------------------------------------------------------------------------------


def test_truncated_backprop_bounds_activation_memory():
    """Deep recurrence must not cost deep activation memory."""
    shallow = _hybrid(
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=4, coda_layers=2,
            truncated_backprop_steps=2,
        )
    )
    deep = _hybrid(
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=4, coda_layers=2,
            truncated_backprop_steps=8,
        )
    )
    a = training_memory(shallow, loop_k=8).activations_gb
    b = training_memory(deep, loop_k=8).activations_gb
    assert b > a


def test_eight_bit_optimizer_cuts_optimizer_state_fourfold():
    cfg = _hybrid()
    full = training_memory(cfg, optimizer_bytes_per_param=8.0)
    eight = training_memory(cfg, optimizer_bytes_per_param=2.0)
    assert full.optimizer_gb == pytest.approx(4 * eight.optimizer_gb, rel=1e-6)
    assert eight.total_gb < full.total_gb


def test_recurrent_depth_costs_bandwidth_not_capacity():
    """Looping the core more times must not change what has to fit in memory."""
    cfg = _hybrid()
    shallow = inference_profile(cfg, device="rtx5090", loop_k=1)
    deep = inference_profile(cfg, device="rtx5090", loop_k=8)
    assert deep.weights_gb == pytest.approx(shallow.weights_gb)
    assert deep.decode_tok_s < shallow.decode_tok_s
    assert deep.flops_per_token > shallow.flops_per_token


def test_recurrent_layers_give_context_independent_state():
    """A pure-recurrent stack must not grow its state with context length."""
    cfg = _hybrid(mixer=MixerConfig(pattern=["gdn"], n_heads=16, n_kv_heads=4))
    short = inference_profile(cfg, context_len=4096)
    long = inference_profile(cfg, context_len=131072)
    assert long.kv_state_gb == pytest.approx(short.kv_state_gb, rel=1e-6)


def test_full_attention_state_grows_with_context():
    cfg = _hybrid(mixer=MixerConfig(pattern=["full_attn"], n_heads=16, n_kv_heads=4))
    short = inference_profile(cfg, context_len=4096)
    long = inference_profile(cfg, context_len=131072)
    assert long.kv_state_gb > 20 * short.kv_state_gb


def test_device_registry_is_ordered_by_bandwidth():
    """Sanity-check the hardware table the whole project reasons against."""
    assert DEVICES["a100_80gb"].bandwidth_gb_s > DEVICES["rtx5090"].bandwidth_gb_s
    assert DEVICES["rtx5090"].bandwidth_gb_s > DEVICES["mac_studio_ultra"].bandwidth_gb_s
    assert DEVICES["mac_studio_ultra"].bandwidth_gb_s > DEVICES["iphone17pro"].bandwidth_gb_s
    assert DEVICES["iphone17pro"].memory_gb < DEVICES["rtx5090"].memory_gb


def test_token_budget_scales_inversely_with_active_params():
    small = _hybrid(d_model=768)
    large = _hybrid(d_model=2048)
    assert tokens_affordable(small)["tokens"] > tokens_affordable(large)["tokens"]


def test_report_renders():
    text = report(_hybrid(), a100_hours=300)
    assert "Budget report" in text
    assert "active / token" in text


# --------------------------------------------------------------------------------------
# Design invariants
# --------------------------------------------------------------------------------------


def test_attention_inside_the_looped_core_is_flagged():
    """The mistake that actually shipped.

    ``configs/prophet_500m_probe.json`` omitted ``core_pattern``, so the global attention
    pattern applied inside the loop and the only full-attention layer ended up there --
    duplicating its KV cache per iteration, which is precisely the invariant decision D1
    exists to protect. It validated, it trained, and it would have confounded every
    ablation built on it.
    """
    cfg = ProphetConfig(
        d_model=1024,
        mixer=MixerConfig(pattern=["gdn", "gdn", "gdn", "full_attn"], n_heads=8, n_kv_heads=2),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=4, coda_layers=2, default_loop_k=4
        ),
    )
    warnings = cfg.design_warnings()
    assert any("inside the looped core" in w for w in warnings)
    assert any("x4 at the default depth" in w for w in warnings)


def test_a_correct_stack_trips_no_invariant():
    cfg = ProphetConfig(
        d_model=1024,
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=8, n_kv_heads=2,
            nope_layers=(1,), linear_beta_max=2.0,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=4, coda_layers=2,
            core_pattern=["gdn"], default_loop_k=4,
        ),
    )
    assert cfg.design_warnings() == []


def test_a_stack_with_no_attention_outside_the_core_is_flagged():
    cfg = ProphetConfig(
        d_model=1024,
        mixer=MixerConfig(pattern=["gdn"], n_heads=8, n_kv_heads=2, nope_layers=(1,)),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=2, coda_layers=2,
            core_pattern=["gdn"],
        ),
    )
    assert any("no attention in the prelude or coda" in w for w in cfg.design_warnings())


def test_missing_nope_layers_is_flagged():
    cfg = ProphetConfig(
        d_model=1024,
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=8, n_kv_heads=2, nope_layers=(),
            linear_beta_max=2.0,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=2, coda_layers=2,
            core_pattern=["gdn"],
        ),
    )
    assert any("nope_layers is empty" in w for w in cfg.design_warnings())


def test_a_parity_blind_write_strength_is_flagged():
    """With beta bounded by 1, every state-transition eigenvalue stays positive and no
    product of them can flip sign -- so parity is out of reach. It costs one
    multiplication to fix, and nothing in a loss curve would reveal it."""
    cfg = ProphetConfig(
        d_model=1024,
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=8, n_kv_heads=2, nope_layers=(1,),
            linear_beta_max=1.0,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=2, core_layers=2, coda_layers=2,
            core_pattern=["gdn"],
        ),
    )
    assert any("cannot express parity" in w for w in cfg.design_warnings())


def test_shipped_configs_satisfy_every_invariant():
    """The generated configurations are the ones ablations will run on."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    shipped = sorted(root.glob("prophet_*.json"))
    assert shipped, "no shipped configurations found"

    for path in shipped:
        cfg = ProphetConfig.from_json(path)
        cfg.validate()
        assert cfg.design_warnings() == [], f"{path.name}: {cfg.design_warnings()}"


def test_shipped_configs_keep_attention_cache_independent_of_depth():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    for path in sorted(root.glob("prophet_*.json")):
        cfg = ProphetConfig.from_json(path)
        counts = {}
        for k in (1, 8):
            counts[k] = sum(
                1 for *_, kind in cfg.cache_slots(loop_k=k)
                if kind in ("full_attn", "swa")
            )
        assert counts[1] == counts[8], f"{path.name}: attention slots scale with k"
