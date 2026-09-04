"""Correctness tests for the Prophet modules.

The tests that matter most here are *equivalence* tests: a cache is only useful if
incremental decoding produces exactly what a full forward pass would have. That class of
bug is silent — the model still generates fluent text — so it has to be caught by
assertion rather than by inspection.
"""

from __future__ import annotations

import pytest
import torch

from prophet.budget import count_parameters
from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MemoryConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.modeling.layers import (
    AttentionCache,
    CausalSelfAttention,
    GatedDeltaNet,
    RecurrentState,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary,
)
from prophet.modeling.model import ProphetCache, ProphetModel
from prophet.modeling.moe import SparseMoE

torch.manual_seed(0)


def tiny_config(**kw) -> ProphetConfig:
    base = dict(
        name="tiny",
        d_model=128,
        max_seq_len=128,
        frontend=FrontendConfig(mode="bpe", vocab_size=256),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"],
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            sliding_window=8,
            attention_sink_tokens=2,
            linear_heads=2,
            linear_head_dim=16,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True,
            prelude_layers=1,
            core_layers=2,
            coda_layers=1,
            core_pattern=["gdn"],
            default_loop_k=3,
            truncated_backprop_steps=2,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    base.update(kw)
    return ProphetConfig(**base)


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------


def test_rmsnorm_normalises_to_unit_rms():
    norm = RMSNorm(64, elementwise_affine=False)
    x = torch.randn(4, 7, 64) * 17.0
    rms = norm(x).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_is_scale_invariant():
    """The property that makes it useful: activation magnitude cannot blow up the block."""
    norm = RMSNorm(32, elementwise_affine=False)
    x = torch.randn(2, 3, 32)
    assert torch.allclose(norm(x), norm(x * 1000.0), atol=1e-4)


def test_rotary_preserves_norm_and_relative_position():
    rope = RotaryEmbedding(64, theta=10000.0)
    cos, sin = rope(torch.arange(16).unsqueeze(0))
    q = torch.randn(1, 4, 16, 64)
    rotated = apply_rotary(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)

    # A rotary dot product must depend only on the offset between positions.
    a = torch.randn(1, 1, 1, 64)
    ra_2 = apply_rotary(a, cos[:, 2:3], sin[:, 2:3])
    ra_5 = apply_rotary(a, cos[:, 5:6], sin[:, 5:6])
    rb_7 = apply_rotary(a, cos[:, 7:8], sin[:, 7:8])
    rb_10 = apply_rotary(a, cos[:, 10:11], sin[:, 10:11])
    assert torch.allclose((ra_2 * rb_7).sum(), (ra_5 * rb_10).sum(), atol=1e-4)


def test_rotary_rejects_indivisible_position_dims():
    with pytest.raises(ValueError, match="divisible"):
        RotaryEmbedding(64, position_dims=3)


def test_rotary_multi_dim_reserves_head_space_for_images():
    """The R12 hook: 2-D/3-D positions must work without changing trunk weights."""
    rope = RotaryEmbedding(96, position_dims=3)
    cos, sin = rope(torch.zeros(2, 5, 3))
    assert cos.shape == (2, 5, 96)


# --------------------------------------------------------------------------------------
# Attention: prefill/decode equivalence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("window,sinks", [(None, 0), (8, 0), (8, 2)])
def test_incremental_decode_matches_full_forward(window, sinks):
    """The single most important correctness property of the cache."""
    torch.manual_seed(1)
    attn = CausalSelfAttention(
        64, n_heads=4, n_kv_heads=2, head_dim=16, window=window, sink_tokens=sinks
    ).eval()
    rope = RotaryEmbedding(16, theta=10000.0)
    x = torch.randn(2, 20, 64)

    cos, sin = rope(torch.arange(20).unsqueeze(0).expand(2, 20))
    with torch.no_grad():
        full = attn(x, cos=cos, sin=sin)

        cache = AttentionCache()
        steps = []
        for t in range(20):
            c, s = rope(torch.full((2, 1), t))
            steps.append(attn(x[:, t : t + 1], cos=c, sin=s, cache=cache))
        incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-4), (
        f"max diff {(full - incremental).abs().max():.2e}"
    )


def test_prefill_then_decode_matches_full_forward():
    """Mixed path: a long prefill followed by single-token steps."""
    torch.manual_seed(2)
    attn = CausalSelfAttention(64, n_heads=4, n_kv_heads=4, head_dim=16).eval()
    rope = RotaryEmbedding(16, theta=10000.0)
    x = torch.randn(1, 12, 64)
    cos, sin = rope(torch.arange(12).unsqueeze(0))

    with torch.no_grad():
        full = attn(x, cos=cos, sin=sin)
        cache = AttentionCache()
        head = attn(x[:, :8], cos=cos[:, :8], sin=sin[:, :8], cache=cache)
        tail = [
            attn(x[:, t : t + 1], cos=cos[:, t : t + 1], sin=sin[:, t : t + 1], cache=cache)
            for t in range(8, 12)
        ]
    assert torch.allclose(full, torch.cat([head, *tail], dim=1), atol=1e-4)


def test_sliding_window_actually_restricts_attention():
    """A windowed layer must be blind to tokens beyond the window."""
    torch.manual_seed(3)
    attn = CausalSelfAttention(
        64, n_heads=4, n_kv_heads=4, head_dim=16, window=4, sink_tokens=0
    ).eval()
    x = torch.randn(1, 16, 64)
    perturbed = x.clone()
    perturbed[:, 0] += 50.0  # far outside the window of the last position

    with torch.no_grad():
        a = attn(x)[:, -1]
        b = attn(perturbed)[:, -1]
    assert torch.allclose(a, b, atol=1e-4)


def test_attention_sinks_remain_visible_beyond_the_window():
    torch.manual_seed(4)
    attn = CausalSelfAttention(
        64, n_heads=4, n_kv_heads=4, head_dim=16, window=4, sink_tokens=2
    ).eval()
    x = torch.randn(1, 16, 64)
    perturbed = x.clone()
    perturbed[:, 0] += 50.0  # a sink token: must still influence the last position

    with torch.no_grad():
        a = attn(x)[:, -1]
        b = attn(perturbed)[:, -1]
    assert not torch.allclose(a, b, atol=1e-3)


def test_windowed_cache_memory_is_bounded():
    attn = CausalSelfAttention(
        64, n_heads=4, n_kv_heads=2, head_dim=16, window=8, sink_tokens=2
    ).eval()
    cache = AttentionCache()
    with torch.no_grad():
        for _ in range(200):
            attn(torch.randn(1, 1, 64), cache=cache)
    assert cache.keys.shape[2] <= 8 + 2
    assert cache.seen == 200


# --------------------------------------------------------------------------------------
# Gated delta recurrence
# --------------------------------------------------------------------------------------


def test_gated_delta_incremental_matches_full_sequence():
    torch.manual_seed(5)
    gdn = GatedDeltaNet(64, n_heads=2, head_dim=16, expand=2.0, conv_kernel=4).eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        full = gdn(x)
        state = RecurrentState()
        steps = [gdn(x[:, t : t + 1], state=state) for t in range(16)]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_gated_delta_state_is_independent_of_sequence_length():
    """The property the hybrid stack is built on: memory does not grow with context."""
    gdn = GatedDeltaNet(64, n_heads=2, head_dim=16, expand=2.0).eval()
    sizes = []
    for length in (16, 256, 4096):
        state = RecurrentState()
        with torch.no_grad():
            gdn(torch.randn(1, length, 64), state=state)
        sizes.append(state.n_bytes())
    assert len(set(sizes)) == 1, f"state grew with context: {sizes}"


def test_gated_delta_migrates_a_restored_state_to_the_activation_device_and_dtype():
    gdn = GatedDeltaNet(
        64, n_heads=2, head_dim=16, expand=2.0, conv_kernel=4, allow_fused=False
    ).eval()
    state = RecurrentState(
        state=torch.zeros(1, 2, 32, 16, dtype=torch.float64),
        conv_state=torch.zeros(1, gdn.conv.in_channels, 3, dtype=torch.float64),
    )
    x = torch.randn(1, 2, 64, dtype=torch.float32)

    with torch.no_grad():
        gdn(x, state=state)

    assert state.state.device == x.device
    assert state.state.dtype == torch.float32
    assert state.conv_state.device == x.device
    assert state.conv_state.dtype == x.dtype


def test_gated_delta_erases_before_writing():
    """The delta rule's defining behaviour: rewriting a key replaces its value.

    Plain linear attention accumulates instead, so the old association keeps
    interfering. This is why the delta family handles associative recall.
    """
    torch.manual_seed(6)
    b, h, dk, dv = 1, 1, 8, 8
    S = torch.zeros(b, h, dv, dk)
    k = torch.nn.functional.normalize(torch.randn(b, h, dk, 1), dim=2)
    v1 = torch.randn(b, h, dv, 1)
    v2 = torch.randn(b, h, dv, 1)
    alpha, beta = 1.0, 1.0

    for v in (v1, v2):
        retrieved = S @ k
        S = alpha * S + beta * (v - alpha * retrieved) @ k.transpose(-1, -2)

    assert torch.allclose((S @ k), v2, atol=1e-5)
    assert not torch.allclose((S @ k), v1, atol=1e-2)


# --------------------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------------------


def test_moe_routes_to_top_k_experts_only():
    torch.manual_seed(7)
    moe = SparseMoE(64, n_experts=8, top_k=2, expert_hidden=32, n_shared=0).eval()
    x = torch.randn(4, 16, 64)
    moe(x)
    assert int(moe.last_stats.expert_counts.sum().item()) == 4 * 16 * 2


def test_moe_bias_balancing_reduces_expert_collapse():
    """Balance must improve without an auxiliary loss touching the gradient."""
    torch.manual_seed(8)
    moe = SparseMoE(
        32, n_experts=8, top_k=2, expert_hidden=16, n_shared=0,
        bias_balancing=True, bias_update_rate=0.05,
    )
    moe.train()
    # Deliberately skew the router so one expert dominates at the start.
    with torch.no_grad():
        moe.router.weight[0] *= 10.0
    x = torch.randn(8, 32, 32)
    first = None
    for _ in range(60):
        moe(x)
        if first is None:
            first = moe.last_stats.max_share
    assert moe.last_stats.max_share < first


def test_moe_shared_expert_sees_every_token():
    moe = SparseMoE(32, n_experts=4, top_k=1, expert_hidden=16, n_shared=1).eval()
    x = torch.randn(2, 4, 32)
    moe(x).sum().backward()
    assert moe.shared.gate_proj.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------


def test_inference_is_deterministic():
    """Two identical eval calls must give identical logits.

    The recurrent core is randomly initialised during training as a regulariser; if that
    leaks into inference the same prompt answers differently every time and no cache can
    ever match a full forward pass.
    """
    model = ProphetModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        a = model(ids, loop_k=3).logits
        b = model(ids, loop_k=3).logits
    assert torch.equal(a, b)


def test_training_randomises_the_core_state():
    """...but during training the randomisation must actually happen."""
    model = ProphetModel(tiny_config())
    model.train()
    ids = torch.randint(0, 256, (1, 8))
    a = model(ids, loop_k=3).logits
    b = model(ids, loop_k=3).logits
    assert not torch.allclose(a, b, atol=1e-5)


def test_model_runs_at_every_recurrence_depth():
    model = ProphetModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (2, 12))
    outputs = {}
    with torch.no_grad():
        for k in (1, 2, 4, 8):
            out = model(ids, loop_k=k)
            assert out.logits.shape == (2, 12, 256)
            assert out.loop_k == k
            outputs[k] = out.logits
    # Different depths must produce different computations, or the loop is decorative.
    assert not torch.allclose(outputs[1], outputs[8], atol=1e-3)


def test_parameter_count_is_independent_of_recurrence_depth():
    """The bet, asserted on the real model rather than the estimate."""
    model = ProphetModel(tiny_config())
    n = model.num_parameters()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        model(ids, loop_k=1)
        model(ids, loop_k=16)
    assert model.num_parameters() == n


def test_attention_cache_does_not_grow_with_recurrence_depth():
    """Looping a recurrent core must not multiply KV memory.

    This is the property that separates Prophet's design from a naive looped
    transformer, where every iteration needs its own KV cache.
    """
    cfg = tiny_config()
    model = ProphetModel(cfg).eval()
    ids = torch.randint(0, 256, (1, 16))

    sizes = {}
    for k in (1, 8):
        cache = ProphetCache()
        with torch.no_grad():
            model(ids, cache=cache, loop_k=k)
        sizes[k] = cache.summary()

    assert sizes[1]["attention_bytes"] == sizes[8]["attention_bytes"]
    # The recurrent state does scale with k, but each slot is a fixed-size matrix
    # rather than a cache that grows with context — that is the whole point.
    assert sizes[8]["recurrent_bytes"] == 8 * sizes[1]["recurrent_bytes"]

    # And the recurrent state stays constant as context grows, while attention does not.
    long_cache = ProphetCache()
    with torch.no_grad():
        model(torch.randint(0, 256, (1, 128)), cache=long_cache, loop_k=8)
    assert long_cache.summary()["recurrent_bytes"] == sizes[8]["recurrent_bytes"]


def test_model_incremental_decode_matches_full_forward():
    torch.manual_seed(9)
    model = ProphetModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        full = model(ids, loop_k=2, return_mtp=False).logits
        cache = ProphetCache()
        steps = [
            model(ids[:, t : t + 1], cache=cache, loop_k=2, return_mtp=False).logits
            for t in range(10)
        ]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=2e-3)


def test_truncated_backprop_cuts_the_graph_before_the_prelude():
    """Without input injection, truncation must isolate the prelude from the loss."""
    cfg = tiny_config(
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
            core_pattern=["gdn"], truncated_backprop_steps=1,
            inject_input_each_step=False,
        )
    )
    model = ProphetModel(cfg)
    model.train()
    model(torch.randint(0, 256, (1, 8)), loop_k=6).logits.sum().backward()

    prelude_grad = model.sections["prelude"][0].ffn.gate_proj.weight.grad
    core_grad = model.sections["core"][0].ffn.gate_proj.weight.grad
    assert core_grad is not None and core_grad.abs().sum() > 0
    assert prelude_grad is None or prelude_grad.abs().sum() == 0


def test_input_injection_restores_the_gradient_path_to_the_prelude():
    """With injection on, the prelude is re-read at every iteration, so it keeps a
    gradient path even under aggressive truncation.

    This is a feature, not a leak: without it the prelude would only ever be trained
    through the loop's first step, and deep recurrence would drift away from the prompt.
    """
    cfg = tiny_config(
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
            core_pattern=["gdn"], truncated_backprop_steps=1,
            inject_input_each_step=True,
        )
    )
    model = ProphetModel(cfg)
    model.train()
    model(torch.randint(0, 256, (1, 8)), loop_k=6).logits.sum().backward()
    assert model.sections["prelude"][0].ffn.gate_proj.weight.grad.abs().sum() > 0


def test_full_backprop_reaches_the_prelude():
    cfg = tiny_config(
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
            core_pattern=["gdn"], truncated_backprop_steps=8,
        )
    )
    model = ProphetModel(cfg)
    model.train()
    model(torch.randint(0, 256, (1, 8)), loop_k=4).logits.sum().backward()
    assert model.sections["prelude"][0].ffn.gate_proj.weight.grad.abs().sum() > 0


def test_moe_model_trains_and_reports_router_stats():
    cfg = tiny_config(
        ffn=FeedForwardConfig(
            kind="moe", n_experts=8, n_experts_per_token=2, n_shared_experts=1,
            expert_hidden_mult=0.5, moe_first_dense_layers=0,
        )
    )
    model = ProphetModel(cfg)
    model.train()
    out = model(torch.randint(0, 256, (2, 8)), loop_k=2)
    assert out.router_stats and out.aux_loss is not None
    (out.logits.sum() + out.aux_loss).backward()


def test_auxiliary_heads_produce_expected_shapes():
    cfg = tiny_config(heads=HeadsConfig(n_multi_token_predict=2, confidence_head=True))
    model = ProphetModel(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 256, (2, 6)))
    assert len(out.mtp_logits) == 2
    assert all(t.shape == (2, 6, 256) for t in out.mtp_logits)
    assert out.confidence.shape == (2, 6)


def test_generate_extends_the_sequence():
    model = ProphetModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (2, 5))
    out = model.generate(ids, max_new_tokens=6, temperature=0.0)
    assert out.shape == (2, 11)
    assert torch.equal(out[:, :5], ids)


def test_generate_masks_tokenizer_capacity_that_has_no_token():
    model = ProphetModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (2, 5))
    out = model.generate(ids, max_new_tokens=3, temperature=0.0, valid_token_ids={7})
    assert torch.equal(out[:, -3:], torch.full((2, 3), 7))


def test_generate_rejects_an_empty_or_out_of_range_token_mask():
    model = ProphetModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (1, 4))
    with pytest.raises(ValueError, match="at least one"):
        model.generate(ids, max_new_tokens=1, valid_token_ids=set())
    with pytest.raises(ValueError, match="outside"):
        model.generate(ids, max_new_tokens=1, valid_token_ids={256})


def test_budget_estimate_tracks_the_real_parameter_count():
    """The budget calculator gates every expensive decision, so it must not drift far
    from the model it claims to describe."""
    cfg = tiny_config(
        recurrent=RecurrentCoreConfig(enabled=False),
        n_layers=4,
        mixer=MixerConfig(pattern=["full_attn"], n_heads=4, n_kv_heads=2, head_dim=32),
        heads=HeadsConfig(),
    )
    estimated = count_parameters(cfg).total
    actual = ProphetModel(cfg).num_parameters()
    assert 0.85 < estimated / actual < 1.15, f"estimate {estimated} vs actual {actual}"


def test_recurrent_memory_layers_use_global_not_section_local_indices():
    """A coda ledger at global layer 3 must not be confused with local coda layer 0."""

    class CountingLedger(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            return torch.zeros_like(hidden)

    cfg = tiny_config(
        memory=MemoryConfig(
            enabled=True,
            kind="product_key",
            layers=(3,),
            memory_dim=8,
            n_slots=64,
        )
    )
    model = ProphetModel(cfg).eval()
    spy = CountingLedger()
    model.ledgers["3"] = spy

    with torch.no_grad():
        model(torch.randint(0, 256, (1, 8)), loop_k=4)

    assert spy.calls == 1


def test_unimplemented_frontend_fails_loudly():
    cfg = tiny_config(frontend=FrontendConfig(mode="byte_patch"))
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        ProphetModel(cfg)


# --------------------------------------------------------------------------------------
# Learned halting: making depth depend on the input
# --------------------------------------------------------------------------------------


def halting_config(**kw) -> ProphetConfig:
    return tiny_config(
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
            core_pattern=["gdn"], default_loop_k=6, halting="ponder",
            halting_loss_weight=0.05, halting_target_steps=3.0, **kw,
        )
    )


def test_halting_produces_a_proper_distribution_over_stopping_times():
    """The distribution enables input-adaptive average compute under a fixed ceiling."""
    torch.manual_seed(0)
    model = ProphetModel(halting_config()).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 256, (2, 8)), loop_k=6)

    assert out.halt_probs is not None
    assert out.halt_probs.shape == (2, 8, 6)
    assert torch.allclose(out.halt_probs.sum(-1), torch.ones(2, 8), atol=1e-5)
    assert (out.halt_probs >= -1e-6).all()


def test_expected_depth_lies_within_the_loop_budget():
    torch.manual_seed(0)
    model = ProphetModel(halting_config()).eval()
    with torch.no_grad():
        depth = model(torch.randint(0, 256, (2, 8)), loop_k=6).expected_depth()
    assert 1.0 <= depth <= 6.0


def test_halting_is_absent_when_not_configured():
    model = ProphetModel(tiny_config()).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 256, (1, 6)))
    assert out.halt_probs is None and out.expected_depth() is None


def test_a_higher_threshold_buys_more_iterations():
    """The runtime dial that a halting head is supposed to provide."""
    torch.manual_seed(0)
    model = ProphetModel(halting_config()).eval()
    ids = torch.randint(0, 256, (2, 8))

    used = {}
    with torch.no_grad():
        for threshold in (0.1, 0.5, 0.95):
            used[threshold] = model(ids, loop_k=12, halt_threshold=threshold).halt_probs.shape[-1]

    assert used[0.1] <= used[0.5] <= used[0.95]
    assert used[0.95] > used[0.1]


def test_halting_never_exceeds_the_loop_budget():
    torch.manual_seed(0)
    model = ProphetModel(halting_config()).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 256, (1, 6)), loop_k=4, halt_threshold=0.999)
    assert out.halt_probs.shape[-1] <= 4


def test_halting_reports_the_iterations_actually_executed():
    model = ProphetModel(halting_config()).eval()
    with torch.no_grad():
        model.halt_head[1].weight.zero_()
        model.halt_head[1].bias.fill_(20.0)
        out = model(
            torch.randint(0, 256, (2, 6)), loop_k=8, halt_threshold=0.9
        )

    assert out.loop_k == 1
    assert out.halt_probs.shape[-1] == 1


def test_invalid_runtime_loop_and_halt_limits_are_rejected():
    model = ProphetModel(halting_config()).eval()
    ids = torch.randint(0, 256, (1, 4))
    with pytest.raises(ValueError, match="loop_k"):
        model(ids, loop_k=0)
    with pytest.raises(ValueError, match="halt_threshold"):
        model(ids, halt_threshold=1.1)


def test_adaptive_halting_refuses_an_incremental_cache_until_backfill_exists():
    model = ProphetModel(halting_config()).eval()
    with pytest.raises(ValueError, match="incremental cache"):
        model(
            torch.randint(0, 256, (1, 4)),
            cache=ProphetCache(),
            loop_k=4,
            halt_threshold=0.9,
        )


def test_per_step_hidden_states_are_returned_for_the_ponder_loss():
    torch.manual_seed(0)
    model = ProphetModel(halting_config())
    out = model(torch.randint(0, 256, (2, 6)), loop_k=5)
    assert out.hidden_per_step is not None and len(out.hidden_per_step) == 5
    assert all(h.shape == (2, 6, model.cfg.d_model) for h in out.hidden_per_step)


def test_halting_probe_passes_do_not_corrupt_the_cache():
    """The bug this guards against is silent and severe.

    Halting applies the coda once per iteration to score each candidate stopping point.
    All those calls share one cache slot, so if they write to it they append the same
    positions k times -- and incremental decoding then produces fluent, plausible, wrong
    output with nothing to indicate why. The probe passes are cache-free; the real coda
    runs once.
    """
    torch.manual_seed(0)
    model = ProphetModel(halting_config()).eval()
    ids = torch.randint(0, 256, (1, 10))

    with torch.no_grad():
        full = model(ids, loop_k=4, return_mtp=False).logits
        cache = ProphetCache()
        steps = [
            model(ids[:, t : t + 1], cache=cache, loop_k=4, return_mtp=False).logits
            for t in range(10)
        ]

    assert torch.allclose(full, torch.cat(steps, dim=1), atol=2e-3)

    coda_slots = [k for k in cache.slots if k[0] == "coda"]
    for key in coda_slots:
        slot = cache.slots[key]
        if hasattr(slot, "keys") and slot.keys is not None:
            assert slot.keys.shape[2] == 10, (
                f"coda cache holds {slot.keys.shape[2]} positions for 10 tokens; "
                "the per-iteration probe passes are writing to it"
            )
