"""Tests for donor-to-Prophet conversion.

The project's chosen path trains Prophet-mini from random initialisation and produces
Prophet-main by converting an open donor, so these tests guard the half of the plan that
inherits someone else's pretraining. The failures they exist to catch are all silent: a
licence that follows the derivative, a tied embedding quietly overwritten on load, a
conversion whose coverage is so low it is really pretraining in disguise.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from prophet.budget import count_parameters
from prophet.convert.donors import (
    DONORS,
    DonorSpec,
    LicenceProblem,
    assert_donor_is_usable,
    get_donor,
)
from prophet.convert.plan import plan_conversion, prophet_config_for_donor
from prophet.convert.weights import DONOR_KEYS, convert_state_dict
from prophet.modeling.model import ProphetModel, _swiglu_hidden


def tiny_donor(**kw) -> DonorSpec:
    """A stand-in donor small enough to convert in a test."""
    base = dict(
        n_layers=12, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        ffn_hidden=512, vocab_size=1024,
    )
    base.update(kw)
    return replace(get_donor("qwen3-0.6b"), **base)


def synthetic_donor_state(d: DonorSpec, *, embed: torch.Tensor | None = None) -> dict:
    state = {
        DONOR_KEYS["embed"]: embed if embed is not None else torch.randn(d.vocab_size, d.d_model),
        DONOR_KEYS["final_norm"]: torch.randn(d.d_model),
    }
    shapes = {
        "norm1": (d.d_model,),
        "norm2": (d.d_model,),
        "q_proj": (d.n_heads * d.head_dim, d.d_model),
        "k_proj": (d.n_kv_heads * d.head_dim, d.d_model),
        "v_proj": (d.n_kv_heads * d.head_dim, d.d_model),
        "o_proj": (d.d_model, d.n_heads * d.head_dim),
        "q_norm": (d.head_dim,),
        "k_norm": (d.head_dim,),
        "gate_proj": (d.ffn_hidden, d.d_model),
        "up_proj": (d.ffn_hidden, d.d_model),
        "down_proj": (d.d_model, d.ffn_hidden),
    }
    for i in range(d.n_layers):
        for name, shape in shapes.items():
            state[DONOR_KEYS[name].format(i=i)] = torch.randn(*shape)
    return state


# --------------------------------------------------------------------------------------
# Licence
# --------------------------------------------------------------------------------------


def test_permissive_donors_pass():
    for key in ("qwen3-1.7b", "qwen3-4b", "qwen3-0.6b", "smollm3-3b"):
        assert_donor_is_usable(get_donor(key))


def test_a_donor_whose_licence_follows_the_derivative_is_refused():
    """Found after conversion, this costs the whole budget spent on it."""
    with pytest.raises(LicenceProblem, match="follows the derivative"):
        assert_donor_is_usable(get_donor("llama-3.2-1b"))


def test_restricted_donor_can_be_used_with_an_explicit_override():
    assert_donor_is_usable(get_donor("llama-3.2-1b"), allow_restricted=True)


def test_naming_constraint_is_surfaced_in_the_error():
    with pytest.raises(LicenceProblem, match="must begin with 'Llama'"):
        assert_donor_is_usable(get_donor("llama-3.2-1b"))


def test_unknown_donor_raises():
    with pytest.raises(KeyError, match="unknown donor"):
        get_donor("gpt-5")


def test_donor_specs_reproduce_their_advertised_sizes():
    """A cross-check we can run without the Hub: a spec whose parameter estimate does not
    match the size in its own name has a transcription error somewhere."""
    expected = {
        "qwen3-0.6b": 0.6e9, "qwen3-1.7b": 1.7e9, "qwen3-4b": 4.0e9,
        "smollm3-3b": 3.0e9, "llama-3.2-1b": 1.2e9,
    }
    for key, target in expected.items():
        estimate = DONORS[key].params_estimate
        assert 0.85 < estimate / target < 1.15, (
            f"{key}: estimate {estimate / 1e9:.2f}B against advertised {target / 1e9:.1f}B"
        )


def test_all_donors_are_marked_unverified():
    """They were written while the Hub was unreachable. Flipping this flag must be a
    deliberate act after running the verification script."""
    assert all(not d.verified for d in DONORS.values())


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


def test_generated_config_matches_the_donor_shapes():
    """Matching head_dim and kv-head count is what lets attention transfer by direct copy
    instead of by interpolation."""
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d)
    assert cfg.d_model == d.d_model
    assert cfg.head_dim == d.head_dim
    assert cfg.mixer.n_kv_heads == d.n_kv_heads
    assert cfg.frontend.vocab_size == d.vocab_size


def test_generated_ffn_width_matches_the_donor_exactly():
    """Off by even one, and every FFN matrix in the model fails to transfer."""
    for key in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "smollm3-3b"):
        d = get_donor(key)
        cfg = prophet_config_for_donor(d)
        assert _swiglu_hidden(cfg.d_model, cfg.ffn.hidden_mult) == d.ffn_hidden, key


def test_prelude_and_coda_take_the_donor_ends():
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d, prelude_layers=4, core_layers=4, coda_layers=4)
    plan = plan_conversion(d, cfg)

    prelude = [b for b in plan.blocks if b.section == "prelude"]
    coda = [b for b in plan.blocks if b.section == "coda"]
    assert [b.donor_layers for b in prelude] == [(0,), (1,), (2,), (3,)]
    assert [b.donor_layers for b in coda] == [(24,), (25,), (26,), (27,)]


def test_core_averages_the_donor_middle():
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d, prelude_layers=4, core_layers=4, coda_layers=4)
    plan = plan_conversion(d, cfg, core_init="average")

    core = [b for b in plan.blocks if b.section == "core"]
    assert len(core) == 4
    covered = [i for b in core for i in b.donor_layers]
    assert covered == list(range(4, 24))  # every middle layer used exactly once
    assert all(len(b.donor_layers) > 1 for b in core)


@pytest.mark.parametrize("mode", ["average", "stride", "first"])
def test_every_core_init_mode_assigns_all_blocks(mode):
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d, prelude_layers=2, core_layers=4, coda_layers=2)
    plan = plan_conversion(d, cfg, core_init=mode)
    assert all(b.donor_layers for b in plan.blocks if b.section == "core")


def test_recurrence_recovers_the_donor_depth_with_fewer_weights():
    """The point of converting into a recurrent trunk: same effective depth, fewer
    parameters to store."""
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d, prelude_layers=4, core_layers=4, coda_layers=4, loop_k=5)
    assert cfg.effective_depth() == d.n_layers
    assert cfg.parameterised_depth() < d.n_layers
    assert count_parameters(cfg).total < d.params_estimate


def test_coverage_is_high_for_a_shape_matched_donor():
    """Below roughly half, a 'conversion' is really pretraining and should be recognised
    as such before the budget is committed."""
    d = get_donor("qwen3-1.7b")
    plan = plan_conversion(d, prophet_config_for_donor(d))
    assert plan.coverage()["coverage"] > 0.8


def test_width_mismatch_is_warned_about():
    """Nothing transfers by direct copy at a different width, so this has to be loud."""
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d)
    cfg.d_model = 1536
    plan = plan_conversion(d, cfg)
    assert any("width mismatch" in w for w in plan.warnings)


def test_head_dim_mismatch_is_warned_about():
    d = get_donor("qwen3-1.7b")
    cfg = prophet_config_for_donor(d)
    cfg.mixer.head_dim = 64
    plan = plan_conversion(d, cfg)
    assert any("head_dim mismatch" in w for w in plan.warnings)


def test_a_donor_shorter_than_prelude_plus_coda_is_warned_about():
    d = replace(get_donor("qwen3-1.7b"), n_layers=4)
    cfg = prophet_config_for_donor(d, prelude_layers=4, core_layers=4, coda_layers=4)
    plan = plan_conversion(d, cfg)
    assert any("exceeds the donor" in w for w in plan.warnings)


def test_unverified_donor_produces_a_warning():
    d = get_donor("qwen3-1.7b")
    plan = plan_conversion(d, prophet_config_for_donor(d))
    assert any("unverified" in w for w in plan.warnings)


def test_plan_report_renders():
    d = get_donor("qwen3-1.7b")
    report = plan_conversion(d, prophet_config_for_donor(d)).report()
    assert "Parameter coverage" in report and "Qwen3-1.7B" in report


# --------------------------------------------------------------------------------------
# Weight transfer
# --------------------------------------------------------------------------------------


@pytest.fixture()
def converted():
    d = tiny_donor()
    cfg = prophet_config_for_donor(d, prelude_layers=2, core_layers=2, coda_layers=2, loop_k=4)
    plan = plan_conversion(d, cfg)
    model = ProphetModel(cfg)
    donor_state = synthetic_donor_state(d)
    state, report = convert_state_dict(donor_state, plan, copy.deepcopy(model.state_dict()))
    return d, cfg, plan, model, donor_state, state, report


def test_converted_model_loads_and_runs(converted):
    _, cfg, _, model, _, state, _ = converted
    model.load_state_dict(state)
    out = model(torch.randint(0, cfg.frontend.vocab_size, (2, 8)))
    assert out.logits.shape == (2, 8, cfg.frontend.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_attention_weights_are_copied_verbatim(converted):
    d, _, _, model, donor_state, state, _ = converted
    model.load_state_dict(state)
    expected = donor_state[DONOR_KEYS["q_proj"].format(i=0)]
    assert torch.allclose(model.sections["prelude"][0].mixer.q_proj.weight, expected)


def test_ffn_weights_are_copied_verbatim(converted):
    _, _, _, model, donor_state, state, _ = converted
    model.load_state_dict(state)
    expected = donor_state[DONOR_KEYS["gate_proj"].format(i=0)]
    assert torch.allclose(model.sections["prelude"][0].ffn.gate_proj.weight, expected)


def test_core_ffn_is_the_mean_of_its_donor_layers(converted):
    d, _, plan, model, donor_state, state, _ = converted
    model.load_state_dict(state)
    block = next(b for b in plan.blocks if b.section == "core" and b.index == 0)
    expected = torch.stack(
        [donor_state[DONOR_KEYS["gate_proj"].format(i=i)] for i in block.donor_layers]
    ).mean(0)
    assert torch.allclose(model.sections["core"][0].ffn.gate_proj.weight, expected, atol=1e-6)


def test_tied_embedding_survives_a_deep_copied_state_dict(converted):
    """Aliasing in a live state dict hides this; copy or serialise it first and the stale
    lm_head entry silently overwrites the donor embedding. The model still runs."""
    d = tiny_donor()
    cfg = prophet_config_for_donor(d, prelude_layers=1, core_layers=1, coda_layers=1)
    assert cfg.frontend.tie_word_embeddings
    plan = plan_conversion(d, cfg)
    model = ProphetModel(cfg)

    marker = torch.arange(d.vocab_size * d.d_model, dtype=torch.float32).reshape(
        d.vocab_size, d.d_model
    )
    donor_state = synthetic_donor_state(d, embed=marker)
    state, _ = convert_state_dict(donor_state, plan, copy.deepcopy(model.state_dict()))

    fresh = ProphetModel(cfg)
    fresh.load_state_dict(state)
    assert torch.allclose(fresh.embed.weight, marker)
    assert fresh.lm_head.weight is fresh.embed.weight


def test_gated_delta_query_projection_is_seeded_from_attention(converted):
    _, _, _, model, _, state, report = converted
    model.load_state_dict(state)
    assert any("core" in name and "q_proj" in name for name in report.seeded)
    assert model.sections["core"][0].mixer.q_proj.weight.abs().sum() > 0


def test_gated_delta_output_projection_starts_inert_in_its_widened_half(converted):
    """The expansion factor doubles the value path. Zeroing the new half keeps the
    layer's initial function as close to the donor's attention as a bounded-state mixer
    can be."""
    d, _, _, model, _, state, _ = converted
    model.load_state_dict(state)
    mixer = model.sections["core"][0].mixer
    weight = mixer.o_proj.weight
    # Widened *per head*: donor head h lands in the first slot of GDN head h's block, so
    # the inert half is the trailing half of every head block, not of the whole matrix.
    per_head = weight.view(weight.shape[0], mixer.n_heads, mixer.head_v)
    half = mixer.head_v // 2
    assert per_head[:, :, half:].abs().sum() == 0
    assert bool((per_head[:, :, :half].abs().sum(dim=(0, 2)) > 0).all())


def test_nothing_important_is_left_at_fresh_initialisation(converted):
    """Everything fresh must be a component the donor genuinely lacks."""
    _, _, _, _, _, _, report = converted
    allowed = ("mtp", "confidence", ".conv.", "a_proj", "b_proj", "o_norm", "lm_head")
    unexpected = [n for n in report.fresh if not any(a in n for a in allowed)]
    assert unexpected == [], f"unexpectedly uninitialised: {unexpected}"


def test_no_shape_mismatches_for_a_matched_donor(converted):
    _, _, _, _, _, _, report = converted
    assert report.mismatched == []


def test_shape_mismatch_is_reported_rather_than_forced(converted):
    """A wrongly shaped copy is far worse than a clean random start."""
    d = tiny_donor()
    cfg = prophet_config_for_donor(d, prelude_layers=1, core_layers=1, coda_layers=1)
    plan = plan_conversion(d, cfg)
    model = ProphetModel(cfg)

    donor_state = synthetic_donor_state(d)
    donor_state[DONOR_KEYS["q_proj"].format(i=0)] = torch.randn(8, 8)  # wrong shape
    _, report = convert_state_dict(donor_state, plan, copy.deepcopy(model.state_dict()))
    assert any("q_proj" in m for m in report.mismatched)


def test_transfer_report_renders(converted):
    _, _, _, _, _, _, report = converted
    text = report.summary()
    assert "direct copy" in text and "have a donor origin" in text
