"""Tests for the reasoning-channel bandwidth measurement.

The claim these support is that chain-of-thought is an information bottleneck, and the
point of the module is that the claim becomes *measurable* rather than rhetorical. So the
tests are mostly calibration: do the estimators return the right answer on inputs whose
answer is known independently?
"""

from __future__ import annotations

import math

import pytest
import torch

from prophet.analysis.bandwidth import (
    BITS_PER_EFFECTIVE_DIM,
    ChannelMeasurement,
    effective_rank,
    measure_channels,
    report,
    state_bits,
    token_entropy_bits,
)
from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.modeling.model import ProphetModel


# --------------------------------------------------------------------------------------
# Effective rank
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 4, 16])
def test_effective_rank_recovers_a_known_low_rank(rank):
    """Nominal width overstates a residual stream's capacity badly; this is the correction,
    so it has to be right on inputs where the answer is known."""
    torch.manual_seed(0)
    states = torch.randn(400, rank) @ torch.randn(rank, 128)
    measured = effective_rank(states)
    assert rank * 0.7 <= measured <= rank * 1.2, f"rank {rank} measured as {measured:.2f}"


def test_effective_rank_never_exceeds_the_nominal_width():
    torch.manual_seed(0)
    states = torch.randn(500, 64)
    assert effective_rank(states) <= 64


def test_effective_rank_is_scale_invariant():
    torch.manual_seed(0)
    states = torch.randn(200, 32)
    assert effective_rank(states) == pytest.approx(effective_rank(states * 1000), rel=1e-3)


def test_effective_rank_is_one_for_a_single_direction():
    direction = torch.randn(1, 48)
    states = torch.arange(1, 101, dtype=torch.float32).unsqueeze(1) * direction
    assert effective_rank(states) == pytest.approx(1.0, abs=0.1)


def test_entropy_method_agrees_on_low_rank_inputs():
    torch.manual_seed(0)
    states = torch.randn(300, 8) @ torch.randn(8, 64)
    a = effective_rank(states, method="participation")
    b = effective_rank(states, method="entropy")
    assert 0.5 < a / b < 2.0


def test_effective_rank_rejects_wrong_shapes():
    with pytest.raises(ValueError, match=r"\(n, d\)"):
        effective_rank(torch.randn(2, 3, 4))


def test_degenerate_inputs_do_not_crash():
    assert effective_rank(torch.randn(1, 16)) == 1.0
    assert effective_rank(torch.zeros(10, 16)) == 1.0


# --------------------------------------------------------------------------------------
# Token entropy
# --------------------------------------------------------------------------------------


def test_uniform_distribution_carries_log2_vocab_bits():
    """The nominal figure, which only a uniform distribution actually reaches."""
    assert token_entropy_bits(torch.zeros(1, 1024)) == pytest.approx(math.log2(1024), abs=1e-3)


def test_a_confident_distribution_carries_almost_nothing():
    """A model mid-reasoning is usually confident, so the real channel is far narrower
    than the vocabulary suggests. This is the correction that makes the ratio honest."""
    logits = torch.full((1, 1024), -20.0)
    logits[0, 0] = 20.0
    assert token_entropy_bits(logits) < 0.01


def test_entropy_is_bounded_by_the_vocabulary():
    torch.manual_seed(0)
    for vocab in (32, 256, 4096):
        bits = token_entropy_bits(torch.randn(8, vocab))
        assert 0 <= bits <= math.log2(vocab) + 1e-6


def test_state_bits_uses_the_conservative_constant():
    torch.manual_seed(0)
    states = torch.randn(300, 8) @ torch.randn(8, 64)
    assert state_bits(states) == pytest.approx(
        effective_rank(states) * BITS_PER_EFFECTIVE_DIM
    )


# --------------------------------------------------------------------------------------
# End-to-end measurement
# --------------------------------------------------------------------------------------


def _probe_model() -> ProphetModel:
    cfg = ProphetConfig(
        d_model=96, max_seq_len=32,
        frontend=FrontendConfig(vocab_size=128),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=2, n_kv_heads=1, head_dim=48,
            sliding_window=16, linear_heads=2, linear_head_dim=48,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
            core_pattern=["gdn"], default_loop_k=2,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
    )
    return ProphetModel(cfg).eval()


def test_measurement_reports_both_channels():
    torch.manual_seed(0)
    model = _probe_model()
    m = measure_channels(model, torch.randint(0, 128, (4, 16)), loop_k=2)

    assert 1.0 <= m.effective_dims <= m.d_model
    assert 0 < m.token_channel_bits <= math.log2(m.vocab_size)
    assert m.measured_ratio > 0
    assert 0 < m.dimension_utilisation <= 1.0


def test_measured_ratio_is_smaller_than_the_nominal_one():
    """Both corrections -- effective rank below nominal width, entropy below log2(vocab) --
    are real, and the nominal 2000:1 figure should not survive contact with measurement in
    the same form it was asserted."""
    torch.manual_seed(0)
    model = _probe_model()
    m = measure_channels(model, torch.randint(0, 128, (8, 16)), loop_k=2)
    assert m.nominal_ratio > 100
    assert m.measured_ratio < m.nominal_ratio


def test_measurement_records_the_depth_it_ran_at():
    torch.manual_seed(0)
    model = _probe_model()
    for k in (1, 4):
        assert measure_channels(model, torch.randint(0, 128, (2, 8)), loop_k=k).loop_k == k


def test_report_states_the_question_it_answers():
    torch.manual_seed(0)
    model = _probe_model()
    ms = [measure_channels(model, torch.randint(0, 128, (4, 12)), loop_k=k) for k in (1, 4)]
    text = report(ms)
    assert "Effective dims" in text
    assert "latent depth is doing work" in text


def test_report_handles_a_single_measurement():
    torch.manual_seed(0)
    model = _probe_model()
    text = report([measure_channels(model, torch.randint(0, 128, (2, 8)))])
    assert "Nominal ratio" in text
