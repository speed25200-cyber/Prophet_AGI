"""Tests for the evaluation harness.

The property under test throughout: the harness must refuse to let a meaningless number
decide anything. Most small-model benchmark scores are indistinguishable from guessing,
and reporting them as if they were not is the most common route to a confidently wrong
ablation.
"""

from __future__ import annotations

import math

import pytest
import torch

from prophet.eval.harness import (
    EXCLUDED,
    RESERVED,
    TIER0,
    TIER1,
    evaluate_bpb,
    run_suite,
)
from prophet.eval.metrics import (
    bits_per_byte,
    chance_level,
    cross_entropy_nats,
    is_above_chance,
    multiple_choice_accuracy,
)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_bits_per_byte_matches_the_definition():
    r = bits_per_byte(total_nats=math.log(2) * 1000, n_tokens=100, n_bytes=1000)
    assert r.bits_per_byte == pytest.approx(1.0)


def test_bits_per_byte_is_comparable_across_tokenizers():
    """The reason for the metric: a tokenizer that packs more text per token must not
    look better for free."""
    nats = 5000.0
    coarse = bits_per_byte(nats, n_tokens=1000, n_bytes=5000)
    fine = bits_per_byte(nats, n_tokens=2500, n_bytes=5000)
    assert coarse.bits_per_byte == pytest.approx(fine.bits_per_byte)
    assert coarse.nats_per_token > fine.nats_per_token  # per-token loss is not comparable


def test_fertility_is_reported_alongside():
    r = bits_per_byte(1000.0, n_tokens=250, n_bytes=1000)
    assert r.bytes_per_token == pytest.approx(4.0)


def test_zero_bytes_does_not_divide_by_zero():
    assert math.isnan(bits_per_byte(1.0, 10, 0).bits_per_byte)


def test_cross_entropy_returns_a_sum_and_a_count():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.randint(0, 7, (2, 5))
    nats, counted = cross_entropy_nats(logits, targets)
    assert counted == 2 * 4  # the shift drops one position per row
    assert nats > 0


def test_cross_entropy_skips_ignored_positions():
    logits = torch.randn(1, 4, 5)
    targets = torch.full((1, 4), -100)
    assert cross_entropy_nats(logits, targets) == (0.0, 0)


def test_chance_detection_matches_intuition():
    assert not is_above_chance(0.26, 500, 4)   # 4-choice chance is 25%
    assert is_above_chance(0.40, 500, 4)
    assert not is_above_chance(0.55, 50, 2)    # too few items to tell
    assert is_above_chance(0.55, 20_000, 2)    # enough items to tell


def test_accuracy_reports_both_normalisations():
    scores = [[1.0, 2.0], [3.0, 1.0]]
    lengths = [[1, 4], [1, 1]]
    out = multiple_choice_accuracy(scores, [0, 0], lengths=lengths)
    assert out["acc"] == 0.5          # raw picks choice 1 on the first item
    assert out["acc_norm"] == 1.0     # length normalisation fixes it


def test_accuracy_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="score rows"):
        multiple_choice_accuracy([[1.0, 2.0]], [0, 1])


def test_chance_level():
    assert chance_level(4) == 0.25
    assert chance_level(0) == 1.0


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


def test_at_chance_results_are_flagged_not_hidden():
    """A missing row reads as an oversight; a flagged row reads as what it is."""
    suite = run_suite(
        TIER1,
        {"hellaswag": lambda: (0.26, "acc", {"n": 1000.0})},
        model_params=1e9,
    )
    result = next(r for r in suite.results if r.task == "hellaswag")
    assert not result.informative
    assert "chance" in result.note


def test_under_scale_tasks_are_flagged():
    suite = run_suite(
        TIER1,
        {"mmlu_continuation": lambda: (1.2, "bpb", {})},
        model_params=1.3e8,
    )
    result = next(r for r in suite.results if r.task == "mmlu_continuation")
    assert not result.informative
    assert "130M" in result.note


def test_only_bpb_tasks_decide_ablations():
    """R11's central rule, asserted so it cannot erode."""
    suite = run_suite(
        TIER1,
        {
            "bpb_web": lambda: (0.9, "bpb", {}),
            "bpb_code": lambda: (1.1, "bpb", {}),
            "piqa": lambda: (0.71, "acc", {"n": 500.0}),
        },
        model_params=1.3e8,
    )
    assert suite.decision_metric() == pytest.approx(1.0)


def test_decision_metric_is_none_without_deciding_tasks():
    suite = run_suite(TIER1, {"piqa": lambda: (0.7, "acc", {"n": 500.0})}, model_params=1e9)
    assert suite.decision_metric() is None


def test_reserved_benchmarks_are_skipped_by_default():
    """Seventeen ablations offer ample chance to overfit the scoreboard. The defence is
    not looking."""
    runners = {t.name: (lambda: (0.5, "acc", {"n": 500.0})) for t in RESERVED}
    suite = run_suite(RESERVED, runners, model_params=2e9)
    assert suite.results == []

    final = run_suite(RESERVED, runners, model_params=2e9, skip_reserved=False)
    assert len(final.results) == len(RESERVED)


def test_excluded_tasks_carry_a_recorded_reason():
    """So the decision is not relitigated every time someone notices they are missing."""
    assert "inverse scaling" in EXCLUDED["truthfulqa"]
    for name in ("truthfulqa", "boolq", "gsm8k", "humaneval"):
        assert EXCLUDED[name]
        assert name not in {t.name for t in TIER1}


def test_tier0_is_small_enough_to_run_every_checkpoint():
    assert len(TIER0) <= 2


def test_report_names_the_uninformative_count():
    suite = run_suite(
        TIER1,
        {"bpb_web": lambda: (0.9, "bpb", {}), "hellaswag": lambda: (0.25, "acc", {"n": 800.0})},
        model_params=1.3e8,
    )
    report = suite.report(model_params=1.3e8)
    assert "not informative" in report
    assert "must not be used to decide" in report


def test_evaluate_bpb_runs_against_the_real_model():
    from prophet.config import FrontendConfig, MixerConfig, ProphetConfig, RecurrentCoreConfig
    from prophet.modeling.model import ProphetModel

    cfg = ProphetConfig(
        d_model=64, n_layers=2,
        frontend=FrontendConfig(vocab_size=64),
        mixer=MixerConfig(pattern=["full_attn"], n_heads=4, n_kv_heads=2, head_dim=16),
        recurrent=RecurrentCoreConfig(enabled=False),
    )
    model = ProphetModel(cfg)
    batches = [(torch.randint(0, 64, (2, 16)), 128) for _ in range(3)]
    result = evaluate_bpb(model, batches, domain="test")

    # An untrained model over a 64-token vocabulary should sit near log2(64) = 6 bits per
    # token, which at 4 bytes per token is about 1.5 bits per byte.
    assert 0.5 < result.bits_per_byte < 4.0
    assert result.n_tokens == 3 * 2 * 15
