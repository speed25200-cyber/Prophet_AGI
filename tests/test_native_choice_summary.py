"""Independent arithmetic checks for native question-level paired analysis."""

import copy
import math

import numpy as np
import pytest

from scripts import summarize_arc_native as summary
from scripts.eval_arc_native import evaluate
from scripts.prepare_arc_recovery_eval import prepare_rows
from tests.test_choice_eval import CharacterTokenizer, FixedDistribution, source_row


def fixture():
    tokenizer = CharacterTokenizer()
    tokenizer.pad_id = 0
    items = prepare_rows([source_row("one"), source_row("two")], tokenizer)
    score = evaluate(
        FixedDistribution(), items, tokenizer, batch_size=2, seq_len=64, device="cpu", loop_k=4
    )
    manifest = {"rows": 2, "batch_size": 2, "buckets": score["buckets"]}
    reports = {}
    for arm in summary.ARMS:
        for depth in (4, 6):
            reports[f"{arm}-k{depth}"] = {
                "complete": True,
                "protocol": "arc-easy-native-r04-evaluation-v1",
                "arm": arm,
                "loop_k": depth,
                "manifest": copy.deepcopy(manifest),
                "manifest_sha256": "fixture",
                "source": {"revision": "fixture"},
                "runtime": {"device": "cpu"},
                "numerical_policy": {"fp32": True},
                "checkpoint": {"arm": arm},
                "evaluation": copy.deepcopy(score),
            }
    return reports, items, manifest


def test_complete_six_reports_recompute_and_identical_models_have_zero_contrasts():
    reports, items, manifest = fixture()
    result = summary.summarize(reports, items, manifest, "fixture", replicates=100)
    assert result["complete"] and len(result["scores"]) == 6 and len(result["paired"]) == 9
    for contrast in result["paired"].values():
        for metric in contrast.values():
            assert metric["difference"] == 0 and metric["ci95"] == [0, 0]
        assert contrast["raw"]["same_correctness"] == 2
        assert contrast["raw"]["different_predictions"] == 0


def test_question_bootstrap_uses_paired_ratios_and_correct_accuracy_direction():
    left = {
        "raw": np.array([1, 0]),
        "normalized": np.array([1, 1]),
        "nats": np.array([2.0, 20.0]),
        "bytes": np.array([1, 10]),
        "tokens": np.array([1, 5]),
        "prediction": np.array([0, 0]),
        "normalized_prediction": np.array([0, 1]),
    }
    right = {
        **left,
        "raw": np.array([0, 1]),
        "normalized": np.array([0, 1]),
        "nats": np.array([4.0, 10.0]),
        "prediction": np.array([1, 1]),
    }
    # Enumerate all four two-question resamples, independent of the RNG implementation.
    draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    actual = summary.paired(left, right, draws)
    assert actual["raw"]["difference"] == 0
    assert actual["raw"]["ci95"] == pytest.approx([-92.5, 92.5])
    assert actual["raw"]["left_only_correct"] == actual["raw"]["right_only_correct"] == 1
    assert actual["raw"]["different_predictions"] == 2
    assert actual["normalized"]["difference"] == 50
    assert actual["nats_per_token"]["difference"] == pytest.approx(8 / 6)
    assert actual["bits_per_byte"]["difference"] == pytest.approx(8 / 11 / math.log(2))
    # Hand-calculated per-resample totals: (-4/2), (8/11), (8/11), (20/20).
    expected = np.quantile(np.array([-2, 8 / 11, 8 / 11, 1]) / math.log(2), [0.025, 0.975])
    assert actual["bits_per_byte"]["ci95"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "corruption", ["missing", "aggregate", "ranking", "order", "checkpoint", "precision", "oracle"]
)
def test_corrupted_or_incomplete_evidence_is_rejected(corruption):
    reports, items, manifest = fixture()
    report = reports["learned_mix-k6"]
    if corruption == "missing":
        reports.pop("fixed_sum-k4")
    elif corruption == "aggregate":
        report["evaluation"]["accuracy"] += 0.5
    elif corruption == "ranking":
        report["evaluation"]["items"][0]["prediction"] = 999
    elif corruption == "order":
        report["evaluation"]["items"].reverse()
    elif corruption == "checkpoint":
        report["checkpoint"]["changed"] = True
    elif corruption == "precision":
        report["numerical_policy"]["fp32"] = False
    else:
        report["evaluation"]["batch_oracle"]["candidates"] = 0
    with pytest.raises(ValueError):
        summary.summarize(reports, items, manifest, "fixture", replicates=100)
