"""Guard the scientific decision against mismatched reports and partial successes."""

import copy
import math

import pytest

from scripts.summarize_depth_adaptation import summarize


def evaluation(k, losses=(4.0, 4.0)):
    rows = [
        {
            "index": i,
            "sha256": str(i) * 64,
            "scored_tokens": n,
            "scored_bytes": 2 * n,
            "total_nats": ce * n,
        }
        for i, (n, ce) in enumerate(zip((100, 300), losses, strict=True))
    ]
    total = sum(row["total_nats"] for row in rows)
    return {
        "loop_k": k,
        "seq_len": 2048,
        "batch_size": 8,
        "precision": "bf16 autocast",
        "protocol": "fixture",
        "documents": rows,
        "total_nats": total,
        "scored_tokens": 400,
        "scored_bytes": 800,
        "nats_per_token": total / 400,
        "bits_per_byte": total / 800 / math.log(2),
    }


def fixture():
    plan = {"steps_each": 4, "additional_tokens_each": 65536, "parent_checkpoint": {"step": 4096}}
    identity = {
        "mode": "train",
        "protocol": "r04-depth-adaptation-v2",
        "numerical_policy": {"deterministic_algorithms": True},
        "plan_sha256": "abc",
        "parent_checkpoint": plan["parent_checkpoint"],
        "runtime": "same",
    }
    fixed = {
        "identity": {**identity, "arm": "fixed4"},
        "complete": True,
        "step": 4,
        "checkpoint": {"step": 4},
        "tokens_seen": 65536,
        "skipped_nonfinite": 0,
        "depth_history": [4] * 4,
        "depth_counts": {"4": 4},
        "loader_step": 32800,
        "results": {str(k): evaluation(k) for k in (1, 2, 4, 6, 8)},
    }
    variable = copy.deepcopy(fixed)
    variable["identity"]["arm"] = "uniform2to6"
    variable["depth_history"] = [2, 3, 5, 6]
    variable["depth_counts"] = {str(k): 1 for k in (2, 3, 5, 6)}
    variable["results"]["6"] = evaluation(6, (3.9, 3.9))
    return fixed, variable, plan, "abc", evaluation(4)


def test_all_conditions_and_document_weighting():
    args = fixture()
    report = summarize(*args)
    assert report["seed0_screen_passed"]
    assert report["variable_k6_minus_k4_ce"] == pytest.approx(-0.1)
    assert report["paired_document_95pct_ce_interval"] == pytest.approx([-0.1, -0.1])
    args[1]["results"]["6"] = evaluation(6, (3.0, 4.2))
    report = summarize(*args)
    assert report["variable_k6_minus_k4_ce"] == pytest.approx(-0.1)
    assert report["paired_document_95pct_ce_interval"] == pytest.approx([-1.0, 0.2])
    assert not report["seed0_screen_passed"]  # Positive upper interval defeats a good mean.


@pytest.mark.parametrize("case", ["small_gain", "k4_regression", "control_k6_better"])
def test_partial_success_cannot_pass(case):
    args = fixture()
    if case == "small_gain":
        args[1]["results"]["6"] = evaluation(6, (3.99, 3.99))
    elif case == "k4_regression":
        args[1]["results"]["4"] = evaluation(4, (4.1, 4.1))
    else:
        args[0]["results"]["6"] = evaluation(6, (3.8, 3.8))
    assert not summarize(*args)["seed0_screen_passed"]


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "identity",
        "document",
        "aggregate",
        "nonfinite",
        "history",
        "missing_depth",
        "depth",
        "loader",
        "plan",
    ],
)
def test_corrupt_or_unmatched_reports_rejected(case):
    args = fixture()
    v = args[1]
    if case == "partial":
        v["complete"] = False
    elif case == "identity":
        v["identity"]["runtime"] = "different"
    elif case == "document":
        v["results"]["6"]["documents"][0]["sha256"] = "different"
    elif case == "aggregate":
        v["results"]["6"]["bits_per_byte"] += 0.1
    elif case == "nonfinite":
        v["results"]["6"]["documents"][0]["total_nats"] = float("nan")
    elif case == "history":
        v["depth_history"][0] = 8
    elif case == "missing_depth":
        del v["results"]["8"]
    elif case == "depth":
        v["results"]["6"]["loop_k"] = 4
    elif case == "loader":
        v["loader_step"] += 8
    else:
        v["identity"]["plan_sha256"] = "different"
    with pytest.raises(ValueError):
        summarize(*args)
