"""The pilot summary computes the pre-registered checks from the round records alone."""

import json

import pytest

from scripts.summarize_closed_loop import load_runs, markdown, paired_gain, summarise


def record(round_index, verified, *, bpb, compute, promoted):
    half = len(verified) // 2
    return {
        "round": round_index,
        "success_mean": sum(verified) / len(verified),
        "bench": [
            {"seed": 7, "tasks": half, "verified": verified[:half]},
            {"seed": 11, "tasks": len(verified) - half, "verified": verified[half:]},
        ],
        "bpb": {"bpb": bpb, "docs": 10},
        "promoted_total": promoted,
        "compute_seconds": compute,
        "generation": {"tokens": 100 * round_index},
    }


def write(root, arm, seed, rounds):
    path = root / f"{arm}-seed{seed}" / "rounds.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rounds))


def test_paired_gain_counts_wins_and_losses_with_an_interval():
    first = record(0, [False] * 8 + [True] * 2, bpb=2.0, compute=0, promoted=0)
    last = record(3, [True] * 7 + [False] * 3, bpb=2.1, compute=100, promoted=9)
    gain = paired_gain(first, last)
    assert gain["tasks"] == 10 and gain["gain"] == pytest.approx(0.5)
    assert gain["won"] == 7 and gain["lost"] == 2
    low, high = gain["interval_95"]
    assert low <= 0.5 <= high
    with pytest.raises(ValueError, match="paired"):
        paired_gain(first, record(1, [True] * 4, bpb=2.0, compute=1, promoted=0))


def test_summary_applies_the_four_checks(tmp_path):
    for seed in (0, 1):
        write(
            tmp_path,
            "closed",
            seed,
            [
                record(0, [False] * 16 + [True] * 4, bpb=2.00, compute=0, promoted=0),
                record(1, [False] * 10 + [True] * 10, bpb=2.02, compute=600, promoted=8),
                record(2, [False] * 4 + [True] * 16, bpb=2.03, compute=1200, promoted=20),
            ],
        )
        write(
            tmp_path,
            "oracle",
            seed,
            [
                record(0, [False] * 16 + [True] * 4, bpb=2.00, compute=0, promoted=0),
                record(1, [False] * 6 + [True] * 14, bpb=2.01, compute=500, promoted=30),
                record(2, [False] * 2 + [True] * 18, bpb=2.01, compute=1000, promoted=60),
            ],
        )
        write(
            tmp_path,
            "frozen",
            seed,
            [record(0, [False] * 16 + [True] * 4, bpb=2.00, compute=0, promoted=0)],
        )
    summary = summarise(load_runs(tmp_path))
    checks = summary["checks"]
    assert checks["H1_self_improvement"]["pass"] is True
    assert checks["H1_self_improvement"]["pooled"] == {"tasks": 40, "won": 24, "lost": 0}
    assert checks["H2_yield"]["closed_over_oracle_gain"] == pytest.approx(0.6 / 0.7)
    assert checks["H3_forgetting"]["pass"] is True
    assert checks["H4_compute"]["pass"] is True
    closed = summary["arms"]["closed"]["seeds"]["0"]
    assert closed["curve"][2]["gain_per_hour"] == pytest.approx(0.3 / (600 / 3600))
    assert closed["compute_hours"] == pytest.approx(1200 / 3600)
    text = markdown(summary)
    assert "| closed | 0 | 2 |" in text and "H1_self_improvement" in text


def test_a_loop_that_forgets_or_stalls_fails_the_checks(tmp_path):
    for seed in (0, 1):
        write(
            tmp_path,
            "closed",
            seed,
            [
                record(0, [False] * 10 + [True] * 10, bpb=2.00, compute=0, promoted=0),
                record(1, [False] * 10 + [True] * 10, bpb=2.40, compute=600, promoted=5),
            ],
        )
        write(
            tmp_path,
            "oracle",
            seed,
            [
                record(0, [False] * 10 + [True] * 10, bpb=2.00, compute=0, promoted=0),
                record(1, [False] * 4 + [True] * 16, bpb=2.05, compute=600, promoted=30),
            ],
        )
    checks = summarise(load_runs(tmp_path))["checks"]
    assert checks["H1_self_improvement"]["pass"] is False
    assert checks["H3_forgetting"]["pass"] is False
    assert checks["H4_compute"]["pass"] is False


def test_h5_compares_the_klpo_arm_with_the_closed_arm(tmp_path):
    base = record(0, [False] * 16 + [True] * 4, bpb=2.00, compute=0, promoted=0)
    for seed in (0, 1):
        write(
            tmp_path,
            "closed",
            seed,
            [base, record(1, [False] * 10 + [True] * 10, bpb=2.03, compute=600, promoted=8)],
        )
        write(
            tmp_path,
            "closed-klpo",
            seed,
            [base, record(1, [False] * 8 + [True] * 12, bpb=2.02, compute=700, promoted=8)],
        )
    checks = summarise(load_runs(tmp_path))["checks"]
    assert "H2_yield" not in checks  # no oracle arm here
    h5 = checks["H5_klpo"]
    assert h5["pass"] is True
    assert h5["klpo_gain"] == pytest.approx(0.4) and h5["closed_gain"] == pytest.approx(0.3)
    # Same gain but more drift fails: both conditions are required.
    write(
        tmp_path / "worse",
        "closed",
        0,
        [base, record(1, [False] * 10 + [True] * 10, bpb=2.03, compute=600, promoted=8)],
    )
    write(
        tmp_path / "worse",
        "closed-klpo",
        0,
        [base, record(1, [False] * 10 + [True] * 10, bpb=2.05, compute=700, promoted=8)],
    )
    assert summarise(load_runs(tmp_path / "worse"))["checks"]["H5_klpo"]["pass"] is False
    assert "closed-klpo" in markdown(summarise(load_runs(tmp_path)))
