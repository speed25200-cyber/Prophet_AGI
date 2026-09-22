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


def test_h8_requires_no_collapse_and_no_loss_of_gain(tmp_path):
    base = record(0, [False] * 16 + [True] * 4, bpb=2.00, compute=0, promoted=0)
    for seed in (0, 1):
        write(
            tmp_path,
            "closed",
            seed,
            [
                base,
                record(1, [False] * 19 + [True] * 1, bpb=2.02, compute=600, promoted=5),  # collapse
                record(2, [False] * 10 + [True] * 10, bpb=2.03, compute=1200, promoted=8),
            ],
        )
        write(
            tmp_path,
            "closed-clean",
            seed,
            [
                base,
                record(1, [False] * 14 + [True] * 6, bpb=2.02, compute=600, promoted=4),
                record(2, [False] * 9 + [True] * 11, bpb=2.04, compute=1200, promoted=8),
            ],
        )
    h8 = summarise(load_runs(tmp_path))["checks"]["H8_clean"]
    assert h8["no_round_below_start_minus_allowance"] is True and h8["pass"] is True
    assert h8["clean_gain"] == pytest.approx(0.35) and h8["closed_gain"] == pytest.approx(0.3)


def family_record(round_index, by_family, *, trained=None, bpb=2.0, compute=0, promoted=0):
    """A round record whose benches carry their family (docs/31 amendment 14), with the
    generation record naming the families the arm looped on."""
    benches = []
    for family, verified in by_family.items():
        half = len(verified) // 2
        benches.append({"family": family, "seed": 7, "tasks": half, "verified": verified[:half]})
        benches.append(
            {
                "family": family,
                "seed": 11,
                "tasks": len(verified) - half,
                "verified": verified[half:],
            }
        )
    flags = [v for verified in by_family.values() for v in verified]
    generation = {"tokens": 100 * round_index}
    if trained:
        generation["by_family"] = {f: {"tasks": 3} for f in trained}
    return {
        "round": round_index,
        "success_mean": sum(flags) / len(flags),
        "success_by_family": {f: sum(v) / len(v) for f, v in by_family.items()},
        "bench": benches,
        "bpb": {"bpb": bpb, "docs": 10},
        "promoted_total": promoted,
        "compute_seconds": compute,
        "generation": generation,
    }


def flags(n_true, n=10):
    return [True] * n_true + [False] * (n - n_true)


def multi_family_layout(root, *, mixed_files_gain=4, mono_lookup_files_final=2, seeds=(0,)):
    """The v10 layout: a mixed closed-clean arm, one mono arm per family benched on both,
    the mixed oracle. Round 0 starts every arm at lookup 4/10, files 2/10."""
    start = {"lookup": flags(4), "files": flags(2)}
    for seed in seeds:
        write(
            root,
            "closed-clean",
            seed,
            [
                family_record(0, start),
                family_record(
                    1,
                    {"lookup": flags(7), "files": flags(2 + mixed_files_gain)},
                    trained=["lookup", "files"],
                    compute=600,
                    promoted=10,
                ),
            ],
        )
        write(
            root,
            "closed-clean-lookup",
            seed,
            [
                family_record(0, start),
                family_record(
                    1,
                    {"lookup": flags(8), "files": flags(mono_lookup_files_final)},
                    trained=["lookup"],
                    compute=500,
                    promoted=6,
                ),
            ],
        )
        write(
            root,
            "closed-clean-files",
            seed,
            [
                family_record(0, start),
                family_record(
                    1, {"lookup": flags(4), "files": flags(5)}, trained=["files"], compute=500
                ),
            ],
        )
        write(
            root,
            "oracle",
            seed,
            [
                family_record(0, start),
                family_record(
                    1,
                    {"lookup": flags(9), "files": flags(8)},
                    trained=["lookup", "files"],
                    compute=400,
                    promoted=12,
                ),
            ],
        )


def test_families_get_their_own_gains_and_h14_passes_on_a_loop_that_learns_both(tmp_path):
    multi_family_layout(tmp_path)
    summary = summarise(load_runs(tmp_path))
    mixed = summary["arms"]["closed-clean"]
    assert mixed["trained"] == ["lookup", "files"] and mixed["bench_families"] == [
        "lookup",
        "files",
    ]
    by_family = mixed["seeds"]["0"]["by_family"]
    assert by_family["lookup"]["curve"] == [0.4, 0.7]
    assert by_family["files"]["gain"]["gain"] == pytest.approx(0.4)
    assert mixed["seeds"]["0"]["gain"]["gain"] == pytest.approx(0.35)
    assert summary["arms"]["closed-clean-lookup"]["trained"] == ["lookup"]
    check = summary["checks"]["H14_multi_family"]
    assert check["mixed_learns_both_every_seed"] is True
    assert check["summed_gain_mixed"] == pytest.approx(0.7)
    assert check["summed_gain_mono"] == {
        "closed-clean-lookup": pytest.approx(0.4),
        "closed-clean-files": pytest.approx(0.3),
    }
    assert check["no_interference"] is True
    assert check["omission"]["closed-clean-lookup/seed0/files"]["transfer_gain"] == pytest.approx(
        0.0
    )
    assert check["no_forgetting_by_omission"] is True
    assert check["yield_mixed_over_oracle"] == pytest.approx(0.35 / 0.55)
    assert check["pass"] is True
    text = markdown(summary)
    assert "| closed-clean | 0 | files | 0.200 → 0.600 |" in text and "H14_multi_family" in text


def test_h14_fails_on_interference_or_on_forgetting_by_omission(tmp_path):
    # The mixed loop gains nothing on files: the mono lookup arm's summed gain wins.
    multi_family_layout(tmp_path / "interference", mixed_files_gain=0)
    check = summarise(load_runs(tmp_path / "interference"))["checks"]["H14_multi_family"]
    assert check["mixed_learns_both_every_seed"] is False
    assert check["no_interference"] is False and check["pass"] is False
    # Looping on lookup alone drops files from 0.2 to 0.0: forgetting by omission.
    multi_family_layout(tmp_path / "omission", mono_lookup_files_final=0)
    check = summarise(load_runs(tmp_path / "omission"))["checks"]["H14_multi_family"]
    assert check["no_forgetting_by_omission"] is False and check["pass"] is False
    assert check["omission"]["closed-clean-lookup/seed0/files"]["held"] is False


def test_h14_is_not_reported_without_the_layout(tmp_path):
    multi_family_layout(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "closed-clean-files-seed0")
    assert "H14_multi_family" not in summarise(load_runs(tmp_path))["checks"]
