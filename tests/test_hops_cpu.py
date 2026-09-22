"""The miniature depth-as-a-dial probe (docs/36): the task is what it claims, the arms
read their schedule, and a run writes every hop count for every arm and seed."""

import json
import random

import pytest

from scripts import hops_cpu as h


def test_the_answer_is_the_table_applied_hops_times():
    rng = random.Random(3)
    for hops in range(1, h.MAX_HOPS + 1):
        ids, answer = h.make_example(rng, hops=hops)
        assert len(ids) == h.LENGTH and ids[-1] == h.EQ and ids[-2] == h.HOP0 + hops - 1
        table = {ids[i]: ids[i + 1] for i in range(0, 3 * h.TABLE, 3)}
        assert all(ids[i + 2] == h.SEP for i in range(0, 3 * h.TABLE, 3))
        assert sorted(table) == sorted(table.values())  # a permutation: every hop defined
        node = ids[-3]
        for _ in range(hops):
            node = table[node]
        assert node == answer
    with pytest.raises(ValueError):
        h.make_example(rng, hops=h.MAX_HOPS + 1)


def test_the_schedule_ties_or_fixes_the_loop_count():
    assert [h.loop_k("tied", n) for n in (1, 3, 6)] == [1, 3, 6]
    assert {h.loop_k("fixed", n) for n in (1, 3, 6)} == {h.FIXED_K}
    assert h.config("attn").recurrent.core_pattern == ["full_attn"]
    assert h.config("gdn").recurrent.core_pattern == ["gdn"]


def test_a_run_reports_every_hop_count_for_every_arm_and_seed(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--arms",
        "gdn-tied,attn-fixed",
        "--steps",
        "2",
        "--eval-n",
        "4",
        "--seeds",
        "0,1",
        "--warmup",
        "1",
    ]
    assert h.main(argv) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["arms"]) == {
        "gdn-tied-seed0",
        "attn-fixed-seed0",
        "gdn-tied-seed1",
        "attn-fixed-seed1",
    }
    for arm in report["arms"].values():
        assert set(arm["accuracy_by_hops"]) == {str(n) for n in range(1, h.MAX_HOPS + 1)}
        assert set(arm["k_sweep"]) == {"2", "3", "4"}
    with pytest.raises(SystemExit):
        h.main(["--out", str(tmp_path), "--arms", "rnn-tied"])
