"""The miniature recall-against-depth probe (docs/37): the task is what it claims, the
state arms have no attention and the layout arm has it, and a run reports every m."""

import json
import random

import pytest

from scripts import recall_cpu as r


def test_each_query_is_answered_by_the_value_stored_under_its_key():
    rng = random.Random(5)
    for pairs in (1, 4, 32):
        ids, positions, answers = r.make_example(rng, pairs=pairs)
        store = {ids[i]: ids[i + 1] for i in range(0, 2 * pairs, 2)}
        assert len(store) == pairs and ids[2 * pairs] == r.SEP
        assert len(positions) == len(answers) == r.QUERIES
        for p, a in zip(positions, answers, strict=True):
            assert store[ids[p]] == a == ids[p + 1]  # the target follows the key
    with pytest.raises(ValueError):
        r.make_example(rng, pairs=r.KEYS + 1)


def test_the_state_arm_has_no_attention_and_the_layout_arm_does():
    state = r.config("state", dk=8, max_pairs=32)
    layout = r.config("layout", dk=16, max_pairs=32)
    assert state.mixer.pattern == ["gdn"] and state.recurrent.coda_pattern == ["gdn", "gdn"]
    assert state.mixer.linear_head_dim == 8
    assert "full_attn" in layout.recurrent.coda_pattern
    for cfg in (state, layout):
        assert not any("qk_norm" in w for w in cfg.design_warnings())


def test_a_run_reports_every_pair_count_for_every_arm(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--dk",
        "8",
        "--k",
        "1,2",
        "--pairs",
        "2,4",
        "--layout",
        "--steps",
        "2",
        "--eval-n",
        "2",
        "--warmup",
        "1",
    ]
    assert r.main(argv) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report["arms"]) == {"state-d8-k1", "state-d8-k2", "layout-d16-k1", "layout-d16-k2"}
    for arm in report["arms"].values():
        assert set(arm["accuracy_by_pairs"]) == {"2", "4"}


def test_an_empty_dk_runs_the_control_alone(tmp_path):
    argv = [
        "--out",
        str(tmp_path),
        "--dk",
        "",
        "--k",
        "1",
        "--pairs",
        "2",
        "--layout",
        "--steps",
        "1",
        "--eval-n",
        "2",
        "--warmup",
        "1",
    ]
    assert r.main(argv) == 0
    assert set(json.loads((tmp_path / "report.json").read_text())["arms"]) == {"layout-d16-k1"}
