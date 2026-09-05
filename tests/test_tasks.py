"""Verifiable task families, the rendered agent dataset, and the corpus and benchmark
preparation cores (offline: fake streams, no Hub)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from prophet.agent.actions import Action
from prophet.agent.render import render_episode
from prophet.agent.state import AgentState
from prophet.agent.tasks import FAMILIES, make_tasks, perfect_trajectory, tools_for, verifier_for
from prophet.data.tokenizer import ProphetTokenizer
from prophet.modeling.action import build_action_targets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_agent_dataset import build  # noqa: E402
from fetch_benchmarks import BENCHMARKS, row_text, write_benchmark  # noqa: E402
from prepare_corpus import load_manifest, write_shards  # noqa: E402

TOK = ProphetTokenizer(merges=[])


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_is_deterministic_and_its_perfect_trajectory_passes(family):
    a, b = make_tasks(4, family=family, seed=3), make_tasks(4, family=family, seed=3)
    assert [t.goal for t in a] == [t.goal for t in b]
    assert [t.goal for t in make_tasks(4, family=family, seed=4)] != [t.goal for t in a]
    for task in a:
        tools = tools_for(task)
        state = AgentState(goal=task.goal)
        for step in perfect_trajectory(task):
            action = step["action"]
            if action["name"] == "note":
                state.notes = action["args"]["text"]
            elif action["name"] != "done":
                tools.run(Action(action["name"], action["args"]))
        assert verifier_for(task)(state), (family, task.goal)


def test_verifier_fails_a_wrong_answer():
    task = make_tasks(1, family="calc", seed=1)[0]
    assert not verifier_for(task)(AgentState(goal="g", notes="42 is not it"))
    task = make_tasks(1, family="replace", seed=1)[0]
    assert not verifier_for(task)(AgentState(goal="g"))  # file untouched


def test_calc_tool_refuses_anything_but_arithmetic():
    task = make_tasks(1, family="calc", seed=2)[0]
    assert tools_for(task).run(Action("calc", {"expression": "__import__('os')"})).startswith("error")
    assert tools_for(task).run(Action("calc", {"expression": "12 * 3"})) == "36"


def test_rendered_episodes_carry_copyable_targets_for_every_copy_family():
    for family in ("files", "calc", "lookup", "count"):
        task = make_tasks(1, family=family, seed=5)[0]
        text = render_episode(task.goal, tools_for(task), perfect_trajectory(task))
        t = build_action_targets(torch.tensor([TOK.encode(text, parse_special=True)]), TOK)
        assert t.counts["values"] >= 2 and t.counts["copyable"] == t.counts["values"], (family, t.counts)
    # replace: the written text is generated, not copied.
    task = make_tasks(1, family="replace", seed=5)[0]
    text = render_episode(task.goal, tools_for(task), perfect_trajectory(task))
    t = build_action_targets(torch.tensor([TOK.encode(text, parse_special=True)]), TOK)
    assert t.counts["values"] == 3 and t.counts["copyable"] < t.counts["values"]


def test_dataset_builder_writes_families_and_a_manifest(tmp_path):
    manifest = build(tmp_path, per_family=3, seed=1, eval_seed=7, families=["files", "calc"])
    assert set(manifest["families"]) == {"files", "calc"}
    rows = [json.loads(l) for l in (tmp_path / "calc.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["family"] == "calc" and "<|call|>" in rows[0]["text"]
    assert json.loads((tmp_path / "manifest.json").read_text())["eval_seed"] == 7
    with pytest.raises(SystemExit, match="must differ"):
        build(tmp_path, per_family=1, seed=7, eval_seed=7, families=["files"])


# --------------------------------------------------------------------------------------
# Corpus preparation core
# --------------------------------------------------------------------------------------


def test_write_shards_caps_resumes_and_never_counts_a_partial_shard(tmp_path):
    docs = [f"document {i}" for i in range(25)]
    m = write_shards("web", iter(docs), tmp_path, shard_docs=10, max_docs=15)
    assert m["docs"] == 15 and m["stopped_by_cap"] and len(m["shards"]) == 2
    assert not list((tmp_path / "web").glob("*.tmp"))
    # A rerun with a larger cap continues after the documents already written.
    m2 = write_shards("web", iter(docs[m["docs"]:]), tmp_path, shard_docs=10, max_docs=None)
    assert m2["docs"] == 15 or m2["complete"]  # a completed source is left alone
    loaded = load_manifest(tmp_path / "web")
    assert loaded["complete"]
    lines = sum(1 for p in sorted((tmp_path / "web").glob("part-*.jsonl")) for _ in p.open())
    assert lines == loaded["docs"]


def test_write_shards_resumes_an_incomplete_source(tmp_path):
    docs = [f"d{i}" for i in range(12)]
    directory = tmp_path / "src"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"docs": 5, "bytes": 10, "shards": ["part-00000.jsonl"], "complete": False}))
    m = write_shards("src", iter(docs[5:]), tmp_path, shard_docs=4)
    assert m["docs"] == 12 and m["shards"][0] == "part-00000.jsonl" and m["shards"][1] == "part-00001.jsonl"


def test_max_bytes_cap_stops_a_source(tmp_path):
    m = write_shards("big", iter(["x" * 100] * 50), tmp_path, shard_docs=5, max_bytes=1200)
    assert m["stopped_by_cap"] and m["bytes"] <= 1500


# --------------------------------------------------------------------------------------
# Benchmark fetch core
# --------------------------------------------------------------------------------------


def test_row_text_joins_fields_lists_and_choice_dicts():
    row = {"question": "Why?", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "endings": ["x", "y"]}
    assert row_text(row, ("question", "choices", "endings")) == "Why?\na\nb\nx\ny"


def test_write_benchmark_refuses_an_empty_set(tmp_path):
    n = write_benchmark("t", [{"q": "one"}, {"q": ""}, {"other": 1}], ("q",), tmp_path)
    assert n == 1 and (tmp_path / "t.jsonl").exists()
    with pytest.raises(ValueError, match="nothing written"):
        write_benchmark("empty", [{"q": ""}], ("q",), tmp_path)
    assert not (tmp_path / "empty.jsonl").exists()


def test_benchmark_specs_cover_the_harness_tiers():
    from prophet.eval.harness import TIER1

    names = {b.name for b in BENCHMARKS}
    for task in TIER1:
        if task.kind == "multiple_choice" or task.name in ("arc_challenge", "commonsense_qa", "winogrande"):
            assert task.name in names, task.name
