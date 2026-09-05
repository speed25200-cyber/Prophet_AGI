"""The executable agent benchmark: verifiers decide, and the curve is per block."""

from __future__ import annotations

import json

import torch
from torch import nn

from prophet.agent.loop import AgentConfig
from prophet.agent.quarantine import Quarantine
from prophet.data.tokenizer import ProphetTokenizer
from prophet.eval.agent_bench import (
    BenchReport,
    EpisodeReport,
    file_tools,
    make_tasks,
    run_bench,
    verifier_for,
)
from prophet.modeling.model import ProphetOutput

TOK = ProphetTokenizer(merges=[])
OPEN = (TOK.special_id("<|think|>"), TOK.special_id("<|call|>"))
CLOSE = (TOK.special_id("<|/think|>"), TOK.special_id("<|/call|>"))


class _Scripted(nn.Module):
    """Speaks a per-episode script inside spans (see tests/test_agent.py)."""

    def __init__(self, scripts: list[list[int]], confidence: float = 5.0) -> None:
        super().__init__()
        self.scripts = scripts
        self.episode = -1
        self.cursor = 0
        self.speaking = False
        self.confidence = confidence
        self.dummy = nn.Parameter(torch.zeros(1))
        self.modality_embed = None

    def _project(self, h):
        return torch.zeros(*h.shape[:-1], 600)

    def forward(self, ids, *, cache=None, loop_k=None, return_mtp=True, halt_threshold=None,
                modality_ids=None, **kw):
        b, s = ids.shape
        if cache is not None:
            if cache.position == 0:  # a fresh episode
                self.episode += 1
                self.cursor = 0
                self.speaking = False
            cache.position += s
        if int(ids[0, -1]) in OPEN:
            self.speaking = True
        logits = torch.full((b, s, 600), -20.0)
        nxt = TOK.eos_id
        script = self.scripts[min(self.episode, len(self.scripts) - 1)]
        if self.speaking and self.cursor < len(script):
            nxt = script[self.cursor]
            self.cursor += 1
            if nxt in CLOSE:
                self.speaking = False
        logits[:, -1, nxt] = 20.0
        return ProphetOutput(logits=logits, hidden=torch.zeros(b, s, 8), loop_k=1,
                             confidence=torch.full((b, s), self.confidence))


def _ids(*spans: str) -> list[int]:
    out: list[int] = []
    for span in spans:
        out += TOK.encode(span, parse_special=True)
    return out


def _call(action: dict) -> str:
    return json.dumps(action, separators=(",", ":")) + "<|/call|>"


def _perfect(task) -> list[int]:
    return _ids(
        "<|/think|>", _call({"name": "grep", "args": {"word": task.goal.split("word ")[1].split("?")[0]}}),
        "<|/think|>", _call({"name": "note", "args": {"text": f"answer: {task.answer}"}}),
        "<|/think|>", _call({"name": "done"}),
    )


def _wrong(task) -> list[int]:
    other = next(n for n in task.files if n != task.answer)
    return _ids(
        "<|/think|>", _call({"name": "note", "args": {"text": f"answer: {other}"}}),
        "<|/think|>", _call({"name": "done"}),
    )


def test_tasks_are_deterministic_and_grounded():
    a, b = make_tasks(4, seed=3), make_tasks(4, seed=3)
    assert [t.goal for t in a] == [t.goal for t in b]
    for task in a:
        word = task.goal.split("word ")[1].split("?")[0]
        holders = [n for n, text in task.files.items() if word in text.split()]
        assert holders == [task.answer]
        assert file_tools(task).run(__import__("prophet.agent.actions", fromlist=["Action"]).Action("grep", {"word": word})) == task.answer


def test_verifier_reads_the_notes():
    task = make_tasks(1)[0]
    from prophet.agent.state import AgentState

    assert verifier_for(task)(AgentState(goal="g", notes=f"it is {task.answer.upper()}"))
    assert not verifier_for(task)(AgentState(goal="g", notes="no idea"))


def _cfg() -> AgentConfig:
    return AgentConfig(max_steps=4, think_budget=3, action_budget=96, halt_threshold=None)


def test_perfect_script_passes_and_wrong_answer_is_refused(tmp_path):
    tasks = make_tasks(3, seed=1)
    good = run_bench(_Scripted([_perfect(t) for t in tasks]), TOK, tasks, _cfg())
    assert good.success_rate == 1.0 and all(e.verified for e in good.episodes)
    assert good.episodes[0].tool_calls == 1 and good.episodes[0].steps == 3

    q = Quarantine(tmp_path / "q.json")
    bad = run_bench(_Scripted([_wrong(t) for t in tasks]), TOK, tasks, _cfg(), quarantine=q)
    assert bad.success_rate == 0.0
    # The wrong `done` was refused by the verifier, so no episode counts as finished.
    assert all(not e.finished for e in bad.episodes)
    assert q.summary()["entries"] == 3 and q.summary()["promoted"] == 0


def test_learning_curve_is_per_block_in_order():
    r = BenchReport([
        EpisodeReport("t", finished=ok, verified=ok, steps=2, tool_calls=1, malformed=0, copied=0,
                      asked=False, reason="done")
        for ok in (False, False, True, True, True, True)
    ])
    assert r.learning_curve(block=2) == [0.0, 1.0, 1.0]
    assert r.learning_curve(block=4) == [0.5, 1.0]
    assert "curve [" in r.summary()


def test_session_can_be_carried_across_bench_episodes():
    tasks = make_tasks(2, seed=2)
    report = run_bench(_Scripted([_perfect(t) for t in tasks]), TOK, tasks, _cfg(), carry_session=True)
    assert report.n == 2
