"""An executable agent benchmark, and the learning curve the continual-learning wall
asks for.

Track R11's rule holds for agents as it does for language modelling: under 500M
parameters most benchmarks sit at chance, so the harness must produce a signal that
moves. Two properties make this one usable before any weight is trained:

**Every task has an executable verifier.** Success is not judged by a model; the
``done`` action is accepted only when the verifier passes (``AgentLoop``'s ground-truth
tier), so a reported success is a real one and "finished but wrong" is counted as a
failure, not a partial credit.

**The curve, not the point.** A single success rate cannot tell a skill from a memory
(W3). :meth:`BenchReport.learning_curve` reports success per block of consecutive
episodes, with the model frozen; anything that rises does so through what was carried
between episodes -- the session state, the quarantine's promoted episodes, a ledger --
and that is the quantity the memory bets have to move. A flat curve on a frozen model
is the honest baseline, and it is what this harness reports today.

The task family is deliberately small and deterministic: find which of a few files
contains a word, note the file name, finish. Small enough that a scripted model can pass
it (which tests the harness), hard enough that a random one cannot (which tests the
verifier), and grounded in exactly the failure modes A2 measured -- omission, the
wrong tool, and an argument value that was sitting in context.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from prophet.agent.actions import ToolRegistry, ToolSchema
from prophet.agent.loop import AgentConfig, AgentLoop, EpisodeResult
from prophet.agent.state import AgentState

__all__ = [
    "FileTask",
    "make_tasks",
    "file_tools",
    "verifier_for",
    "EpisodeReport",
    "BenchReport",
    "run_bench",
]

_WORDS = [
    "anchor", "beacon", "cinder", "delta", "ember", "falcon", "garnet", "harbor", "iris",
    "jasper", "kestrel", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz",
    "ripple", "saffron", "timber", "umber", "violet", "willow", "zephyr",
]


@dataclass
class FileTask:
    name: str
    files: dict[str, str]
    goal: str
    answer: str
    """What the final notes must contain for the verifier to pass."""
    family: str = "files"


def make_tasks(n: int, *, seed: int = 0, n_files: int = 3, words_per_file: int = 6) -> list[FileTask]:
    """Deterministic tasks: one file holds the target word, the others do not."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        names = [f"{rng.choice(_WORDS)}_{j}.txt" for j in range(n_files)]
        target = rng.choice(_WORDS)
        holder = rng.randrange(n_files)
        files = {}
        for j, name in enumerate(names):
            pool = [w for w in _WORDS if w != target]
            body = " ".join(rng.choice(pool) for _ in range(words_per_file))
            if j == holder:
                at = rng.randrange(words_per_file)
                parts = body.split()
                parts[at] = target
                body = " ".join(parts)
            files[name] = body
        goal = (
            f"Which file contains the word {target}? Use the tools, note the file name "
            "with the note action, then finish."
        )
        tasks.append(FileTask(f"task-{seed}-{i}", files, goal, names[holder]))
    return tasks


def file_tools(task: FileTask) -> ToolRegistry:
    reg = ToolRegistry()
    reg.add(ToolSchema("list_files", "List the file names", {"type": "object", "properties": {}}))
    reg.add(ToolSchema("read_file", "Read one file", {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
    }))
    reg.add(ToolSchema("grep", "Names of the files containing a word", {
        "type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"],
    }))
    reg.bind("list_files", lambda: "\n".join(sorted(task.files)))
    reg.bind("read_file", lambda path: task.files.get(path, f"no such file: {path}"))
    reg.bind("grep", lambda word: "\n".join(sorted(n for n, t in task.files.items() if word in t.split())) or "(none)")
    return reg


def verifier_for(task: FileTask) -> Callable[[AgentState], bool]:
    def check(state: AgentState) -> bool:
        return task.answer.lower() in state.notes.lower()

    return check


@dataclass
class EpisodeReport:
    task: str
    finished: bool
    verified: bool
    steps: int
    tool_calls: int
    malformed: int
    copied: int
    asked: bool
    reason: str

    @property
    def success(self) -> bool:
        return self.finished and self.verified


@dataclass
class BenchReport:
    episodes: list[EpisodeReport] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return sum(e.success for e in self.episodes) / max(self.n, 1)

    @property
    def malformed_rate(self) -> float:
        total = sum(e.steps for e in self.episodes)
        return sum(e.malformed for e in self.episodes) / max(total, 1)

    @property
    def mean_steps(self) -> float:
        return sum(e.steps for e in self.episodes) / max(self.n, 1)

    def learning_curve(self, block: int = 8) -> list[float]:
        """Success rate per block of consecutive episodes, in order."""
        if block < 1:
            raise ValueError("block must be >= 1")
        return [
            sum(e.success for e in self.episodes[i : i + block]) / len(self.episodes[i : i + block])
            for i in range(0, self.n, block)
        ]

    def summary(self) -> str:
        curve = " ".join(f"{x:.2f}" for x in self.learning_curve())
        return (
            f"{self.n} episodes: success {self.success_rate:.1%}, "
            f"{self.mean_steps:.1f} steps/episode, malformed {self.malformed_rate:.1%}, "
            f"asked {sum(e.asked for e in self.episodes)}; curve [{curve}]"
        )


def run_bench(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[FileTask],
    cfg: AgentConfig,
    *,
    quarantine: Any | None = None,
    carry_session: bool = False,
    scorer: Any | None = None,
    make_loop: Callable[..., AgentLoop] | None = None,
) -> BenchReport:
    """Run every task once, in order, and report.

    ``carry_session`` passes each episode's recurrent state to the next -- the R03 bet
    applied to the agent -- which is the one thing that can bend the curve on a frozen
    model. ``quarantine`` collects the episodes for promotion and later consolidation.
    """
    report = BenchReport()
    session = None
    for task in tasks:
        tools = file_tools(task)
        loop = (make_loop or AgentLoop)(
            model, tokenizer, tools, cfg, quarantine=quarantine, verifier_tool=verifier_for(task),
            scorer=scorer,
        )
        result: EpisodeResult = loop.run(task.goal, session=session if carry_session else None)
        if carry_session:
            session = result.session
        report.episodes.append(EpisodeReport(
            task=task.name,
            finished=result.finished,
            verified=result.verified_before_done,
            steps=len(result.steps),
            tool_calls=sum(1 for s in result.steps if s.action is not None and s.action.name in tools.names
                           and s.action.name not in ("note", "verify", "ask", "done", "rollback")),
            malformed=sum(1 for s in result.steps if s.gated == "malformed"),
            copied=sum(s.copied for s in result.steps),
            asked=result.asked_user is not None,
            reason=result.reason,
        ))
    return report
