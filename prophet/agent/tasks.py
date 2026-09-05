"""Verifiable task families for the agent: the dataset the agentic recipe trains on.

Every task here is generated from a seed, comes with the tools it needs, an executable
verifier that decides success, and a *perfect trajectory* -- the shortest sequence of
calls that passes. Rendered through :mod:`prophet.agent.render`, the perfect
trajectories are the supervised set (no licence, no annotator, no leakage from any
benchmark); the verifier makes the same tasks a benchmark; and the seed ranges keep the
two apart. Five families, chosen to cover the failure modes track A2 measured:

| family    | the agent must                                          | what it exercises            |
|-----------|---------------------------------------------------------|------------------------------|
| files     | find which file holds a word, note the name             | tool choice, copy from output|
| calc      | evaluate an expression with the calculator, note it     | copy from the goal, no CoT   |
| lookup    | read a JSON file, note one field's value                | copy from a structured output|
| count     | count a word across files, note the number              | tool over reasoning          |
| replace   | rewrite a file with one word replaced, then finish      | a generated long value       |

``replace`` is the deliberately hard one: its argument is not in context verbatim, so
the copy pointer cannot fill it and the model has to generate a whole file.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from prophet.agent.actions import Action, ToolRegistry, ToolSchema
from prophet.agent.state import AgentState

__all__ = ["Task", "FAMILIES", "make_tasks", "tools_for", "verifier_for", "perfect_trajectory"]

_WORDS = [
    "anchor", "beacon", "cinder", "delta", "ember", "falcon", "garnet", "harbor", "iris",
    "jasper", "kestrel", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz",
    "ripple", "saffron", "timber", "umber", "violet", "willow", "zephyr",
]


@dataclass
class Task:
    name: str
    family: str
    goal: str
    answer: str
    """What the verifier looks for (in the notes, or in a file for ``replace``)."""
    files: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------


def _files(rng: random.Random, n_files: int, words_per_file: int, target: str | None) -> tuple[dict[str, str], str]:
    names = [f"{rng.choice(_WORDS)}_{j}.txt" for j in range(n_files)]
    holder = rng.randrange(n_files)
    files = {}
    for j, name in enumerate(names):
        pool = [w for w in _WORDS if w != target]
        body = " ".join(rng.choice(pool) for _ in range(words_per_file))
        if target is not None and j == holder:
            parts = body.split()
            parts[rng.randrange(words_per_file)] = target
            body = " ".join(parts)
        files[name] = body
    return files, names[holder]


def _make_files(rng: random.Random, i: int, seed: int) -> Task:
    target = rng.choice(_WORDS)
    files, holder = _files(rng, 3, 6, target)
    goal = (f"Which file contains the word {target}? Use the tools, note the file name "
            "with the note action, then finish.")
    return Task(f"files-{seed}-{i}", "files", goal, holder, files, {"word": target})


def _make_calc(rng: random.Random, i: int, seed: int) -> Task:
    a, b = rng.randrange(10, 999), rng.randrange(10, 999)
    op = rng.choice(["+", "-", "*"])
    expression = f"{a} {op} {b}"
    result = str(eval(expression))  # noqa: S307 - integers and one operator, built here
    goal = f"Compute {expression} with the calc tool, note the result, then finish."
    return Task(f"calc-{seed}-{i}", "calc", goal, result, {}, {"expression": expression})


def _make_lookup(rng: random.Random, i: int, seed: int) -> Task:
    fields = {"city": rng.choice(["Lyon", "Oslo", "Kyoto", "Quito", "Perth"]),
              "year": str(rng.randrange(1900, 2030)), "code": rng.choice(_WORDS)}
    key = rng.choice(list(fields))
    name = f"{rng.choice(_WORDS)}.json"
    files = {name: json.dumps(fields, separators=(",", ":"))}
    goal = f"Read {name} and note the value of the field {key}, then finish."
    return Task(f"lookup-{seed}-{i}", "lookup", goal, fields[key], files, {"key": key, "file": name})


def _make_count(rng: random.Random, i: int, seed: int) -> Task:
    word = rng.choice(_WORDS)
    files, _ = _files(rng, 3, 8, None)
    # Plant the word a known number of times across the files.
    total = rng.randrange(1, 6)
    names = list(files)
    for _ in range(total):
        name = rng.choice(names)
        parts = files[name].split()
        parts[rng.randrange(len(parts))] = word
        files[name] = " ".join(parts)
    total = sum(f.split().count(word) for f in files.values())
    goal = f"How many times does the word {word} appear across the files? Use the count tool, note the number, then finish."
    return Task(f"count-{seed}-{i}", "count", goal, str(total), files, {"word": word})


def _make_replace(rng: random.Random, i: int, seed: int) -> Task:
    old = rng.choice(_WORDS)
    new = rng.choice([w for w in _WORDS if w != old])
    files, holder = _files(rng, 1, 5, old)
    expected = files[holder].replace(old, new)
    goal = f"In {holder}, replace the word {old} by {new} using read_file and write_file, then finish."
    return Task(f"replace-{seed}-{i}", "replace", goal, expected, files, {"old": old, "new": new, "file": holder})


FAMILIES: dict[str, Callable[[random.Random, int, int], Task]] = {
    "files": _make_files, "calc": _make_calc, "lookup": _make_lookup, "count": _make_count,
    "replace": _make_replace,
}


def make_tasks(n: int, *, family: str, seed: int = 0) -> list[Task]:
    """Deterministic tasks of one family. Use disjoint seeds for training and evaluation:
    the generator is the only source of tasks, so a seed is a split."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    rng = random.Random(f"{family}-{seed}")
    return [FAMILIES[family](rng, i, seed) for i in range(n)]


# --------------------------------------------------------------------------------------
# Tools, verifiers, perfect trajectories
# --------------------------------------------------------------------------------------


def tools_for(task: Task) -> ToolRegistry:
    reg = ToolRegistry()
    files = task.files  # shared, mutable: write_file edits it and the verifier reads it
    reg.add(ToolSchema("list_files", "List the file names", {"type": "object", "properties": {}}))
    reg.add(ToolSchema("read_file", "Read one file", {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
    }))
    reg.bind("list_files", lambda: "\n".join(sorted(files)))
    reg.bind("read_file", lambda path: files.get(path, f"no such file: {path}"))
    if task.family in ("files",):
        reg.add(ToolSchema("grep", "Names of the files containing a word", {
            "type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"],
        }))
        reg.bind("grep", lambda word: "\n".join(sorted(n for n, t in files.items() if word in t.split())) or "(none)")
    if task.family == "calc":
        reg.add(ToolSchema("calc", "Evaluate an arithmetic expression", {
            "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"],
        }))
        reg.bind("calc", _safe_calc)
    if task.family == "count":
        reg.add(ToolSchema("count", "How many times a word appears across all files", {
            "type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"],
        }))
        reg.bind("count", lambda word: str(sum(t.split().count(word) for t in files.values())))
    if task.family == "replace":
        reg.add(ToolSchema("write_file", "Replace a file's content", {
            "type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        }, irreversible=True))

        def write_file(path: str, text: str) -> str:
            files[path] = text
            return f"wrote {len(text)} chars to {path}"

        reg.bind("write_file", write_file)
    return reg


def _safe_calc(expression: str) -> str:
    """Integers and + - * only; anything else is an error string the agent can read."""
    allowed = set("0123456789 +-*")
    if not expression or set(expression) - allowed:
        return "error: only integers and + - * are supported"
    try:
        return str(eval(expression))  # noqa: S307 - character set restricted above
    except Exception as exc:  # noqa: BLE001 - reported to the agent, never raised
        return f"error: {exc}"


def verifier_for(task: Task) -> Callable[[AgentState], bool]:
    if task.family == "replace":
        return lambda state: task.files.get(task.extra["file"]) == task.answer
    return lambda state: task.answer.lower() in state.notes.lower()


def perfect_trajectory(task: Task) -> list[dict[str, Any]]:
    """The shortest passing episode, with real observations from the tools."""
    tools = tools_for(task)

    def step(i: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
        obs = "" if name in ("note", "done") else str(tools.run(Action(name, args)))
        return {"step": i, "think": "", "action": {"name": name, "args": args},
                "p_correct": None, "tier": None, "observation": obs}

    if task.family == "files":
        return [step(0, "grep", {"word": task.extra["word"]}),
                step(1, "note", {"text": task.answer}), step(2, "done", {})]
    if task.family == "calc":
        return [step(0, "calc", {"expression": task.extra["expression"]}),
                step(1, "note", {"text": task.answer}), step(2, "done", {})]
    if task.family == "lookup":
        return [step(0, "read_file", {"path": task.extra["file"]}),
                step(1, "note", {"text": task.answer}), step(2, "done", {})]
    if task.family == "count":
        return [step(0, "count", {"word": task.extra["word"]}),
                step(1, "note", {"text": task.answer}), step(2, "done", {})]
    if task.family == "replace":
        # Read, then write the edited content; the trajectory is built on a copy so the
        # task's files are left as generated for the verifier.
        original = dict(task.files)
        s0 = step(0, "read_file", {"path": task.extra["file"]})
        s1 = step(1, "write_file", {"path": task.extra["file"], "text": task.answer})
        task.files.clear()
        task.files.update(original)
        return [s0, s1, step(2, "done", {})]
    raise KeyError(task.family)
