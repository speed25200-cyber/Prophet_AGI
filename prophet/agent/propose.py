"""Programme 3 (docs/33): the model proposes its own tasks under program-verified rules.

The generator of ``prophet.agent.tasks`` is a human's idea of what a task looks like; the
closed loop of docs/31 can never learn past it. Here the model emits a *specification*
through a typed tool call, a program checks it against rules that are grammar, not
judgement, and derives the task -- files, goal in the generator's own template, answer --
so that the usual executable verifier applies. The model never supplies an answer: the
rules compute it from the specification, the way the rules of a game score a position.

One family for now, ``lookup``, in two formats. ``lists`` (the default):
``{"file": "orchid.json", "keys": "city,year,code", "values": "Lyon,1939,meadow",
"ask": "code"}``, everything a string so the action grammar sees flat JSON. ``object``
(docs/33 amendment 9): ``{"file": "orchid.json", "fields": {"city": "Lyon", "year":
"1939", "code": "meadow"}, "ask": "code"}`` -- the fields in the very form the solver
reads in the file, with no two lists to align by position.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from prophet.agent.actions import ToolRegistry, ToolSchema
from prophet.agent.tasks import _WORDS, Task, _safe_calc

__all__ = [
    "FORMATS",
    "Spec",
    "propose_goal",
    "PROPOSE_GOAL",
    "propose_schema",
    "propose_registry",
    "validate",
    "task_from_spec",
    "spec_from_task",
    "proposal_trajectory",
    "novel",
    "make_hard_lookup",
    "CalcSpec",
    "make_hard",
    "make_hard_calc",
]

FAMILIES = ("lookup", "calc")
CALC_EXPRESSION = re.compile(r"^\d{1,4}( [+*-] \d{1,4}){1,3}$")
CALC_MAX_LENGTH = 32
GENERATOR_OPERANDS = (10, 998)
"""The calc generator draws two integers in this range, one operator (tasks._make_calc)."""
FORMATS = ("lists", "object")
"""How a proposal writes its fields: two comma-separated lists aligned by position, or
one JSON object shaped like the file the task will hold (docs/33 amendment 9)."""
MIN_FIELDS, MAX_FIELDS = 2, 6
TOKEN = re.compile(r"^[A-Za-z0-9]{1,12}$")
FILENAME = re.compile(r"^[A-Za-z0-9_]{1,12}\.json$")
GENERATOR_KEYS = ("city", "year", "code")
"""The keys the generator of ``prophet.agent.tasks`` ever uses."""
HARD_KEYS = ("owner", "color", "size", "port", "zone", "shape")
"""Keys the generator never uses: the out-of-distribution bench draws two of them."""
CITIES = ("Lyon", "Oslo", "Kyoto", "Quito", "Perth")

PROPOSE_GOAL = {
    "lookup": (
        "Propose a new lookup task: a JSON file name, its field keys and values "
        "(comma-separated), and the key to ask for."
    ),
    "calc": "Propose a new calc task: an arithmetic expression of integers with + - *.",
}
PROPOSE_GOAL_OBJECT = {
    "lookup": (
        "Propose a new lookup task: a JSON file name, its fields as an object of keys and "
        "values, and the key to ask for."
    ),
}


def propose_goal(family: str, fmt: str = "lists") -> str:
    """The goal of a proposal episode in format ``fmt``."""
    _check_format(fmt)
    if family == "calc":
        return PROPOSE_GOAL["calc"]  # one field: the format question does not arise
    return (PROPOSE_GOAL_OBJECT if fmt == "object" else PROPOSE_GOAL)[family]


def _check_format(fmt: str) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"proposal format {fmt!r} not in {FORMATS}")


@dataclass(frozen=True)
class Spec:
    """A validated proposal: what the program derives a task from."""

    family: str
    file: str
    keys: tuple[str, ...]
    values: tuple[str, ...]
    ask: str

    def as_args(self, fmt: str = "lists") -> dict[str, Any]:
        _check_format(fmt)
        if fmt == "object":
            return {
                "file": self.file,
                "fields": dict(zip(self.keys, self.values, strict=True)),
                "ask": self.ask,
            }
        return {
            "file": self.file,
            "keys": ",".join(self.keys),
            "values": ",".join(self.values),
            "ask": self.ask,
        }

    def signature(self) -> str:
        """What makes two proposals the same task."""
        return json.dumps([self.family, self.file, self.keys, self.values, self.ask])

    def size(self) -> int:
        """The number of fields, the proposal's size in the counts."""
        return len(self.keys)


@dataclass(frozen=True)
class CalcSpec:
    """A validated calc proposal: one expression, whose value the executor computes
    (docs/39 SI-1). One field, where the lookup proposer failed on two aligned lists."""

    family: str
    expression: str

    def as_args(self, fmt: str = "lists") -> dict[str, Any]:
        return {"expression": self.expression}

    def signature(self) -> str:
        return json.dumps([self.family, self.expression])

    def operands(self) -> list[int]:
        return [int(t) for t in self.expression.split()[::2]]

    def size(self) -> int:
        """The number of operands (the generator always writes two)."""
        return len(self.operands())


def propose_schema(family: str, fmt: str = "lists") -> ToolSchema:
    if family not in FAMILIES:
        raise KeyError(f"no proposal grammar for family {family!r}")
    _check_format(fmt)
    if family == "calc":
        return ToolSchema(
            "propose_calc",
            "Propose a calc task: an arithmetic expression of integers with + - *",
            {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
    if fmt == "object":
        return ToolSchema(
            f"propose_{family}",
            "Propose a lookup task: a file name, its fields as an object, the key to ask",
            {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "fields": {"type": "object"},
                    "ask": {"type": "string"},
                },
                "required": ["file", "fields", "ask"],
            },
        )
    return ToolSchema(
        f"propose_{family}",
        "Propose a lookup task: a file name, comma-separated keys and values, the key to ask",
        {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "keys": {"type": "string"},
                "values": {"type": "string"},
                "ask": {"type": "string"},
            },
            "required": ["file", "keys", "values", "ask"],
        },
    )


def propose_registry(family: str, fmt: str = "lists") -> ToolRegistry:
    """The one-tool registry a proposal episode runs with; the tool just acknowledges."""
    reg = ToolRegistry()
    reg.add(propose_schema(family, fmt))
    reg.bind(f"propose_{family}", lambda **_: "ok")
    return reg


def validate(family: str, args: Any) -> Spec | CalcSpec | str:
    """The rules. Returns a :class:`Spec` (or :class:`CalcSpec`), or the reason the proposal
    is refused. None of these is a judgement of interest or difficulty: only what a task
    must be to exist."""
    if family not in FAMILIES:
        return f"no proposal grammar for family {family!r}"
    if not isinstance(args, dict):
        return "arguments are not an object"
    if family == "calc":
        expression = args.get("expression")
        if not isinstance(expression, str):
            return "missing or non-string expression"
        expression = expression.strip()
        if len(expression) > CALC_MAX_LENGTH or not CALC_EXPRESSION.match(expression):
            return "expression must be 2 to 4 integers of 1 to 4 digits joined by ' + ', ' - ' or ' * '"
        if _safe_calc(expression).startswith("error"):
            return "the executor refuses the expression"
        return CalcSpec("calc", expression)
    if "fields" in args:
        # The object format (amendment 9): the same rules, read off the object.
        for key in ("file", "ask"):
            if not isinstance(args.get(key), str):
                return f"missing or non-string {key}"
        if not isinstance(args["fields"], dict):
            return "fields is not an object"
        if not all(isinstance(v, str) for v in args["fields"].values()):
            return "a field value is not a string"
        keys = tuple(k.strip() for k in args["fields"])
        values = tuple(v.strip() for v in args["fields"].values())
    else:
        for key in ("file", "keys", "values", "ask"):
            if not isinstance(args.get(key), str):
                return f"missing or non-string {key}"
        keys = tuple(k.strip() for k in args["keys"].split(","))
        values = tuple(v.strip() for v in args["values"].split(","))
    file = args["file"].strip()
    if not FILENAME.match(file):
        return "file name must be alphanumeric and end in .json"
    if len(keys) != len(values):
        return "keys and values differ in number"
    if not MIN_FIELDS <= len(keys) <= MAX_FIELDS:
        return f"between {MIN_FIELDS} and {MAX_FIELDS} fields"
    for token in (*keys, *values):
        if not TOKEN.match(token):
            return f"key or value {token!r} is not a short alphanumeric token"
    if len(set(keys)) != len(keys):
        return "duplicate key"
    ask = args["ask"].strip()
    if ask not in keys:
        return "ask is not one of the keys"
    return Spec(family, file, keys, values, ask)


def task_from_spec(spec: Spec | CalcSpec, *, name: str) -> Task:
    """The task a specification defines, in the generator's own template so the solver's
    policy applies unchanged; the answer is computed here, never taken from the model."""
    if isinstance(spec, CalcSpec):
        goal = f"Compute {spec.expression} with the calc tool, note the result, then finish."
        return Task(
            name,
            "calc",
            goal,
            _safe_calc(spec.expression),
            {},
            {"expression": spec.expression, "proposed": True, "n_fields": spec.size()},
        )
    fields = dict(zip(spec.keys, spec.values, strict=True))
    files = {spec.file: json.dumps(fields, separators=(",", ":"))}
    goal = f"Read {spec.file} and note the value of the field {spec.ask}, then finish."
    return Task(
        name,
        spec.family,
        goal,
        fields[spec.ask],
        files,
        {"key": spec.ask, "file": spec.file, "proposed": True, "n_fields": len(spec.keys)},
    )


def spec_from_task(task: Task) -> Spec | CalcSpec:
    """A generator task as the specification the model would have had to propose."""
    if task.family == "calc":
        return CalcSpec("calc", task.extra["expression"])
    if task.family not in FAMILIES or len(task.files) != 1:
        raise ValueError(f"{task.name}: not a single-file {FAMILIES} task")
    ((file, body),) = task.files.items()
    fields = json.loads(body)
    return Spec(
        task.family, file, tuple(fields), tuple(str(v) for v in fields.values()), task.extra["key"]
    )


def proposal_trajectory(spec: Spec, fmt: str = "lists") -> list[dict[str, Any]]:
    """The one-step episode that proposes ``spec`` in format ``fmt``, in the shape the
    renderer and the quarantine take."""
    return [
        {
            "step": 0,
            "think": "",
            "action": {"name": f"propose_{spec.family}", "args": spec.as_args(fmt)},
            "p_correct": None,
            "tier": None,
            "observation": "ok",
        }
    ]


def novel(spec: Spec | CalcSpec, seen: Iterable[Spec | CalcSpec]) -> bool:
    """Whether a proposal leaves the amorce's distribution: a number of fields or a key
    that no seen specification has (docs/33 H25); for calc, at least two operators or an
    integer outside the generator's range (docs/39 SI-1, H25c)."""
    if isinstance(spec, CalcSpec):
        lo, hi = GENERATOR_OPERANDS
        return spec.size() > 2 or any(not lo <= v <= hi for v in spec.operands())
    seen = list(seen)
    counts = {len(s.keys) for s in seen}
    keys = {k for s in seen for k in s.keys}
    return len(spec.keys) not in counts or any(k not in keys for k in spec.keys)


def make_hard_lookup(n: int, *, seed: int = 0) -> list[Task]:
    """The out-of-distribution bench: five fields, two of them under keys the generator
    never uses, the question on any of the five. Same template, same tools, same verifier
    as ``lookup``; no arm ever trains on it."""
    rng = random.Random(f"lookup-hard-{seed}")
    tasks = []
    for i in range(n):
        extra = rng.sample(HARD_KEYS, 2)
        fields = {
            "city": rng.choice(CITIES),
            "year": str(rng.randrange(1900, 2030)),
            "code": rng.choice(_WORDS),
            extra[0]: rng.choice(_WORDS),
            extra[1]: str(rng.randrange(1, 100)),
        }
        order = list(fields)
        rng.shuffle(order)
        fields = {k: fields[k] for k in order}
        key = rng.choice(order)
        name = f"{rng.choice(_WORDS)}.json"
        tasks.append(
            Task(
                f"lookup-hard-{seed}-{i}",
                "lookup",
                f"Read {name} and note the value of the field {key}, then finish.",
                fields[key],
                {name: json.dumps(fields, separators=(",", ":"))},
                {"key": key, "file": name, "hard": True, "n_fields": 5},
            )
        )
    return tasks


def make_hard_calc(n: int, *, seed: int = 0) -> list[Task]:
    """The out-of-distribution calc bench (docs/39 SI-1): three operands, which the
    generator never writes. Same template, tool and verifier as ``calc``."""
    rng = random.Random(f"calc-hard-{seed}")
    tasks = []
    for i in range(n):
        a, b, c = (rng.randrange(10, 999) for _ in range(3))
        expression = f"{a} {rng.choice('+-*')} {b} {rng.choice('+-*')} {c}"
        goal = f"Compute {expression} with the calc tool, note the result, then finish."
        tasks.append(
            Task(
                f"calc-hard-{seed}-{i}",
                "calc",
                goal,
                _safe_calc(expression),
                {},
                {"expression": expression, "hard": True},
            )
        )
    return tasks


def make_hard(family: str, n: int, *, seed: int = 0) -> list[Task]:
    """The out-of-distribution bench of a proposal family."""
    if family == "lookup":
        return make_hard_lookup(n, seed=seed)
    if family == "calc":
        return make_hard_calc(n, seed=seed)
    raise KeyError(f"no out-of-distribution bench for family {family!r}")
