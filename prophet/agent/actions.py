"""Typed actions: what an agent can do, and the grammar that keeps it well-formed.

Track A3's finding reframed this module. The dominant tool-call failures at 1-4B
parameters are not formatting: they are *omission* (about two thirds of failures) and
wrong argument *values* (most of the rest). Grammar-constrained decoding fixes the minor
class -- syntax, hallucinated names, missing required keys -- at negligible cost, and
must apply **only inside the call span**: constraining the reasoning that precedes a call
costs capacity-limited models 28-36 points. So: think free, act typed.

Three things live here:

- :class:`ToolSchema` / :class:`Action` -- the typed objects, with a canonical hash so a
  loop detector can recognise "the same action again".
- :class:`ActionGrammar` -- a prefix validator for the JSON call syntax against the
  tool's schema. Given a partial call string it answers "could this still become a valid
  call?", which is exactly what constrained decoding needs at every step.
- :class:`ConstrainedDecoder` -- the reference decoder: masks the LM head to tokens whose
  decoded text keeps the prefix valid. It validates the top candidates one by one, which
  is fine for a reference and wrong for production; a compiled automaton (XGrammar-style)
  is the production path and this is the oracle it must agree with.

Reserved actions that every tool set has, from track A2: ``note`` rewrites the pinned
notes, ``verify`` runs a check, ``ask`` puts a question to the user with the action that
would resolve it, ``done`` claims the goal, ``rollback`` restores an earlier step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "ToolSchema",
    "Action",
    "RESERVED_ACTIONS",
    "IRREVERSIBLE_DEFAULT",
    "ToolRegistry",
    "ActionGrammar",
    "PrefixState",
    "ConstrainedDecoder",
]

RESERVED_ACTIONS: dict[str, dict[str, Any]] = {
    "note": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    "verify": {"type": "object", "properties": {"what": {"type": "string"}}, "required": []},
    "ask": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "proposed_action": {"type": "string"}},
        "required": ["question"],
    },
    "done": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": []},
    "rollback": {"type": "object", "properties": {"step": {"type": "integer"}}, "required": ["step"]},
}

#: Tool classes whose effects cannot be undone by the agent. The confidence gate applies
#: only to these; gating every action would cost a step per step for no protection.
IRREVERSIBLE_DEFAULT: frozenset[str] = frozenset(
    {"write_file", "delete", "git_commit", "submit", "purchase", "send", "execute_sql_write"}
)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]
    """A JSON-schema object: ``{"type": "object", "properties": {...}, "required": [...]}``."""
    irreversible: bool = False

    def body(self) -> str:
        """The schema as compact JSON -- what sits between the two control tokens."""
        return json.dumps(
            {"name": self.name, "description": self.description, "parameters": self.parameters},
            separators=(",", ":"),
        )

    def render(self) -> str:
        """The text form the model reads, one block per tool, closed by the anchor. A
        prompt builder splices the control ids explicitly (see ``AgentLoop``); this text
        form is for datasets and for ``parse_special=True`` encoding."""
        return f"<|tool_def|>{self.body()}<|/tool_def|>"

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.parameters.get("required", ()))

    @property
    def properties(self) -> dict[str, dict[str, Any]]:
        return dict(self.parameters.get("properties", {}))


@dataclass(frozen=True)
class Action:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> str:
        return json.dumps({"name": self.name, "args": self.args}, sort_keys=True,
                          separators=(",", ":"))

    def hash(self) -> str:
        """Stable identity for loop detection: the same tool with the same arguments."""
        return hashlib.blake2b(self.canonical().encode(), digest_size=8).hexdigest()

    @classmethod
    def parse(cls, text: str) -> "Action":
        data = json.loads(text)
        if not isinstance(data, dict) or "name" not in data:
            raise ValueError("an action is a JSON object with a 'name'")
        args = data.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("'args' must be an object")
        return cls(name=str(data["name"]), args=args)


class ToolRegistry:
    """The tools a task exposes, plus the reserved ones, plus their implementations."""

    def __init__(self, tools: Iterable[ToolSchema] = (), *, irreversible: Iterable[str] = ()) -> None:
        self._schemas: dict[str, ToolSchema] = {}
        self._impl: dict[str, Callable[..., Any]] = {}
        self._irreversible = set(IRREVERSIBLE_DEFAULT) | set(irreversible)
        for name, params in RESERVED_ACTIONS.items():
            self._schemas[name] = ToolSchema(name, f"reserved action: {name}", params)
        for t in tools:
            self.add(t)

    def add(self, schema: ToolSchema, impl: Callable[..., Any] | None = None) -> None:
        if schema.name in RESERVED_ACTIONS:
            raise ValueError(f"{schema.name!r} is a reserved action name")
        self._schemas[schema.name] = schema
        if impl is not None:
            self._impl[schema.name] = impl
        if schema.irreversible:
            self._irreversible.add(schema.name)

    def bind(self, name: str, impl: Callable[..., Any]) -> None:
        if name not in self._schemas:
            raise KeyError(f"unknown tool {name!r}")
        self._impl[name] = impl

    def schema(self, name: str) -> ToolSchema:
        return self._schemas[name]

    @property
    def names(self) -> list[str]:
        return list(self._schemas)

    def is_irreversible(self, name: str) -> bool:
        return name in self._irreversible

    def render(self) -> str:
        """Every non-reserved schema, in registration order, for the pinned prefix."""
        return "".join(s.render() for n, s in self._schemas.items() if n not in RESERVED_ACTIONS)

    def schemas(self) -> list[ToolSchema]:
        """Non-reserved schemas in registration order -- the anchors' order."""
        return [s for n, s in self._schemas.items() if n not in RESERVED_ACTIONS]

    def run(self, action: Action) -> Any:
        if action.name in RESERVED_ACTIONS:
            raise ValueError(f"reserved action {action.name!r} is handled by the loop, not a tool")
        if action.name not in self._impl:
            raise KeyError(f"tool {action.name!r} has no implementation bound")
        self.validate(action)
        return self._impl[action.name](**action.args)

    def validate(self, action: Action) -> None:
        """Check arguments against the schema: required keys, no unknown keys, types."""
        schema = self._schemas.get(action.name)
        if schema is None:
            raise KeyError(f"unknown tool {action.name!r}")
        props = schema.properties
        missing = [k for k in schema.required if k not in action.args]
        if missing:
            raise ValueError(f"{action.name}: missing required {missing}")
        unknown = [k for k in action.args if k not in props]
        if unknown and props:
            raise ValueError(f"{action.name}: unknown arguments {unknown}")
        for key, value in action.args.items():
            expected = props.get(key, {}).get("type")
            if expected and not _json_type_ok(value, expected):
                raise TypeError(f"{action.name}.{key}: expected {expected}, got {type(value).__name__}")


def _json_type_ok(value: Any, expected: str) -> bool:
    return {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }.get(expected, lambda v: True)(value)


# --------------------------------------------------------------------------------------
# Prefix grammar
# --------------------------------------------------------------------------------------


@dataclass
class PrefixState:
    """Result of validating a partial call: still viable, complete, or dead."""

    viable: bool
    complete: bool = False
    reason: str = ""


class ActionGrammar:
    """Prefix validator for ``{"name": <tool>, "args": {<schema-typed>}}``.

    Answers, for any partial string, whether it can still be extended into a valid call
    to one of the registered tools. Implemented as a tolerant JSON prefix scanner with a
    small amount of schema awareness on top: tool names must be a prefix of a known
    name, argument keys must be a prefix of a schema key, and a value's first character
    must be compatible with its declared type.

    It is deliberately conservative in one direction only: it never says "viable" for a
    string that cannot be completed. It may say "dead" late rather than early -- for
    example a duplicated key is only rejected when the key closes -- which costs a
    wasted token, not a malformed call.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.names = registry.names
        self._all_names = tuple(registry.names)

    def restrict(self, names: "set[str] | None") -> None:
        """Limit the tool names the grammar accepts -- what the selection head decided --
        or ``None`` to accept every registered name again. Reserved actions are never
        cut: the head's "none" option is exactly "one of those"."""
        if names is None:
            self.names = self._all_names
        else:
            keep = set(names) | set(RESERVED_ACTIONS)
            self.names = tuple(n for n in self._all_names if n in keep)

    # -- public ------------------------------------------------------------------------

    def check(self, partial: str) -> PrefixState:
        try:
            return self._scan(partial)
        except _Dead as dead:
            return PrefixState(False, False, str(dead))

    def complete(self, text: str) -> Action | None:
        """Parse a finished call, validated against its schema; ``None`` if not valid."""
        try:
            action = Action.parse(text)
            self.registry.validate(action)
            return action
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    # -- scanner -----------------------------------------------------------------------

    def _scan(self, s: str) -> PrefixState:
        i = _skip_ws(s, 0)
        if i == len(s):
            return PrefixState(True)
        i = _expect(s, i, "{")
        if i is None:
            return PrefixState(True)
        # "name"
        i = _skip_ws(s, i)
        key, i = _scan_string(s, i)
        if key is None:
            return PrefixState(True)
        if key.done and key.value != "name":
            raise _Dead("first key must be 'name'")
        if not key.done:
            if not "name".startswith(key.value):
                raise _Dead("first key must be 'name'")
            return PrefixState(True)
        i = _skip_ws(s, i)
        i2 = _expect(s, i, ":")
        if i2 is None:
            return PrefixState(True)
        i = _skip_ws(s, i2)
        name, i = _scan_string(s, i)
        if name is None:
            return PrefixState(True)
        if not name.done:
            if not any(n.startswith(name.value) for n in self.names):
                raise _Dead(f"no tool name starts with {name.value!r}")
            return PrefixState(True)
        if name.value not in self.names:
            raise _Dead(f"unknown tool {name.value!r}")
        schema = self.registry.schema(name.value)

        i = _skip_ws(s, i)
        if i == len(s):
            return PrefixState(True)
        if s[i] == "}":
            if schema.required:
                raise _Dead(f"{name.value} requires {list(schema.required)}")
            return PrefixState(True, complete=True)
        i = _expect(s, i, ",")
        if i is None:
            return PrefixState(True)
        i = _skip_ws(s, i)
        key, i = _scan_string(s, i)
        if key is None:
            return PrefixState(True)
        if not key.done:
            if not "args".startswith(key.value):
                raise _Dead("second key must be 'args'")
            return PrefixState(True)
        if key.value != "args":
            raise _Dead("second key must be 'args'")
        i = _skip_ws(s, i)
        i2 = _expect(s, i, ":")
        if i2 is None:
            return PrefixState(True)
        i = _skip_ws(s, i2)
        i2 = _expect(s, i, "{")
        if i2 is None:
            return PrefixState(True)
        seen, i, closed = self._scan_args(s, i2, schema)
        if not closed:
            return PrefixState(True)
        missing = [r for r in schema.required if r not in seen]
        if missing:
            raise _Dead(f"{name.value}: missing required {missing}")
        i = _skip_ws(s, i)
        if i == len(s):
            return PrefixState(True)
        if s[i] != "}":
            raise _Dead("expected closing brace")
        tail = s[i + 1:].strip()
        if tail:
            raise _Dead("trailing characters after the call")
        return PrefixState(True, complete=True)

    def _scan_args(self, s: str, i: int, schema: ToolSchema) -> tuple[set[str], int, bool]:
        props = schema.properties
        seen: set[str] = set()
        while True:
            i = _skip_ws(s, i)
            if i == len(s):
                return seen, i, False
            if s[i] == "}":
                return seen, i + 1, True
            if seen:
                i2 = _expect(s, i, ",")
                if i2 is None:
                    return seen, i, False
                i = _skip_ws(s, i2)
            key, i = _scan_string(s, i)
            if key is None:
                return seen, i, False
            if not key.done:
                if props and not any(k.startswith(key.value) for k in props):
                    raise _Dead(f"no parameter of {schema.name} starts with {key.value!r}")
                return seen, i, False
            if props and key.value not in props:
                raise _Dead(f"{schema.name} has no parameter {key.value!r}")
            if key.value in seen:
                raise _Dead(f"duplicate parameter {key.value!r}")
            i = _skip_ws(s, i)
            i2 = _expect(s, i, ":")
            if i2 is None:
                return seen, i, False
            i = _skip_ws(s, i2)
            if i == len(s):
                return seen, i, False
            expected = props.get(key.value, {}).get("type")
            done, i = _scan_value(s, i, expected)
            if not done:
                return seen, i, False
            seen.add(key.value)


class _Dead(Exception):
    pass


@dataclass
class _Str:
    value: str
    done: bool


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _expect(s: str, i: int, ch: str) -> int | None:
    if i >= len(s):
        return None
    if s[i] != ch:
        raise _Dead(f"expected {ch!r} at {i}, found {s[i]!r}")
    return i + 1


def _scan_string(s: str, i: int) -> tuple[_Str | None, int]:
    if i >= len(s):
        return None, i
    if s[i] != '"':
        raise _Dead(f"expected a string at {i}")
    j = i + 1
    out = []
    while j < len(s):
        c = s[j]
        if c == "\\":
            if j + 1 >= len(s):
                return _Str("".join(out), False), j
            out.append(s[j + 1])
            j += 2
            continue
        if c == '"':
            return _Str("".join(out), True), j + 1
        out.append(c)
        j += 1
    return _Str("".join(out), False), j


def _scan_value(s: str, i: int, expected: str | None) -> tuple[bool, int]:
    """Scan one JSON value; return (complete, index_after)."""
    c = s[i]
    if c == '"':
        if expected not in (None, "string"):
            raise _Dead(f"expected {expected}, found a string")
        v, j = _scan_string(s, i)
        return (v is not None and v.done), j
    if c in "-0123456789":
        if expected not in (None, "integer", "number"):
            raise _Dead(f"expected {expected}, found a number")
        j = i
        while j < len(s) and s[j] in "-+.eE0123456789":
            j += 1
        if expected == "integer" and any(ch in s[i:j] for ch in ".eE"):
            raise _Dead("expected an integer")
        # A number is only known to be complete once a non-number character follows.
        return (j < len(s)), j
    if c in "tf":
        if expected not in (None, "boolean"):
            raise _Dead(f"expected {expected}, found a boolean")
        word = "true" if c == "t" else "false"
        frag = s[i:i + len(word)]
        if not word.startswith(frag):
            raise _Dead("malformed literal")
        return (frag == word), i + len(frag)
    if c == "n":
        if expected not in (None, "null"):
            raise _Dead(f"expected {expected}, found null")
        frag = s[i:i + 4]
        if not "null".startswith(frag):
            raise _Dead("malformed literal")
        return (frag == "null"), i + len(frag)
    if c == "[":
        if expected not in (None, "array"):
            raise _Dead(f"expected {expected}, found an array")
        return _scan_container(s, i, "[", "]")
    if c == "{":
        if expected not in (None, "object"):
            raise _Dead(f"expected {expected}, found an object")
        return _scan_container(s, i, "{", "}")
    raise _Dead(f"unexpected character {c!r}")


def _scan_container(s: str, i: int, open_ch: str, close_ch: str) -> tuple[bool, int]:
    depth = 0
    in_str = False
    j = i
    while j < len(s):
        c = s[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return True, j + 1
        j += 1
    return False, j


# --------------------------------------------------------------------------------------
# Reference constrained decoder
# --------------------------------------------------------------------------------------


class ConstrainedDecoder:
    """Mask the LM head so the decoded call stays a valid prefix.

    Reference implementation: at each step it decodes the top ``candidates`` tokens and
    keeps those whose text extends the prefix validly. Correct, simple, and roughly
    ``candidates`` grammar checks per step -- fine as the oracle a compiled automaton is
    tested against, not as the thing that runs on a phone.
    """

    def __init__(self, grammar: ActionGrammar, decode_token: Callable[[int], str],
                 *, candidates: int = 64, end_id: int | None = None) -> None:
        self.grammar = grammar
        self.decode_token = decode_token
        self.candidates = candidates
        self.end_id = end_id

    def allowed(self, prefix: str, ranked_token_ids: Sequence[int]) -> list[int]:
        """Token ids, from the ranked candidates, that keep ``prefix`` viable."""
        state = self.grammar.check(prefix)
        if not state.viable:
            return []
        out: list[int] = []
        for tid in list(ranked_token_ids)[: self.candidates]:
            if tid == self.end_id:
                if state.complete:
                    out.append(tid)
                continue
            piece = self.decode_token(tid)
            if not piece:
                continue
            if self.grammar.check(prefix + piece).viable:
                out.append(tid)
        return out
