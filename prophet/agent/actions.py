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
    "done": {"type": "object", "properties": {}, "required": []},
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
        if unknown:
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
    """Result of validating a partial call: still viable, complete, or dead.

    ``value_start`` marks a prefix that ends exactly where an argument value begins
    (after ``"key":``) -- the one place a copy pointer may fire; ``key`` and
    ``expected_type`` say which argument and what the schema expects there."""

    viable: bool
    complete: bool = False
    reason: str = ""
    value_start: bool = False
    tool: str | None = None
    key: str | None = None
    expected_type: str | None = None
    in_string: bool = False
    """The prefix ends inside an unterminated string value: the one place a sampled
    token cannot break the structure (``AgentConfig.sample_scope``)."""


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

    def __init__(self, registry: ToolRegistry, *, compact: bool = True,
                 ordered: bool = False) -> None:
        self.registry = registry
        self.names = registry.names
        self._all_names = tuple(registry.names)
        self.compact = compact
        self.ordered = ordered
        """Require the argument keys in the schema's order (docs/33 amendment 6). The
        renderer writes them in that order, so a model trained on rendered episodes has
        never seen another; leaving the order free lets a small model put a list where
        a word belongs."""
        """Reject whitespace outside strings. Calls are rendered compact
        (``separators=(",", ":")``), so a model trained on rendered episodes has never
        seen a space in a call; admitting one at decode let a drifting model open the
        span with indentation, wander off its training distribution and die before
        ``{`` -- the failure of the first closed-loop pilot (docs/32). ``compact=False``
        restores the tolerant JSON scanner."""

    def restrict(self, names: "set[str] | None", *, exclude: "frozenset[str]" = frozenset()) -> None:
        """Limit the tool names the grammar accepts -- what the selection head decided --
        or ``None`` to accept every registered name again. Reserved actions are never
        cut by ``names``: the head's "none" option is exactly "one of those". ``exclude``
        removes names after that, reserved ones included: the loop uses it to forbid the
        action of the previous step (``AgentConfig.no_repeat_action``)."""
        if names is None:
            keep = set(self._all_names)
        else:
            keep = set(names) | set(RESERVED_ACTIONS)
        self.names = tuple(n for n in self._all_names if n in keep and n not in exclude)

    # -- public ------------------------------------------------------------------------

    def _ws(self, s: str, i: int) -> int:
        """Whitespace outside strings: skipped when tolerant, dead when compact."""
        if not self.compact:
            return _skip_ws(s, i)
        if i < len(s) and s[i] in " \t\n\r":
            raise _Dead("whitespace outside strings; calls are rendered compact")
        return i


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
        i = self._ws(s, 0)
        if i == len(s):
            return PrefixState(True)
        i = _expect(s, i, "{")
        if i is None:
            return PrefixState(True)
        # "name"
        i = self._ws(s, i)
        key, i = _scan_string(s, i)
        if key is None:
            return PrefixState(True)
        if key.done and key.value != "name":
            raise _Dead("first key must be 'name'")
        if not key.done:
            if not "name".startswith(key.value):
                raise _Dead("first key must be 'name'")
            return PrefixState(True)
        i = self._ws(s, i)
        i2 = _expect(s, i, ":")
        if i2 is None:
            return PrefixState(True)
        i = self._ws(s, i2)
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

        i = self._ws(s, i)
        if i == len(s):
            return PrefixState(True)
        if s[i] == "}":
            if schema.required:
                raise _Dead(f"{name.value} requires {list(schema.required)}")
            return PrefixState(True, complete=True)
        i = _expect(s, i, ",")
        if i is None:
            return PrefixState(True)
        i = self._ws(s, i)
        key, i = _scan_string(s, i)
        if key is None:
            return PrefixState(True)
        if not key.done:
            if not "args".startswith(key.value):
                raise _Dead("second key must be 'args'")
            return PrefixState(True)
        if key.value != "args":
            raise _Dead("second key must be 'args'")
        i = self._ws(s, i)
        i2 = _expect(s, i, ":")
        if i2 is None:
            return PrefixState(True)
        i = self._ws(s, i2)
        i2 = _expect(s, i, "{")
        if i2 is None:
            return PrefixState(True)
        seen, i, closed, at_value, in_string = self._scan_args(s, i2, schema)
        if not closed:
            if at_value is not None:
                key_name, expected = at_value
                return PrefixState(
                    True, value_start=True, tool=name.value, key=key_name, expected_type=expected
                )
            return PrefixState(True, tool=name.value, in_string=in_string)
        missing = [r for r in schema.required if r not in seen]
        if missing:
            raise _Dead(f"{name.value}: missing required {missing}")
        i = self._ws(s, i)
        if i == len(s):
            return PrefixState(True)
        if s[i] != "}":
            raise _Dead("expected closing brace")
        tail = s[i + 1:].strip()
        if tail:
            raise _Dead("trailing characters after the call")
        return PrefixState(True, complete=True)

    def _scan_args(
        self, s: str, i: int, schema: ToolSchema
    ) -> tuple[set[str], int, bool, tuple[str, str | None] | None, bool]:
        """Returns ``(seen keys, index, closed, at_value, in_string)`` where ``at_value``
        is ``(key, expected type)`` when the prefix ends exactly at a value start and
        ``in_string`` says the prefix ends inside an unterminated string value."""
        props = schema.properties
        seen: set[str] = set()
        while True:
            i = self._ws(s, i)
            if i == len(s):
                return seen, i, False, None, False
            if s[i] == "}":
                return seen, i + 1, True, None, False
            if seen:
                if s[i] == "," and all(k in seen for k in props):
                    # Every parameter is given: the only continuation is the closing
                    # brace. A comma here led the decoder into a key that cannot exist.
                    raise _Dead(f"{schema.name}: all parameters given, expected '}}'")
                i2 = _expect(s, i, ",")
                if i2 is None:
                    return seen, i, False, None, False
                i = self._ws(s, i2)
            if not props:
                # An empty schema means "no parameters", not "anything goes": the renderer
                # never writes a key here, so the decoder must not admit one (docs/32, mode c).
                raise _Dead(f"{schema.name} takes no parameters")
            key, i = _scan_string(s, i)
            if key is None:
                return seen, i, False, None, False
            if not key.done:
                # A partial key must open a parameter not given yet: a prefix of a key
                # already seen is a dead end the decoder would otherwise walk into.
                candidates = [k for k in props if k.startswith(key.value) and k not in seen]
                if self.ordered:
                    # Strict order: the next key is the first one not given yet.
                    candidates = [k for k in candidates if k == next(p for p in props if p not in seen)]
                if not candidates:
                    raise _Dead(f"no unseen parameter of {schema.name} starts with {key.value!r}")
                return seen, i, False, None, False
            if key.value not in props:
                raise _Dead(f"{schema.name} has no parameter {key.value!r}")
            if self.ordered:
                expected_key = next(k for k in props if k not in seen)
                if key.value != expected_key:
                    raise _Dead(f"{schema.name}: expected parameter {expected_key!r}, "
                                f"found {key.value!r} (schema order)")
            if key.value in seen:
                raise _Dead(f"duplicate parameter {key.value!r}")
            i = self._ws(s, i)
            i2 = _expect(s, i, ":")
            if i2 is None:
                return seen, i, False, None, False
            i = self._ws(s, i2)
            expected = props.get(key.value, {}).get("type")
            if i == len(s):
                return seen, i, False, (key.value, expected), False
            done, i, in_string = _scan_value(s, i, expected, self._ws)
            if not done:
                return seen, i, False, None, in_string
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


_ESCAPES = frozenset('"\\/bfnrtu')
_HEX = frozenset("0123456789abcdefABCDEF")


def _escape_end(s: str, j: int) -> int | None:
    """Index after the escape at ``s[j] == "\\"``, or ``None`` when the prefix ends inside it.

    JSON admits eight one-letter escapes and ``\\uXXXX``; anything else makes a call that
    ``json.loads`` refuses, so the prefix is dead (docs/33 amendment 8)."""
    if j + 1 >= len(s):
        return None
    esc = s[j + 1]
    if esc not in _ESCAPES:
        raise _Dead(f"invalid escape \\{esc} in a string")
    if esc != "u":
        return j + 2
    digits = s[j + 2 : j + 6]
    if not all(h in _HEX for h in digits):
        raise _Dead("invalid \\u escape in a string")
    return j + 6 if len(digits) == 4 else None


def _check_string_char(c: str) -> None:
    """A raw control character (U+0000 to U+001F) inside a string: ``json.loads`` refuses
    the finished call, so no continuation can complete it. The renderer never writes one
    (``json.dumps`` escapes it); the decoder must not admit one either (docs/33
    amendment 8 -- a sampled newline in a proposal kept a doomed span "viable" to the
    end of its budget)."""
    if c < " ":
        raise _Dead(f"raw control character {c!r} in a string")


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
            end = _escape_end(s, j)
            if end is None:
                return _Str("".join(out), False), j
            out.append(s[j + 1])
            j = end
            continue
        if c == '"':
            return _Str("".join(out), True), j + 1
        _check_string_char(c)
        out.append(c)
        j += 1
    return _Str("".join(out), False), j


def _scan_value(
    s: str, i: int, expected: str | None, ws: Callable[[str, int], int]
) -> tuple[bool, int, bool]:
    """Scan one JSON value; return (complete, index_after, ends_inside_a_string). The last
    is true for an open string value and for an open string anywhere inside an array or
    an object, which is where value-only sampling draws (docs/33 amendment 9)."""
    c = s[i]
    if c == '"':
        if expected not in (None, "string"):
            raise _Dead(f"expected {expected}, found a string")
        v, j = _scan_string(s, i)
        done = v is not None and v.done
        return done, j, not done
    if c in "-0123456789":
        if expected not in (None, "integer", "number"):
            raise _Dead(f"expected {expected}, found a number")
        j = i
        while j < len(s) and s[j] in "-+.eE0123456789":
            j += 1
        if expected == "integer" and any(ch in s[i:j] for ch in ".eE"):
            raise _Dead("expected an integer")
        # A number is only known to be complete once a non-number character follows.
        return (j < len(s)), j, False
    if c in "tf":
        if expected not in (None, "boolean"):
            raise _Dead(f"expected {expected}, found a boolean")
        word = "true" if c == "t" else "false"
        frag = s[i:i + len(word)]
        if not word.startswith(frag):
            raise _Dead("malformed literal")
        return (frag == word), i + len(frag), False
    if c == "n":
        if expected not in (None, "null"):
            raise _Dead(f"expected {expected}, found null")
        frag = s[i:i + 4]
        if not "null".startswith(frag):
            raise _Dead("malformed literal")
        return (frag == "null"), i + len(frag), False
    if c == "[":
        if expected not in (None, "array"):
            raise _Dead(f"expected {expected}, found an array")
        return _scan_container(s, i, "[", "]", ws)
    if c == "{":
        if expected not in (None, "object"):
            raise _Dead(f"expected {expected}, found an object")
        return _scan_container(s, i, "{", "}", ws)
    raise _Dead(f"unexpected character {c!r}")


def _scan_container(
    s: str, i: int, open_ch: str, close_ch: str, ws: Callable[[str, int], int]
) -> tuple[bool, int, bool]:
    """Scan an array or an object as strict JSON: members separated by commas, an object's
    keys as strings followed by a colon, nested values recursively, whitespace as ``ws``
    says. The former scanner only counted brackets and strings, so ``{"a""b"}`` or
    ``[1 2]`` were "complete" calls that ``json.loads`` refuses; viable must mean
    completable (docs/33 amendment 9). Returns (complete, index_after, ends_inside_a_string)."""
    j = i + 1
    first = True
    while True:
        j = ws(s, j)
        if j == len(s):
            return False, j, False
        if s[j] == close_ch and first:
            return True, j + 1, False
        if not first:
            if s[j] == close_ch:
                return True, j + 1, False
            if s[j] != ",":
                raise _Dead(f"expected ',' or {close_ch!r} at {j}, found {s[j]!r}")
            j = ws(s, j + 1)
            if j == len(s):
                return False, j, False
        if open_ch == "{":
            key, j = _scan_string(s, j)
            if key is None:
                return False, j, False
            if not key.done:
                return False, j, True
            j = ws(s, j)
            if j == len(s):
                return False, j, False
            if s[j] != ":":
                raise _Dead(f"expected ':' at {j}, found {s[j]!r}")
            j = ws(s, j + 1)
            if j == len(s):
                return False, j, False
        done, j, in_string = _scan_value(s, j, None, ws)
        if not done:
            return False, j, in_string
        first = False


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

    def allowed(self, prefix: str, ranked_token_ids: Sequence[int], *,
                limit: int | None = -1) -> list[int]:
        """Token ids, from the ranked candidates, that keep ``prefix`` viable. ``limit``
        caps how many candidates are checked (default: ``candidates``; ``None``: all of
        them, the fallback when the model's head of the distribution has no viable
        continuation, e.g. after a sampled sub-word)."""
        state = self.grammar.check(prefix)
        if not state.viable:
            return []
        out: list[int] = []
        cap = self.candidates if limit == -1 else limit
        for tid in list(ranked_token_ids)[:cap]:
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
