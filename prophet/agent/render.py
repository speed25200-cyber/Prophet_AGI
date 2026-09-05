"""From promoted episodes back to training text.

The quarantine (A4) holds what the agent did and whether it was verified; the loader
(:mod:`prophet.data.corpus`) reads documents. This module is the bridge that closes the
loop the continual-learning wall (W3) leaves open for an agent: a *verified* episode
becomes a document rendered in the same control-id format the action heads' targets
are read from (:func:`prophet.modeling.action.build_action_targets`), so consolidating
experience into the weights is the ordinary training path with one more source.

Only promoted entries are rendered. Promotion is the quarantine's job -- ground truth
immediately, consensus after agreement, learned never -- and a rendered episode carries
no provenance of its own, so nothing here may widen what promotion allowed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

from prophet.agent.actions import ToolRegistry
from prophet.agent.quarantine import Entry, Quarantine

__all__ = ["render_episode", "QuarantineSource"]


def render_episode(
    goal: str,
    tools: ToolRegistry,
    trajectory: list[dict[str, Any]],
    *,
    notes: str = "",
    final: str = "<|eos|>",
) -> str:
    """The episode as the text the agent loop would have produced.

    Layout mirrors ``AgentLoop``: system goal, one ``<|tool_def|>`` block per schema,
    notes, then per step an optional think span, the call (or ``<|nocall|>``), and the
    observation as a ``<|tool|>`` turn. Malformed steps (no action) are dropped: they
    are not a decision to learn.
    """
    parts = [f"<|system|>Goal: {goal}\n", tools.render(), f"\nNotes:\n{notes}\n<|assistant|>"]
    for step in trajectory:
        action = step.get("action")
        if action is None and step.get("gated") == "malformed":
            continue
        think = step.get("think") or ""
        if think:
            parts.append(f"<|think|>{think}<|/think|>")
        if action is None:
            parts.append("<|nocall|>")
        else:
            body = json.dumps(
                {"name": action["name"], "args": action.get("args", {})}, separators=(",", ":")
            )
            parts.append(f"<|call|>{body}<|/call|>")
        observation = step.get("observation") or ""
        if observation:
            parts.append(f"<|tool|>{observation}<|assistant|>")
    parts.append(final)
    return "".join(parts)


class QuarantineSource:
    """Promoted episodes as a :class:`prophet.data.corpus.DocumentSource`.

    Wrap it in ``TokenisedSource(..., parse_special=True)``: the text is rendered by
    this project, so its control strings are meant as control ids. ``registries`` maps
    a task family to the tool registry its episodes ran with; a family without one is
    skipped, since a call without its schema in context has no anchor to learn from.
    """

    def __init__(
        self,
        quarantine: Quarantine,
        registries: Mapping[str, ToolRegistry] | ToolRegistry,
        *,
        name: str = "quarantine",
        weight: float = 1.0,
        families: list[str] | None = None,
    ) -> None:
        self.quarantine = quarantine
        self.registries = registries
        self.name = name
        self.weight = weight
        self.families = families

    def _registry(self, family: str) -> ToolRegistry | None:
        if isinstance(self.registries, ToolRegistry):
            return self.registries
        return self.registries.get(family)

    def entries(self) -> list[Entry]:
        out = []
        for entry in self.quarantine.promoted():
            if self.families is not None and entry.family not in self.families:
                continue
            if self._registry(entry.family) is None:
                continue
            out.append(entry)
        return out

    def n_documents(self) -> int:
        return len(self.entries())

    def open(self, start: int) -> Iterator[str]:
        for entry in self.entries()[start:]:
            registry = self._registry(entry.family)
            assert registry is not None
            yield render_episode(entry.goal, registry, entry.trajectory)
