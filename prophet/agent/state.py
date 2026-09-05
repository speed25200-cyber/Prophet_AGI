"""Agent state: the pinned prefix, the observation window, and O(1) rollback.

Three findings from track A2 shape this module. Context loss is about a quarter of
design-level long-horizon failures, and *compaction* -- summarising old context to make
room -- turns a zero constraint-violation rate into 30-59%. So the goal, constraints,
tool schemas and a short notes span are **pinned**: cached once and never compacted.
Tool I/O is kept verbatim in a bounded window; older observations are evicted from the
attention cache and survive only as what the bounded recurrent state carried forward.
That is state-carried compaction: nothing is rewritten, so nothing is rewritten wrongly.

The third finding is that agents cannot recover from a wrong action because they cannot
get back to the state before it. A snapshot of the cache after every step makes
``rollback(step)`` an O(1) restore rather than a re-execution, at a per-step cost that
is the size of the recurrent state plus the bounded attention windows -- constant in the
length of the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from prophet.modeling.layers import AttentionCache, RecurrentState
from prophet.modeling.model import ProphetCache

__all__ = ["CacheSnapshot", "snapshot_cache", "restore_cache", "Observation", "AgentState"]


@dataclass
class CacheSnapshot:
    """A complete, detached copy of a :class:`ProphetCache` at one point in time."""

    slots: dict[tuple[str, int, int], AttentionCache | RecurrentState]
    position: int
    loop_k: int | None

    def n_bytes(self) -> int:
        return sum(s.n_bytes() for s in self.slots.values())


def snapshot_cache(cache: ProphetCache) -> CacheSnapshot:
    """Clone every slot's tensors. Tensors inside a slot are replaced, never mutated in
    place, so a clone here is sufficient for the snapshot to stay valid."""
    slots: dict[tuple[str, int, int], AttentionCache | RecurrentState] = {}
    for key, slot in cache.slots.items():
        if isinstance(slot, AttentionCache):
            slots[key] = AttentionCache(
                keys=None if slot.keys is None else slot.keys.clone(),
                values=None if slot.values is None else slot.values.clone(),
                positions=None if slot.positions is None else slot.positions.clone(),
                window=slot.window, sink_tokens=slot.sink_tokens, seen=slot.seen,
            )
        else:
            slots[key] = RecurrentState(
                state=None if slot.state is None else slot.state.clone(),
                conv_state=None if slot.conv_state is None else slot.conv_state.clone(),
                seen=slot.seen,
            )
    return CacheSnapshot(slots=slots, position=cache.position, loop_k=cache.loop_k)


def restore_cache(cache: ProphetCache, snap: CacheSnapshot) -> None:
    """Put a cache back exactly as it was.

    The snapshot's tensors are cloned on the way in, so the snapshot itself is never
    aliased by the live cache and can be restored any number of times.
    """
    tmp = ProphetCache(slots=dict(snap.slots), position=snap.position, loop_k=snap.loop_k)
    fresh = snapshot_cache(tmp)
    cache.slots = fresh.slots
    cache.position = fresh.position
    cache.loop_k = fresh.loop_k


@dataclass
class Observation:
    step: int
    action_name: str
    text: str
    n_tokens: int
    cache_start: int
    """Absolute position where this observation's tokens begin in the cache."""
    cache_end: int


@dataclass
class AgentState:
    """Everything the loop needs to keep straight between steps."""

    goal: str
    notes: str = ""
    """The one span the model may rewrite, through the ``note`` action, capped in size.
    It is part of the pinned prefix in spirit, re-issued rather than edited in place."""
    notes_cap_tokens: int = 2048
    window_steps: int = 8
    """How many recent observations stay verbatim in the attention window."""

    step: int = 0
    window: list[Observation] = field(default_factory=list)
    snapshots: dict[int, CacheSnapshot] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    attempts_on_current: int = 0

    # -- snapshots and rollback --------------------------------------------------------

    def record_snapshot(self, cache: ProphetCache) -> None:
        self.snapshots[self.step] = snapshot_cache(cache)

    def rollback(self, cache: ProphetCache, to_step: int) -> bool:
        """Restore the cache to the state *before* ``to_step`` ran. O(1) in episode
        length: it is a restore, not a replay."""
        snap = self.snapshots.get(to_step)
        if snap is None:
            return False
        restore_cache(cache, snap)
        self.step = to_step
        self.window = [o for o in self.window if o.step < to_step]
        self.trajectory = [t for t in self.trajectory if t["step"] < to_step]
        for s in list(self.snapshots):
            if s > to_step:
                del self.snapshots[s]
        return True

    def snapshot_bytes(self) -> int:
        return sum(s.n_bytes() for s in self.snapshots.values())

    # -- loop detection -----------------------------------------------------------------

    def note_action(self, action_hash: str) -> int:
        """Count repeats of an identical action; step repetition is ~16% of failures."""
        self.action_counts[action_hash] = self.action_counts.get(action_hash, 0) + 1
        return self.action_counts[action_hash]

    # -- window ---------------------------------------------------------------------------

    def push_observation(self, obs: Observation) -> list[Observation]:
        """Add an observation; return the ones that fell out of the verbatim window and
        should be evicted from the attention cache."""
        self.window.append(obs)
        evicted: list[Observation] = []
        while len(self.window) > self.window_steps:
            evicted.append(self.window.pop(0))
        return evicted

    def evict_from_attention(self, cache: ProphetCache, obs: Observation) -> int:
        """Drop an observation's span from every attention slot. The recurrent state is
        untouched: it already folded the observation in, and that is the compaction."""
        dropped = 0
        for slot in cache.slots.values():
            if not isinstance(slot, AttentionCache) or slot.positions is None:
                continue
            keep = (slot.positions < obs.cache_start) | (slot.positions >= obs.cache_end)
            n_before = int(slot.positions.numel())
            if bool(keep.all()):
                continue
            idx = keep.nonzero(as_tuple=True)[0]
            slot.keys = slot.keys.index_select(2, idx)
            slot.values = slot.values.index_select(2, idx)
            slot.positions = slot.positions.index_select(0, idx)
            dropped += n_before - int(slot.positions.numel())
        return dropped
