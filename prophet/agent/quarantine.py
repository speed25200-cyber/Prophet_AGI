"""Quarantine: where an agent's experience waits before it is allowed to become memory.

Nothing an agent does is written to the ledger live. Track A2's failure taxonomy and
track A4's arithmetic both point the same way: a trajectory that *looks* successful is
wrong often enough (about one SWE-bench pass in ten is lucky; a learned verifier admits
30-40% wrong answers) that writing it to memory caps future accuracy below what
recomputing would give. The published record of repeated consolidation is utility rising
and then falling *below* the no-memory baseline.

So an episode enters quarantine with **provenance** -- which tier checked it, which
verifier version, the score, the depth disagreement, the attempt count -- and is promoted
by a rule, never by a flag. This replaces the ``verified: bool`` that track W4 rightly
said could not be audited, revoked, or used for eviction.

Promotion rule (from the verifier's tier semantics):

- ``GROUND_TRUTH`` promotes immediately.
- ``CONSENSUS`` promotes after three later consensus hits on the same family, or one
  ground-truth hit.
- ``LEARNED`` never promotes. It may have been acted on; it is not remembered.
- ``UNVERIFIED`` is refused at the door.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prophet.agent.verify import Tier

__all__ = ["Provenance", "Entry", "Quarantine"]


@dataclass
class Provenance:
    tier: int
    verifier_version: str
    p_correct: float
    depth_disagreement: float | None
    attempts: int
    agreements: int = 0
    recorded_at: float = field(default_factory=time.time)


@dataclass
class Entry:
    family: str
    """Task family the episode belongs to; promotion and consolidation are per family."""
    goal: str
    trajectory: list[dict[str, Any]]
    """Serialised steps: ``{"action": ..., "observation": ..., "verdict": ...}``."""
    outcome_passed: bool
    process_ok: bool
    """The agent verified before claiming done, and the pass was not lucky."""
    provenance: Provenance
    promoted: bool = False
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Entry":
        d = dict(d)
        d["provenance"] = Provenance(**d["provenance"])
        return cls(**d)


class Quarantine:
    """Durable, inspectable holding area with a promotion rule."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[Entry] = []
        if self.path and self.path.exists():
            self.entries = [Entry.from_dict(d) for d in json.loads(self.path.read_text())]

    # -- admission -------------------------------------------------------------------

    def add(self, entry: Entry) -> bool:
        """Admit an episode. Returns False when the tier does not permit admission."""
        if entry.provenance.tier == Tier.UNVERIFIED:
            return False
        entry.id = entry.id or f"{entry.family}:{len(self.entries)}:{int(entry.provenance.recorded_at)}"
        self.entries.append(entry)
        self._promote(entry)
        self._save()
        return True

    def _promote(self, entry: Entry) -> None:
        tier = entry.provenance.tier
        if tier == Tier.GROUND_TRUTH and entry.outcome_passed and entry.process_ok:
            entry.promoted = True
            # A ground-truth hit vouches for earlier consensus entries of the family.
            for e in self.entries:
                if e.family == entry.family and e.provenance.tier == Tier.CONSENSUS and e.outcome_passed:
                    e.promoted = True
        elif tier == Tier.CONSENSUS and entry.outcome_passed:
            later = [
                e for e in self.entries
                if e.family == entry.family and e.provenance.tier == Tier.CONSENSUS
                and e.outcome_passed
            ]
            if len(later) >= 3:
                for e in later:
                    e.promoted = True

    # -- queries ----------------------------------------------------------------------

    def promoted(self, family: str | None = None) -> list[Entry]:
        return [e for e in self.entries if e.promoted and (family is None or e.family == family)]

    def pending(self, family: str | None = None) -> list[Entry]:
        return [e for e in self.entries if not e.promoted and (family is None or e.family == family)]

    def replay(self, family: str | None = None, *, limit: int = 64) -> list[Entry]:
        """Promoted entries to interleave during consolidation, so writing new material
        does not quietly displace old."""
        out = self.promoted(family)
        return out[-limit:]

    def families(self) -> list[str]:
        return sorted({e.family for e in self.entries})

    def revoke(self, verifier_version: str) -> int:
        """Demote everything a discredited verifier version admitted. Provenance is what
        makes this possible; a boolean could not have been revoked."""
        n = 0
        for e in self.entries:
            if e.provenance.verifier_version == verifier_version and e.promoted:
                e.promoted = False
                n += 1
        self._save()
        return n

    def summary(self) -> dict[str, Any]:
        by_tier: dict[str, int] = {}
        for e in self.entries:
            by_tier[Tier(e.provenance.tier).name] = by_tier.get(Tier(e.provenance.tier).name, 0) + 1
        return {
            "entries": len(self.entries),
            "promoted": len(self.promoted()),
            "families": len(self.families()),
            "by_tier": by_tier,
        }

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([e.to_dict() for e in self.entries], indent=1))
