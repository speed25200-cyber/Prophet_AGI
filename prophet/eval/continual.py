"""Metrics and acceptance gates for continual-learning experiments.

Recall on examples that were written to memory is not evidence of learning.  The
distinction Prophet needs is between *episode recall* and transfer to held-out members of
the same family after the original context has been removed.  This module makes that
distinction executable and keeps a potentially destructive consolidation delta separate
until it passes every safety gate.

All quality values below are bits per byte (BPB), where lower is better.  The module has
no PyTorch dependency so a proposed consolidation can be rejected before an accelerator
job or model merge is attempted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MIN_STABLE_RECALL_GAIN",
    "SkillMeasurement",
    "ConsolidationEvidence",
    "ConsolidationGate",
    "GateDecision",
    "evaluate_consolidation",
]


# Ratios below this denominator are numerical stories, not evidence.  The stricter
# experiment-level acquisition threshold lives in ConsolidationGate.
MIN_STABLE_RECALL_GAIN = 1e-6


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class SkillMeasurement:
    """Memory-off/on BPB on written examples and held-out sibling examples.

    ``seen`` contains only episodes used by the write or consolidation pass.  ``heldout``
    contains disjoint examples from the same generative family.  Context and warm session
    state must be cleared in both arms; otherwise the measurement confounds memory with a
    longer prompt.

    The skill ratio is ``transfer_gain / recall_gain``.  A value near zero describes a
    lookup table.  A positive value means some benefit reaches unseen sibling examples.
    Values above one and negative values are retained rather than clipped: both are useful
    diagnostics and clipping would hide leakage or interference.
    """

    seen_memory_off_bpb: float
    seen_memory_on_bpb: float
    heldout_memory_off_bpb: float
    heldout_memory_on_bpb: float

    def __post_init__(self) -> None:
        for name, value in (
            ("seen_memory_off_bpb", self.seen_memory_off_bpb),
            ("seen_memory_on_bpb", self.seen_memory_on_bpb),
            ("heldout_memory_off_bpb", self.heldout_memory_off_bpb),
            ("heldout_memory_on_bpb", self.heldout_memory_on_bpb),
        ):
            _finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @property
    def recall_gain(self) -> float:
        """BPB reduction on episodes that were written (higher is better)."""
        return self.seen_memory_off_bpb - self.seen_memory_on_bpb

    @property
    def transfer_gain(self) -> float:
        """BPB reduction on held-out siblings (higher is better)."""
        return self.heldout_memory_off_bpb - self.heldout_memory_on_bpb

    @property
    def skill_ratio(self) -> float | None:
        """Transfer divided by recall, or ``None`` without a stable recall signal."""
        if self.recall_gain <= MIN_STABLE_RECALL_GAIN:
            return None
        return self.transfer_gain / self.recall_gain


@dataclass(frozen=True)
class ConsolidationEvidence:
    """Held-out evidence collected before a reversible delta may be merged."""

    general_bpb_before: float
    general_bpb_after: float
    skill_before: SkillMeasurement
    skill_after: SkillMeasurement
    recall_error_before: float
    recall_error_after: float
    addressing_jaccard: float
    injection_success_before: float
    injection_success_after: float

    def __post_init__(self) -> None:
        if not isinstance(self.skill_before, SkillMeasurement) or not isinstance(
            self.skill_after, SkillMeasurement
        ):
            raise TypeError("skill_before and skill_after must be SkillMeasurement instances")
        for name, value in vars(self).items():
            if name in {"skill_before", "skill_after"}:
                continue
            _finite(name, value)
        for name in (
            "general_bpb_before",
            "general_bpb_after",
            "recall_error_before",
            "recall_error_after",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "addressing_jaccard",
            "injection_success_before",
            "injection_success_after",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")


@dataclass(frozen=True)
class ConsolidationGate:
    """Thresholds that must all pass; there is no weighted average escape hatch."""

    max_general_bpb_increase: float = 0.005
    min_recall_gain: float = 0.01
    min_recall_retention: float = 0.95
    min_skill_ratio_after: float = 0.05
    min_skill_ratio_gain: float = 0.10
    max_recall_error_increase: float = 0.05
    min_addressing_jaccard: float = 0.80
    max_injection_success_increase: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("min_recall_retention", "min_addressing_jaccard"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class GateDecision:
    """Auditable result of a merge decision."""

    accepted: bool
    failures: tuple[str, ...]
    general_bpb_delta: float
    recall_gain_before: float
    recall_gain_after: float
    recall_gain_retention: float | None
    skill_ratio_before: float | None
    skill_ratio_after: float | None
    skill_ratio_delta: float | None
    recall_error_delta: float
    injection_success_delta: float

    def require_acceptance(self) -> None:
        """Raise before a caller mutates the base checkpoint."""
        if not self.accepted:
            raise ValueError("consolidation rejected: " + "; ".join(self.failures))


def evaluate_consolidation(
    evidence: ConsolidationEvidence,
    gate: ConsolidationGate | None = None,
) -> GateDecision:
    """Evaluate every merge invariant and return all failures, not only the first."""
    gate = gate or ConsolidationGate()
    general_delta = evidence.general_bpb_after - evidence.general_bpb_before
    recall_before = evidence.skill_before.recall_gain
    recall_after = evidence.skill_after.recall_gain
    recall_retention = recall_after / recall_before if recall_before > 0 else None
    skill_before = evidence.skill_before.skill_ratio
    skill_after = evidence.skill_after.skill_ratio
    skill_delta = (
        skill_after - skill_before
        if skill_before is not None and skill_after is not None
        else None
    )
    recall_delta = evidence.recall_error_after - evidence.recall_error_before
    injection_delta = evidence.injection_success_after - evidence.injection_success_before

    failures: list[str] = []
    if general_delta > gate.max_general_bpb_increase:
        failures.append(
            f"general BPB increased by {general_delta:.6f} "
            f"(limit {gate.max_general_bpb_increase:.6f})"
        )
    if recall_before < gate.min_recall_gain:
        failures.append(
            f"baseline recall gain is {recall_before:.6f} "
            f"(minimum {gate.min_recall_gain:.6f})"
        )
    if recall_after < gate.min_recall_gain:
        failures.append(
            f"candidate recall gain is {recall_after:.6f} "
            f"(minimum {gate.min_recall_gain:.6f})"
        )
    if recall_retention is None or recall_retention < gate.min_recall_retention:
        rendered = "undefined" if recall_retention is None else f"{recall_retention:.6f}"
        failures.append(
            f"recall-gain retention is {rendered} "
            f"(minimum {gate.min_recall_retention:.6f})"
        )
    if skill_after is None or skill_after < gate.min_skill_ratio_after:
        rendered = "undefined" if skill_after is None else f"{skill_after:.6f}"
        failures.append(
            f"candidate skill ratio is {rendered} "
            f"(minimum {gate.min_skill_ratio_after:.6f})"
        )
    if skill_delta is None or skill_delta < gate.min_skill_ratio_gain:
        rendered = "undefined" if skill_delta is None else f"{skill_delta:.6f}"
        failures.append(
            f"skill ratio improved by {rendered} "
            f"(minimum {gate.min_skill_ratio_gain:.6f})"
        )
    if recall_delta > gate.max_recall_error_increase:
        failures.append(
            f"recall error increased by {recall_delta:.6f} "
            f"(limit {gate.max_recall_error_increase:.6f})"
        )
    if evidence.addressing_jaccard < gate.min_addressing_jaccard:
        failures.append(
            f"addressing Jaccard is {evidence.addressing_jaccard:.6f} "
            f"(minimum {gate.min_addressing_jaccard:.6f})"
        )
    if injection_delta > gate.max_injection_success_increase:
        failures.append(
            f"injection success increased by {injection_delta:.6f} "
            f"(limit {gate.max_injection_success_increase:.6f})"
        )

    return GateDecision(
        accepted=not failures,
        failures=tuple(failures),
        general_bpb_delta=general_delta,
        recall_gain_before=recall_before,
        recall_gain_after=recall_after,
        recall_gain_retention=recall_retention,
        skill_ratio_before=skill_before,
        skill_ratio_after=skill_after,
        skill_ratio_delta=skill_delta,
        recall_error_delta=recall_delta,
        injection_success_delta=injection_delta,
    )
