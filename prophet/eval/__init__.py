"""Evaluation primitives for quality, reasoning, and continual learning."""

from prophet.eval.continual import (
    MIN_STABLE_RECALL_GAIN,
    ConsolidationEvidence,
    ConsolidationGate,
    GateDecision,
    SkillMeasurement,
    evaluate_consolidation,
)

__all__ = [
    "MIN_STABLE_RECALL_GAIN",
    "ConsolidationEvidence",
    "ConsolidationGate",
    "GateDecision",
    "SkillMeasurement",
    "evaluate_consolidation",
]
