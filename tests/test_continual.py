"""CPU-only tests for the memory-versus-skill metric and merge gate."""

from __future__ import annotations

import pytest

from prophet.eval.continual import (
    ConsolidationEvidence,
    SkillMeasurement,
    evaluate_consolidation,
)


def test_skill_ratio_distinguishes_a_lookup_table_from_transfer():
    lookup = SkillMeasurement(2.0, 1.5, 2.0, 2.0)
    skill = SkillMeasurement(2.0, 1.5, 2.0, 1.75)

    assert lookup.recall_gain == pytest.approx(0.5)
    assert lookup.transfer_gain == pytest.approx(0.0)
    assert lookup.skill_ratio == pytest.approx(0.0)
    assert skill.skill_ratio == pytest.approx(0.5)


def test_skill_ratio_preserves_negative_transfer():
    result = SkillMeasurement(2.0, 1.5, 2.0, 2.1)
    assert result.skill_ratio == pytest.approx(-0.2)


def test_skill_ratio_is_undefined_without_a_recall_gain():
    result = SkillMeasurement(2.0, 2.0, 2.0, 1.8)
    assert result.skill_ratio is None


def test_skill_ratio_is_undefined_for_a_numerically_tiny_recall_gain():
    result = SkillMeasurement(2.0, 2.0 - 1e-12, 2.0, 1.0)
    assert result.skill_ratio is None


def passing_evidence(**changes: object) -> ConsolidationEvidence:
    values = dict(
        general_bpb_before=1.500,
        general_bpb_after=1.504,
        skill_before=SkillMeasurement(2.0, 1.8, 2.0, 1.98),
        skill_after=SkillMeasurement(2.0, 1.8, 2.0, 1.94),
        recall_error_before=0.10,
        recall_error_after=0.14,
        addressing_jaccard=0.85,
        injection_success_before=0.01,
        injection_success_after=0.01,
    )
    values.update(changes)
    return ConsolidationEvidence(**values)


def test_consolidation_is_accepted_only_when_every_gate_passes():
    decision = evaluate_consolidation(passing_evidence())
    assert decision.accepted
    assert decision.failures == ()
    decision.require_acceptance()


def test_consolidation_reports_every_failed_gate():
    decision = evaluate_consolidation(
        passing_evidence(
            general_bpb_after=1.51,
            skill_after=SkillMeasurement(2.0, 1.999, 2.0, 2.01),
            recall_error_after=0.20,
            addressing_jaccard=0.50,
            injection_success_after=0.02,
        )
    )

    assert not decision.accepted
    assert len(decision.failures) == 8
    with pytest.raises(ValueError, match="consolidation rejected"):
        decision.require_acceptance()


def test_invalid_rates_are_rejected_before_a_merge_decision():
    with pytest.raises(ValueError, match="addressing_jaccard"):
        passing_evidence(addressing_jaccard=1.1)


def test_gate_rejects_an_exploding_ratio_from_negligible_acquisition():
    decision = evaluate_consolidation(
        passing_evidence(
            skill_before=SkillMeasurement(2.0, 2.0 - 1e-12, 2.0, 2.0),
            skill_after=SkillMeasurement(2.0, 2.0 - 1e-12, 2.0, 1.0),
        )
    )
    assert not decision.accepted
    assert any("recall gain" in failure for failure in decision.failures)
    assert decision.skill_ratio_after is None


def test_gate_rejects_less_negative_transfer_even_if_ratio_delta_improves():
    decision = evaluate_consolidation(
        passing_evidence(
            skill_before=SkillMeasurement(2.0, 1.8, 2.0, 2.06),
            skill_after=SkillMeasurement(2.0, 1.8, 2.0, 2.02),
        )
    )
    assert decision.skill_ratio_delta == pytest.approx(0.2)
    assert not decision.accepted
    assert any("candidate skill ratio" in failure for failure in decision.failures)


def test_gate_requires_matched_recall_acquisition():
    decision = evaluate_consolidation(
        passing_evidence(
            skill_before=SkillMeasurement(2.0, 1.8, 2.0, 1.98),
            skill_after=SkillMeasurement(2.0, 1.9, 2.0, 1.94),
        )
    )
    assert decision.recall_gain_retention == pytest.approx(0.5)
    assert not decision.accepted
    assert any("recall-gain retention" in failure for failure in decision.failures)
