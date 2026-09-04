"""Tests for the compute allocator.

Both bugs this file guards against were found by reading the allocator's *output* and
noticing the plan was wrong -- twice, in the same family. An allocator that quietly
produces a bad plan is worse than no allocator, because the plan looks authoritative.
"""

from __future__ import annotations

import pytest

from prophet.plan import ASKS, Ask, allocate, plan_report


def test_every_ask_is_well_formed():
    for ask in ASKS:
        assert ask.hours > 0, ask.name
        assert 1 <= ask.priority <= 5, ask.name
        assert ask.rationale, ask.name
        assert ask.kind in ("gate", "ablation", "production", "optional"), ask.name


def test_allocation_respects_the_reserve():
    _, summary = allocate(300.0, reserve_frac=0.10)
    assert summary["reserve_hours"] == pytest.approx(30.0)
    assert summary["allocated_hours"] <= summary["available_hours"] + 1e-9


def test_ties_break_by_declaration_order_not_by_cost():
    """The first bug: cheapest-first funded three small requests ahead of the 45-hour
    post-training stage, leaving a base model that is not a deliverable."""
    asks, _ = allocate(300.0)
    funded = [a for a in asks if a.satisfied]
    priorities = [a.priority for a in funded]
    assert priorities == sorted(priorities), "funding crossed priority bands out of order"

    # Within priority 1, an expensive item declared early must beat a cheap one after it.
    p1 = [a.name for a in funded if a.priority == 1]
    assert "Prophet-mini pretraining" in p1
    assert "post-training" in p1


def test_no_backfill_across_the_funding_line():
    """The second bug: an item that did not fit was skipped, and cheaper lower-priority
    work behind it was funded instead -- trading the thing the plan depends on for two
    things it does not."""
    asks, _ = allocate(300.0)
    order = {id(a): i for i, a in enumerate(sorted(asks, key=lambda x: (x.priority,)))}
    del order  # ordering is by declaration within a band; checked structurally below

    seen_unfunded = False
    for ask in sorted(
        enumerate(asks), key=lambda pair: (pair[1].priority, pair[0])
    ):
        _, a = ask
        if not a.satisfied:
            seen_unfunded = True
        elif seen_unfunded:
            pytest.fail(
                f"{a.track}/{a.name} was funded after an earlier request went unfunded"
            )


def test_persistent_memory_is_funded():
    """An explicit project decision: it is the one capability no competitor has."""
    asks, _ = allocate(300.0)
    memory = next(a for a in asks if a.name == "two-tier memory")
    assert memory.satisfied, "persistent memory dropped below the funding line"


def test_the_cheap_gates_are_funded_first():
    """Gates exist to close expensive tracks before they spend anything, so a gate that
    is itself cut has failed at its only job."""
    asks, _ = allocate(300.0)
    cheap_gates = [a for a in asks if a.kind == "gate" and a.hours <= 3.0]
    assert cheap_gates
    assert all(a.satisfied for a in cheap_gates)


def test_a_larger_budget_funds_at_least_as_much():
    small, _ = allocate(200.0)
    large, _ = allocate(600.0)
    assert sum(a.hours for a in large if a.satisfied) >= sum(
        a.hours for a in small if a.satisfied
    )


def test_oversubscription_is_reported_honestly():
    _, summary = allocate(300.0)
    assert summary["oversubscription"] > 1.0
    assert summary["requested_hours"] > summary["available_hours"]


def test_report_names_what_was_cut():
    text = plan_report(300.0)
    assert "Not funded" in text
    assert "no backfill" in text
    assert "Gates run first" in text


def test_gate_dependencies_point_at_real_work():
    names = {a.name for a in ASKS}
    for ask in ASKS:
        for blocked in ask.blocks:
            assert any(
                blocked.endswith(n) or n in blocked for n in names
            ), f"{ask.name} blocks unknown work {blocked!r}"
