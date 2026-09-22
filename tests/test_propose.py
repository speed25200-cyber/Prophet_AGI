"""Programme 3 (docs/33): proposals are validated by rules, tasks are derived by the
program, the out-of-distribution bench stays out of the generator's reach."""

import json

import pytest

from prophet.agent import tasks as task_families
from prophet.agent.actions import Action, ActionGrammar
from prophet.agent.propose import (
    GENERATOR_KEYS,
    HARD_KEYS,
    PROPOSE_GOAL,
    Spec,
    make_hard_lookup,
    novel,
    proposal_trajectory,
    propose_registry,
    spec_from_task,
    task_from_spec,
    validate,
)
from prophet.agent.render import render_episode
from prophet.agent.state import AgentState

GOOD = {
    "file": "orchid.json",
    "keys": "city,year,code",
    "values": "Lyon,1939,meadow",
    "ask": "code",
}


def test_rules_accept_a_well_formed_proposal_and_derive_the_task():
    spec = validate("lookup", GOOD)
    assert isinstance(spec, Spec) and spec.keys == ("city", "year", "code") and spec.ask == "code"
    task = task_from_spec(spec, name="p-0")
    assert task.family == "lookup" and task.answer == "meadow"
    assert task.goal == "Read orchid.json and note the value of the field code, then finish."
    assert json.loads(task.files["orchid.json"]) == {
        "city": "Lyon",
        "year": "1939",
        "code": "meadow",
    }
    # The usual tools and verifier apply: the answer is what the verifier looks for.
    assert task_families.verifier_for(task)(AgentState(goal=task.goal, notes="it is meadow"))
    assert not task_families.verifier_for(task)(AgentState(goal=task.goal, notes="Lyon"))
    assert (
        task_families.tools_for(task).run(Action("read_file", {"path": "orchid.json"}))
        == task.files["orchid.json"]
    )


@pytest.mark.parametrize(
    "bad, reason",
    [
        ({**GOOD, "file": "orchid.txt"}, "file name"),
        ({**GOOD, "file": "a b.json"}, "file name"),
        ({**GOOD, "keys": "city", "values": "Lyon"}, "between"),
        ({**GOOD, "keys": "a,b,c,d,e,f,g", "values": "1,2,3,4,5,6,7"}, "between"),
        ({**GOOD, "keys": "city,year", "values": "Lyon"}, "differ in number"),
        ({**GOOD, "keys": "city,city,code"}, "duplicate"),
        ({**GOOD, "ask": "owner"}, "not one of the keys"),
        ({**GOOD, "values": "Ly on,1939,meadow"}, "alphanumeric"),
        ({**GOOD, "values": "averyveryverylongvalue,1939,meadow"}, "alphanumeric"),
        ({"file": "x.json", "keys": "a,b", "values": "1,2"}, "missing"),
        ({**GOOD, "ask": 3}, "non-string"),
        ("not an object", "not an object"),
    ],
)
def test_rules_refuse_what_is_not_a_task(bad, reason):
    verdict = validate("lookup", bad)
    assert isinstance(verdict, str) and reason in verdict


def test_generator_tasks_round_trip_through_their_specification():
    for task in task_families.make_tasks(20, family="lookup", seed=3):
        spec = spec_from_task(task)
        again = task_from_spec(spec, name=task.name)
        assert again.goal == task.goal and again.answer == task.answer and again.files == task.files
        assert validate("lookup", spec.as_args()) == spec


def test_a_proposal_renders_as_a_call_the_compact_grammar_accepts():
    spec = validate("lookup", GOOD)
    reg = propose_registry("lookup")
    text = render_episode(PROPOSE_GOAL["lookup"], reg, proposal_trajectory(spec))
    body = text.split("<|call|>")[1].split("<|/call|>")[0]
    assert json.loads(body) == {"name": "propose_lookup", "args": GOOD}
    grammar = ActionGrammar(reg, compact=True)
    state = grammar.check(body)
    assert state.viable and state.complete
    assert reg.run(Action.parse(body)) == "ok"


def test_novelty_is_a_field_count_or_a_key_the_amorce_never_had():
    amorce = [spec_from_task(t) for t in task_families.make_tasks(30, family="lookup", seed=1)]
    assert not novel(validate("lookup", GOOD), amorce)
    assert novel(
        validate("lookup", {**GOOD, "keys": "city,owner,code", "values": "Lyon,ana,meadow"}), amorce
    )
    assert novel(
        validate("lookup", {**GOOD, "keys": "city,year", "values": "Lyon,1939", "ask": "year"}),
        amorce,
    )


def test_hard_bench_is_out_of_the_generators_distribution_and_deterministic():
    hard = make_hard_lookup(30, seed=17)
    assert [t.name for t in hard] == [t.name for t in make_hard_lookup(30, seed=17)]
    assert hard[0].files != make_hard_lookup(30, seed=19)[0].files
    for task in hard:
        fields = json.loads(next(iter(task.files.values())))
        assert len(fields) == 5 and task.extra["hard"]
        assert sum(k in HARD_KEYS for k in fields) == 2 and all(k in fields for k in GENERATOR_KEYS)
        assert task.answer == fields[task.extra["key"]] and task.family == "lookup"
        assert task.goal.startswith("Read ") and "the field " in task.goal
    # The solver's usual tools serve it.
    assert "read_file" in task_families.tools_for(hard[0]).names
