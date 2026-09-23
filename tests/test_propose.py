"""Programme 3 (docs/33): proposals are validated by rules, tasks are derived by the
program, the out-of-distribution bench stays out of the generator's reach."""

import json

import pytest

from prophet.agent import tasks as task_families
from prophet.agent.actions import Action, ActionGrammar
from prophet.agent.propose import (
    FORMATS,
    GENERATOR_KEYS,
    HARD_KEYS,
    PROPOSE_GOAL,
    Spec,
    make_hard_lookup,
    novel,
    proposal_trajectory,
    propose_goal,
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


def test_object_format_round_trips_and_renders_as_a_call_the_ordered_grammar_accepts():
    """docs/33 amendment 9: the fields as one JSON object, shaped like the file the task
    will hold, instead of two lists aligned by position. Same specification, same rules,
    and the rendered call is exactly what the grammar the proposer decodes with accepts."""
    assert FORMATS == ("lists", "object")
    spec = validate("lookup", GOOD)
    args = spec.as_args("object")
    assert args == {
        "file": "orchid.json",
        "fields": {"city": "Lyon", "year": "1939", "code": "meadow"},
        "ask": "code",
    }
    assert validate("lookup", args) == spec
    for task in task_families.make_tasks(10, family="lookup", seed=5):
        s = spec_from_task(task)
        assert validate("lookup", s.as_args("object")) == s
        # The object is the file's own content: what the solver reads is what is proposed.
        assert (
            json.dumps(s.as_args("object")["fields"], separators=(",", ":")) in task.files.values()
        )
    reg = propose_registry("lookup", "object")
    assert propose_goal("lookup", "object") != propose_goal("lookup") == PROPOSE_GOAL["lookup"]
    text = render_episode(
        propose_goal("lookup", "object"), reg, proposal_trajectory(spec, "object")
    )
    body = text.split("<|call|>")[1].split("<|/call|>")[0]
    assert json.loads(body) == {"name": "propose_lookup", "args": args}
    grammar = ActionGrammar(reg, compact=True, ordered=True)
    assert grammar.check(body).complete
    assert reg.run(Action.parse(body)) == "ok"
    # The lists grammar does not admit the object call, nor the object grammar the lists one.
    assert not ActionGrammar(propose_registry("lookup"), ordered=True).check(body).viable
    with pytest.raises(ValueError):
        propose_registry("lookup", "pairs")


@pytest.mark.parametrize(
    "fields, reason",
    [
        ("city", "fields is not an object"),
        ({"city": "Lyon", "year": 1939}, "a field value is not a string"),
        ({"city": "Lyon"}, "between 2 and 6 fields"),
        ({"city": "Lyon", "year": "19 39"}, "is not a short alphanumeric token"),
    ],
)
def test_object_format_is_held_to_the_same_rules(fields, reason):
    verdict = validate("lookup", {"file": "orchid.json", "fields": fields, "ask": "city"})
    assert isinstance(verdict, str) and reason in verdict
    missing = validate(
        "lookup", {"file": "orchid.json", "fields": {"a": "b", "c": "d"}, "ask": "e"}
    )
    assert missing == "ask is not one of the keys"


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


def test_calc_proposals_are_one_field_whose_answer_the_executor_computes():
    """docs/39 SI-1: a calc proposal is one expression; the rules are grammar only, the
    answer is the executor's, and novelty is two operators or an integer outside the
    generator's 10-998 range (H25c)."""
    from prophet.agent.propose import CalcSpec

    for task in task_families.make_tasks(20, family="calc", seed=4):
        spec = spec_from_task(task)
        assert isinstance(spec, CalcSpec) and validate("calc", spec.as_args()) == spec
        again = task_from_spec(spec, name=task.name)
        assert again.goal == task.goal and again.answer == task.answer
        assert not novel(spec, [])  # the generator's own form is never new
    spec = validate("calc", {"expression": " 12 * 34 + 5 "})
    assert spec.expression == "12 * 34 + 5" and spec.size() == 3 and novel(spec, [])
    assert task_from_spec(spec, name="p").answer == "413"
    assert novel(validate("calc", {"expression": "1234 - 5"}), [])  # out of 10-998
    for bad in ("12*34", "12 / 3", "12", "99999 + 1", "1 + 2 + 3 + 4 + 5", 42):
        assert isinstance(validate("calc", {"expression": bad}), str), bad
    reg = propose_registry("calc")
    body = render_episode(propose_goal("calc"), reg, proposal_trajectory(spec))
    body = body.split("<|call|>")[1].split("<|/call|>")[0]
    assert json.loads(body) == {"name": "propose_calc", "args": {"expression": "12 * 34 + 5"}}
    assert ActionGrammar(reg, compact=True, ordered=True).check(body).complete


def test_the_calc_hard_bench_has_three_operands_and_is_deterministic():
    from prophet.agent.propose import make_hard, make_hard_calc

    a, b = make_hard_calc(10, seed=17), make_hard("calc", 10, seed=17)
    assert [t.goal for t in a] == [t.goal for t in b]
    for task in a:
        assert task.family == "calc" and len(task.extra["expression"].split()) == 5
        assert task.answer == str(eval(task.extra["expression"]))  # noqa: S307
    generator = {
        t.extra["expression"] for t in task_families.make_tasks(500, family="calc", seed=1)
    }
    assert not generator & {t.extra["expression"] for t in a}
    with pytest.raises(KeyError):
        make_hard("files", 2)
