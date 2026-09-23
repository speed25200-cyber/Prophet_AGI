"""The closed loop on a miniature model: rounds, promotion, accounting, resume, three arms."""

import json
import random

import pytest

from prophet.agent.quarantine import Quarantine
from prophet.data.tokenizer import ProphetTokenizer
from prophet.modeling.model import ProphetModel
from prophet.train.checkpoint import CheckpointManager
from scripts import closed_loop
from scripts.closed_loop import clean_trajectory, main
from tests.test_loop_core_corpus import paragraph
from tests.test_run_loop_core import tiny_config


def agent_tiny_config():
    """The miniature model with a two-layer coda whose global layer is NoPE: the copy
    pointer of the action heads scores its keys there (``heads.action_head``)."""
    import dataclasses

    cfg = tiny_config("gdn")
    return dataclasses.replace(
        cfg,
        n_layers=4,
        recurrent=dataclasses.replace(
            cfg.recurrent, coda_layers=2, coda_pattern=["swa", "full_attn"]
        ),
    )


SEQ_LEN = 256


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """What ``first_run_cpu.py`` leaves behind: tokenizer, corpus, held-out text, checkpoint."""
    root = tmp_path_factory.mktemp("first-run")
    ProphetTokenizer([], vocab_size=512).save(root / "tokenizer.json")
    rng = random.Random(0)
    for name in ("prose", "code"):
        path = root / "corpus" / name / "part-00000.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps({"text": paragraph(rng, 60)}) + "\n" for _ in range(20)))
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "heldout.jsonl").write_text(
        "".join(json.dumps({"text": paragraph(rng, 60)}) + "\n" for _ in range(6))
    )
    cfg = agent_tiny_config()
    cfg.to_json(root / "tiny.json")
    model = ProphetModel(cfg)
    CheckpointManager(root / "checkpoints").save({"model": model.state_dict(), "step": 3}, 3)
    return root


def run(work, out, arm, **extra):
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        arm,
        "--family",
        "calc",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--rounds",
        "2",
        "--tasks-per-round",
        "3",
        "--attempts",
        "2",
        "--steps-per-round",
        "2",
        "--seed-episodes",
        "3",
        "--seed-steps",
        "2",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "3",
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    assert main(argv) == 0
    return [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]


def test_oracle_arm_promotes_every_task_trains_and_accounts(work, tmp_path, capsys):
    out = tmp_path / "oracle"
    rounds = run(work, out, "oracle")
    assert "RUN_COMPLETE" in capsys.readouterr().out
    assert [r["round"] for r in rounds] == [0, 1, 2]
    assert (
        rounds[0]["seed"]["episodes"]["episodes"] == 3 and rounds[0]["seed"]["train"]["steps"] == 2
    )
    assert rounds[1]["generation"]["promoted_new"] == 3 and rounds[2]["promoted_total"] == 6
    assert rounds[1]["train"]["rows"] == 3 and rounds[2]["train"]["rows"] == 6
    assert rounds[2]["train"]["steps"] == 2 and rounds[2]["compute_seconds"] > 0
    for r in rounds:
        assert len(r["bench"]) == 2 and 0.0 <= r["success_mean"] <= 1.0
        assert r["bpb"]["docs"] == 3 and r["bpb"]["bpb"] > 0
    quarantine = json.loads((out / "quarantine.json").read_text())
    assert len(quarantine) == 6 and all(
        e["promoted"] and e["provenance"]["tier"] == 0 for e in quarantine
    )
    report = json.loads((out / "report.json").read_text())
    assert report["complete"] and report["quarantine"]["promoted"] == 6
    assert (out / "seed" / "seed.json").exists()

    # Resuming a finished run adds nothing; a changed recipe is refused.
    assert run(work, out, "oracle") == rounds
    with pytest.raises(ValueError, match="protocol changed"):
        run(work, out, "oracle", steps_per_round=3)


def test_closed_arm_only_learns_from_verified_episodes(work, tmp_path):
    out = tmp_path / "closed"
    rounds = run(work, out, "closed", seed_dir=tmp_path / "oracle-seed-missing")
    generation = rounds[1]["generation"]
    assert generation["tasks"] == 3 and generation["attempts"] == 2
    assert generation["episodes"] >= 3 and generation["tokens"] > 0 and generation["seconds"] > 0
    assert generation["promoted_new"] == generation["solved"] <= 3
    # Nothing admitted (unverified episodes are refused at the door) leaves no file.
    path = out / "quarantine.json"
    quarantine = json.loads(path.read_text()) if path.exists() else []
    assert all(e["promoted"] == (e["outcome_passed"] and e["process_ok"]) for e in quarantine)
    assert all(e["provenance"]["tier"] in (0, 2) for e in quarantine)
    total = sum(r["generation"]["promoted_new"] for r in rounds[1:])
    assert rounds[-1]["promoted_total"] == total
    if total == 0:
        assert rounds[1]["train"]["steps"] == 0, "nothing verified, nothing trained"
    else:
        assert rounds[-1]["train"]["rows"] == total


def test_frozen_arm_never_trains_and_shares_the_seed(work, tmp_path):
    seed_dir = tmp_path / "oracle" / "seed"
    run(work, tmp_path / "oracle", "oracle")
    rounds = run(work, tmp_path / "frozen", "frozen", seed_dir=seed_dir)
    assert rounds[0]["seed"]["reused"] == str(seed_dir) or "from_step" not in rounds[0]["seed"]
    assert all(r["train"] is None for r in rounds)
    assert all(r["generation"] is None or r["generation"]["episodes"] == 0 for r in rounds)
    assert rounds[1]["success_mean"] == rounds[2]["success_mean"], (
        "greedy bench on unchanged weights"
    )


def test_resume_continues_at_the_next_round(work, tmp_path):
    out = tmp_path / "resume"
    first = run(work, out, "oracle", rounds=1)
    assert [r["round"] for r in first] == [0, 1]
    # The same protocol with more rounds is a change; the same protocol resumed is not.
    with pytest.raises(ValueError, match="protocol changed"):
        run(work, out, "oracle", rounds=2)
    again = run(work, out, "oracle", rounds=1)
    assert again == first


def test_closed_klpo_arm_keeps_every_episode_and_runs_klpo_updates(work, tmp_path):
    out = tmp_path / "klpo"
    rounds = run(work, out, "closed-klpo", klpo_steps=2, klpo_draws=3)
    generation = rounds[1]["generation"]
    assert generation["episodes"] >= 3 and generation["policy_tokens"] > 0
    assert 0 <= generation["rewarded_episodes"] <= generation["episodes"]
    episodes = json.loads((out / "round-001-episodes.json").read_text())
    assert len(episodes) == generation["episodes"]
    assert all(e["reward"] in (0, 1) and e["ids"] and e["sampled"] for e in episodes)
    assert all(len(r["mc_ids"]) == 3 for e in episodes for r in e["sampled"])
    klpo = rounds[1]["klpo"]
    assert klpo["steps"] == 2 and len(klpo["losses"]) == 2 and klpo["seconds"] > 0
    assert klpo["beta"] == 0.1 and klpo["episodes"] == generation["episodes"]
    assert rounds[1]["compute_seconds"] > generation["seconds"] + (rounds[1]["train"] or {}).get(
        "seconds", 0
    )
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["klpo"]["draws"] == 3 and protocol["klpo"]["steps"] == 2


def test_lr_scale_reaches_the_round_trainer_but_not_the_seed(work, tmp_path, monkeypatch):
    """docs/31 amendment 5: --lr-scale multiplies the peak rates of the per-round training,
    is frozen in the protocol, and leaves the amorce at the full rate."""
    captured = []
    real = closed_loop.Trainer

    class Spy(real):
        def __init__(self, model, loader, tc, **kw):
            captured.append(tc)
            super().__init__(model, loader, tc, **kw)

    monkeypatch.setattr(closed_loop, "Trainer", Spy)
    out = tmp_path / "scaled"
    run(work, out, "oracle", lr_scale=0.25)
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["lr_scale"] == 0.25
    assert len(captured) == 3  # the seed, then one trainer per round
    assert captured[0].peak_lr_muon == pytest.approx(0.01)
    assert captured[0].peak_lr_adamw == pytest.approx(2e-3)
    for tc in captured[1:]:
        assert tc.peak_lr_muon == pytest.approx(0.0025)
        assert tc.peak_lr_adamw == pytest.approx(5e-4)


def test_resume_forgets_a_round_that_was_generated_but_never_recorded(work, tmp_path, capsys):
    """A crash between a round's generation and its record must not make the resumed run
    train on that round twice: entries carry their round and orphans are dropped."""
    out = tmp_path / "orphans"
    rounds = run(work, out, "oracle", rounds=2)
    assert [r["promoted_total"] for r in rounds] == [0, 3, 6]
    quarantine = Quarantine(out / "quarantine.json")
    assert {e.provenance.verifier_version for e in quarantine.promoted("calc")} == {
        "oracle-round-1",
        "oracle-round-2",
    }
    # Simulate the crash: round 2 was generated (its entries are in the quarantine) but
    # never recorded.
    lines = (out / "rounds.jsonl").read_text().splitlines(keepends=True)
    (out / "rounds.jsonl").write_text("".join(lines[:2]))
    rounds = run(work, out, "oracle", rounds=2)
    assert "PRUNED" in capsys.readouterr().out
    assert [r["promoted_total"] for r in rounds] == [0, 3, 6]
    assert len(Quarantine(out / "quarantine.json").promoted("calc")) == 6


def step(name, args=None, **extra):
    return {"step": 0, "think": "", "action": {"name": name, "args": args or {}}, **extra}


def test_clean_trajectory_is_the_canonical_form():
    assert clean_trajectory(
        [step("read_file", {"path": "a"}), step("note", {"text": "x"}), step("done")]
    )
    assert not clean_trajectory([])
    assert not clean_trajectory([step("read_file", {"path": "a"}), step("note", {"text": "x"})])
    assert not clean_trajectory(
        [
            step("read_file", {"path": "a"}),
            step("note", {"text": "x"}),
            step("note", {"text": "x"}),
            step("done"),
        ]
    )
    assert not clean_trajectory(
        [
            step("read_file", {"path": "a"}),
            {"step": 1, "think": "", "action": None, "gated": "malformed"},
            step("done"),
        ]
    )
    assert not clean_trajectory(
        [
            step("read_file", {"path": "a"}),
            step("done", observation="verification failed"),
            step("note", {"text": "x"}),
            step("done"),
        ]
    )


def test_closed_clean_arm_records_demotions_and_trains_on_canonical_episodes_only(work, tmp_path):
    out = tmp_path / "clean"
    rounds = run(work, out, "closed-clean")
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["arm"] == "closed-clean"
    for r in rounds[1:]:
        assert "demoted_sloppy" in r["generation"]
        assert r["generation"]["promoted_new"] >= 0
    quarantine = Quarantine(out / "quarantine.json")
    assert all(clean_trajectory(e.trajectory) for e in quarantine.promoted("calc"))


def test_no_repeat_action_flag_reaches_the_protocol_and_the_generation_config(work, tmp_path):
    out = tmp_path / "norepeat"
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        "frozen",
        "--family",
        "calc",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--rounds",
        "1",
        "--tasks-per-round",
        "2",
        "--attempts",
        "1",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
        "--no-repeat-action",
        "--no-repeat-emitted",
        "--sample-copy",
    ]
    assert main(argv) == 0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["no_repeat_action"] is True
    # docs/31 amendment 24: the refinement is recorded and reaches bench and generation.
    assert protocol["no_repeat_emitted"] is True and closed_loop.NO_REPEAT_EMITTED is True
    assert closed_loop.generation_config("calc", temperature=0.0).no_repeat_emitted
    closed_loop.NO_REPEAT_EMITTED = False
    assert closed_loop.NO_REPEAT_ACTION is True
    assert protocol["sample_copy"] is True and closed_loop.SAMPLE_COPY is True
    assert closed_loop.generation_config("calc", temperature=0.7, sample_copy=True).sample_copy
    assert not closed_loop.generation_config(
        "calc", temperature=0.0
    ).sample_copy  # the bench stays greedy
    closed_loop.SAMPLE_COPY = False
    assert closed_loop.generation_config(
        "calc", temperature=0.0, no_repeat_action=closed_loop.NO_REPEAT_ACTION
    ).no_repeat_action
    closed_loop.NO_REPEAT_ACTION = False
    # The refinement alone is refused: it says which name no_repeat_action forbids.
    with pytest.raises(SystemExit):
        main([a for a in argv if a != "--no-repeat-action"])
    closed_loop.NO_REPEAT_ACTION = closed_loop.NO_REPEAT_EMITTED = False


def test_two_families_share_one_loop_and_are_benched_apart(work, tmp_path):
    """docs/31 amendment 14: the rounds draw their tasks family by family, the quarantine
    files every episode under its own family, the rows are rendered with the registry of
    the episode's family, and the bench reports each family."""
    out = tmp_path / "mixed"
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        "oracle",
        "--family",
        "calc",
        "--family",
        "lookup",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--rounds",
        "1",
        "--tasks-per-round",
        "2",
        "--attempts",
        "1",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
    ]
    assert main(argv) == 0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["family"] == "calc+lookup" and "bench_families" not in protocol
    rounds = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    for record in rounds:
        assert [b["family"] for b in record["bench"]] == ["calc", "calc", "lookup", "lookup"]
        assert [b["seed"] for b in record["bench"]] == [7, 11, 7, 11]
        assert set(record["success_by_family"]) == {"calc", "lookup"}
        assert record["success_mean"] == pytest.approx(
            sum(record["success_by_family"].values()) / 2
        )
    generation = rounds[1]["generation"]
    assert generation["tasks"] == 4 and generation["promoted_new"] == 4
    assert {f: g["promoted_new"] for f, g in generation["by_family"].items()} == {
        "calc": 2,
        "lookup": 2,
    }
    assert rounds[1]["promoted_total"] == 4 and rounds[1]["train"]["rows"] == 4
    entries = Quarantine(out / "quarantine.json").promoted()
    assert sorted(e.family for e in entries) == ["calc", "calc", "lookup", "lookup"]
    # The amorce holds the perfect trajectories of both families, interleaved.
    assert json.loads((out / "seed" / "seed.json").read_text())["episodes"]["rows"] == 4


def test_the_bench_can_measure_families_the_rounds_do_not_train(work, tmp_path):
    out = tmp_path / "bench-wide"
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        "frozen",
        "--family",
        "calc",
        "--bench-family",
        "calc",
        "--bench-family",
        "lookup",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--rounds",
        "1",
        "--tasks-per-round",
        "2",
        "--attempts",
        "1",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
    ]
    assert main(argv) == 0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["family"] == "calc" and protocol["bench_families"] == "calc+lookup"
    rounds = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    assert [b["family"] for b in rounds[1]["bench"]] == ["calc", "calc", "lookup", "lookup"]
    assert rounds[1]["generation"]["tasks"] == 2 and "by_family" not in rounds[1]["generation"]
    # A family named twice is refused.
    with pytest.raises(SystemExit):
        main(argv + ["--family", "calc"])


def test_a_single_family_writes_the_protocol_as_before(work, tmp_path):
    """A run started before ``--family`` could repeat resumes under the new code: its
    protocol has ``family`` as a plain name and no ``bench_families``."""
    out = tmp_path / "single"
    rounds = run(work, out, "frozen", rounds=0)
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["family"] == "calc" and "bench_families" not in protocol
    assert [b["family"] for b in rounds[0]["bench"]] == ["calc", "calc"]
    assert rounds[0]["success_by_family"] == {"calc": rounds[0]["success_mean"]}


def test_copy_topk_explores_from_the_second_attempt_only(work, tmp_path, monkeypatch):
    """docs/31 amendment 15: the first attempt keeps the policy's pointer, the retries
    draw it among the best K positions; the protocol records the option only when set."""
    from prophet.agent import tasks as task_families
    from prophet.agent.quarantine import Quarantine as Q
    from scripts.closed_loop import generate_round

    seen = []
    real = closed_loop.run_bench

    def spy(model, tokenizer, tasks, cfg, **kw):
        seen.append(cfg.copy_topk)
        return real(model, tokenizer, tasks, cfg, **kw)

    monkeypatch.setattr(closed_loop, "run_bench", spy)
    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    model = ProphetModel(agent_tiny_config()).eval()
    tasks = task_families.make_tasks(2, family="calc", seed=5)
    generation = generate_round(
        model,
        tokenizer,
        "calc",
        tasks,
        Q(tmp_path / "q.json"),
        round_index=1,
        attempts=3,
        temperature=0.7,
        copy_topk=3,
    )
    assert seen[:1] == [0] and all(k == 3 for k in seen[1:]) and len(seen) >= 1
    assert generation["promoted_explored"] >= 0
    # docs/39 amendment 2: the end draw rides on the same exploring attempts.
    ends = []
    monkeypatch.setattr(
        closed_loop,
        "run_bench",
        lambda model, tokenizer, tasks, cfg, **kw: (
            ends.append(cfg.copy_explore_end) or real(model, tokenizer, tasks, cfg, **kw)
        ),
    )
    generate_round(
        model,
        tokenizer,
        "calc",
        tasks,
        Q(tmp_path / "q3.json"),
        round_index=1,
        attempts=2,
        temperature=0.7,
        copy_topk=3,
        explore_from_attempt=1,
        copy_explore_end=True,
    )
    assert ends and all(ends)
    monkeypatch.setattr(closed_loop, "run_bench", spy)
    seen.clear()
    generate_round(
        model,
        tokenizer,
        "calc",
        tasks,
        Q(tmp_path / "q2.json"),
        round_index=1,
        attempts=2,
        temperature=0.7,
        copy_topk=3,
        explore_from_attempt=1,
    )
    assert seen[:1] == [3]
    out = tmp_path / "topk"
    run(work, out, "frozen", rounds=0, copy_topk=3)
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["copy_topk"] == 3 and protocol["explore_from_attempt"] == 2
    run(work, tmp_path / "plain", "frozen", rounds=0)
    assert "copy_topk" not in json.loads((tmp_path / "plain" / "protocol.json").read_text())
    run(work, tmp_path / "obs", "frozen", rounds=0, copy_topk=2, copy_explore="observations")
    protocol = json.loads((tmp_path / "obs" / "protocol.json").read_text())
    assert protocol["copy_explore"] == "observations" and protocol["copy_topk"] == 2
    assert (
        closed_loop.generation_config(
            "calc", temperature=0.7, copy_topk=2, copy_explore="observations"
        ).copy_explore
        == "observations"
    )
    run(work, tmp_path / "bounds", "frozen", rounds=0, copy_boundaries="explore")
    protocol = json.loads((tmp_path / "bounds" / "protocol.json").read_text())
    assert protocol["copy_boundaries"] == "explore" and closed_loop.COPY_BOUNDARIES == "explore"
    # The bench and the generation read the same switch.
    assert closed_loop.generation_config("calc", temperature=0.0).copy_boundaries == "explore"
    run(work, tmp_path / "plain2", "frozen", rounds=0)
    assert "copy_boundaries" not in json.loads((tmp_path / "plain2" / "protocol.json").read_text())
    assert closed_loop.COPY_BOUNDARIES == "off"


def test_device_and_replay_names_reach_the_trainer_and_the_protocol(work, tmp_path, monkeypatch):
    """The A100 launcher passes --device cuda and the loop-core corpus names; the CPU
    pilots never set either, and their protocol is unchanged."""
    seen = {}
    real_trainer = closed_loop.Trainer
    real_replay = closed_loop.replay_source

    class Spy(real_trainer):
        def __init__(self, model, loader, tc, **kw):
            seen.setdefault("devices", []).append(tc.device)
            super().__init__(model, loader, tc, **kw)

    def replay_spy(work_dir, tokenizer, weight, names=("prose", "code")):
        seen.setdefault("names", []).append(tuple(names))
        return real_replay(work_dir, tokenizer, weight, names=names)

    monkeypatch.setattr(closed_loop, "Trainer", Spy)
    monkeypatch.setattr(closed_loop, "replay_source", replay_spy)
    out = tmp_path / "dev"
    run(work, out, "oracle", rounds=1, device="cpu", replay_names="prose,code")
    protocol = json.loads((out / "protocol.json").read_text())
    assert "device" not in protocol and "replay_names" not in protocol
    assert set(seen["devices"]) == {"cpu"} and set(seen["names"]) == {("prose", "code")}
    seen.clear()
    (work / "corpus" / "other").mkdir(exist_ok=True)
    for name in ("prose", "code"):
        for path in (work / "corpus" / name).glob("*.jsonl"):
            (work / "corpus" / "other" / f"{name}-{path.name}").write_text(path.read_text())
    out2 = tmp_path / "names"
    run(work, out2, "oracle", rounds=1, replay_names="other")
    protocol = json.loads((out2 / "protocol.json").read_text())
    assert protocol["replay_names"] == ["other"] and set(seen["names"]) == {("other",)}


def test_cuda_is_refused_when_absent(work, tmp_path):
    import torch

    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present; the refusal cannot be exercised")
    with pytest.raises(SystemExit):
        run(work, tmp_path / "cuda", "frozen", rounds=0, device="cuda")


def test_proposal_rows_render_perfect_proposals_for_the_amorce(work):
    from prophet.agent.propose import validate
    from scripts.closed_loop import proposal_rows

    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    # The bare byte-level test tokenizer needs ~580 ids for a proposal row.
    rows, specs, stats = proposal_rows(tokenizer, 3, family="lookup", seed=5, seq_len=768)
    assert stats["rows"] == 3 and stats["truncated"] == 0 and all(len(r) == 768 for r in rows)
    assert all(validate("lookup", s.as_args()) == s for s in specs)
    text = tokenizer.decode(rows[0], skip_special=False)
    assert "propose_lookup" in text and '"ask":"' in text


def test_propose_round_counts_malformed_invalid_duplicate_and_valid(work, monkeypatch):
    import types

    from prophet.agent import loop as loop_module
    from prophet.agent.actions import Action
    from scripts.closed_loop import propose_round

    good = {
        "file": "orchid.json",
        "keys": "city,year,code",
        "values": "Lyon,1939,meadow",
        "ask": "code",
    }
    scripted = iter(
        [
            Action("propose_lookup", good),
            Action("propose_lookup", good),  # the same task again: a duplicate
            Action("propose_lookup", {**good, "ask": "owner"}),  # refused by the rules
            None,  # no call at all: malformed
            Action(
                "propose_lookup",
                {**good, "keys": "city,owner", "values": "Lyon,ana", "ask": "owner"},
            ),
        ]
    )

    budgets = []
    scopes = []

    def fake_run(self, goal, **kw):
        budgets.append(self.cfg.action_budget)
        scopes.append(
            (
                self.cfg.sample_scope,
                self.cfg.decoder_widen,
                self.cfg.sample_actions,
                self.cfg.sample_topk,
                self.cfg.allow_copy,
                self.cfg.ordered_keys,
                self.cfg.copy_keys,
            )
        )
        action = next(scripted)
        if action is None:  # a malformed call: the loop keeps its span
            step = types.SimpleNamespace(action=None, gated="malformed", span='{"file":"orc')
        else:
            step = types.SimpleNamespace(action=action, gated="")
        return types.SimpleNamespace(steps=[step], tokens=7)

    monkeypatch.setattr(loop_module.AgentLoop, "run", fake_run)
    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    model = ProphetModel(agent_tiny_config()).eval()
    seen = set()
    proposals, counts = propose_round(
        model, tokenizer, "lookup", 5, seen=seen, amorce_specs=[], temperature=0.7, round_index=1
    )
    assert counts["valid"] == 2 and counts["duplicate"] == 1
    assert counts["invalid"] == 1 and counts["malformed"] == 1 and counts["tokens"] == 35
    assert counts["novel"] == 2 and counts["n_fields"] == {"3": 1, "2": 1}
    assert [t.answer for _, t in proposals] == ["meadow", "ana"]
    # Every proposal is kept for reading, in order (docs/33 amendment 7).
    samples = counts["samples"]
    assert [x["verdict"] for x in samples] == [
        "valid",
        "duplicate",
        "invalid",
        "malformed",
        "valid",
    ]
    assert samples[0]["args"] == good and samples[2]["reason"] == "ask is not one of the keys"
    assert samples[3] == {"verdict": "malformed", "span": '{"file":"orc'}
    assert all(t.family == "lookup" and t.extra["proposed"] for _, t in proposals)
    assert len(seen) == 2
    # A proposal call is ~60 tokens: the loop's 64-token action budget is raised for it
    # (docs/33 amendment 1).
    assert budgets and all(b >= 160 for b in budgets)
    assert set(scopes) == {("values", True, True, 5, False, True, None)}  # amendments 2 to 6
    # Amendment 10: --propose-copy ask reopens the pointer for ask, and for ask only.
    scripted = iter([Action("propose_lookup", good)])
    scopes.clear()
    propose_round(
        model,
        tokenizer,
        "lookup",
        1,
        seen=set(),
        amorce_specs=[],
        temperature=0.7,
        round_index=2,
        copy="ask",
    )
    assert scopes == [("values", True, True, 5, True, True, ("ask",))]


def test_the_object_format_reaches_the_amorce_the_probe_and_the_protocol(work, monkeypatch):
    """docs/33 amendment 9: --propose-format object is read by the amorce rows, the
    proposal episodes and the promotion, and is recorded in protocol.json only when it is
    not the default."""
    import types

    from prophet.agent import loop as loop_module
    from prophet.agent.actions import Action
    from prophet.agent.propose import propose_goal
    from scripts.closed_loop import proposal_rows, propose_round

    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    rows, specs, stats = proposal_rows(
        tokenizer, 2, family="lookup", seed=3, seq_len=2048, fmt="object"
    )
    text = tokenizer.decode(rows[0], skip_special=False)
    assert '"fields":{"' in text and '"keys":' not in text and stats["truncated"] == 0
    lists_text = tokenizer.decode(
        proposal_rows(tokenizer, 2, family="lookup", seed=3, seq_len=2048)[0][0],
        skip_special=False,
    )
    assert '"keys":"' in lists_text and '"fields":' not in lists_text

    seen_params = []

    def fake_run(self, goal, **kw):
        seen_params.append((goal, tuple(self.tools.schema("propose_lookup").properties)))
        args = {"file": "orchid.json", "fields": {"city": "Lyon", "code": "ash"}, "ask": "code"}
        step = types.SimpleNamespace(action=Action("propose_lookup", args), gated="")
        return types.SimpleNamespace(steps=[step], tokens=5)

    monkeypatch.setattr(loop_module.AgentLoop, "run", fake_run)
    model = ProphetModel(agent_tiny_config()).eval()
    proposals, counts = propose_round(
        model,
        tokenizer,
        "lookup",
        1,
        seen=set(),
        amorce_specs=specs,
        temperature=0.7,
        round_index=0,
        fmt="object",
    )
    assert counts["valid"] == 1 and proposals[0][1].answer == "ash"
    assert seen_params == [(propose_goal("lookup", "object"), ("file", "fields", "ask"))]


def test_calibration_refuses_an_amorce_with_nothing_left_to_learn():
    """The A100 launcher retained a rung as soon as every family succeeded canonically,
    even when all of them were saturated: its eight rounds would then only have measured
    retention. At least one family must start below 0.95 (docs/31 amendment 20)."""
    from pathlib import Path

    from scripts.closed_loop import calibration_verdict

    def record(success, canonical):
        return {"success_by_family": success, "canonical_by_family": canonical}

    fam = ("calc", "lookup", "files")
    saturated = record(dict.fromkeys(fam, 1.0), dict.fromkeys(fam, 1.0))
    assert calibration_verdict(saturated) == (False, list(fam))
    mixed = record({"calc": 1.0, "lookup": 0.4, "files": 0.97}, dict.fromkeys(fam, 0.3))
    assert calibration_verdict(mixed) == (True, ["calc", "files"])
    never = record({"calc": 0.5, "lookup": 0.0, "files": 0.5}, dict.fromkeys(fam, 0.3))
    assert not calibration_verdict(never)[0]
    uncanonical = record(dict.fromkeys(fam, 0.5), {"calc": 0.3, "lookup": 0.0, "files": 0.3})
    assert not calibration_verdict(uncanonical)[0]
    # The launcher delegates to this function rather than repeating the rule.
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_a100.sh"
    assert "from closed_loop import calibration_verdict" in launcher.read_text()


def test_a_gpu_run_records_its_card_so_a_resume_on_another_refuses(monkeypatch):
    """docs/34 §9: Colab may hand out another card than the one a run started on. The
    frozen protocol names the card, so the protocol check refuses such a resume; CPU runs
    record nothing new and resume as before."""
    import torch

    from scripts.closed_loop import device_protocol

    assert device_protocol("cpu") == {}
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA A100-SXM4-80GB")
    a100 = device_protocol("cuda")
    assert a100 == {"device": "cuda", "device_name": "NVIDIA A100-SXM4-80GB"}
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA H100 80GB HBM3")
    assert device_protocol("cuda") != a100


def test_closed_propose_runs_on_calc_with_its_own_hard_bench(work, tmp_path):
    """docs/39 SI-1: the proposal arm accepts calc, trains its two-stage amorce with calc
    proposals, probes them, and measures the three-operand bench."""
    out = tmp_path / "calc"
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        "closed-propose",
        "--family",
        "calc",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--tasks-per-round",
        "2",
        "--propose-n",
        "2",
        "--attempts",
        "1",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
        "--rounds",
        "0",
        "--propose-amorce",
        "2",
        "--propose-amorce-steps",
        "1",
        "--hard-bench",
        "--seed-dir",
        str(tmp_path / "calc-seed"),
    ]
    assert main(argv) == 0
    r0 = json.loads((out / "rounds.jsonl").read_text().splitlines()[0])
    assert r0["proposal_probe"]["emitted"] == 2
    assert all(b["family"] == "calc" for b in r0["bench_hard"])
    with pytest.raises(SystemExit):
        main([a if a != "calc" else "files" for a in argv])


def test_proposals_are_promoted_only_when_solved_on_a_retry(work, tmp_path):
    from prophet.agent.propose import task_from_spec, validate
    from prophet.agent.quarantine import Quarantine as Q
    from scripts.closed_loop import promote_proposals

    specs = [
        validate("lookup", {"file": f"f{i}.json", "keys": "a,b", "values": "x,y", "ask": "b"})
        for i in range(3)
    ]
    proposals = [(s, task_from_spec(s, name=f"p{i}")) for i, s in enumerate(specs)]
    q = Q(tmp_path / "q.json")
    promoted = promote_proposals(
        proposals, {"p0": 1, "p1": 2, "p2": 3}, q, family="lookup", round_index=4
    )
    assert promoted == 2
    entries = q.promoted("propose-lookup")
    assert len(entries) == 2 and all(e.promoted for e in entries)
    assert entries[0].trajectory[0]["action"]["name"] == "propose_lookup"
    assert entries[0].provenance.verifier_version == "round-4"
    assert q.promoted("lookup") == []


def test_closed_propose_arm_runs_with_a_proposal_amorce_and_the_hard_bench(work, tmp_path):
    out = tmp_path / "propose"
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        "closed-propose",
        "--family",
        "lookup",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--rounds",
        "1",
        "--tasks-per-round",
        "2",
        "--propose-n",
        "2",
        "--propose-amorce",
        "2",
        "--hard-bench",
        "--attempts",
        "2",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
    ]
    assert main(argv) == 0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["propose"] == {"n": 2} and protocol["propose_amorce"] == 2
    assert protocol["hard_bench"] is True
    seed = json.loads((out / "seed" / "seed.json").read_text())
    assert seed["episodes"]["rows"] == 2 and seed["proposals"]["rows"] == 2
    rounds = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    for record in rounds:
        assert [b["family"] for b in record["bench_hard"]] == ["lookup", "lookup"]
        assert [b["seed"] for b in record["bench_hard"]] == [17, 19]
        assert 0.0 <= record["success_hard"] <= 1.0
    proposals = rounds[1]["generation"]["proposals"]
    assert proposals["emitted"] == 2
    assert (
        proposals["valid"] + proposals["invalid"] + proposals["malformed"] + proposals["duplicate"]
        == 2
    )
    assert (
        proposals["solved_first"] + proposals["solved_retry"] + proposals["unsolved"]
        == proposals["valid"]
    )
    assert proposals["promoted"] <= proposals["solved_retry"]
    # A closed-propose arm refuses a family without a proposal grammar (lookup and calc
    # have one: docs/33, docs/39).
    with pytest.raises(SystemExit):
        main([a if a != "lookup" else "files" for a in argv])


def test_two_stage_amorce_is_recorded_once_and_reused_and_the_probe_is_recorded(work, tmp_path):
    """docs/33 amendment 1: the second amorce stage trains proposals with solver rows,
    is recorded in the shared seed directory, and a later arm reuses it; the
    closed-propose arm records a proposal probe at round 0."""
    seed_dir = tmp_path / "amorce" / "seed"
    common = dict(
        rounds=0, propose_amorce=2, propose_amorce_steps=2, seed_dir=str(seed_dir), hard_bench=""
    )
    out = tmp_path / "first"
    argv_extra = []
    for key, value in common.items():
        argv_extra += [f"--{key.replace('_', '-')}"] + ([str(value)] if value != "" else [])
    argv = [
        "--work",
        str(work),
        "--out",
        str(out),
        "--arm",
        "closed-propose",
        "--family",
        "lookup",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--tasks-per-round",
        "2",
        "--propose-n",
        "2",
        "--attempts",
        "1",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
    ] + argv_extra
    assert main(argv) == 0
    # The proposer decodes without copying, so its calls train no copy gate (docs/39
    # amendment 1); the solver's calls train it everywhere.
    assert closed_loop.GATE_KEYS == {"propose_lookup": ()}
    recorded = json.loads((seed_dir / "seed.json").read_text())
    assert recorded["episodes"]["rows"] == 2 and "proposals" not in recorded
    stage = recorded["proposal_stage"]
    assert (
        stage["proposals"]["rows"] == 2
        and stage["solver_rows"] == 2
        and stage["train"]["steps"] == 2
    )
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["propose_amorce_steps"] == 2
    assert "propose_format" not in protocol  # the default leaves runs in progress resumable
    rounds = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    probe = rounds[0]["proposal_probe"]
    assert probe["emitted"] == 2 and "valid" in probe and "malformed" in probe
    # A second arm on the same seed directory does not retrain the stage.
    stamp = (seed_dir / "seed.json").stat().st_mtime_ns
    out2 = tmp_path / "second"
    argv2 = [a if a != str(out) else str(out2) for a in argv]
    argv2[argv2.index("closed-propose")] = "frozen"
    assert main(argv2) == 0
    assert (seed_dir / "seed.json").stat().st_mtime_ns == stamp
    rounds2 = [json.loads(line) for line in (out2 / "rounds.jsonl").read_text().splitlines()]
    assert rounds2[0]["proposal_probe"] is None
    assert rounds2[0]["seed"]["proposal_stage"]["train"]["steps"] == 2
    # An arm in the object format refuses the lists-format amorce of this seed directory,
    # and records its own format when it trains one (docs/33 amendment 9).
    argv3 = [a if a != str(out) else str(tmp_path / "third") for a in argv] + [
        "--propose-format",
        "object",
    ]
    with pytest.raises(ValueError, match="trained in format 'lists'"):
        main(argv3)
    fresh = tmp_path / "amorce-object" / "seed"
    argv4 = [
        a if a != str(seed_dir) else str(fresh)
        for a in (a if a != str(out) else str(tmp_path / "fourth") for a in argv)
    ] + ["--propose-format", "object"]
    assert main(argv4) == 0
    assert (
        json.loads((tmp_path / "fourth" / "protocol.json").read_text())["propose_format"]
        == "object"
    )
    assert json.loads((fresh / "seed.json").read_text())["proposal_stage"]["format"] == "object"


def test_train_rows_hands_the_gate_keys_to_the_trainer(work, tmp_path, monkeypatch):
    """main() derives GATE_KEYS from --propose-copy (docs/39 amendment 1); every
    training call of the run carries it."""
    captured = []

    class FakeTrainer:
        def __init__(self, model, loader, tc, **kw):
            captured.append(tc)
            self.step, self.skipped_nonfinite = 0, 0

        def train(self):
            return []

    monkeypatch.setattr(closed_loop, "Trainer", FakeTrainer)
    monkeypatch.setattr(closed_loop, "GATE_KEYS", {"propose_calc": ()})
    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    closed_loop.train_rows(
        None,
        None,
        tokenizer,
        [[1, 2, 3]],
        work=work,
        steps=1,
        seq_len=8,
        batch_size=1,
        replay_fraction=0.0,
        checkpoint_dir=tmp_path,
        seed=0,
    )
    assert [tc.gate_keys for tc in captured] == [{"propose_calc": ()}]
    assert closed_loop.PROPOSE_COPY_KEYS == {"none": (), "ask": ("ask",)}


def test_copy_explore_end_is_recorded_when_set_and_needs_copy_topk(work, tmp_path):
    """docs/39 amendment 2: the option enters protocol.json only when set, and without
    --copy-topk nothing would read it, so the script refuses it."""
    base = [
        "--work",
        str(work),
        "--arm",
        "frozen",
        "--family",
        "calc",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--rounds",
        "0",
        "--tasks-per-round",
        "3",
        "--attempts",
        "2",
        "--steps-per-round",
        "2",
        "--seed-episodes",
        "3",
        "--seed-steps",
        "2",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "3",
    ]
    out = tmp_path / "end"
    assert main(base + ["--out", str(out), "--copy-topk", "2", "--copy-explore-end"]) == 0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["copy_explore_end"] is True and "copy_end_boundaries" not in protocol
    with pytest.raises(SystemExit):
        main(base + ["--out", str(tmp_path / "bare"), "--copy-explore-end"])
    # The end-boundary switch: recorded when set, read by every loop config of the run,
    # and refused in its "explore" mode without --copy-topk.
    out = tmp_path / "words"
    argv = base + ["--out", str(out), "--copy-topk", "2", "--copy-end-boundaries", "explore"]
    assert main(argv) == 0
    assert json.loads((out / "protocol.json").read_text())["copy_end_boundaries"] == "explore"
    assert closed_loop.generation_config("calc", temperature=0.0).copy_end_boundaries == "explore"
    with pytest.raises(SystemExit):
        main(base + ["--out", str(tmp_path / "bare2"), "--copy-end-boundaries", "explore"])
    assert main(base + ["--out", str(tmp_path / "plain")]) == 0
    assert closed_loop.generation_config("calc", temperature=0.0).copy_end_boundaries == "off"


def test_bench_checkpoint_replays_the_runs_greedy_bench_on_an_ood_set(work, tmp_path):
    """docs/39 amendment 2: a checkpoint is benched after the run on an out-of-distribution
    set, with the model config and decoding switches of the run's protocol.json."""
    from scripts import bench_checkpoint

    out = tmp_path / "frozen"
    run(work, out, "frozen", rounds=0, copy_topk=2, copy_end_boundaries="explore")
    closed_loop.COPY_END_BOUNDARIES = closed_loop.NO_REPEAT_ACTION = "reset"
    result = bench_checkpoint.main(
        [
            "--work",
            str(work),
            "--run",
            str(out),
            "--bench",
            "calc-digits",
            "--seeds",
            "17,19",
            "--n",
            "2",
        ]
    )
    assert result["round"] == 0 and len(result["verified"]) == 4
    assert result["success"] == sum(result["verified"]) / 4
    # The switches come from the protocol, not from whatever the process held.
    assert closed_loop.COPY_END_BOUNDARIES == "explore" and closed_loop.NO_REPEAT_ACTION is False
    with pytest.raises(SystemExit):
        bench_checkpoint.main(["--work", str(work), "--run", str(out), "--bench", "nope"])


def test_the_rules_ceiling_and_the_extra_benches_are_read_and_recorded(work, tmp_path, monkeypatch):
    """docs/39 amendment 5: --calc-max-digits reaches the proposal rules, --extra-bench is
    measured every round; both enter protocol.json only when set, and nonsense is refused."""
    import types

    from prophet.agent import loop as loop_module
    from prophet.agent.actions import Action
    from scripts.closed_loop import propose_round

    out = tmp_path / "extra"
    rounds = run(work, out, "frozen", rounds=0, calc_max_digits=6, extra_bench="calc-digits5")
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["calc_max_digits"] == 6 and protocol["extra_bench"] == ["calc-digits5"]
    extra = rounds[0]["bench_extra"]["calc-digits5"]
    assert 0.0 <= extra["success"] <= 1.0 and len(extra["verified"]) == 2 * 2  # 2 seeds x 2
    plain = tmp_path / "plain"
    rounds = run(work, plain, "frozen", rounds=0)
    protocol = json.loads((plain / "protocol.json").read_text())
    assert "calc_max_digits" not in protocol and "extra_bench" not in protocol
    assert "bench_extra" not in rounds[0]
    for bad in (["--calc-max-digits", "3"], ["--extra-bench", "calc-digits9"]):
        with pytest.raises(SystemExit):
            run(
                work, tmp_path / "bad", "frozen", rounds=0, **{bad[0][2:].replace("-", "_"): bad[1]}
            )

    # The proposal rules read the ceiling main() set.
    def fake_run(self, goal, **kw):
        step = types.SimpleNamespace(
            action=Action("propose_calc", {"expression": "123456 + 7"}), gated=""
        )
        return types.SimpleNamespace(steps=[step], tokens=1)

    monkeypatch.setattr(loop_module.AgentLoop, "run", fake_run)
    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    model = ProphetModel(agent_tiny_config()).eval()
    for digits, valid in ((4, 0), (6, 1)):
        monkeypatch.setattr(closed_loop, "MAX_DIGITS", digits)
        _, counts = propose_round(
            model, tokenizer, "calc", 1, seen=set(), amorce_specs=[], temperature=0.7, round_index=1
        )
        assert counts["valid"] == valid, digits


def test_learnability_probe_counts_and_runs(work, tmp_path):
    """docs/39 amendment 6: the probe reads per-task successes into what a learnability
    reward could pay -- tasks sometimes solved, greedy failures some sample rescues -- and
    runs end to end on a run directory."""
    from scripts import learnability_probe

    greedy = [True, True, False, False, False]
    wins = [4, 2, 0, 1, 3]
    assert learnability_probe.counts(greedy, wins, 4) == {
        "tasks": 5,
        "greedy": 2,
        "all_k": 1,
        "none": 1,
        "learnable": 3,
        "greedy_failures": 3,
        "rescued_by_sampling": 2,
    }
    out = tmp_path / "frozen"
    run(work, out, "frozen", rounds=0)
    results = learnability_probe.main(
        ["--work", str(work), "--run", str(out), "--bench", "calc-hard", "--n", "2", "--k", "2"]
    )
    assert len(results) == 1 and results[0]["tasks"] == 4 and results[0]["round"] == 0
    assert results[0]["all_k"] + results[0]["none"] + results[0]["learnable"] == 4
    with pytest.raises(SystemExit):
        learnability_probe.main(["--work", str(work), "--run", str(out), "--bench", "nope"])


def test_train_rows_gives_the_share_rows_a_fixed_part_of_the_stream(work, tmp_path, monkeypatch):
    """docs/39 amendment 7: with a share, the proposal rows take that fraction of the
    episode stream however few they are; alone, they are the stream."""
    captured = []

    class FakeLoader:
        def __init__(self, sources, **kw):
            captured.append({s.name: (s.weight, len(s.documents)) for s in sources})

    class FakeTrainer:
        def __init__(self, model, loader, tc, **kw):
            self.step, self.skipped_nonfinite = 0, 0

        def train(self):
            return []

    monkeypatch.setattr(closed_loop, "StreamingLoader", FakeLoader)
    monkeypatch.setattr(closed_loop, "Trainer", FakeTrainer)
    tokenizer = ProphetTokenizer.load(work / "tokenizer.json")
    common = dict(
        work=work, steps=1, seq_len=8, batch_size=1, replay_fraction=0.0, checkpoint_dir=tmp_path
    )
    solutions, proposals = [[1, 2, 3]] * 9, [[4, 5, 6]]
    closed_loop.train_rows(None, None, tokenizer, solutions, seed=0, **common)
    closed_loop.train_rows(
        None, None, tokenizer, solutions, seed=0, share_rows=proposals, share=0.25, **common
    )
    closed_loop.train_rows(
        None, None, tokenizer, [], seed=0, share_rows=proposals, share=0.25, **common
    )
    assert captured[0] == {"episodes": (1.0, 9)}
    assert captured[1] == {"episodes": (0.75, 9), "proposals": (0.25, 1)}
    assert captured[2] == {"episodes": (1.0, 1)}


def test_propose_share_is_read_by_the_round_and_recorded(work, tmp_path):
    """docs/39 amendment 7: --propose-share reaches the round's training (the promoted
    proposals and the amorce's, counted), enters protocol.json, and is refused where
    nothing would read it."""
    base = [
        "--work",
        str(work),
        "--family",
        "calc",
        "--config",
        str(work / "tiny.json"),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "2",
        "--tasks-per-round",
        "2",
        "--propose-n",
        "2",
        "--attempts",
        "1",
        "--steps-per-round",
        "1",
        "--seed-episodes",
        "2",
        "--seed-steps",
        "1",
        "--bench-tasks",
        "2",
        "--bpb-docs",
        "2",
        "--rounds",
        "1",
        "--propose-amorce",
        "2",
        "--propose-amorce-steps",
        "1",
        "--seed-dir",
        str(tmp_path / "seed"),
    ]
    out = tmp_path / "share"
    assert (
        main(base + ["--arm", "closed-propose", "--out", str(out), "--propose-share", "0.25"]) == 0
    )
    assert json.loads((out / "protocol.json").read_text())["propose_share"] == 0.25
    rounds = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    assert rounds[1]["train"]["share_rows"] >= 2  # at least the amorce's two proposals
    with pytest.raises(SystemExit):
        main(
            base
            + ["--arm", "closed-clean", "--out", str(tmp_path / "c"), "--propose-share", "0.25"]
        )
    with pytest.raises(SystemExit):
        main(
            base
            + ["--arm", "closed-propose", "--out", str(tmp_path / "d"), "--propose-share", "1.5"]
        )
