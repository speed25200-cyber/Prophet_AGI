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
        "--sample-copy",
    ]
    assert main(argv) == 0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["no_repeat_action"] is True
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
