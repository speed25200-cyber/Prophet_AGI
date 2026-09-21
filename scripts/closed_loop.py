#!/usr/bin/env python3
"""The closed loop: a model improves on a verifiable task family from its own episodes.

    python scripts/closed_loop.py --work FIRST_RUN --out OUT --arm closed --family calc \
        --seed 0 --rounds 6 --tasks-per-round 40 --steps-per-round 100

One round, for the ``closed`` arm:

1. draw ``--tasks-per-round`` tasks of the family that no round, seed set or bench has
   used (the generator is the only source of tasks; a seed is a split);
2. run the agent loop on each, up to ``--attempts`` times, with the task's executable
   verifier deciding; every verified success enters the quarantine at tier 0 and is
   promoted at once, every failure is discarded (nothing learned is ever memorised);
3. train ``--steps-per-round`` steps on the promoted episodes rendered exactly as the
   loop produced them, with ``--replay-fraction`` of the rows drawn from the base
   corpus;
4. measure on tasks never trained on (two fixed seeds), held-out bits per byte, and
   the compute the round consumed: tokens generated, seconds of generation and of
   training.

Arms: ``closed`` as above; ``oracle`` promotes the *perfect* trajectory of every round
task instead of generating (the upper reference: what the loop would learn from a
teacher at the same task budget); ``frozen`` never trains (the floor and the bench's
own variance). All arms start from the same seed model: the base checkpoint fine-tuned
on ``--seed-episodes`` perfect trajectories, so that the loop has a nonzero success
rate to start from -- a model at zero generates nothing to learn from.

Every round is checkpointed; repeating the command resumes at the next round. The
report (``rounds.jsonl``, ``report.json``) is the record; nothing here decides.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.agent import tasks as task_families  # noqa: E402
from prophet.agent.loop import (
    AgentConfig,  # noqa: E402
    AgentLoop,  # noqa: E402
)
from prophet.agent.quarantine import Entry, Provenance, Quarantine  # noqa: E402
from prophet.agent.render import render_episode  # noqa: E402
from prophet.agent.verify import Tier  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.streaming import StreamingLoader, sources_from_iterables  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.agent_bench import run_bench  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.checkpoint import CheckpointManager  # noqa: E402
from prophet.train.klpo import klpo_update  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from scripts.first_agent_run_cpu import (  # noqa: E402
    CONFIG,
    agent_config,
    build_rows,
    heldout_bpb,
    replay_source,
)

ARMS = ("closed", "oracle", "frozen", "closed-klpo", "closed-clean")
BENCH_SEEDS = (7, 11)
SEED_TASK_BASE = 1_000
ROUND_TASK_BASE = 10_000


def generation_config(
    family: str, *, temperature: float, verifier_version: str = "prior-0"
) -> AgentConfig:
    """The bench's loop settings (docs/09), with the family named so the quarantine
    files the episodes under it and a temperature the caller chooses: sampled for
    generation, greedy for the held-out bench."""
    return AgentConfig(
        max_steps=4,
        think_budget=4,
        action_budget=64,
        halt_threshold=None,
        k_decide=2,
        tau_done=0.0,
        tau_act=0.0,
        tau_ask=0.0,
        sample_temperature=temperature,
        family=family,
        verifier_version=verifier_version,
    )


def rows_from_entries(
    entries: list[Entry], registry, tokenizer: ProphetTokenizer, *, seq_len: int
) -> tuple[list[list[int]], dict]:
    """Promoted episodes as training rows, one per row, rendered as the loop produced them."""
    rows, truncated, longest = [], 0, 0
    for entry in entries:
        text = render_episode(entry.goal, registry, entry.trajectory)
        ids = [tokenizer.bos_id] + tokenizer.encode(text, parse_special=True)
        longest = max(longest, len(ids))
        if len(ids) > seq_len:
            truncated += 1
            ids = ids[:seq_len]
        rows.append(ids + [tokenizer.pad_id] * (seq_len - len(ids)))
    return rows, {"rows": len(rows), "longest": longest, "truncated": truncated}


def train_rows(
    model,
    cfg,
    tokenizer,
    rows: list[list[int]],
    *,
    work: Path,
    steps: int,
    seq_len: int,
    batch_size: int,
    replay_fraction: float,
    checkpoint_dir: Path,
    seed: int,
    lr_scale: float = 1.0,
) -> dict:
    """``steps`` updates on ``rows`` mixed with the base corpus; a fresh schedule each call.

    ``lr_scale`` multiplies both peak learning rates (docs/31 amendment 5: the per-round
    recipe is the suspected cause of the drift and the forgetting).
    """
    if not rows or steps < 1:
        return {
            "steps": 0,
            "seconds": 0.0,
            "loss_first": None,
            "loss_last": None,
            "rows": len(rows),
        }
    sources = sources_from_iterables({"episodes": (1.0 - replay_fraction, rows)})
    if replay_fraction > 0:
        sources.append(replay_source(work, tokenizer, replay_fraction))
    loader = StreamingLoader(sources, seq_len=seq_len, batch_size=batch_size, seed=seed)
    tc = TrainConfig(
        total_steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        peak_lr_muon=0.01 * lr_scale,
        peak_lr_adamw=2e-3 * lr_scale,
        warmup_frac=0.05,
        decay_frac=0.3,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=10**9,
        log_every=max(steps // 4, 1),
        device="cpu",
        mtp_weight=0.0,
        seed=seed,
    )
    trainer = Trainer(model, loader, tc, model_config=cfg, tokenizer=tokenizer)
    started = time.time()
    history = trainer.train()
    return {
        "steps": trainer.step,
        "seconds": time.time() - started,
        "rows": len(rows),
        "loss_first": history[0].loss if history else None,
        "loss_last": history[-1].loss if history else None,
        "sel_accuracy_last": history[-1].extra.get("action/sel_accuracy") if history else None,
        "skipped_nonfinite": trainer.skipped_nonfinite,
    }


def bench_family(model, tokenizer, family: str, *, n_tasks: int, seed: int) -> dict:
    tasks = task_families.make_tasks(n_tasks, family=family, seed=seed)
    started = time.time()
    report = run_bench(
        model,
        tokenizer,
        tasks,
        generation_config(family, temperature=0.0),
        tools_for=task_families.tools_for,
        verifier_for_task=task_families.verifier_for,
    )
    return {
        "seed": seed,
        "tasks": report.n,
        "success_rate": report.success_rate,
        "malformed_rate": report.malformed_rate,
        "mean_tokens": report.mean_tokens,
        "tokens_per_success": report.tokens_per_success,
        "copied_values": sum(e.copied for e in report.episodes),
        "seconds": time.time() - started,
        "verified": [bool(e.verified) for e in report.episodes],
    }


def evaluate(
    model, tokenizer, family: str, *, work: Path, bench_tasks: int, bpb_docs: int, seq_len: int
) -> dict:
    model.eval()
    benches = [
        bench_family(model, tokenizer, family, n_tasks=bench_tasks, seed=s) for s in BENCH_SEEDS
    ]
    mean = sum(b["success_rate"] for b in benches) / len(benches)
    bpb = (
        heldout_bpb(work, model, tokenizer, seq_len=min(seq_len, 256), max_docs=bpb_docs)
        if bpb_docs
        else None
    )
    return {"bench": benches, "success_mean": mean, "bpb": bpb}


def generate_round(
    model,
    tokenizer,
    family: str,
    tasks,
    quarantine: Quarantine,
    *,
    round_index: int,
    attempts: int,
    temperature: float,
) -> dict:
    """Run the loop on every task, up to ``attempts`` times each; verified successes
    enter the quarantine through the loop itself (tier 0, promoted)."""
    remaining = list(tasks)
    started = time.time()
    tokens = episodes = 0
    solved = set()
    before = len(quarantine.promoted(family))
    for _attempt in range(attempts):
        if not remaining:
            break
        report = run_bench(
            model,
            tokenizer,
            remaining,
            generation_config(
                family, temperature=temperature, verifier_version=f"round-{round_index}"
            ),
            quarantine=quarantine,
            tools_for=task_families.tools_for,
            verifier_for_task=task_families.verifier_for,
        )
        tokens += sum(e.tokens for e in report.episodes)
        episodes += report.n
        solved.update(e.task for e in report.episodes if e.verified)
        remaining = [t for t, e in zip(remaining, report.episodes, strict=True) if not e.verified]
    return {
        "tasks": len(tasks),
        "episodes": episodes,
        "attempts": attempts,
        "solved": len(solved),
        "promoted_new": len(quarantine.promoted(family)) - before,
        "tokens": tokens,
        "seconds": time.time() - started,
    }


def generate_round_klpo(
    model,
    tokenizer,
    family: str,
    tasks,
    quarantine: Quarantine,
    *,
    round_index: int,
    attempts: int,
    temperature: float,
    draws: int,
) -> tuple[dict, list[dict]]:
    """Like ``generate_round`` but every episode is kept with its terminal reward and
    the sampler's records (docs/research/A5_klpo.md): the action span is sampled and
    every drawn token carries the sampler's log-probability and auxiliary draws."""
    cfg = generation_config(
        family, temperature=temperature, verifier_version=f"round-{round_index}"
    )
    cfg.sample_actions = True
    cfg.record_sampling = True
    cfg.mc_draws = draws
    remaining = list(tasks)
    started = time.time()
    tokens = 0
    episodes: list[dict] = []
    solved: set[str] = set()
    before = len(quarantine.promoted(family))
    for _attempt in range(attempts):
        if not remaining:
            break
        still = []
        for task in remaining:
            loop = AgentLoop(
                model,
                tokenizer,
                task_families.tools_for(task),
                cfg,
                quarantine=quarantine,
                verifier_tool=task_families.verifier_for(task),
            )
            result = loop.run(task.goal)
            tokens += result.tokens
            reward = 1 if result.verified_before_done else 0
            episodes.append(
                {
                    "task": task.name,
                    "reward": reward,
                    "ids": result.ids,
                    "sampled": result.sampled,
                    "tokens": result.tokens,
                    "policy_tokens": len(result.sampled or []),
                }
            )
            if reward:
                solved.add(task.name)
            else:
                still.append(task)
        remaining = still
    generation = {
        "tasks": len(tasks),
        "episodes": len(episodes),
        "attempts": attempts,
        "solved": len(solved),
        "promoted_new": len(quarantine.promoted(family)) - before,
        "tokens": tokens,
        "policy_tokens": sum(e["policy_tokens"] for e in episodes),
        "rewarded_episodes": sum(e["reward"] for e in episodes),
        "seconds": time.time() - started,
    }
    return generation, episodes


def clean_trajectory(trajectory: list[dict]) -> bool:
    """Canonical form (docs/31 amendment 7): every step parsed, no refused ``done``, no step
    identical to the previous one, and ``done`` last. The executable verifier judges the
    outcome; this only refuses to *teach* a sloppy way of reaching it."""
    if not trajectory:
        return False
    previous = None
    for step in trajectory:
        action = step.get("action")
        if action is None or step.get("gated"):
            return False
        if str(step.get("observation") or "").startswith("verification failed"):
            return False
        current = (action.get("name"), json.dumps(action.get("args", {}), sort_keys=True))
        if current == previous:
            return False
        previous = current
    return trajectory[-1]["action"].get("name") == "done"


def entry_round(entry: Entry) -> int | None:
    """The round an entry was generated in, from its provenance tag (``round-N`` or
    ``oracle-round-N``); ``None`` for entries without a tag (seed episodes, old runs)."""
    m = re.search(r"round-(\d+)$", entry.provenance.verifier_version)
    return int(m.group(1)) if m else None


def oracle_round(family: str, tasks, quarantine: Quarantine, *, round_index: int) -> dict:
    before = len(quarantine.promoted(family))
    for task in tasks:
        quarantine.add(
            Entry(
                family=family,
                goal=task.goal,
                trajectory=task_families.perfect_trajectory(task),
                outcome_passed=True,
                process_ok=True,
                provenance=Provenance(
                    tier=int(Tier.GROUND_TRUTH),
                    verifier_version=f"oracle-round-{round_index}",
                    p_correct=1.0,
                    depth_disagreement=None,
                    attempts=1,
                ),
            )
        )
    return {
        "tasks": len(tasks),
        "episodes": len(tasks),
        "attempts": 0,
        "solved": len(tasks),
        "promoted_new": len(quarantine.promoted(family)) - before,
        "tokens": 0,
        "seconds": 0.0,
    }


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--work",
        type=Path,
        required=True,
        help="first run directory: tokenizer, corpus, checkpoint",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--family", default="calc", choices=sorted(task_families.FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tasks-per-round", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--steps-per-round", type=int, default=100)
    ap.add_argument("--seed-episodes", type=int, default=100)
    ap.add_argument("--seed-steps", type=int, default=200)
    ap.add_argument(
        "--seed-dir", type=Path, default=None, help="reuse a seed checkpoint trained by another arm"
    )
    ap.add_argument("--replay-fraction", type=float, default=0.5)
    ap.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="multiplies the peak learning rates of the per-round training (not the seed)",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="sampling temperature during generation"
    )
    ap.add_argument("--bench-tasks", type=int, default=40)
    ap.add_argument("--bpb-docs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--minutes", type=float, default=120.0, help="wall budget for this launch; resume later"
    )
    ap.add_argument(
        "--klpo-steps",
        type=int,
        default=None,
        help="KLPO updates per round (default: --steps-per-round)",
    )
    ap.add_argument("--klpo-beta", type=float, default=0.1)
    ap.add_argument("--klpo-lr", type=float, default=5e-4)
    ap.add_argument(
        "--klpo-draws", type=int, default=8, help="auxiliary token draws per prefix (M)"
    )
    ap.add_argument(
        "--klpo-temperature", type=float, default=1.0, help="sampling temperature of the KLPO arm"
    )
    args = ap.parse_args(argv)
    if args.rounds < 0 or args.tasks_per_round < 1 or args.attempts < 1 or args.steps_per_round < 0:
        ap.error("rounds >= 0, tasks and attempts >= 1, steps >= 0")
    if not 0 <= args.replay_fraction < 1:
        ap.error("replay fraction in [0, 1)")
    began = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = ProphetTokenizer.load(args.work / "tokenizer.json")
    cfg = agent_config(ProphetConfig.from_json(args.config))
    cfg.validate()
    torch.manual_seed(args.seed)
    model = ProphetModel(cfg)
    protocol = {
        "arm": args.arm,
        "family": args.family,
        "seed": args.seed,
        "config": cfg.to_dict(),
        "rounds": args.rounds,
        "tasks_per_round": args.tasks_per_round,
        "attempts": args.attempts,
        "steps_per_round": args.steps_per_round,
        "seed_episodes": args.seed_episodes,
        "seed_steps": args.seed_steps,
        "replay_fraction": args.replay_fraction,
        "lr_scale": args.lr_scale,
        "temperature": args.temperature,
        "bench_tasks": args.bench_tasks,
        "bench_seeds": list(BENCH_SEEDS),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "task_seeds": {
            "seed": SEED_TASK_BASE + args.seed,
            "rounds": f"{ROUND_TASK_BASE} * (seed + 1) + round",
        },
        "klpo": {
            "steps": args.klpo_steps if args.klpo_steps is not None else args.steps_per_round,
            "beta": args.klpo_beta,
            "lr": args.klpo_lr,
            "draws": args.klpo_draws,
            "temperature": args.klpo_temperature,
        },
    }
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("protocol changed; use a fresh output directory")
    else:
        write_json(protocol_path, protocol)
    quarantine = Quarantine(args.out / "quarantine.json")
    rounds_path = args.out / "rounds.jsonl"
    rounds = (
        [json.loads(line) for line in rounds_path.read_text().splitlines() if line.strip()]
        if rounds_path.exists()
        else []
    )
    manager = CheckpointManager(args.out / "checkpoints")

    def record(entry: dict) -> None:
        rounds.append(entry)
        with rounds_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, default=str) + "\n")
        print(
            "ROUND",
            json.dumps(
                {
                    k: entry[k]
                    for k in ("round", "success_mean", "promoted_total", "compute_seconds")
                }
            ),
            flush=True,
        )

    if manager.has_checkpoint():
        state, meta = manager.load_latest()
        model.load_state_dict(state["model"], strict=True)
        print(
            "RESUMED",
            {"round": meta.step, "promoted": len(quarantine.promoted(args.family))},
            flush=True,
        )
    else:
        # Round 0: the seed model, shared by every arm by construction.
        seed_dir = args.seed_dir or (args.out / "seed")
        seed_manager = CheckpointManager(seed_dir / "checkpoints")
        if seed_manager.has_checkpoint():
            state, _ = seed_manager.load_latest()
            model.load_state_dict(state["model"], strict=True)
            recorded = (
                json.loads((seed_dir / "seed.json").read_text())
                if (seed_dir / "seed.json").exists()
                else {}
            )
            seed_report = {"reused": str(seed_dir), **recorded}
        else:
            base, meta = CheckpointManager(args.work / "checkpoints").load_latest()
            missing, unexpected = model.load_state_dict(base["model"], strict=False)
            seed_report = {
                "from_step": meta.step,
                "fresh_params": sorted(missing)[:8],
                "unexpected": len(unexpected),
            }
            if args.seed_episodes:
                rows, stats = build_rows(
                    tokenizer,
                    args.seed_episodes,
                    seed=SEED_TASK_BASE + args.seed,
                    seq_len=args.seq_len,
                    families=[args.family],
                )
                seed_report["episodes"] = stats
                seed_report["train"] = train_rows(
                    model,
                    cfg,
                    tokenizer,
                    rows,
                    work=args.work,
                    steps=args.seed_steps,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    replay_fraction=args.replay_fraction,
                    checkpoint_dir=seed_dir / "scratch",
                    seed=args.seed,
                )
            seed_manager.save({"model": model.state_dict(), "step": 0}, 0)
            write_json(seed_dir / "seed.json", seed_report)
        started = time.time()
        measured = evaluate(
            model,
            tokenizer,
            args.family,
            work=args.work,
            bench_tasks=args.bench_tasks,
            bpb_docs=args.bpb_docs,
            seq_len=args.seq_len,
        )
        manager.save({"model": model.state_dict(), "step": 0}, 0)
        record(
            {
                "round": 0,
                "seed": seed_report,
                **measured,
                "eval_seconds": time.time() - started,
                "promoted_total": len(quarantine.promoted(args.family)),
                "compute_seconds": 0.0,
                "generation": None,
                "train": None,
            }
        )

    done = max(r["round"] for r in rounds)
    # A crash between a round's generation and its record leaves that round's entries in
    # the quarantine; regenerating the round would then train on them twice. Entries are
    # tagged with their round, so drop the ones beyond the last recorded round.
    dropped = quarantine.discard(lambda e: (entry_round(e) or 0) > done)
    if dropped:
        print("PRUNED", {"orphan_entries": dropped, "after_round": done}, flush=True)
    compute = rounds[-1]["compute_seconds"]
    for r in range(done + 1, args.rounds + 1):
        if (time.time() - began) / 60 > args.minutes:
            print("SESSION_COMPLETE_RESUME_REQUIRED", flush=True)
            return 0
        torch.manual_seed(args.seed * 1_000 + r)
        tasks = task_families.make_tasks(
            args.tasks_per_round, family=args.family, seed=ROUND_TASK_BASE * (args.seed + 1) + r
        )
        if args.arm in ("closed", "closed-clean"):
            generation = generate_round(
                model,
                tokenizer,
                args.family,
                tasks,
                quarantine,
                attempts=args.attempts,
                temperature=args.temperature,
                round_index=r,
            )
            if args.arm == "closed-clean":
                # Verified but sloppy episodes are not taught (docs/31 amendment 7).
                demoted = quarantine.discard(
                    lambda e: (
                        entry_round(e) == r and e.promoted and not clean_trajectory(e.trajectory)
                    )
                )
                generation["demoted_sloppy"] = demoted
                generation["promoted_new"] -= demoted
        elif args.arm == "closed-klpo":
            generation, klpo_episodes = generate_round_klpo(
                model,
                tokenizer,
                args.family,
                tasks,
                quarantine,
                attempts=args.attempts,
                temperature=args.klpo_temperature,
                draws=args.klpo_draws,
                round_index=r,
            )
            write_json(args.out / f"round-{r:03d}-episodes.json", klpo_episodes)
        elif args.arm == "oracle":
            generation = oracle_round(args.family, tasks, quarantine, round_index=r)
        else:
            generation = {
                "tasks": len(tasks),
                "episodes": 0,
                "attempts": 0,
                "solved": 0,
                "promoted_new": 0,
                "tokens": 0,
                "seconds": 0.0,
            }
        train = None
        if args.arm != "frozen":
            promoted = quarantine.promoted(args.family)
            registry = task_families.tools_for(tasks[0])
            rows, row_stats = rows_from_entries(promoted, registry, tokenizer, seq_len=args.seq_len)
            train = {
                **row_stats,
                **train_rows(
                    model,
                    cfg,
                    tokenizer,
                    rows,
                    work=args.work,
                    steps=args.steps_per_round,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    replay_fraction=args.replay_fraction,
                    checkpoint_dir=args.out / "scratch",
                    seed=args.seed * 1_000 + r,
                    lr_scale=args.lr_scale,
                ),
            }
        klpo = None
        if args.arm == "closed-klpo":
            klpo_started = time.time()
            klpo = klpo_update(
                model,
                klpo_episodes,
                pad_id=tokenizer.pad_id,
                steps=protocol["klpo"]["steps"],
                beta=args.klpo_beta,
                lr=args.klpo_lr,
                batch_size=args.batch_size,
                seed=args.seed * 1_000 + r,
            )
            klpo["seconds"] = time.time() - klpo_started
        started = time.time()
        measured = evaluate(
            model,
            tokenizer,
            args.family,
            work=args.work,
            bench_tasks=args.bench_tasks,
            bpb_docs=args.bpb_docs,
            seq_len=args.seq_len,
        )
        compute += generation["seconds"] + (train["seconds"] if train else 0.0)
        compute += klpo["seconds"] if klpo else 0.0
        manager.save({"model": model.state_dict(), "step": r}, r)
        record(
            {
                "round": r,
                "generation": generation,
                "train": train,
                "klpo": klpo,
                **measured,
                "eval_seconds": time.time() - started,
                "promoted_total": len(quarantine.promoted(args.family)),
                "compute_seconds": compute,
            }
        )
    write_json(
        args.out / "report.json",
        {
            "protocol": protocol,
            "rounds": rounds,
            "quarantine": quarantine.summary(),
            "complete": True,
        },
    )
    print("RUN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
