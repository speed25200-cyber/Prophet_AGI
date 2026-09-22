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

Several families share one loop when ``--family`` is repeated (docs/31 amendment 14):
each round draws its tasks family by family (the same draws as a single-family run of
that seed), the quarantine files every episode under its own family, the rows are
rendered with the tool registry of the episode's family, and the bench reports the
success per family. ``--bench-family`` measures an arm on families it does not train.

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
from prophet.agent.propose import (  # noqa: E402
    PROPOSE_GOAL,
    make_hard_lookup,
    novel,
    proposal_trajectory,
    propose_registry,
    spec_from_task,
    task_from_spec,
    validate,
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

SAMPLE_COPY = False  # set by main() from --sample-copy; generation only (the bench is greedy)
NO_REPEAT_ACTION = False  # set by main() from --no-repeat-action; read by generation_config callers
COPY_BOUNDARIES = "off"  # set by main() from --copy-boundaries; bench and generation alike
ARMS = ("closed", "oracle", "frozen", "closed-klpo", "closed-clean", "closed-propose")
BENCH_SEEDS = (7, 11)
HARD_BENCH_SEEDS = (17, 19)  # the out-of-distribution bench of docs/33, never trained on
SEED_TASK_BASE = 1_000
ROUND_TASK_BASE = 10_000


def generation_config(
    family: str,
    *,
    temperature: float,
    verifier_version: str = "prior-0",
    no_repeat_action: bool = False,
    sample_copy: bool = False,
    copy_topk: int = 0,
    copy_explore: str = "all",
    copy_boundaries: str | None = None,
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
        no_repeat_action=no_repeat_action,
        sample_copy=sample_copy,
        copy_topk=copy_topk,
        copy_explore=copy_explore,
        copy_boundaries=COPY_BOUNDARIES if copy_boundaries is None else copy_boundaries,
    )


def registry_for(family: str):
    """The tool registry an episode of ``family`` is rendered with: the schemas depend on
    the family only, so any task of it gives the pinned prompt the loop wrote."""
    return task_families.tools_for(task_families.make_tasks(1, family=family, seed=0)[0])


def rows_from_entries(
    entries: list[Entry], registries: dict, tokenizer: ProphetTokenizer, *, seq_len: int
) -> tuple[list[list[int]], dict]:
    """Promoted episodes as training rows, one per row, rendered as the loop produced them,
    each with the registry of its family (``registries[entry.family]``)."""
    rows, truncated, longest = [], 0, 0
    for entry in entries:
        text = render_episode(entry.goal, registries[entry.family], entry.trajectory)
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
    device: str = "cpu",
    replay_names: tuple[str, ...] = ("prose", "code"),
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
        sources.append(replay_source(work, tokenizer, replay_fraction, names=replay_names))
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
        device=device,
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


def bench_family(model, tokenizer, family: str, *, n_tasks: int, seed: int, tasks=None) -> dict:
    """The greedy bench on ``n_tasks`` generator tasks of ``family`` at ``seed``, or on
    ``tasks`` when given (the out-of-distribution bench of docs/33)."""
    tasks = (
        tasks if tasks is not None else task_families.make_tasks(n_tasks, family=family, seed=seed)
    )
    started = time.time()
    report = run_bench(
        model,
        tokenizer,
        tasks,
        generation_config(family, temperature=0.0, no_repeat_action=NO_REPEAT_ACTION),
        tools_for=task_families.tools_for,
        verifier_for_task=task_families.verifier_for,
    )
    return {
        "family": family,
        "seed": seed,
        "tasks": report.n,
        "success_rate": report.success_rate,
        "canonical_rate": report.canonical_rate,
        "malformed_rate": report.malformed_rate,
        "mean_tokens": report.mean_tokens,
        "tokens_per_success": report.tokens_per_success,
        "copied_values": sum(e.copied for e in report.episodes),
        "seconds": time.time() - started,
        "verified": [bool(e.verified) for e in report.episodes],
    }


def evaluate(
    model,
    tokenizer,
    families: list[str],
    *,
    work: Path,
    bench_tasks: int,
    bpb_docs: int,
    seq_len: int,
    device: str = "cpu",
    hard: bool = False,
) -> dict:
    """The benches of every family (family-major, then the bench seeds), their mean, the
    mean per family, and the held-out bits per byte; with ``hard``, the out-of-distribution
    lookup bench of docs/33 as well (``bench_hard``, ``success_hard``)."""
    model.eval()
    benches = [
        bench_family(model, tokenizer, family, n_tasks=bench_tasks, seed=s)
        for family in families
        for s in BENCH_SEEDS
    ]
    mean = sum(b["success_rate"] for b in benches) / len(benches)
    by_family = {
        family: sum(b["success_rate"] for b in benches if b["family"] == family) / len(BENCH_SEEDS)
        for family in families
    }
    canonical_by_family = {
        family: sum(b["canonical_rate"] for b in benches if b["family"] == family)
        / len(BENCH_SEEDS)
        for family in families
    }
    bpb = (
        heldout_bpb(
            work, model, tokenizer, seq_len=min(seq_len, 256), max_docs=bpb_docs, device=device
        )
        if bpb_docs
        else None
    )
    measured = {
        "bench": benches,
        "success_mean": mean,
        "success_by_family": by_family,
        "canonical_by_family": canonical_by_family,
        "bpb": bpb,
    }
    if hard:
        hard_benches = [
            bench_family(
                model,
                tokenizer,
                "lookup",
                n_tasks=bench_tasks,
                seed=s,
                tasks=make_hard_lookup(bench_tasks, seed=s),
            )
            for s in HARD_BENCH_SEEDS
        ]
        measured["bench_hard"] = hard_benches
        measured["success_hard"] = sum(b["success_rate"] for b in hard_benches) / len(hard_benches)
    return measured


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
    copy_topk: int = 0,
    explore_from_attempt: int = 2,
    copy_explore: str = "all",
) -> dict:
    """Run the loop on every task, up to ``attempts`` times each; verified successes
    enter the quarantine through the loop itself (tier 0, promoted).

    ``copy_topk`` > 0 makes the attempts from ``explore_from_attempt`` on draw the copy
    pointer's start among its best ``copy_topk`` positions (docs/31 amendment 15): the
    first attempt keeps the policy, the retries explore where it failed.
    ``promoted_explored`` counts what those retries verified.
    """
    remaining = list(tasks)
    started = time.time()
    tokens = episodes = promoted_explored = 0
    solved = set()
    solved_at: dict[str, int] = {}
    before = len(quarantine.promoted(family))
    for attempt in range(1, attempts + 1):
        if not remaining:
            break
        exploring = copy_topk > 0 and attempt >= explore_from_attempt
        before_attempt = len(quarantine.promoted(family))
        report = run_bench(
            model,
            tokenizer,
            remaining,
            generation_config(
                family,
                temperature=temperature,
                verifier_version=f"round-{round_index}",
                no_repeat_action=NO_REPEAT_ACTION,
                sample_copy=SAMPLE_COPY,
                copy_topk=copy_topk if exploring else 0,
                copy_explore=copy_explore,
            ),
            quarantine=quarantine,
            tools_for=task_families.tools_for,
            verifier_for_task=task_families.verifier_for,
        )
        tokens += sum(e.tokens for e in report.episodes)
        episodes += report.n
        for t, e in zip(remaining, report.episodes, strict=True):
            if e.verified:
                solved.add(t.name)
                solved_at.setdefault(t.name, attempt)
        if exploring:
            promoted_explored += len(quarantine.promoted(family)) - before_attempt
        remaining = [t for t, e in zip(remaining, report.episodes, strict=True) if not e.verified]
    return {
        "tasks": len(tasks),
        "episodes": episodes,
        "attempts": attempts,
        "solved": len(solved),
        "solved_at": solved_at,
        "promoted_new": len(quarantine.promoted(family)) - before,
        "promoted_explored": promoted_explored,
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
        family,
        temperature=temperature,
        verifier_version=f"round-{round_index}",
        no_repeat_action=NO_REPEAT_ACTION,
        sample_copy=SAMPLE_COPY,
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


def proposal_rows(
    tokenizer: ProphetTokenizer, n: int, *, family: str, seed: int, seq_len: int
) -> tuple[list[list[int]], list, dict]:
    """``n`` perfect proposal episodes for the amorce (docs/33 §2): generator tasks turned
    into the specification the model would have had to propose. Returns the rows, the
    specifications (the amorce's distribution, for novelty) and the stats."""
    registry = propose_registry(family)
    specs = [spec_from_task(t) for t in task_families.make_tasks(n, family=family, seed=seed)]
    rows, longest, truncated = [], 0, 0
    for spec in specs:
        text = render_episode(PROPOSE_GOAL[family], registry, proposal_trajectory(spec))
        ids = [tokenizer.bos_id] + tokenizer.encode(text, parse_special=True)
        longest = max(longest, len(ids))
        if len(ids) > seq_len:
            truncated += 1
            ids = ids[:seq_len]
        rows.append(ids + [tokenizer.pad_id] * (seq_len - len(ids)))
    return rows, specs, {"rows": len(rows), "longest": longest, "truncated": truncated}


def propose_round(
    model,
    tokenizer,
    family: str,
    n: int,
    *,
    seen: set[str],
    amorce_specs: list,
    temperature: float,
    round_index: int,
) -> tuple[list, dict]:
    """``n`` proposal episodes (one step each, action span sampled), validated by the
    rules; returns ``[(spec, task), ...]`` and the counts. ``seen`` holds the signatures
    already proposed in this run, so a task is never proposed twice."""
    cfg = generation_config(
        family,
        temperature=temperature,
        verifier_version=f"round-{round_index}",
        no_repeat_action=NO_REPEAT_ACTION,
    )
    cfg.max_steps = 1
    cfg.sample_actions = True
    registry = propose_registry(family)
    counts = {
        "emitted": n,
        "malformed": 0,
        "invalid": 0,
        "duplicate": 0,
        "valid": 0,
        "novel": 0,
        "n_fields": {},
        "tokens": 0,
    }
    proposals = []
    for i in range(n):
        result = AgentLoop(model, tokenizer, registry, cfg).run(PROPOSE_GOAL[family])
        counts["tokens"] += int(getattr(result, "tokens", 0))
        action = result.steps[0].action if result.steps else None
        if action is None or action.name != f"propose_{family}":
            counts["malformed"] += 1
            continue
        verdict = validate(family, action.args)
        if isinstance(verdict, str):
            counts["invalid"] += 1
            continue
        if verdict.signature() in seen:
            counts["duplicate"] += 1
            continue
        seen.add(verdict.signature())
        counts["valid"] += 1
        counts["novel"] += int(novel(verdict, amorce_specs))
        counts["n_fields"][str(len(verdict.keys))] = (
            counts["n_fields"].get(str(len(verdict.keys)), 0) + 1
        )
        proposals.append((verdict, task_from_spec(verdict, name=f"proposed-{round_index}-{i}")))
    return proposals, counts


def promote_proposals(
    proposals: list,
    solved_at: dict[str, int],
    quarantine: Quarantine,
    *,
    family: str,
    round_index: int,
) -> int:
    """The proposer's only reward (docs/33 §2): a proposal is promoted when its task was
    solved on a retry and not at the first attempt -- at the edge of what the solver can
    do. Returns how many were promoted."""
    promoted = 0
    for spec, task in proposals:
        if solved_at.get(task.name, 0) < 2:
            continue
        quarantine.add(
            Entry(
                family=f"propose-{family}",
                goal=PROPOSE_GOAL[family],
                trajectory=proposal_trajectory(spec),
                outcome_passed=True,
                process_ok=True,
                provenance=Provenance(
                    tier=int(Tier.GROUND_TRUTH),
                    verifier_version=f"round-{round_index}",
                    p_correct=1.0,
                    depth_disagreement=None,
                    attempts=1,
                ),
            )
        )
        promoted += 1
    return promoted


def merge_generation(parts: dict[str, dict]) -> dict:
    """One round's generation record from the per-family records: the counts add up,
    ``attempts`` is common, and the parts stay under ``by_family``."""
    merged = {
        key: sum(part[key] for part in parts.values())
        for key in ("tasks", "episodes", "solved", "promoted_new", "tokens", "seconds")
    }
    for key in ("policy_tokens", "rewarded_episodes", "promoted_explored"):
        if all(key in part for part in parts.values()):
            merged[key] = sum(part[key] for part in parts.values())
    if all("solved_at" in part for part in parts.values()):
        merged["solved_at"] = {
            k: v for part in parts.values() for k, v in part["solved_at"].items()
        }
    merged["attempts"] = next(iter(parts.values()))["attempts"]
    merged["by_family"] = parts
    return merged


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
    ap.add_argument(
        "--family",
        action="append",
        choices=sorted(task_families.FAMILIES),
        help="task family of the rounds; repeat for several in one loop (default: calc)",
    )
    ap.add_argument(
        "--bench-family",
        action="append",
        choices=sorted(task_families.FAMILIES),
        help="families of the held-out bench (default: the round families)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tasks-per-round", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--steps-per-round", type=int, default=100)
    ap.add_argument(
        "--seed-episodes", type=int, default=100, help="perfect trajectories per family"
    )
    ap.add_argument("--seed-steps", type=int, default=200)
    ap.add_argument(
        "--seed-dir", type=Path, default=None, help="reuse a seed checkpoint trained by another arm"
    )
    ap.add_argument("--replay-fraction", type=float, default=0.5)
    ap.add_argument(
        "--sample-copy",
        action="store_true",
        help="sample the copy pointer at generation instead of its argmax (docs/31 amendment 12)",
    )
    ap.add_argument(
        "--no-repeat-action",
        action="store_true",
        help="forbid at each step the action name of the previous step (docs/31 amendment 11)",
    )
    ap.add_argument(
        "--copy-topk",
        type=int,
        default=0,
        help="retries draw the copy pointer's start among its best K positions "
        "(docs/31 amendment 15); 0 keeps the argmax",
    )
    ap.add_argument(
        "--copy-explore",
        choices=("all", "observations"),
        default="all",
        help="which copy events --copy-topk explores: all, or only spans read from a tool "
        "observation (docs/31 amendment 17)",
    )
    ap.add_argument(
        "--copy-boundaries",
        choices=("off", "explore", "always"),
        default="off",
        help="restrict the copy pointer's start to word boundaries: for the exploratory "
        "draw only, or for the argmax too (docs/31 amendment 19)",
    )
    ap.add_argument(
        "--explore-from-attempt",
        type=int,
        default=2,
        help="first attempt that explores with --copy-topk (1 = every attempt)",
    )
    ap.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="multiplies the peak learning rates of the per-round training (not the seed)",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="sampling temperature during generation"
    )
    ap.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="where the model trains and decodes; the CPU pilots never set it",
    )
    ap.add_argument(
        "--replay-names",
        default="prose,code",
        help="comma-separated sub-directories of WORK/corpus the replay draws from "
        "(a loop-core corpus: fineweb-edu,composition)",
    )
    ap.add_argument(
        "--propose-n",
        type=int,
        default=None,
        help="closed-propose: proposals per round (default: --tasks-per-round)",
    )
    ap.add_argument(
        "--propose-amorce",
        type=int,
        default=0,
        help="perfect proposal episodes added to the amorce, for every arm (docs/33 §2)",
    )
    ap.add_argument(
        "--hard-bench",
        action="store_true",
        help="also measure the out-of-distribution lookup bench of docs/33 every round",
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
    global NO_REPEAT_ACTION, SAMPLE_COPY, COPY_BOUNDARIES
    NO_REPEAT_ACTION = bool(args.no_repeat_action)
    SAMPLE_COPY = bool(args.sample_copy)
    COPY_BOUNDARIES = args.copy_boundaries
    if args.rounds < 0 or args.tasks_per_round < 1 or args.attempts < 1 or args.steps_per_round < 0:
        ap.error("rounds >= 0, tasks and attempts >= 1, steps >= 0")
    if not 0 <= args.replay_fraction < 1:
        ap.error("replay fraction in [0, 1)")
    families = args.family or ["calc"]
    bench_families = args.bench_family or list(families)
    if len(set(families)) != len(families) or len(set(bench_families)) != len(bench_families):
        ap.error("a family is named once")
    registries = {f: registry_for(f) for f in families}
    if args.arm == "closed-propose" and families != ["lookup"]:
        ap.error("closed-propose proposes lookup tasks only (docs/33 §5): --family lookup")
    training_families = list(families)
    if args.arm == "closed-propose":
        for f in families:
            training_families.append(f"propose-{f}")
            registries[f"propose-{f}"] = propose_registry(f)
    propose_n = args.propose_n if args.propose_n is not None else args.tasks_per_round
    replay_names = tuple(n for n in args.replay_names.split(",") if n)
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda but no CUDA device is available")
    began = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = ProphetTokenizer.load(args.work / "tokenizer.json")
    cfg = agent_config(ProphetConfig.from_json(args.config))
    cfg.validate()
    torch.manual_seed(args.seed)
    model = ProphetModel(cfg).to(args.device)
    # One family writes the protocol as before this option existed, so a run in progress
    # resumes; several are joined with "+", the bench families only when they differ.
    protocol = {
        "arm": args.arm,
        "family": "+".join(families),
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
        "no_repeat_action": args.no_repeat_action,
        "sample_copy": args.sample_copy,
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
    if bench_families != families:
        protocol["bench_families"] = "+".join(bench_families)
    if args.copy_topk < 0 or args.explore_from_attempt < 1:
        ap.error("copy-topk >= 0, explore-from-attempt >= 1")
    if args.copy_topk:
        protocol["copy_topk"] = args.copy_topk
        protocol["explore_from_attempt"] = args.explore_from_attempt
        if args.copy_explore != "all":
            protocol["copy_explore"] = args.copy_explore
    if args.copy_boundaries != "off":
        protocol["copy_boundaries"] = args.copy_boundaries
    if args.arm == "closed-propose":
        protocol["propose"] = {"n": propose_n}
    if args.propose_amorce:
        protocol["propose_amorce"] = args.propose_amorce
    if args.hard_bench:
        protocol["hard_bench"] = True
    if args.device != "cpu":
        protocol["device"] = args.device
    if replay_names != ("prose", "code"):
        protocol["replay_names"] = list(replay_names)
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

    def promoted_entries() -> list[Entry]:
        """The promoted episodes of the round families (and, for closed-propose, the
        promoted proposals), in the order they were admitted."""
        return [e for e in quarantine.promoted() if e.family in training_families]

    def record(entry: dict) -> None:
        rounds.append(entry)
        with rounds_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, default=str) + "\n")
        line = {k: entry[k] for k in ("round", "success_mean", "promoted_total", "compute_seconds")}
        if len(bench_families) > 1:
            line["by_family"] = entry["success_by_family"]
            line["canonical"] = entry["canonical_by_family"]
        if "success_hard" in entry:
            line["hard"] = entry["success_hard"]
        if (entry.get("generation") or {}).get("proposals"):
            p = entry["generation"]["proposals"]
            line["proposals"] = {k: p[k] for k in ("valid", "solved_retry", "promoted")}
        print("ROUND", json.dumps(line), flush=True)

    if manager.has_checkpoint():
        state, meta = manager.load_latest()
        model.load_state_dict(state["model"], strict=True)
        print("RESUMED", {"round": meta.step, "promoted": len(promoted_entries())}, flush=True)
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
            base, meta = CheckpointManager(args.work / "checkpoints").load_latest(
                map_location=args.device
            )
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
                    families=families,
                )
                seed_report["episodes"] = stats
                if args.propose_amorce:
                    extra_rows, _, proposal_stats = proposal_rows(
                        tokenizer,
                        args.propose_amorce,
                        family=families[0],
                        seed=SEED_TASK_BASE + args.seed,
                        seq_len=args.seq_len,
                    )
                    rows = rows + extra_rows
                    seed_report["proposals"] = proposal_stats
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
                    device=args.device,
                    replay_names=replay_names,
                )
            seed_manager.save({"model": model.state_dict(), "step": 0}, 0)
            write_json(seed_dir / "seed.json", seed_report)
        started = time.time()
        measured = evaluate(
            model,
            tokenizer,
            bench_families,
            work=args.work,
            bench_tasks=args.bench_tasks,
            bpb_docs=args.bpb_docs,
            seq_len=args.seq_len,
            device=args.device,
            hard=args.hard_bench,
        )
        manager.save({"model": model.state_dict(), "step": 0}, 0)
        record(
            {
                "round": 0,
                "seed": seed_report,
                **measured,
                "eval_seconds": time.time() - started,
                "promoted_total": len(promoted_entries()),
                "compute_seconds": 0.0,
                "generation": None,
                "train": None,
            }
        )

    proposed_signatures: set[str] = set()
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
        tasks_by_family = {
            f: task_families.make_tasks(
                args.tasks_per_round, family=f, seed=ROUND_TASK_BASE * (args.seed + 1) + r
            )
            for f in families
        }
        n_tasks = sum(len(t) for t in tasks_by_family.values())
        if args.arm == "closed-propose":
            family = families[0]
            amorce_specs = [
                spec_from_task(t)
                for t in task_families.make_tasks(
                    args.propose_amorce, family=family, seed=SEED_TASK_BASE + args.seed
                )
            ]
            proposals, counts = propose_round(
                model,
                tokenizer,
                family,
                propose_n,
                seen=proposed_signatures,
                amorce_specs=amorce_specs,
                temperature=args.temperature,
                round_index=r,
            )
            proposed_tasks = [t for _, t in proposals]
            if proposed_tasks:
                generation = generate_round(
                    model,
                    tokenizer,
                    family,
                    proposed_tasks,
                    quarantine,
                    attempts=args.attempts,
                    temperature=args.temperature,
                    round_index=r,
                    copy_topk=args.copy_topk,
                    explore_from_attempt=args.explore_from_attempt,
                    copy_explore=args.copy_explore,
                )
            else:
                generation = {
                    "tasks": 0,
                    "episodes": 0,
                    "attempts": args.attempts,
                    "solved": 0,
                    "solved_at": {},
                    "promoted_new": 0,
                    "promoted_explored": 0,
                    "tokens": 0,
                    "seconds": 0.0,
                }
            # Verified but sloppy solutions are not taught (docs/31 amendment 7).
            demoted = quarantine.discard(
                lambda e, r=r, family=family: (
                    e.family == family
                    and entry_round(e) == r
                    and e.promoted
                    and not clean_trajectory(e.trajectory)
                )
            )
            generation["demoted_sloppy"] = demoted
            generation["promoted_new"] -= demoted
            solved_at = generation["solved_at"]
            generation["proposals"] = {
                **counts,
                "solved_first": sum(1 for v in solved_at.values() if v == 1),
                "solved_retry": sum(1 for v in solved_at.values() if v >= 2),
                "unsolved": len(proposed_tasks) - len(solved_at),
                "promoted": promote_proposals(
                    proposals, solved_at, quarantine, family=family, round_index=r
                ),
            }
            generation["tokens"] += counts["tokens"]
        elif args.arm in ("closed", "closed-clean"):
            generation = merge_generation(
                {
                    f: generate_round(
                        model,
                        tokenizer,
                        f,
                        tasks,
                        quarantine,
                        attempts=args.attempts,
                        temperature=args.temperature,
                        round_index=r,
                        copy_topk=args.copy_topk,
                        explore_from_attempt=args.explore_from_attempt,
                        copy_explore=args.copy_explore,
                    )
                    for f, tasks in tasks_by_family.items()
                }
            )
            if args.arm == "closed-clean":
                # Verified but sloppy episodes are not taught (docs/31 amendment 7).
                demoted = quarantine.discard(
                    lambda e, r=r: (
                        entry_round(e) == r and e.promoted and not clean_trajectory(e.trajectory)
                    )
                )
                generation["demoted_sloppy"] = demoted
                generation["promoted_new"] -= demoted
        elif args.arm == "closed-klpo":
            parts, klpo_episodes = {}, []
            for f, tasks in tasks_by_family.items():
                parts[f], family_episodes = generate_round_klpo(
                    model,
                    tokenizer,
                    f,
                    tasks,
                    quarantine,
                    attempts=args.attempts,
                    temperature=args.klpo_temperature,
                    draws=args.klpo_draws,
                    round_index=r,
                )
                klpo_episodes.extend(family_episodes)
            generation = merge_generation(parts)
            write_json(args.out / f"round-{r:03d}-episodes.json", klpo_episodes)
        elif args.arm == "oracle":
            generation = merge_generation(
                {
                    f: oracle_round(f, tasks, quarantine, round_index=r)
                    for f, tasks in tasks_by_family.items()
                }
            )
        else:
            generation = {
                "tasks": n_tasks,
                "episodes": 0,
                "attempts": 0,
                "solved": 0,
                "promoted_new": 0,
                "tokens": 0,
                "seconds": 0.0,
            }
        train = None
        if args.arm != "frozen":
            rows, row_stats = rows_from_entries(
                promoted_entries(), registries, tokenizer, seq_len=args.seq_len
            )
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
                    device=args.device,
                    replay_names=replay_names,
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
                device=args.device,
            )
            klpo["seconds"] = time.time() - klpo_started
        started = time.time()
        measured = evaluate(
            model,
            tokenizer,
            bench_families,
            work=args.work,
            bench_tasks=args.bench_tasks,
            bpb_docs=args.bpb_docs,
            seq_len=args.seq_len,
            device=args.device,
            hard=args.hard_bench,
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
                "promoted_total": len(promoted_entries()),
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
