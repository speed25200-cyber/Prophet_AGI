#!/usr/bin/env python3
"""Are a checkpoint's failures at the frontier stochastic or systematic? (docs/39
amendment 6.) A self-play proposer is paid for tasks its solver solves *sometimes*
(Absolute Zero's learnability, 0 < successes < k). That signal exists only if sampling
rescues some of the greedy failures. On each out-of-distribution bench: the greedy
bench, then ``k`` attempts with the copy pointer sampled, and the counts.

    python scripts/learnability_probe.py --work W --run OUT/closed-propose-seed0 \\
        --bench calc-hard,calc-digits6 --k 4

At 7M (SI-8a seed 0, round 10): 2 of 18 and 1 of 38 greedy failures rescued by any
sample -- the failures are systematic, and the reward would pay noise, not the frontier.
The same probe on the programme-1 checkpoint is the first gate of self-play at 375M.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.agent import tasks as task_families  # noqa: E402
from prophet.agent.propose import OOD_BENCHES, make_bench  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.agent_bench import run_bench  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.checkpoint import CheckpointManager  # noqa: E402
from scripts import closed_loop  # noqa: E402


def outcomes(model, tokenizer, tasks, cfg) -> list[bool]:
    report = run_bench(
        model,
        tokenizer,
        tasks,
        cfg,
        tools_for=task_families.tools_for,
        verifier_for_task=task_families.verifier_for,
    )
    return [bool(e.verified) for e in report.episodes]


def counts(greedy: list[bool], wins: list[int], k: int) -> dict:
    """The probe's reading of per-task successes: ``greedy`` (one greedy attempt) and
    ``wins`` (successes out of ``k`` sampled attempts)."""
    failures = [w for w, g in zip(wins, greedy, strict=True) if not g]
    return {
        "tasks": len(greedy),
        "greedy": sum(greedy),
        "all_k": sum(1 for w in wins if w == k),
        "none": sum(1 for w in wins if w == 0),
        "learnable": sum(1 for w in wins if 0 < w < k),
        "greedy_failures": len(failures),
        "rescued_by_sampling": sum(1 for w in failures if w > 0),
    }


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="a closed_loop.py output directory")
    ap.add_argument("--checkpoints", type=Path, help="default: RUN/checkpoints")
    ap.add_argument("--bench", default="calc-hard", help="comma-separated, among OOD_BENCHES")
    ap.add_argument("--seeds", default="17,19")
    ap.add_argument("--n", type=int, default=30, help="tasks per seed")
    ap.add_argument("--k", type=int, default=4, help="sampled attempts per task")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args(argv)
    benches = [b for b in args.bench.split(",") if b]
    if not benches or set(benches) - set(OOD_BENCHES) or args.k < 2:
        ap.error(f"--bench among {', '.join(OOD_BENCHES)}; --k >= 2")
    protocol = json.loads((args.run / "protocol.json").read_text())
    closed_loop.NO_REPEAT_ACTION = bool(protocol.get("no_repeat_action", False))
    closed_loop.NO_REPEAT_EMITTED = bool(protocol.get("no_repeat_emitted", False))
    closed_loop.COPY_BOUNDARIES = protocol.get("copy_boundaries", "off")
    closed_loop.COPY_END_BOUNDARIES = protocol.get("copy_end_boundaries", "off")
    tokenizer = ProphetTokenizer.load(args.work / "tokenizer.json")
    model = ProphetModel(ProphetConfig.from_dict(protocol["config"]))
    state, _ = CheckpointManager(args.checkpoints or args.run / "checkpoints").load_latest()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    results = []
    for name in benches:
        tasks = [t for s in args.seeds.split(",") for t in make_bench(name, args.n, seed=int(s))]
        greedy = outcomes(
            model,
            tokenizer,
            tasks,
            closed_loop.generation_config("calc", temperature=0.0, no_repeat_action=True),
        )
        wins = [0] * len(tasks)
        sampled = closed_loop.generation_config(
            "calc", temperature=args.temperature, no_repeat_action=True, sample_copy=True
        )
        for j in range(args.k):
            torch.manual_seed(j)
            for i, ok in enumerate(outcomes(model, tokenizer, tasks, sampled)):
                wins[i] += ok
        result = {
            "bench": name,
            "k": args.k,
            "round": state.get("step"),
            **counts(greedy, wins, args.k),
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


if __name__ == "__main__":
    main()
