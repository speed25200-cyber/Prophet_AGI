#!/usr/bin/env python3
"""Bench a closed-loop checkpoint on an out-of-distribution set, after the run (docs/39
amendment 2). The model config and the decoding switches are read from the run's
``protocol.json``, so the bench is the greedy one the run itself used.

    python scripts/bench_checkpoint.py --work W --run OUT/closed-propose-seed0 \\
        --bench calc-digits --seeds 17,19 --n 30

Prints one JSON line: success rate and per-task outcomes, in a fixed order, so that two
checkpoints on the same bench can be compared task by task (paired gain).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.agent.propose import make_hard_calc, make_hard_calc_digits  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.checkpoint import CheckpointManager  # noqa: E402
from scripts import closed_loop  # noqa: E402

BENCHES = {"calc-hard": make_hard_calc, "calc-digits": make_hard_calc_digits}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="the run's work directory")
    ap.add_argument("--run", type=Path, required=True, help="a closed_loop.py output directory")
    ap.add_argument(
        "--checkpoints", type=Path, help="checkpoint directory (default: RUN/checkpoints)"
    )
    ap.add_argument("--bench", choices=sorted(BENCHES), required=True)
    ap.add_argument("--seeds", default="17,19")
    ap.add_argument("--n", type=int, default=30, help="tasks per seed")
    args = ap.parse_args(argv)
    protocol = json.loads((args.run / "protocol.json").read_text())
    # The switches the run's bench read, as main() of closed_loop.py sets them.
    closed_loop.NO_REPEAT_ACTION = bool(protocol.get("no_repeat_action", False))
    closed_loop.NO_REPEAT_EMITTED = bool(protocol.get("no_repeat_emitted", False))
    closed_loop.COPY_BOUNDARIES = protocol.get("copy_boundaries", "off")
    closed_loop.COPY_END_BOUNDARIES = protocol.get("copy_end_boundaries", "off")
    tokenizer = ProphetTokenizer.load(args.work / "tokenizer.json")
    model = ProphetModel(ProphetConfig.from_dict(protocol["config"]))
    state, _ = CheckpointManager(args.checkpoints or args.run / "checkpoints").load_latest()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    verified: list[bool] = []
    for seed in (int(s) for s in args.seeds.split(",")):
        tasks = BENCHES[args.bench](args.n, seed=seed)
        record = closed_loop.bench_family(
            model, tokenizer, "calc", n_tasks=args.n, seed=seed, tasks=tasks
        )
        verified += record["verified"]
    result = {
        "bench": args.bench,
        "run": str(args.run),
        "round": state.get("step"),  # closed_loop.py saves the round as the step
        "seeds": args.seeds,
        "success": sum(verified) / len(verified),
        "verified": verified,
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
