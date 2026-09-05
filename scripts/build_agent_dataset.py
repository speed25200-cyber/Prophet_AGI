#!/usr/bin/env python3
"""Render the agentic training set: perfect trajectories of every verifiable family.

    python scripts/build_agent_dataset.py --out data/agent --per-family 2000 --seed 1

Writes one ``<family>.jsonl`` per task family (``{"text": ..., "family": ..., "task":
..., "answer": ...}``), rendered in the control-id stream the action heads read their
targets from, plus ``manifest.json`` with counts, byte sizes, the generator seed and the
seed reserved for evaluation. The text is rendered by this project, so a loader must
encode it with ``parse_special=True`` (``TokenisedSource(..., parse_special=True)``).

Seeds are the split: ``--seed`` generates the training tasks, ``--eval-seed`` is written
to the manifest as the seed the benchmark must use, and the script refuses to build a
set whose seeds coincide. No benchmark item can leak into this set because none is
used to make it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.agent.render import render_episode  # noqa: E402
from prophet.agent.tasks import FAMILIES, make_tasks, perfect_trajectory, tools_for  # noqa: E402


def build(out: Path, *, per_family: int, seed: int, eval_seed: int, families: list[str]) -> dict:
    if seed == eval_seed:
        raise SystemExit("--seed and --eval-seed must differ: the seed is the train/eval split")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": seed, "eval_seed": eval_seed, "per_family": per_family, "families": {}}
    try:
        manifest["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        manifest["commit"] = None
    for family in families:
        path = out / f"{family}.jsonl"
        n_bytes = 0
        with path.open("w", encoding="utf-8") as f:
            for task in make_tasks(per_family, family=family, seed=seed):
                text = render_episode(task.goal, tools_for(task), perfect_trajectory(task))
                row = {"text": text, "family": family, "task": task.name, "answer": task.answer}
                line = json.dumps(row, ensure_ascii=False)
                f.write(line + "\n")
                n_bytes += len(text.encode("utf-8"))
        manifest["families"][family] = {"episodes": per_family, "bytes": n_bytes,
                                        "mean_bytes": n_bytes / max(per_family, 1), "path": path.name}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()
    manifest = build(Path(args.out), per_family=args.per_family, seed=args.seed, eval_seed=args.eval_seed,
                     families=[f for f in args.families.split(",") if f])
    for family, stats in manifest["families"].items():
        print(f"{family:10s} {stats['episodes']:6d} episodes  {stats['bytes'] / 1e6:6.2f} MB  {stats['mean_bytes']:.0f} B/episode")
    print(f"manifest   {Path(args.out) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
