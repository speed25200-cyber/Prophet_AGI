#!/usr/bin/env python3
"""Summarise a closed-loop pilot (docs/31): the curves per arm and seed, and the four
pre-registered checks, computed from the round records and nothing else.

    python scripts/summarize_closed_loop.py --root OUT --out OUT/summary.json

``OUT`` holds ``<arm>-seed<s>/rounds.jsonl`` directories written by
``scripts/closed_loop.py``. The checks are the ones of docs/31 §1; the script reports
them, it does not soften them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ARMS = ("closed", "oracle", "frozen")
H3_ALLOWANCE = 0.05


def load_runs(root: Path) -> dict[str, dict[int, list[dict]]]:
    runs: dict[str, dict[int, list[dict]]] = {}
    for path in sorted(root.glob("*-seed*/rounds.jsonl")):
        arm, seed = path.parent.name.rsplit("-seed", 1)
        rounds = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rounds or rounds[0]["round"] != 0:
            raise ValueError(f"{path}: rounds must start at 0")
        runs.setdefault(arm, {})[int(seed)] = rounds
    if not runs:
        raise ValueError(f"{root}: no rounds.jsonl found")
    return runs


def outcomes(round_record: dict) -> list[bool]:
    """Per-task bench outcomes across the bench seeds, in a fixed order."""
    flags: list[bool] = []
    for bench in round_record["bench"]:
        if "verified" not in bench:
            raise ValueError("bench record lacks per-task outcomes")
        flags.extend(bool(v) for v in bench["verified"])
    return flags


def paired_gain(first: dict, last: dict, *, draws: int = 10_000, seed: int = 0) -> dict:
    """Success(last) - success(first) on the same tasks, with a paired bootstrap interval."""
    a, b = outcomes(first), outcomes(last)
    if len(a) != len(b) or not a:
        raise ValueError("bench outcomes are not paired")
    n = len(a)
    diff = [int(y) - int(x) for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    resampled = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    low = resampled[int(0.025 * (draws - 1))]
    high = resampled[int(0.975 * (draws - 1))]
    return {
        "tasks": n,
        "gain": sum(diff) / n,
        "won": sum(1 for d in diff if d > 0),
        "lost": sum(1 for d in diff if d < 0),
        "interval_95": [low, high],
    }


def curve(rounds: list[dict]) -> list[dict]:
    out = []
    for previous, current in zip([None] + rounds[:-1], rounds, strict=True):
        hours = current["compute_seconds"] / 3600
        gain_per_hour = None
        if previous is not None:
            delta_hours = (current["compute_seconds"] - previous["compute_seconds"]) / 3600
            if delta_hours > 0:
                gain_per_hour = (current["success_mean"] - previous["success_mean"]) / delta_hours
        out.append(
            {
                "round": current["round"],
                "success": current["success_mean"],
                "bpb": current["bpb"]["bpb"] if current.get("bpb") else None,
                "promoted_total": current["promoted_total"],
                "compute_hours": hours,
                "gain_per_hour": gain_per_hour,
                "generated_tokens": (current.get("generation") or {}).get("tokens", 0),
            }
        )
    return out


def summarise(runs: dict[str, dict[int, list[dict]]]) -> dict:
    arms: dict[str, dict] = {}
    for arm, by_seed in runs.items():
        seeds = {}
        for seed, rounds in sorted(by_seed.items()):
            first, last = rounds[0], rounds[-1]
            seeds[str(seed)] = {
                "rounds": len(rounds) - 1,
                "curve": curve(rounds),
                "gain": paired_gain(first, last, seed=seed),
                "bpb_delta": (last["bpb"]["bpb"] - first["bpb"]["bpb"])
                if first.get("bpb") and last.get("bpb")
                else None,
                "final_success": last["success_mean"],
                "compute_hours": last["compute_seconds"] / 3600,
            }
        gains = [s["gain"]["gain"] for s in seeds.values()]
        deltas = [s["bpb_delta"] for s in seeds.values() if s["bpb_delta"] is not None]
        arms[arm] = {
            "seeds": seeds,
            "mean_gain": sum(gains) / len(gains),
            "mean_bpb_delta": sum(deltas) / len(deltas) if deltas else None,
        }
    checks: dict[str, dict] = {}
    closed = arms.get("closed")
    oracle = arms.get("oracle")
    if closed:
        per_seed = closed["seeds"].values()
        pooled_won = sum(s["gain"]["won"] for s in per_seed)
        pooled_lost = sum(s["gain"]["lost"] for s in per_seed)
        pooled_n = sum(s["gain"]["tasks"] for s in per_seed)
        checks["H1_self_improvement"] = {
            "gain_positive_every_seed": all(s["gain"]["gain"] > 0 for s in per_seed),
            "interval_excludes_zero_every_seed": all(
                s["gain"]["interval_95"][0] > 0 for s in per_seed
            ),
            "pooled": {"tasks": pooled_n, "won": pooled_won, "lost": pooled_lost},
            "pass": all(s["gain"]["gain"] > 0 for s in per_seed)
            and all(s["gain"]["interval_95"][0] > 0 for s in per_seed),
        }
        last_rounds = [s["curve"][-1]["gain_per_hour"] for s in per_seed]
        checks["H4_compute"] = {
            "last_round_gain_per_hour": last_rounds,
            "pass": all(g is not None and g > 0 for g in last_rounds),
        }
    if closed and oracle:
        ratio = closed["mean_gain"] / oracle["mean_gain"] if oracle["mean_gain"] > 0 else None
        checks["H2_yield"] = {"closed_over_oracle_gain": ratio, "pass": None}
        if closed["mean_bpb_delta"] is not None and oracle["mean_bpb_delta"] is not None:
            checks["H3_forgetting"] = {
                "closed_bpb_delta": closed["mean_bpb_delta"],
                "oracle_bpb_delta": oracle["mean_bpb_delta"],
                "allowance": H3_ALLOWANCE,
                "pass": closed["mean_bpb_delta"] <= oracle["mean_bpb_delta"] + H3_ALLOWANCE,
            }
    return {"arms": arms, "checks": checks}


def markdown(summary: dict) -> str:
    lines = [
        "| Bras | Graine | Tours | Succès 0 → R | Gain [IC 95 %] | Δ BPB | Heures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in summary["arms"].items():
        for seed, s in data["seeds"].items():
            first = s["curve"][0]["success"]
            low, high = s["gain"]["interval_95"]
            delta = "—" if s["bpb_delta"] is None else f"{s['bpb_delta']:+.3f}"
            lines.append(
                f"| {arm} | {seed} | {s['rounds']} | {first:.3f} → {s['final_success']:.3f} | "
                f"{s['gain']['gain']:+.3f} [{low:+.3f}, {high:+.3f}] | {delta} | {s['compute_hours']:.2f} |"
            )
    lines.append("")
    for name, check in summary["checks"].items():
        verdict = check.get("pass")
        label = "passe" if verdict else ("échoue" if verdict is False else "rapporté")
        lines.append(
            f"- **{name}** : {label} — {json.dumps({k: v for k, v in check.items() if k != 'pass'}, default=str)}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    summary = summarise(load_runs(args.root))
    text = markdown(summary)
    print(text)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        args.out.with_suffix(".md").write_text(text)
    return 0 if all(math.isfinite(a["mean_gain"]) for a in summary["arms"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
