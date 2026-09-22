#!/usr/bin/env python3
"""Summarise a closed-loop pilot (docs/31): the curves per arm and seed, and the
pre-registered checks (H1-H4; H5 when the KLPO arm ran; H8 when the closed-clean arm
ran; H14 when several families shared one loop), computed from the round records and
nothing else. Bench records that carry a ``family`` give the gains per family.

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

ARMS = ("closed", "oracle", "frozen", "closed-klpo", "closed-clean")
COLLAPSE_ALLOWANCE = 0.10
H8_BPB_ALLOWANCE = 0.02
H3_ALLOWANCE = 0.05
H14_YIELD = 0.6


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


def outcomes(round_record: dict, family: str | None = None) -> list[bool]:
    """Per-task bench outcomes across the bench seeds, in a fixed order; of one family
    when ``family`` is given."""
    flags: list[bool] = []
    for bench in round_record["bench"]:
        if "verified" not in bench:
            raise ValueError("bench record lacks per-task outcomes")
        if family is not None and bench.get("family") != family:
            continue
        flags.extend(bool(v) for v in bench["verified"])
    return flags


def bench_families(round_record: dict) -> list[str] | None:
    """The families the bench measured, in bench order; ``None`` for records written
    before benches carried a family."""
    seen: list[str] = []
    for bench in round_record["bench"]:
        if "family" not in bench:
            return None
        if bench["family"] not in seen:
            seen.append(bench["family"])
    return seen


def trained_families(rounds: list[dict]) -> list[str] | None:
    """The families whose episodes the arm generated or was given, from the first round's
    generation record; ``None`` when the record does not say (frozen arm, older runs)."""
    for record in rounds[1:]:
        by_family = (record.get("generation") or {}).get("by_family")
        if by_family:
            return list(by_family)
    return None


def task_sets(rounds: list[dict], family: str | None = None) -> dict:
    """How the bench tasks behave across the rounds: always solved, never solved, or
    flipping; and the bound the never-solved set puts on what the loop can still learn
    from itself (docs/32 §14: a verifier filters, it does not contradict)."""
    per_task = list(zip(*[outcomes(r, family) for r in rounds], strict=True))
    always = sum(all(t) for t in per_task)
    never = sum(not any(t) for t in per_task)
    return {
        "tasks": len(per_task),
        "always": always,
        "never": never,
        "flip": len(per_task) - always - never,
        "reachable_bound": 1 - never / len(per_task) if per_task else None,
    }


def family_success(round_record: dict, family: str) -> float:
    flags = outcomes(round_record, family)
    return sum(flags) / len(flags)


def paired_gain(
    first: dict, last: dict, *, family: str | None = None, draws: int = 10_000, seed: int = 0
) -> dict:
    """Success(last) - success(first) on the same tasks, with a paired bootstrap interval."""
    a, b = outcomes(first, family), outcomes(last, family)
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
                "task_sets": task_sets(rounds),
            }
            measured = bench_families(first)
            if measured:
                seeds[str(seed)]["by_family"] = {
                    f: {
                        "curve": [family_success(r, f) for r in rounds],
                        "gain": paired_gain(first, last, family=f, seed=seed),
                        "task_sets": task_sets(rounds, f),
                    }
                    for f in measured
                }
        gains = [s["gain"]["gain"] for s in seeds.values()]
        deltas = [s["bpb_delta"] for s in seeds.values() if s["bpb_delta"] is not None]
        arms[arm] = {
            "seeds": seeds,
            "trained": trained_families(next(iter(by_seed.values()))),
            "bench_families": bench_families(next(iter(by_seed.values()))[0]),
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
    klpo = arms.get("closed-klpo")
    if closed and klpo:
        # docs/31 H5: at equal task budget the KLPO arm gains at least as much as the
        # closed arm on the seed mean and drifts no more in bits per byte.
        gain_ok = klpo["mean_gain"] >= closed["mean_gain"]
        deltas_known = closed["mean_bpb_delta"] is not None and klpo["mean_bpb_delta"] is not None
        drift_ok = deltas_known and klpo["mean_bpb_delta"] <= closed["mean_bpb_delta"]
        checks["H5_klpo"] = {
            "klpo_gain": klpo["mean_gain"],
            "closed_gain": closed["mean_gain"],
            "klpo_bpb_delta": klpo["mean_bpb_delta"],
            "closed_bpb_delta": closed["mean_bpb_delta"],
            "pass": bool(gain_ok and drift_ok),
        }
    clean = arms.get("closed-clean")
    if closed and clean:
        # docs/31 H8: canonical-form promotion removes the collapse without losing the gain.
        no_collapse = all(
            min(point["success"] for point in s["curve"])
            >= s["curve"][0]["success"] - COLLAPSE_ALLOWANCE
            for s in clean["seeds"].values()
        )
        gain_ok = clean["mean_gain"] >= closed["mean_gain"]
        deltas_known = closed["mean_bpb_delta"] is not None and clean["mean_bpb_delta"] is not None
        drift_ok = (
            deltas_known and clean["mean_bpb_delta"] <= closed["mean_bpb_delta"] + H8_BPB_ALLOWANCE
        )
        checks["H8_clean"] = {
            "no_round_below_start_minus_allowance": no_collapse,
            "clean_gain": clean["mean_gain"],
            "closed_gain": closed["mean_gain"],
            "clean_bpb_delta": clean["mean_bpb_delta"],
            "closed_bpb_delta": closed["mean_bpb_delta"],
            "pass": bool(no_collapse and gain_ok and drift_ok),
        }
    checks.update(h14_multi_family(arms))
    return {"arms": arms, "checks": checks}


def h14_multi_family(arms: dict[str, dict]) -> dict[str, dict]:
    """docs/31 amendment 14: one closed arm looping on several families, one closed arm
    per family looping on that family alone, every arm benched on all of them, and the
    oracle on the same families. Absent that layout, nothing is reported."""
    closed_arms = {
        name: data
        for name, data in arms.items()
        if name.startswith("closed") and data["trained"] and data["bench_families"]
    }
    mixed = [name for name, data in closed_arms.items() if len(data["trained"]) > 1]
    if len(mixed) != 1:
        return {}
    mixed_name = mixed[0]
    families = arms[mixed_name]["trained"]
    monos = {}
    for f in families:
        for name, data in closed_arms.items():
            if data["trained"] == [f] and set(data["bench_families"]) >= set(families):
                monos[f] = name
    oracle = arms.get("oracle")
    if len(monos) != len(families) or not oracle or oracle["trained"] != families:
        return {}
    seeds = set(arms[mixed_name]["seeds"])
    for name in list(monos.values()) + ["oracle"]:
        seeds &= set(arms[name]["seeds"])
    if not seeds:
        return {}
    seeds = sorted(seeds, key=int)

    def family_gain(name: str, seed: str, f: str) -> float:
        return arms[name]["seeds"][seed]["by_family"][f]["gain"]["gain"]

    def summed_gain(name: str) -> float:
        return sum(sum(family_gain(name, s, f) for f in families) for s in seeds) / len(seeds)

    learns_both = all(
        all(family_gain(mixed_name, s, f) > 0 for f in families)
        and arms[mixed_name]["seeds"][s]["gain"]["interval_95"][0] > 0
        for s in seeds
    )
    mixed_sum = summed_gain(mixed_name)
    mono_sums = {name: summed_gain(name) for name in monos.values()}
    omission = {}
    no_forgetting = True
    for f, name in monos.items():
        for g in families:
            if g == f:
                continue
            for s in seeds:
                curve_g = arms[name]["seeds"][s]["by_family"][g]["curve"]
                held = min(curve_g) >= curve_g[0] - COLLAPSE_ALLOWANCE
                no_forgetting &= held
                omission[f"{name}/seed{s}/{g}"] = {
                    "start": curve_g[0],
                    "min": min(curve_g),
                    "transfer_gain": family_gain(name, s, g),
                    "held": held,
                }
    mixed_union = sum(arms[mixed_name]["seeds"][s]["gain"]["gain"] for s in seeds) / len(seeds)
    oracle_union = sum(oracle["seeds"][s]["gain"]["gain"] for s in seeds) / len(seeds)
    ratio = mixed_union / oracle_union if oracle_union > 0 else None
    return {
        "H14_multi_family": {
            "families": families,
            "seeds": seeds,
            "mixed_learns_both_every_seed": learns_both,
            "summed_gain_mixed": mixed_sum,
            "summed_gain_mono": mono_sums,
            "no_interference": mixed_sum >= max(mono_sums.values()),
            "omission": omission,
            "no_forgetting_by_omission": no_forgetting,
            "yield_mixed_over_oracle": ratio,
            "pass": bool(
                learns_both
                and mixed_sum >= max(mono_sums.values())
                and no_forgetting
                and ratio is not None
                and ratio >= H14_YIELD
            ),
        }
    }


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
    lines += [
        "| Bras | Graine | Tâches | Toujours réussies | Jamais réussies | Basculent | Borne atteignable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in summary["arms"].items():
        for seed, s in data["seeds"].items():
            t = s["task_sets"]
            lines.append(
                f"| {arm} | {seed} | {t['tasks']} | {t['always']} | {t['never']} | {t['flip']} | "
                f"{t['reachable_bound']:.3f} |"
            )
    lines.append("")
    if any("by_family" in s for data in summary["arms"].values() for s in data["seeds"].values()):
        lines += [
            "| Bras | Graine | Famille | Succès 0 → R | Gain [IC 95 %] | Pire tour |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for arm, data in summary["arms"].items():
            for seed, s in data["seeds"].items():
                for family, fs in s.get("by_family", {}).items():
                    low, high = fs["gain"]["interval_95"]
                    lines.append(
                        f"| {arm} | {seed} | {family} | {fs['curve'][0]:.3f} → {fs['curve'][-1]:.3f} | "
                        f"{fs['gain']['gain']:+.3f} [{low:+.3f}, {high:+.3f}] | {min(fs['curve']):.3f} |"
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
