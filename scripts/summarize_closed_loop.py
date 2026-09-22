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
H20_VALID = 0.5  # docs/33: share of proposals the rules accept, every round
H21_EDGE = (0.2, 0.8)  # solve rate of valid proposals that counts as "at the edge"
H21_ROUNDS = 3  # on at least this many rounds
H22_ALLOWANCE = 0.05
H24_BPB_ALLOWANCE = 0.02
H25_NOVEL = 0.5
SATURATED = 0.95  # a family starting here or above is judged on retention, not gain
SATURATION_LOSS = 0.05


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


def hard_outcomes(round_record: dict) -> list[bool]:
    """Per-task outcomes of the out-of-distribution bench (docs/33), in a fixed order."""
    flags: list[bool] = []
    for bench in round_record.get("bench_hard") or []:
        flags.extend(bool(v) for v in bench["verified"])
    return flags


def paired_gain(
    first: dict,
    last: dict,
    *,
    family: str | None = None,
    draws: int = 10_000,
    seed: int = 0,
    hard: bool = False,
) -> dict:
    """Success(last) - success(first) on the same tasks, with a paired bootstrap interval;
    ``hard`` takes the out-of-distribution bench instead of the bench."""
    if hard:
        a, b = hard_outcomes(first), hard_outcomes(last)
    else:
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
            if hard_outcomes(first):
                seeds[str(seed)]["hard"] = {
                    "curve": [r["success_hard"] for r in rounds],
                    "gain": paired_gain(first, last, seed=seed, hard=True),
                    "task_sets": {
                        "tasks": len(hard_outcomes(first)),
                        "never": sum(
                            not any(t)
                            for t in zip(*[hard_outcomes(r) for r in rounds], strict=True)
                        ),
                    },
                }
            proposals = [(r.get("generation") or {}).get("proposals") for r in rounds[1:]]
            if proposals and all(proposals):
                seeds[str(seed)]["proposals"] = {
                    "per_round": proposals,
                    "valid_share": [p["valid"] / max(p["emitted"], 1) for p in proposals],
                    "solve_rate": [
                        (p["solved_first"] + p["solved_retry"]) / p["valid"] if p["valid"] else None
                        for p in proposals
                    ],
                    "novel_share": sum(p["novel"] for p in proposals)
                    / max(sum(p["valid"] for p in proposals), 1),
                    "promoted": sum(p["promoted"] for p in proposals),
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
    checks.update(self_proposal_checks(arms))
    return {"arms": arms, "checks": checks}


def self_proposal_checks(arms: dict[str, dict]) -> dict[str, dict]:
    """docs/33 §3: H20 to H25, when a closed-propose arm ran against the closed-clean
    witness with the out-of-distribution bench on both. Absent that, nothing."""
    propose, clean = arms.get("closed-propose"), arms.get("closed-clean")
    if not propose or not clean:
        return {}
    seeds = sorted(set(propose["seeds"]) & set(clean["seeds"]), key=int)
    if not seeds or any("proposals" not in propose["seeds"][s] for s in seeds):
        return {}
    if any("hard" not in propose["seeds"][s] or "hard" not in clean["seeds"][s] for s in seeds):
        return {}
    checks: dict[str, dict] = {}
    valid_ok = all(
        all(v >= H20_VALID for v in propose["seeds"][s]["proposals"]["valid_share"]) for s in seeds
    )
    checks["H20_validity"] = {
        "valid_share": {s: propose["seeds"][s]["proposals"]["valid_share"] for s in seeds},
        "pass": valid_ok,
    }
    edge_rounds = {
        s: sum(
            1
            for v in propose["seeds"][s]["proposals"]["solve_rate"]
            if v is not None and H21_EDGE[0] <= v <= H21_EDGE[1]
        )
        for s in seeds
    }
    checks["H21_edge"] = {
        "solve_rate": {s: propose["seeds"][s]["proposals"]["solve_rate"] for s in seeds},
        "edge_rounds": edge_rounds,
        "pass": all(n >= H21_ROUNDS for n in edge_rounds.values()),
    }
    transfer = all(
        propose["seeds"][s]["gain"]["gain"] >= clean["seeds"][s]["gain"]["gain"] - H22_ALLOWANCE
        and propose["seeds"][s]["gain"]["interval_95"][0] > 0
        for s in seeds
    )
    checks["H22_transfer"] = {
        "propose_gain": {s: propose["seeds"][s]["gain"]["gain"] for s in seeds},
        "clean_gain": {s: clean["seeds"][s]["gain"]["gain"] for s in seeds},
        "pass": transfer,
    }
    reach = all(
        propose["seeds"][s]["hard"]["gain"]["gain"] > clean["seeds"][s]["hard"]["gain"]["gain"]
        and propose["seeds"][s]["hard"]["gain"]["interval_95"][0] > 0
        for s in seeds
    )
    checks["H23_reach"] = {
        "propose_hard_gain": {s: propose["seeds"][s]["hard"]["gain"] for s in seeds},
        "clean_hard_gain": {s: clean["seeds"][s]["hard"]["gain"]["gain"] for s in seeds},
        "pass": reach,
    }
    deltas_known = propose["mean_bpb_delta"] is not None and clean["mean_bpb_delta"] is not None
    checks["H24_forgetting"] = {
        "propose_bpb_delta": propose["mean_bpb_delta"],
        "clean_bpb_delta": clean["mean_bpb_delta"],
        "pass": bool(
            deltas_known
            and propose["mean_bpb_delta"] <= clean["mean_bpb_delta"] + H24_BPB_ALLOWANCE
        ),
    }
    novel_share = {s: propose["seeds"][s]["proposals"]["novel_share"] for s in seeds}
    checks["H25_novelty"] = {
        "novel_share": novel_share,
        "pass": all(v >= H25_NOVEL for v in novel_share.values()),
    }
    checks["programme_3"] = {
        "pass": bool(valid_ok and transfer and reach),
        "rule": "H20, H22 and H23 (docs/33 §3)",
    }
    return checks


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

    def learns_or_keeps(name: str, seed: str, f: str) -> bool:
        """A family with room to gain must gain; one that starts saturated (docs/31
        amendment 20: at or above SATURATED) must not lose more than the allowance."""
        curve_f = arms[name]["seeds"][seed]["by_family"][f]["curve"]
        if curve_f[0] >= SATURATED:
            return curve_f[-1] >= curve_f[0] - SATURATION_LOSS
        return family_gain(name, seed, f) > 0

    learns_both = all(
        all(learns_or_keeps(mixed_name, s, f) for f in families)
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
    if any("hard" in s for data in summary["arms"].values() for s in data["seeds"].values()):
        lines += [
            "| Bras | Graine | Banc hors distribution 0 → R | Gain [IC 95 %] | Jamais réussies | Propositions valides / tour | Résolues à la reprise | Promues |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for arm, data in summary["arms"].items():
            for seed, s in data["seeds"].items():
                if "hard" not in s:
                    continue
                h = s["hard"]
                low, high = h["gain"]["interval_95"]
                p = s.get("proposals")
                valid = ", ".join(str(r["valid"]) for r in p["per_round"]) if p else "—"
                retry = sum(r["solved_retry"] for r in p["per_round"]) if p else "—"
                lines.append(
                    f"| {arm} | {seed} | {h['curve'][0]:.3f} → {h['curve'][-1]:.3f} | "
                    f"{h['gain']['gain']:+.3f} [{low:+.3f}, {high:+.3f}] | {h['task_sets']['never']} | "
                    f"{valid} | {retry} | {p['promoted'] if p else '—'} |"
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
