"""Recompute the preregistered R04 depth-policy screen from final document losses.

This does not reload weights, rerun inference, or establish training-seed uncertainty.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "docs/experiments/2026-09-20-r04-depth-adaptation-plan/protocol.json"
REFERENCE = ROOT / "docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json"
DEPTHS = {"1", "2", "4", "6", "8"}
DOCUMENT_KEYS = ("index", "sha256", "scored_tokens", "scored_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_report(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def score(result, reference, depth):
    require(result["loop_k"] == depth, "evaluation depth differs")
    for key in ("seq_len", "batch_size", "precision", "protocol"):
        require(result[key] == reference[key], f"evaluation setting differs: {key}")
    rows = result["documents"]
    require(len(rows) == len(reference["documents"]) > 0, "document count differs")
    for row, original in zip(rows, reference["documents"], strict=True):
        require(
            all(row[key] == original[key] for key in DOCUMENT_KEYS),
            "paired document identity differs",
        )
        require(
            row["scored_tokens"] > 0
            and row["scored_bytes"] > 0
            and math.isfinite(row["total_nats"])
            and row["total_nats"] >= 0,
            "invalid document loss or denominator",
        )
    nats = sum(row["total_nats"] for row in rows)
    tokens = sum(row["scored_tokens"] for row in rows)
    size = sum(row["scored_bytes"] for row in rows)
    values = {
        "total_nats": nats,
        "scored_tokens": tokens,
        "scored_bytes": size,
        "nats_per_token": nats / tokens,
        "bits_per_byte": nats / math.log(2) / size,
    }
    for key, value in values.items():
        require(math.isclose(result[key], value, rel_tol=1e-12), f"aggregate differs: {key}")
    return values


def summarize(fixed, variable, plan, plan_sha256, reference, *, experiment="depth"):
    require(experiment in ("depth", "reinjection"), "unknown screen experiment")
    reinjection = experiment == "reinjection"
    arms = ("fixed_sum", "learned_mix") if reinjection else ("fixed4", "uniform2to6")
    protocol = "r04-input-adapter-v1" if reinjection else "r04-depth-adaptation-v2"
    label = "learned_mix" if reinjection else "variable"
    if reinjection:
        require(plan["experiment"] == protocol, "screen plan protocol differs")
    summaries = {}
    identities = []
    for arm, report in zip(arms, (fixed, variable), strict=True):
        identity = report["identity"]
        require(
            identity["arm"] == arm
            and identity["mode"] == "train"
            and identity["protocol"] == protocol
            and identity["numerical_policy"]["deterministic_algorithms"]
            and identity["plan_sha256"] == plan_sha256
            and identity["parent_checkpoint"] == plan["parent_checkpoint"],
            "run contract differs from planned strict-policy arm",
        )
        identities.append({key: value for key, value in identity.items() if key != "arm"})
        require(
            report["complete"]
            and report["step"] == report["checkpoint"]["step"] == plan["steps_each"]
            and report["tokens_seen"] == plan["additional_tokens_each"]
            and report["skipped_nonfinite"] == 0
            and set(report["results"]) == DEPTHS,
            "incomplete or invalid final run",
        )
        history = report["depth_history"]
        require(len(history) == plan["steps_each"], "incomplete depth history")
        allowed = {4} if not reinjection and arm == arms[0] else {2, 3, 4, 5, 6}
        require(set(history) <= allowed, "training depth outside planned support")
        counts = {str(key): value for key, value in Counter(history).items()}
        require(report["depth_counts"] == counts, "depth counts differ")
        summaries[arm] = {
            "depth_counts": counts,
            "mean_training_loops": sum(history) / len(history),
            "scores": {
                depth: score(report["results"][depth], reference, int(depth))
                for depth in sorted(DEPTHS, key=int)
            },
        }
    require(identities[0] == identities[1], "paired run identities differ")
    require(fixed["loader_step"] == variable["loader_step"], "paired loader cursors differ")
    if reinjection:
        require(
            fixed["depth_history"] == variable["depth_history"],
            "paired component training depths differ",
        )

    rows4, rows6 = (variable["results"][str(k)]["documents"] for k in (4, 6))
    tokens = np.array([row["scored_tokens"] for row in rows4], dtype=np.int64)
    delta = np.array([b["total_nats"] - a["total_nats"] for a, b in zip(rows4, rows6, strict=True)])
    # Matches the published frozen-depth analysis: PCG64 and linear quantiles.
    draws = np.random.Generator(np.random.PCG64(0)).integers(0, len(rows4), (10000, len(rows4)))
    samples = delta[draws].sum(axis=1) / tokens[draws].sum(axis=1)
    interval = np.quantile(samples, [0.025, 0.975], method="linear").tolist()
    f, v = (summaries[arm]["scores"] for arm in arms)
    ratio = v["6"]["bits_per_byte"] / v["4"]["bits_per_byte"]
    control_ratio = v["4"]["bits_per_byte"] / f["4"]["bits_per_byte"]
    checks = {
        f"{label}_k6_improves_own_k4_by_at_least_half_percent": ratio <= 0.995,
        "paired_ce_interval_upper_below_zero": interval[1] < 0,
        f"{label}_k4_within_one_percent_of_control_k4": control_ratio <= 1.01,
        f"{label}_k6_improves_control_k6": v["6"]["bits_per_byte"] < f["6"]["bits_per_byte"],
    }
    return {
        "protocol": "r04-input-adapter-screen-v1"
        if reinjection
        else "r04-depth-adaptation-screen-v1",
        "plan_sha256": plan_sha256,
        "identity_without_arm": identities[0],
        "arms": summaries,
        f"{label}_k6_over_k4_bpb": ratio,
        f"{label}_k4_over_control_k4_bpb": control_ratio,
        f"{label}_k6_minus_k4_ce": float(delta.sum() / tokens.sum()),
        "paired_document_95pct_ce_interval": interval,
        "bootstrap": {"draws": 10000, "seed": 0, "rng": "PCG64", "quantile": "linear"},
        "checks": checks,
        "seed0_screen_passed": all(checks.values()),
        "scope": "Recomputed exported document losses, conditional on these checkpoints. No weight reload, repeated inference, training-seed uncertainty, reasoning or architecture-adoption claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--variable", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "use a fresh output report")
    result = summarize(
        read_report(args.fixed),
        read_report(args.variable),
        read_report(PLAN),
        hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        read_report(REFERENCE),
    )
    result["input_files"] = {
        arm: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for arm, path in (("fixed4", args.fixed), ("uniform2to6", args.variable))
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {"seed0_screen_passed": result["seed0_screen_passed"], "checks": result["checks"]}
        )
    )


if __name__ == "__main__":
    main()
