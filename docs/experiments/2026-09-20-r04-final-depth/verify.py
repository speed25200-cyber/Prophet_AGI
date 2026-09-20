"""Verify source identities and recompute frozen R04 depth statistics.

Run from any directory with the repository's Python environment. This reuses
exported per-document losses, not a second model forward pass.
"""

import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def data(name):
    path = HERE / name
    return (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    )


def main():
    manifest = json.loads(data("export-manifest.json"))
    for name, expected in manifest["files"].items():
        raw = data(name)
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
    r = json.loads(data("report.json"))
    original = json.loads(
        (
            ROOT / "docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json"
        ).read_bytes()
    )
    assert r["complete"] and r["planned_depths"] == [4, 1, 2, 6, 8]
    assert r["checkpoint"] == original["checkpoint"] and r["checkpoint"]["step"] == 4096
    assert r["run_protocol"] == original["run_protocol"]
    assert r["run_protocol"]["revision"] == "e5720d0b774977455b1920a6f67b5df078377f11"
    source = subprocess.check_output(
        ["git", "show", manifest["driver_revision"] + ":scripts/eval_r04_depth.py"], cwd=ROOT
    )
    assert hashlib.sha256(source).hexdigest() == r["evaluation_driver_sha256"]
    reference = original["documents"]
    fields = ["index", "sha256", "scored_tokens", "scored_bytes"]
    identities = [{key: row[key] for key in fields} for row in reference]
    assert len(reference) == 376
    summaries = {}
    for k, result in zip(r["planned_depths"], r["results"], strict=True):
        assert result["loop_k"] == k and result["seq_len"] == 2048 and result["batch_size"] == 8
        rows = result["documents"]
        assert [{key: row[key] for key in fields} for row in rows] == identities
        assert all(math.isfinite(row["total_nats"]) and row["total_nats"] > 0 for row in rows)
        targets = sum(row["scored_tokens"] for row in rows)
        size = sum(row["scored_bytes"] for row in rows)
        nats = sum(row["total_nats"] for row in rows)
        assert targets == result["scored_tokens"] == 393040
        assert size == result["scored_bytes"] == 1750592
        assert math.isclose(nats, result["total_nats"], rel_tol=1e-12)
        assert math.isclose(nats / targets, result["nats_per_token"], rel_tol=1e-12)
        assert math.isclose(nats / math.log(2) / size, result["bits_per_byte"], rel_tol=1e-12)
        if k == 4:
            assert rows == reference and result["nats_per_token"] == original["nats_per_token"]
        summaries[k] = {
            "nats_per_token": nats / targets,
            "bits_per_byte": nats / math.log(2) / size,
            "bpb_change_percent_vs_k4": 100 * (nats / original["total_nats"] - 1),
        }
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(reference), size=(10000, len(reference)))
    targets = np.array([row["scored_tokens"] for row in reference])
    base = np.array([row["total_nats"] for row in reference])
    contrasts = {}
    for k, result in zip(r["planned_depths"], r["results"], strict=True):
        if k == 4:
            continue
        delta = np.array([row["total_nats"] for row in result["documents"]]) - base
        sampled = delta[draws].sum(axis=1) / targets[draws].sum(axis=1)
        contrasts[f"k{k}_minus_k4"] = {
            "nats_per_token": float(delta.sum() / targets.sum()),
            "paired_document_unadjusted_95pct_interval": np.quantile(
                sampled, [0.025, 0.975]
            ).tolist(),
        }
    queue = json.loads(data("queue.json"))
    assert (
        queue["status"] == "complete" and queue["elapsed_seconds"] < queue["maximum_seconds"] == 600
    )
    compressed = (HERE / "report.json.gz").read_bytes()
    verification = {
        "complete": True,
        "scores": summaries,
        "contrasts": contrasts,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 0,
        "scope": "Source/checkpoint/runtime/document identities and recomputed aggregate losses; bootstrap conditions on this seed/checkpoint and is unadjusted. No repeated GPU forward or reasoning claim.",
        "original_k4_per_document_scores_exact": True,
        "source_zip_sha256": "24cfe9aa69413292f0934e7c08e3730cf608cca9498ffb4d0193a27b7ce58e38",
        "gzip_bytes": len(compressed),
        "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
