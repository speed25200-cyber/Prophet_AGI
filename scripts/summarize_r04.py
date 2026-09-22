#!/usr/bin/env python3
"""Summarize one matched R04 seed at a common checkpoint; no adoption decision."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.eval.paired import paired_document_difference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    reports, protocols, contracts, arms = {}, {}, {}, {}
    for arm in ("loop", "plain"):
        root = args.root / f"{arm}-seed{args.seed}"
        protocol = read(root / "protocol.json")
        report = read(root / f"evaluation-step-{args.step:06d}.json")
        audit = read(root / f"checkpoint-audit-step-{args.step:06d}.json")
        initial = read(root / "initial.json")
        assert protocol["variant"] == arm and protocol["seed"] == args.seed
        assert report["run_protocol"] == initial["run_protocol"] == protocol
        assert report["step"] == audit["step"] == args.step
        assert report["train_tokens"] == audit["tokens_seen"] == args.step * protocol["batch_size"] * protocol["seq_len"]
        assert report["checkpoint"] == audit["checkpoint"]
        assert audit["all_model_optimizer_tensors_finite"] and audit["skipped_nonfinite"] == 0
        assert 0 < args.step <= protocol["steps"]
        logs = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
        seconds = [row["seconds"] for row in logs if row["step"] <= args.step]
        arms[arm] = {"initial_ce": initial["nats_per_token"], "final_ce": report["nats_per_token"],
                     "final_bpb": report["bits_per_byte"], "checkpoint": audit["checkpoint"],
                     "logged_step_seconds_median": statistics.median(seconds),
                     "timing_samples": len(seconds), "skipped_nonfinite": audit["skipped_nonfinite"]}
        reports[arm], contracts[arm] = report, audit["training_contract"]
        protocols[arm] = {key: value for key, value in protocol.items() if key not in ("variant", "config")}
    assert protocols["loop"] == protocols["plain"] and contracts["loop"] == contracts["plain"]
    result = {"seed": args.seed, "step": args.step, "tokens_per_arm": reports["loop"]["train_tokens"],
              "planned_steps_per_arm": protocols["loop"]["steps"], "arms": arms,
              "paired": paired_document_difference(reports["loop"], reports["plain"]),
              "scope": "single-seed early-learning pilot; document uncertainty excludes seed uncertainty; no architecture adoption decision"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
