#!/usr/bin/env python3
"""Copy an audited R04 checkpoint into a new, independently verified directory.

Run after training stops. Existing snapshots are never overwritten. The completion
marker proves filesystem verification; flush mounted Drive before releasing Colab.
Keep this script beside audit_r04_checkpoint.py when using it outside the repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_r04_checkpoint import audit_checkpoint  # noqa: E402


def snapshot(run: Path, step: int, destination: Path) -> dict:
    run, destination = run.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError("preserve the existing snapshot; choose a new destination")
    if destination.is_relative_to(run) or run.is_relative_to(destination):
        raise ValueError("snapshot and source must be separate directories")
    audit = audit_checkpoint(run, step)
    if not audit["all_model_optimizer_tensors_finite"] or audit["skipped_nonfinite"]:
        raise ValueError("snapshot requires an intact finite run with no skipped steps")
    meta = audit["checkpoint"]
    destination.mkdir(parents=True)
    # A failed copy leaves only a new partial directory; the previous run survives.
    for path in sorted(run.iterdir()):
        if (
            path.name != "SNAPSHOT_COMPLETE.json"
            and path.is_file()
            and path.suffix in (".json", ".jsonl", ".log", ".txt")
        ):
            shutil.copyfile(path, destination / path.name)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir()
    filename = f"ckpt_slot{meta['slot']}.pt"
    shutil.copyfile(run / "checkpoints" / filename, checkpoint_dir / filename)
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps({"checkpoints": [meta]}, indent=2) + "\n", encoding="utf-8"
    )
    verified = audit_checkpoint(destination, step)
    if verified != audit:
        raise ValueError("copied snapshot differs from its source audit")
    (destination / f"checkpoint-audit-step-{step:06d}.json").write_text(
        json.dumps(verified, indent=2) + "\n", encoding="utf-8"
    )
    marker = {
        "complete": True,
        "checkpoint": meta,
        "scope": "verified filesystem copy; flush remote storage before releasing the VM",
    }
    temporary = destination / "SNAPSHOT_COMPLETE.json.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / "SNAPSHOT_COMPLETE.json")
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run, args.step, args.destination), indent=2), flush=True)


if __name__ == "__main__":
    main()
