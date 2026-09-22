#!/usr/bin/env python3
"""Copy a loop-core corpus between persistent storage and local disk, verified.

Training reads shards from local disk (mounted Drive stalled under sustained reads in
the R04 sessions, docs/17); the corpus is built once and kept on Drive. This script
moves it either way and verifies the destination against its manifest, so a torn copy
is never mistaken for a corpus. A destination that already verifies is left alone.

    python scripts/stage_corpus.py --source DRIVE/data/loop-core-v1 --dest data/loop-core-v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_loop_core import verify_corpus  # noqa: E402


def stage(source: Path, dest: Path) -> dict:
    began = time.time()
    if dest.exists():
        try:
            provenance = verify_corpus(dest)
            return {"status": "already-staged", "manifest_sha256": provenance["manifest_sha256"]}
        except (OSError, ValueError, KeyError):
            shutil.rmtree(dest)
    expected = verify_corpus(source)
    staging = dest.with_name(dest.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)
    actual = verify_corpus(staging)
    if (
        actual["manifest_sha256"] != expected["manifest_sha256"]
        or actual["artifacts"] != expected["artifacts"]
    ):
        shutil.rmtree(staging)
        raise ValueError("staged copy differs from the source corpus")
    staging.rename(dest)
    return {
        "status": "staged",
        "manifest_sha256": actual["manifest_sha256"],
        "seconds": time.time() - began,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(stage(args.source, args.dest)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
