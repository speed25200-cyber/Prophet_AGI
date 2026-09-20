#!/usr/bin/env python3
"""Hash an audited initialization independently of its PyTorch archive container."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.train.distillation import state_sha256  # noqa: E402
from scripts.recover_qwen import load_initialization, write_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("initialization", "audit", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve the previous identity audit")
    payload, archive_sha = load_initialization(args.initialization, args.audit)
    config_bytes = json.dumps(payload["config"], sort_keys=True, separators=(",", ":")).encode()
    report = {
        "complete": True,
        "checkpoint_sha256": archive_sha,
        "state_sha256": state_sha256(payload["model"]),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "donor_revision": payload["donor_revision"],
        "donor_weights_sha256": payload["donor_weights_sha256"],
        "state_entries": len(payload["model"]),
        "scope": "all state tensor names, shapes, dtypes and bytes; excludes serialization container metadata",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
