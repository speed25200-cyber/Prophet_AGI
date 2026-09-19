#!/usr/bin/env python3
"""One Colab session of a multi-week run.

    python scripts/colab_session.py --config configs/prophet_mini.json \\
        --work /content/drive/MyDrive/prophet --session-minutes 600 \\
        --tokenizer tokenizer.json --data-root corpus/ --benchmarks benchmarks/ \\
        --tokens 16.1e9 --batch-size 16 --seq-len 4096 --grad-accum 4

A session on a free-tier A100 ends without warning, and a run that spans weeks is a
sequence of such sessions. This script is what each of them runs, in this order:

1. ``scripts/gpu_check.py`` -- the fused kernel must agree with the reference scan on
   this device, or nothing else happens. Its throughput line is the honest estimate of
   how many sessions the run needs.
2. ``scripts/train.py`` with ``--session-minutes`` a little under the session you
   expect, so the run checkpoints and exits on its own terms rather than being killed
   mid-write; the two-slot atomic checkpoint makes even that survivable.
3. On the next session, the same command resumes from the newest intact checkpoint in
   the same data phase, with the same stream.

Everything that must survive the session -- checkpoints, the tokenizer, the corpus
index -- lives under ``--work``, which should be a mounted Drive. Nothing here installs
packages: do that in the notebook cell above, from ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True, help="persistent directory (a mounted Drive)")
    ap.add_argument("--session-minutes", type=float, required=True)
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="everything after '--' goes to scripts/train.py verbatim")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    extra = [a for a in args.train_args if a != "--"]

    if not args.skip_gpu_check:
        check = subprocess.run([sys.executable, str(ROOT / "scripts" / "gpu_check.py"), "--config", args.config])
        if check.returncode != 0:
            print("gpu_check failed: not starting the run", file=sys.stderr)
            return check.returncode

    # Keep a headroom of 5% for the final checkpoint write and the process exit.
    budget = max(1.0, args.session_minutes * 0.95)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--config", args.config,
        "--checkpoint-dir", str(work / "checkpoints"),
        "--session-minutes", f"{budget:.1f}",
        *extra,
    ]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
