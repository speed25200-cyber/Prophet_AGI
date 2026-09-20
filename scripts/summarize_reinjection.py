"""Recompute the fixed R04 input-adapter decision from exported document losses."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adapt_r04_reinjection import PLAN_DIR  # noqa: E402
from scripts.summarize_depth_adaptation import (  # noqa: E402
    REFERENCE,
    read_report,
    require,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-sum", type=Path, required=True)
    parser.add_argument("--learned-mix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "use a fresh output report")
    plan = PLAN_DIR / "protocol.json"
    result = summarize(
        read_report(args.fixed_sum),
        read_report(args.learned_mix),
        read_report(plan),
        hashlib.sha256(plan.read_bytes()).hexdigest(),
        read_report(REFERENCE),
        experiment="reinjection",
    )
    result["input_files"] = {
        arm: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for arm, path in (("fixed_sum", args.fixed_sum), ("learned_mix", args.learned_mix))
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {"seed0_screen_passed": result["seed0_screen_passed"], "checks": result["checks"]}
        )
    )


if __name__ == "__main__":
    main()
