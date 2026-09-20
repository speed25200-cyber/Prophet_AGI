#!/usr/bin/env python3
"""Check that the two Qwen recovery initializations differ only in their core mixer.

Checks the common trainable backbone tensors; auxiliary heads are disabled by the
recovery objective and are excluded. This is not function or quality equivalence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from scripts.recover_qwen import recovery_config  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("hybrid", "attention", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve existing pair audit")
    torch.set_num_threads(2)
    hybrid = torch.load(args.hybrid, weights_only=True, map_location="cpu", mmap=True)
    attention = torch.load(args.attention, weights_only=True, map_location="cpu", mmap=True)
    hc = recovery_config(hybrid["config"], 5, 512).to_dict()
    ac = recovery_config(attention["config"], 5, 512).to_dict()
    if hc["recurrent"]["core_pattern"] != ["gdn"] or ac["recurrent"]["core_pattern"] != ["full_attn"]:
        raise ValueError("pair must compare the GDN and full-attention core")
    hc["recurrent"]["core_pattern"] = ac["recurrent"]["core_pattern"]
    if hc != ac:
        raise ValueError("recovery configurations differ beyond their core mixer")
    checked = []
    for name, value in hybrid["model"].items():
        if (name.startswith(("embed.", "sections.prelude.", "sections.coda.", "norm_out.", "lm_head."))
                or (name.startswith("sections.core.") and ".mixer." not in name)):
            if name not in attention["model"] or not torch.equal(value, attention["model"][name]):
                raise ValueError(f"retained tensor differs: {name}")
            checked.append(name)
    if len(checked) != 111:
        raise ValueError("unexpected backbone layout for this pinned Qwen experiment")
    report = {"complete": True, "hybrid_sha256": digest(args.hybrid),
              "attention_sha256": digest(args.attention),
              "recovery_config_differs_only_in_core_mixer": True,
              "identical_retained_tensors": checked,
              "scope": "initialization comparison only; auxiliary heads unused by recovery; no recovery quality result"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(len(checked), "identical retained tensors; only core mixer differs", flush=True)


if __name__ == "__main__":
    main()
