"""Compare complete checkpoint states and evaluations for two R04 prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_r04_pilot import sha256, write_json  # noqa: E402


def compare_states(left, right, path="state", counts=None):
    if counts is None:
        counts = {"tensors": 0, "tensor_elements": 0}
    if isinstance(left, torch.Tensor):
        if not torch.isfinite(left).all():
            raise ValueError(f"nonfinite checkpoint tensor: {path}")
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise ValueError(f"checkpoint tensor differs: {path}")
        counts["tensors"] += 1
        counts["tensor_elements"] += left.numel()
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"checkpoint keys differ: {path}")
        for key in left:
            compare_states(left[key], right[key], f"{path}.{key}", counts)
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"checkpoint sequence differs: {path}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            compare_states(a, b, f"{path}[{i}]", counts)
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"checkpoint value differs: {path}")
    return counts


def load_run(folder, expected_step):
    report = json.loads((folder / f"evaluation-step-{expected_step:06d}.json").read_bytes())
    meta = report["checkpoint"]
    manifest = json.loads((folder / "checkpoints/manifest.json").read_bytes())
    if (
        report["step"] != expected_step
        or meta["step"] != expected_step
        or meta not in manifest["checkpoints"]
    ):
        raise ValueError("expected evaluated checkpoint is not published")
    path = folder / f"checkpoints/ckpt_slot{meta['slot']}.pt"
    if path.stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
        raise ValueError("checkpoint bytes differ from publication")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state["step"] != expected_step or state["tokens_seen"] != report["tokens_seen"]:
        raise ValueError("checkpoint counters differ from evaluation")
    if state["skipped_nonfinite"] or report["skipped_nonfinite"]:
        raise ValueError("restart gate does not allow skipped nonfinite updates")
    if (
        state["adaptation_depth_history"] != report["depth_history"]
        or state["loader"]["step"] != report["loader_step"]
    ):
        raise ValueError("checkpoint loader/depth history differs from evaluation")
    if state["training_contract"]["run_identity"] != report["identity"]:
        raise ValueError("checkpoint identity differs from evaluation")
    if not report["identity"]["numerical_policy"]["deterministic_algorithms"]:
        raise ValueError("strict numerical policy required")
    return state, report


def audit(left, right, expected_step):
    a, ar = load_run(left, expected_step)
    b, br = load_run(right, expected_step)
    counts = compare_states(a, b)
    if ar["results"] != br["results"]:
        raise ValueError("per-document evaluation differs")
    return {
        "passed": True,
        "step": expected_step,
        "tokens_seen": a["tokens_seen"],
        "depth_history": a["adaptation_depth_history"],
        "identity": ar["identity"],
        "left_checkpoint": ar["checkpoint"],
        "right_checkpoint": br["checkpoint"],
        "state_comparison": counts,
        "per_document_evaluation_equal": True,
        "scope": "Exact equality of every checkpoint tensor/value, including model, optimizers, RNG, loader, depth history, contract and counters, plus all saved evaluation results. Session/process separation must additionally be established by the launch queue.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.expected_step < 1:
        raise ValueError("positive expected step and fresh report path required")
    report = audit(args.left, args.right, args.expected_step)
    write_json(args.out, report)
    print("R04_FULL_STATE_RESTART_EQUAL", report["state_comparison"], flush=True)


if __name__ == "__main__":
    main()
