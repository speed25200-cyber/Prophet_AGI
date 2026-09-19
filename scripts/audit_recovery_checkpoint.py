#!/usr/bin/env python3
"""Verify the exact evaluated recovery checkpoint before inference or persistence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tensor_count(tree):
    if isinstance(tree, torch.Tensor):
        if tree.is_floating_point() or tree.is_complex():
            flat = tree.reshape(-1)
            require(all(bool(torch.isfinite(flat[i:i + 1_048_576]).all())
                        for i in range(0, flat.numel(), 1_048_576)), "nonfinite checkpoint tensor")
        return 1
    if isinstance(tree, dict):
        return sum(finite_tensor_count(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(finite_tensor_count(value) for value in tree)
    if isinstance(tree, float):
        require(math.isfinite(tree), "nonfinite checkpoint scalar")
    return 0


def load_evaluated_checkpoint(run: Path, step: int):
    """Return restricted-loaded state and audit; never fall back to another step.

    Read the checksum and tensors from the same open file, so slot rotation cannot
    substitute a different checkpoint between those operations. This checks local
    artifact consistency, not remote durability or independent data provenance.
    """
    require(isinstance(step, int) and not isinstance(step, bool) and step > 0, "invalid step")
    run = Path(run)
    evaluation_bytes = (run / f"evaluation-step-{step:06d}.json").read_bytes()
    report = json.loads(evaluation_bytes)
    manifest = json.loads((run / "recovery.json").read_bytes())
    meta = report["checkpoint"]
    require(type(meta["slot"]) is int and meta["slot"] in (0, 1), "invalid checkpoint slot")
    require(meta["step"] == report["step"] == step, "evaluated checkpoint step differs")
    require(meta in json.loads((run / "manifest.json").read_bytes())["checkpoints"],
            "evaluated checkpoint has rotated or its manifest differs")
    checkpoint = run / f"ckpt_slot{meta['slot']}.pt"
    with checkpoint.open("rb") as stream:
        require(os.fstat(stream.fileno()).st_size == meta["bytes"], "checkpoint size differs")
        require(hashlib.file_digest(stream, "sha256").hexdigest() == meta["sha256"],
                "checkpoint checksum differs")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    contract = json.loads(json.dumps(state["training_contract"]))
    require(contract == manifest["training_contract"], "training contract differs")
    require(json.loads(json.dumps(state["config"])) == manifest["config"], "model config differs")
    require(contract["run_identity"] == report["identity"], "evaluation identity differs")
    require(state["step"] == step, "saved checkpoint step differs")
    expected_tokens = step * contract["batch_size"] * contract["seq_len"] * contract["grad_accum_steps"]
    require(state["tokens_seen"] == report["tokens_seen"] == expected_tokens, "token budget differs")
    require(state["skipped_nonfinite"] == report["skipped_nonfinite"] == state["consecutive_nonfinite"] == 0,
            "checkpoint includes skipped nonfinite updates")
    cuda_rng = state.get("cuda_rng")
    if contract["device_type"] == "cuda":
        require(isinstance(cuda_rng, list) and len(cuda_rng) == 1 and cuda_rng[0].numel() > 0,
                "single-device CUDA RNG is missing")
    counts = {group: finite_tensor_count(state[group]) for group in ("model", "optimizers")}
    require(all(counts.values()), "checkpoint has no model or optimizer tensors")
    evaluation = report["evaluation"]
    identity = report["identity"]
    require(evaluation["seq_len"] == contract["seq_len"], "evaluation sequence length differs")
    require(evaluation["batch_size"] == identity.get("evaluation_batch_size", contract["batch_size"]),
            "evaluation batch size differs")
    require(evaluation["loop_k"] == identity["evaluation_loop_k"], "evaluation depth differs")
    expected_precision = ("bf16 autocast" if contract["device_type"] == "cuda"
                          and contract["dtype"] == "bfloat16" else "fp32")
    require(evaluation["precision"] == expected_precision, "evaluation precision differs")
    for key in ("total_nats", "scored_tokens", "scored_bytes"):
        total = sum(doc[key] for doc in evaluation["documents"])
        require(math.isfinite(total) and total > 0 and math.isclose(
            total, evaluation[key], rel_tol=1e-10, abs_tol=1e-8), "evaluation aggregate differs")
    for doc in evaluation["documents"]:
        require(math.isfinite(doc["total_nats"]) and doc["total_nats"] >= 0, "invalid document loss")
    require(math.isclose(evaluation["nats_per_token"], evaluation["total_nats"] /
                         evaluation["scored_tokens"], rel_tol=1e-10), "evaluation CE differs")
    require(math.isclose(evaluation["bits_per_byte"], evaluation["total_nats"] /
                         evaluation["scored_bytes"] / math.log(2), rel_tol=1e-10), "evaluation BPB differs")
    audit = {
        "complete": True, "protocol": "evaluated-recovery-checkpoint-audit-v1",
        "step": step, "tokens_seen": expected_tokens, "checkpoint": meta,
        "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "training_contract": contract, "config": state["config"],
        "finite_tensor_counts": counts, "skipped_nonfinite": 0,
        "cuda_rng_saved": cuda_rng is not None, "torch": str(torch.__version__),
        "evaluation_nats_per_token": evaluation["nats_per_token"],
        "scope": "exact evaluated local checkpoint; remote durability and data provenance are separate checks",
    }
    return state, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve previous checkpoint evidence")
    torch.set_num_threads(2)
    state, audit = load_evaluated_checkpoint(args.run, args.step)
    del state
    from scripts.recover_qwen import write_report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, audit)
    print("RECOVERY_CHECKPOINT_AUDIT", audit["step"], audit["checkpoint"]["sha256"], flush=True)


if __name__ == "__main__":
    main()
