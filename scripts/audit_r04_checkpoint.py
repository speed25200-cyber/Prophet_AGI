#!/usr/bin/env python3
"""Audit the exact checkpoint referenced by a completed R04 validation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def audit_checkpoint(run: Path, step: int) -> dict:
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    protocol = read(run / "protocol.json")
    evaluation = read(run / f"evaluation-step-{step:06d}.json")
    meta = evaluation["checkpoint"]
    if evaluation["run_protocol"] != protocol or evaluation["step"] != step:
        raise ValueError("validation protocol or step differs")
    if meta["slot"] not in (0, 1) or meta["step"] != step:
        raise ValueError("invalid published checkpoint identity")
    manifest = read(run / "checkpoints/manifest.json")
    if meta not in manifest["checkpoints"]:
        raise ValueError("published checkpoint has rotated; stop training before audit")
    with (run / f"checkpoints/ckpt_slot{meta['slot']}.pt").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != meta["sha256"] or os.fstat(stream.fileno()).st_size != meta["bytes"]:
            raise ValueError("checkpoint checksum or size differs")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if state["step"] != step or json.loads(json.dumps(state["config"])) != protocol["config"]:
        raise ValueError("checkpoint step or model configuration differs")
    expected_tokens = step * protocol["batch_size"] * protocol["seq_len"]
    if state["tokens_seen"] != expected_tokens or evaluation["train_tokens"] != expected_tokens:
        raise ValueError("checkpoint or evaluation token count differs")
    tensor_count = 0
    nonfinite = []

    def inspect(value, path):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
                value
            ).all():
                nonfinite.append(path)
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(item, f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                inspect(item, f"{path}/{index}")

    inspect(state["model"], "model")
    inspect(state["optimizers"], "optimizers")
    if not tensor_count:
        raise ValueError("checkpoint contains no model or optimizer tensors")
    return {
        "variant": protocol["variant"],
        "seed": protocol["seed"],
        "checkpoint": meta,
        "step": step,
        "tokens_seen": state["tokens_seen"],
        "trainer_state_version": state["trainer_state_version"],
        "skipped_nonfinite": state["skipped_nonfinite"],
        "consecutive_nonfinite": state["consecutive_nonfinite"],
        "all_model_optimizer_tensors_finite": not nonfinite,
        "inspected_tensors": tensor_count,
        "nonfinite_tensor_paths": nonfinite,
        "training_contract": state["training_contract"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    output = args.run / f"checkpoint-audit-step-{args.step:06d}.json"
    if output.exists():
        raise FileExistsError("preserve the existing audit")
    result = audit_checkpoint(args.run, args.step)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)
    if not result["all_model_optimizer_tensors_finite"]:
        raise SystemExit("non-finite checkpoint: audit saved; do not accept this run")


if __name__ == "__main__":
    main()
