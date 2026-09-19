#!/usr/bin/env python3
"""Evaluate frozen R04 weights across depths using their original model source.

This driver may live outside --model-repo, so adding evaluation code never changes
the pinned training checkout. Run only after the training process has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-repo", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--expected-step", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 1, 2, 8])
    args = ap.parse_args()
    if args.expected_step < 1 or not args.depths or any(k < 1 for k in args.depths):
        ap.error("step and depths must be positive")
    if len(set(args.depths)) != len(args.depths) or args.depths[0] != 4:
        ap.error("depths must be unique and start with the trained depth 4")
    if args.out.exists():
        raise FileExistsError("preserve the previous depth experiment; choose a new output")
    protocol = json.loads((args.run / "protocol.json").read_text(encoding="utf-8"))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.model_repo, text=True
    ).strip()
    if revision != protocol["revision"] or protocol["variant"] != "loop":
        raise ValueError("requires the shared model's exact training source")
    subprocess.run(["git", "diff", "HEAD", "--exit-code"], cwd=args.model_repo, check=True)
    if os.environ.get("TRITON_F32_DEFAULT") != "tf32x3":
        raise ValueError("set TRITON_F32_DEFAULT=tf32x3 before starting Python")
    sys.path.insert(0, str(args.model_repo.resolve()))
    import torch

    from prophet.config import ProphetConfig
    from prophet.data.corpus import LocalTextSource
    from prophet.data.tokenizer import ProphetTokenizer
    from prophet.eval.text import evaluate_documents
    from prophet.modeling.model import ProphetModel
    from prophet.train.checkpoint import CheckpointManager, CheckpointMeta
    from scripts.run_r04_pilot import verify_pilot

    if verify_pilot(args.corpus) != protocol["data"]:
        raise ValueError("evaluation corpus differs from the frozen training run")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != protocol["runtime"]["device"]
    ):
        raise RuntimeError("the recorded CUDA device is required")
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fla": version("fla-core"),
        "triton": version("triton"),
        "device": torch.cuda.get_device_name(0),
        "triton_f32_default": "tf32x3",
    }
    if runtime != protocol["runtime"]:
        raise RuntimeError("evaluation runtime differs from the recorded training environment")
    saved = json.loads((args.run / f"evaluation-step-{args.expected_step:06d}.json").read_text())
    checkpoint = CheckpointMeta.from_dict(saved["checkpoint"])
    manager = CheckpointManager(args.run / "checkpoints")
    if checkpoint not in manager.read_manifest() or checkpoint.slot not in (0, 1):
        raise ValueError("published checkpoint has rotated; stop training before evaluation")
    # Select the exact weights referenced by validation, even when a periodic and
    # a session-end save share a step number in the original training revision.
    checkpoint_path = manager.slot_path(checkpoint.slot)
    with checkpoint_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != checkpoint.sha256 or os.fstat(stream.fileno()).st_size != checkpoint.bytes:
            raise ValueError("published checkpoint checksum or size differs")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if checkpoint.step != args.expected_step or state["step"] != args.expected_step:
        raise ValueError("checkpoint rotated or its step differs; stop training before evaluation")
    if json.loads(json.dumps(state["config"])) != protocol["config"]:
        raise ValueError("checkpoint model configuration differs")
    if saved["checkpoint"] != checkpoint.to_dict() or saved["run_protocol"] != protocol:
        raise ValueError("saved validation belongs to different weights or protocol")
    torch.backends.cuda.matmul.allow_tf32 = state["training_contract"]["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = state["training_contract"]["allow_tf32"]
    cfg = ProphetConfig.from_dict(state["config"])
    model = ProphetModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.cuda().eval()
    tokenizer = ProphetTokenizer.load(args.corpus / "tokenizer.json")
    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)
    result = {
        "complete": False,
        "checkpoint": checkpoint.to_dict(),
        "run_protocol": protocol,
        "evaluation_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_depths": args.depths,
        "results": [],
        "memory_scope": "standalone model and evaluation workspace; excludes training optimizer and gradients",
        "scope": "fixed-k-trained model sensitivity; language loss only; no reasoning claim",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.out.with_suffix(args.out.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.out)

    save()
    for depth in args.depths:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        measured = evaluate_documents(
            model,
            validation.open(),
            tokenizer,
            seq_len=2048,
            batch_size=8,
            device="cuda",
            loss_chunk_tokens=512,
            loop_k=depth,
        )
        torch.cuda.synchronize()
        measured["evaluation_seconds_including_io_compile"] = time.perf_counter() - began
        measured["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if depth == 4:
            if not math.isclose(
                measured["nats_per_token"], saved["nats_per_token"], rel_tol=0, abs_tol=1e-5
            ):
                raise ValueError("trained-depth CE does not reproduce the saved evaluation")
            if [
                (d["sha256"], d["scored_tokens"], d["scored_bytes"]) for d in measured["documents"]
            ] != [(d["sha256"], d["scored_tokens"], d["scored_bytes"]) for d in saved["documents"]]:
                raise ValueError("evaluation targets differ from saved validation")
        result["results"].append(measured)
        save()
        print(
            "DEPTH_RESULT",
            depth,
            {key: value for key, value in measured.items() if key != "documents"},
            flush=True,
        )
    result["complete"] = True
    save()
    print("DEPTH_SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
