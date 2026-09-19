#!/usr/bin/env python3
"""One bounded session of a matched R04 arm, with frozen provenance and full validation.

Repeat the same command to resume. Put --out on persistent storage before a long run.
The default 128-step session is an initial stability check inside the 4096-step run,
not a separately annealed smoke model. No shared/unshared quality claim is automatic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.data.tokenizer import ProphetTokenizer  # noqa: E402
from prophet.eval.text import evaluate_documents  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from scripts.gpu_check import kernel_agreement  # noqa: E402

PILOT_TOKENIZER_SHA256 = "7d8d36adb2b2bf9dac7a6060294641ea16c59ce20506e2294c556c7abd99537b"


def tokenizer_semantic_hash(path: Path) -> str:
    canonical = json.dumps(json.loads(path.read_text(encoding="utf-8")),
                           sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_pilot(root: Path) -> dict:
    """Bind the finite recipe to the exact audited corpus, not just its name."""
    expected = json.loads((ROOT / "docs/experiments/2026-09-19-pilot-corpus.json").read_text())
    actual = json.loads((root / "manifest.json").read_text())
    if not actual.get("complete") or actual.get("source_revision") != expected["source_revision"]:
        raise ValueError("pilot source revision or completeness differs")
    for relative, metadata in expected["artifacts"].items():
        if sha256(root / relative) != metadata["sha256"]:
            raise ValueError(f"pilot shard differs: {relative}")
    tokenizer_path = root / "tokenizer.json"
    metadata = json.loads((root / "tokenizer.json.metadata.json").read_text())
    if sha256(tokenizer_path) != metadata["tokenizer_sha256"]:
        raise ValueError("tokenizer file does not match its training metadata")
    train_hash = expected["artifacts"]["train/fineweb-edu/part-00000.jsonl"]["sha256"]
    if [s["fingerprint"]["files"] for s in metadata["sources"]] != [[train_hash]]:
        raise ValueError("tokenizer was not trained exclusively on this training split")
    # Serialization newlines differ across platforms; the semantic vocabulary is fixed.
    semantic_hash = tokenizer_semantic_hash(tokenizer_path)
    if semantic_hash != PILOT_TOKENIZER_SHA256:
        raise ValueError("tokenizer vocabulary differs from the audited pilot")
    return {"source_revision": expected["source_revision"], "artifacts": expected["artifacts"],
            "tokenizer_semantic_sha256": semantic_hash}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze_protocol(path: Path, protocol: dict) -> None:
    canonical = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text()) != canonical:
            raise ValueError("run protocol changed; use the original revision/settings to resume")
    else:
        write_json(path, canonical)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/fineweb-pilot-v1"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", choices=["loop", "plain"], required=True)
    ap.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--max-session-steps", type=int, default=128)
    ap.add_argument("--session-minutes", type=float, default=45)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.max_session_steps < 1 or not math.isfinite(args.session_minutes) or args.session_minutes <= 0:
        ap.error("session limits must be positive")
    provenance = verify_pilot(args.corpus)
    cfg = ProphetConfig.from_json(ROOT / f"configs/prophet_r04_{args.variant}.json")
    cfg.validate()
    tokenizer = ProphetTokenizer.load(args.corpus / "tokenizer.json")
    if tokenizer.vocab_size != cfg.frontend.vocab_size or tokenizer.n_tokens != 32768:
        raise ValueError("R04 requires the complete 32768-entry pilot vocabulary")
    recipe = json.loads((ROOT / "docs/experiments/2026-09-19-pilot-token-counts.json").read_text())
    planned_tokens = 4096 * 8 * 2048
    if planned_tokens > 4 * recipe["splits"]["train"]["tokens"]:
        raise ValueError("planned token budget exceeds the four-epoch cap")
    protocol = {
        "format_version": 1, "variant": args.variant, "seed": args.seed, "config": cfg.to_dict(),
        "data": provenance, "steps": 4096, "batch_size": 8, "seq_len": 2048,
        "loss_chunk_tokens": 512, "muon_lr": 0.01, "adamw_lr": 0.0003,
        "evaluation": "all held-out documents; context 2048; one-token overlap; per-document CE/BPB",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if not torch.cuda.is_available() or not HAS_FLA:
        raise RuntimeError("validated CUDA/FLA environment required; run tests/test_gpu.py first")
    if os.environ.get("TRITON_F32_DEFAULT") != "tf32x3":
        raise RuntimeError("R04 requires TRITON_F32_DEFAULT=tf32x3 before Python starts")
    protocol["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "fla": version("fla-core"), "triton": version("triton"),
                           "device": torch.cuda.get_device_name(0), "triton_f32_default": "tf32x3"}
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "protocol.json").exists() and any(args.out.iterdir()):
        raise ValueError("output contains artifacts without a protocol; use a fresh directory")
    freeze_protocol(args.out / "protocol.json", protocol)
    error = kernel_agreement(cfg, seed=args.seed)
    write_json(args.out / "kernel-check.json", {"max_abs_error": error, "tolerance": 0.002})
    if not math.isfinite(error) or error > 0.002:
        raise RuntimeError("fused/reference gate failed")
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    model = ProphetModel(cfg)
    source = LocalTextSource.from_root(args.corpus / "train", "fineweb-edu", 1.0)
    loader = StreamingLoader([TokenisedSource(source, tokenizer)], seq_len=2048, batch_size=8,
                             seed=args.seed)

    def log(metrics):
        print(metrics.format(), flush=True)
        with (args.out / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(metrics)) + "\n")

    trainer = Trainer(model, loader, TrainConfig(
        total_steps=4096, seq_len=2048, batch_size=8, peak_lr_muon=0.01,
        peak_lr_adamw=0.0003, loss_chunk_tokens=512, seed=args.seed, device="cuda",
        checkpoint_dir=str(args.out / "checkpoints"), checkpoint_every=512, log_every=16,
        max_wall_seconds=args.session_minutes * 60,
    ), model_config=cfg, tokenizer=tokenizer, on_log=log)
    resumed = trainer.maybe_resume()
    if not resumed and (args.out / "train.jsonl").exists():
        raise RuntimeError("training history exists but no checkpoint survived; preserve this attempt and use a new output directory")
    validation = LocalTextSource.from_root(args.corpus / "validation", "fineweb-edu", 1.0)

    def evaluate(label, checkpoint=None):
        result = evaluate_documents(model, validation.open(), tokenizer, device="cuda",
                                    seq_len=2048, batch_size=8)
        result.update({"step": trainer.step, "train_tokens": trainer.tokens_seen,
                       "checkpoint": checkpoint, "run_protocol": protocol})
        write_json(args.out / f"{label}.json", result)
        print(label, {k: result[k] for k in ("step", "nats_per_token", "bits_per_byte", "scored_tokens")}, flush=True)

    if trainer.step == 0:
        evaluate("initial")
    old_handlers = {sig: signal.signal(sig, lambda *_: setattr(trainer, "stop_requested", True))
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        trainer.train(max_steps=args.max_session_steps)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    # Exceptions can interrupt a batch after its loader/RNG advanced. Keep the last
    # completed checkpoint intact; only a normal return is safe to publish here.
    if trainer.step:
        meta = trainer.ckpt.save(trainer.state_dict(), trainer.step)
        print("CHECKPOINT", meta.to_dict(), flush=True)
        evaluate(f"evaluation-step-{trainer.step:06d}", meta.to_dict())
    print("RUN_COMPLETE" if trainer.step == 4096 else "SESSION_COMPLETE_RESUME_REQUIRED", flush=True)


if __name__ == "__main__":
    main()
