#!/usr/bin/env python3
"""Resume-safe CE or frozen-donor KL recovery from an audited Qwen initialization.

This is an experimental training path, not an adopted architecture. Input corpora must
be prepared and audited separately; hashing them does not establish decontamination.
The R04 experiment uses its frozen runner and is unaffected by this entry point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.convert.donors import get_donor  # noqa: E402
from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.donor_tokenizer import DonorByteTokenizer  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.eval.text import evaluate_documents  # noqa: E402
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.distillation import DistillationObjective, DistillationSettings  # noqa: E402
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from scripts.audit_qwen_blocks import REVISION, WEIGHTS_SHA256  # noqa: E402
from scripts.rehearse_qwen_conversion import digest  # noqa: E402
from scripts.verify_donors import compare  # noqa: E402

TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def load_source(source):
    """Verify the pinned local donor and construct the shared text/byte policy."""
    if (digest(source / "model.safetensors") != WEIGHTS_SHA256
            or digest(source / "tokenizer.json") != TOKENIZER_SHA256):
        raise ValueError("source differs from the audited pinned artifacts")
    source_config = json.loads((source / "config.json").read_bytes())
    differences = compare(get_donor("qwen3-0.6b"), source_config)
    if differences:
        raise ValueError(f"donor configuration differs from the audited source: {differences}")
    tokenizer = DonorByteTokenizer(source / "tokenizer.json", eos_id=151645,
                                   pad_id=151643, vocab_size=151936)
    return source_config, tokenizer


def load_initialization(initialization, audit_path):
    """Restricted-load exactly the artifact recorded by a successful conversion audit."""
    audit = json.loads(audit_path.read_bytes())
    initialization_sha = digest(initialization)
    if (not audit.get("complete") or not audit.get("all_parameters_finite")
            or not audit.get("serialization_exact")
            or initialization_sha != audit["checkpoint_sha256"]
            or audit["donor_revision"] != REVISION
            or audit["donor_weights_sha256"] != WEIGHTS_SHA256):
        raise ValueError("initialization differs from the audited pinned artifacts")
    payload = torch.load(initialization, map_location="cpu", weights_only=True, mmap=True)
    if (json.loads(json.dumps(payload["config"])) != audit["config"]
            or payload["donor_revision"] != REVISION
            or payload["donor_weights_sha256"] != WEIGHTS_SHA256):
        raise ValueError("initialization metadata mismatch")
    return payload, initialization_sha


def recovery_config(initial_config, loop_k, seq_len):
    cfg = ProphetConfig.from_dict(initial_config)
    if (cfg.heads.action_head or cfg.recurrent.token_depth or cfg.recurrent.halting != "none"):
        raise ValueError("recovery requires plain-text configuration without action/depth policies")
    if not cfg.recurrent.enabled or loop_k < 1 or not 2 <= seq_len <= cfg.max_seq_len:
        raise ValueError("invalid recovery depth or sequence length")
    cfg.recurrent.train_loop_min = cfg.recurrent.train_loop_max = loop_k
    cfg.recurrent.default_loop_k = loop_k
    cfg.recurrent.truncated_backprop_steps = loop_k
    # Heads stay in the initialized state_dict; their losses are disabled explicitly.
    cfg.heads.mtp_loss_weight = 0.0
    cfg.heads.confidence_loss_weight = 0.0
    cfg.validate()
    return cfg


def read_documents(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                text = json.loads(line)["text"]
                if not isinstance(text, str):
                    raise ValueError("corpus text must be a string")
                yield text


def write_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "initialization", "audit", "train", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--objective", choices=("ce", "kl"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--loop-k", type=int, default=5)
    parser.add_argument("--muon-lr", type=float, required=True)
    parser.add_argument("--adamw-lr", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--max-session-steps", type=int)
    parser.add_argument("--session-minutes", type=float, default=45)
    args = parser.parse_args()
    positive = (args.steps, args.batch_size, args.grad_accum, args.chunk_tokens,
                args.checkpoint_every, args.session_minutes, args.muon_lr, args.adamw_lr)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("steps, batch sizes, intervals, rates and session length must be positive")
    if args.max_session_steps is not None and args.max_session_steps < 1:
        parser.error("--max-session-steps must be positive")
    if args.objective == "ce" and (args.alpha != 0.5 or args.temperature != 1.0):
        parser.error("--alpha and --temperature only apply to the KL arm")
    settings = DistillationSettings(args.alpha, args.temperature, args.chunk_tokens)
    import transformers
    from transformers import AutoModelForCausalLM

    source_config, tokenizer = load_source(args.source)
    payload, initialization_sha = load_initialization(args.initialization, args.audit)
    cfg = recovery_config(payload["config"], args.loop_k, args.seq_len)
    if cfg.frontend.vocab_size != tokenizer.vocab_size:
        raise ValueError("student vocabulary differs from the pinned donor")
    device = torch.device(args.device)
    if device.type == "cuda" and not HAS_FLA:
        raise RuntimeError("CUDA recovery requires the pinned FLA kernel and a separate GPU gate")
    identity = {"protocol": "qwen-recovery-v1", "initialization_sha256": initialization_sha,
                "donor_revision": REVISION, "donor_weights_sha256": WEIGHTS_SHA256,
                "donor_config": source_config, "tokenizer": tokenizer.fingerprint(),
                "train_sha256": digest(args.train), "validation_sha256": digest(args.validation),
                "objective": args.objective, "torch": str(torch.__version__),
                "transformers": transformers.__version__,
                "teacher_dtype": "bfloat16" if device.type == "cuda" else "float32",
                "teacher_attention": "sdpa", "evaluation_loop_k": args.loop_k,
                "cuda": torch.version.cuda,
                "fla": version("fla-core") if device.type == "cuda" else None,
                "triton": version("triton") if device.type == "cuda" else None,
                "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT")}
    torch.manual_seed(args.seed)
    model = ProphetModel(cfg)  # FP32 master weights
    model.load_state_dict(payload["model"], strict=True)
    del payload
    source = TokenisedSource(LocalTextSource("recovery", 1.0, [args.train]), tokenizer, max_epochs=4)
    loader = StreamingLoader([source], seq_len=args.seq_len, batch_size=args.batch_size,
                             seed=args.seed, separator=None)
    objective = None
    if args.objective == "kl":
        teacher = AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        objective = DistillationObjective(teacher, identity=identity, settings=settings)
    # Teacher construction must not move the student's stochastic recurrence stream.
    torch.manual_seed(args.seed)
    train_cfg = TrainConfig(
        total_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum, peak_lr_muon=args.muon_lr,
        peak_lr_adamw=args.adamw_lr, loss_chunk_tokens=args.chunk_tokens,
        mtp_weight=0.0, confidence_weight=0.0, z_loss_weight=0.0, ponder_weight=0.0,
        checkpoint_dir=str(args.out), checkpoint_every=args.checkpoint_every,
        log_every=1, max_wall_seconds=args.session_minutes * 60, seed=args.seed,
        device=args.device,
    )
    metrics_path = args.out / "training.jsonl"

    def log(metrics):
        from dataclasses import asdict
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(metrics)) + "\n")
        print(metrics.format(), flush=True)

    trainer = Trainer(model, loader, train_cfg, model_config=cfg, tokenizer=tokenizer,
                      distillation=objective, run_identity=identity, on_log=log)
    manifest = json.loads(json.dumps({"training_contract": trainer.training_contract(),
                                      "config": cfg.to_dict()}))
    manifest_path = args.out / "recovery.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_bytes()) != manifest:
            raise ValueError("output directory belongs to another recovery experiment")
        if not trainer.maybe_resume():
            raise ValueError("existing run has no valid checkpoint; use a new output directory")
        completed_report = args.out / f"evaluation-step-{trainer.step:06d}.json"
        if trainer.step >= args.steps and completed_report.exists():
            previous = json.loads(completed_report.read_bytes())
            if previous["step"] != trainer.step or previous["identity"] != identity:
                raise ValueError("completed evaluation belongs to another recovery experiment")
            print("RECOVERY_ALREADY_COMPLETE", trainer.step, flush=True)
            return
    else:
        if args.out.exists() and any(args.out.iterdir()):
            raise FileExistsError("new recovery requires an empty output directory")
        args.out.mkdir(parents=True, exist_ok=True)
        write_report(manifest_path, manifest)
        # Step zero is recoverable even if preempted before the first periodic save.
        trainer.ckpt.save(trainer.state_dict(), 0)

    def stop(signum, frame):
        trainer.stop_requested = True
        print(f"signal {signum}: finishing step and checkpointing", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trainer.train(max_steps=args.max_session_steps)
    checkpoint = trainer.ckpt.save(trainer.state_dict(), trainer.step)
    # Free the teacher before evaluation; it is reloaded and rehashed on resume.
    if objective is not None:
        objective.teacher.to("cpu")
    evaluation = evaluate_documents(model, read_documents(args.validation), tokenizer,
                                    seq_len=args.seq_len, batch_size=args.batch_size,
                                    device=args.device, loss_chunk_tokens=args.chunk_tokens,
                                    loop_k=args.loop_k)
    report = {"step": trainer.step, "tokens_seen": trainer.tokens_seen,
              "checkpoint": checkpoint.to_dict(), "evaluation": evaluation,
              "skipped_nonfinite": trainer.skipped_nonfinite, "identity": identity,
              "teacher_tokens_seen": trainer.tokens_seen if objective is not None else 0,
              "timing_protocol": "logged step seconds include teacher forward when KL is enabled",
              "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                             if device.type == "cuda" else None),
              "scope": "experimental recovery; data cleanliness and architecture adoption are separate gates"}
    destination = args.out / f"evaluation-step-{trainer.step:06d}.json"
    if destination.exists():
        raise FileExistsError("preserve the previous evaluation; this session did not advance")
    write_report(destination, report)
    print("RECOVERY_SESSION_COMPLETE", trainer.step, evaluation["nats_per_token"], flush=True)


if __name__ == "__main__":
    main()
