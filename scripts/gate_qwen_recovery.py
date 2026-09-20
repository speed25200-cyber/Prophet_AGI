#!/usr/bin/env python3
"""Gate actual donor-derived weights on CUDA before allocating recovery compute.

Checks full-model training logits/gradients against the sequential GDN reference,
then measures real Trainer updates, including the frozen teacher for the KL arm.
Diagnostic updates are discarded; this is neither a recovery run nor a cache gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.data.corpus import LocalTextSource, TokenisedSource  # noqa: E402
from prophet.data.streaming import StreamingLoader  # noqa: E402
from prophet.modeling.layers import HAS_FLA, GatedDeltaNet  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from prophet.train.distillation import (  # noqa: E402
    DistillationObjective,
    DistillationSettings,
    state_sha256,
)
from prophet.train.loop import TrainConfig, Trainer  # noqa: E402
from prophet.train.loss import compute_loss  # noqa: E402
from scripts import recover_qwen as recovery  # noqa: E402


def tensor_agreement(reference, actual, *, tolerance, max_absolute=None):
    """Finite full-tensor relative L2, plus an optional independent absolute bound."""
    if reference.shape != actual.shape:
        return {"passed": False, "reason": "shape mismatch"}
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        return {"passed": False, "reason": "nonfinite tensor"}
    difference = (reference.float() - actual.float()).detach()
    relative = float(difference.norm() / reference.float().norm().clamp_min(1e-8))
    maximum = float(difference.abs().max()) if difference.numel() else 0.0
    return {"passed": relative <= tolerance and (max_absolute is None or maximum <= max_absolute),
            "relative_l2": relative, "max_absolute": maximum,
            "relative_l2_tolerance": tolerance, "absolute_tolerance": max_absolute}


def diagnostic_gdn_fp32(model):
    """Override only this disposable instance; do not change production/config defaults.

    The normal kernel already accumulates its recurrence in FP32. This probe also
    keeps the GDN projections, convolution and output projection out of autocast,
    isolating whether rounding at the mixer's boundaries explains the full-model
    mixed-precision disagreement. Other modules retain their original policy.
    """
    layers = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    if not layers:
        raise ValueError("GDN FP32 diagnostic requires GDN mixers")
    for layer in layers:
        def forward(x, *, state=None, _original=layer.forward):
            with torch.autocast(device_type=x.device.type, enabled=False):
                return _original(x.float(), state=state)
        layer.forward = forward
    return len(layers)


def training_kernel_agreement(model, ids, *, loop_k, autocast, chunk_tokens=128):
    """Both passes use the same training RNG and full BPTT with checkpointing.

    CPU is useful for testing this harness but cannot establish a fused CUDA gate.
    Auxiliary heads are excluded from this main-CE numerical comparison. The real
    Trainer timing below retains the recovery runner's complete execution policy.
    """
    layers = [(m, m.allow_fused, m.chunk_size) for m in model.modules()
              if isinstance(m, GatedDeltaNet)]
    if not layers:
        return {"passed": True, "applicable": False, "reason": "no GDN mixers"}
    old_mode, old_checkpointing = model.training, model.gradient_checkpointing
    old_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    device = ids.device
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    model.train()
    model.gradient_checkpointing = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.random.fork_rng(devices=devices):
            def run(fused):
                torch.manual_seed(0)
                model.zero_grad(set_to_none=True)
                for layer, _, _ in layers:
                    layer.allow_fused, layer.chunk_size = fused, None
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                    output = model(ids, loop_k=loop_k, return_mtp=False)
                loss = compute_loss(output, ids, mtp_weight=0.0, z_loss_weight=0.0,
                                    loss_chunk_tokens=chunk_tokens).lm
                loss.backward()
                gradients = {name: None if p.grad is None else p.grad.detach().clone()
                             for name, p in model.named_parameters()}
                return output.logits.detach(), float(loss.detach()), gradients

            ref_logits, ref_loss, ref_gradients = run(False)
            logits, loss, gradients = run(True)
            tolerance = 0.03 if autocast else 0.002
            tensors = {"logits": tensor_agreement(ref_logits, logits, tolerance=tolerance,
                                                  max_absolute=None if autocast else 0.002)}
            unused = []
            for name, expected in ref_gradients.items():
                actual = gradients[name]
                if expected is None or actual is None:
                    allowed = (expected is None and actual is None
                               and name.startswith(("mtp_heads.", "confidence_head.")))
                    tensors[name] = {"passed": allowed, "both_unused": expected is None and actual is None}
                    unused.append(name)
                else:
                    tensors[name] = tensor_agreement(expected, actual, tolerance=tolerance)
            return {"passed": math.isfinite(ref_loss) and math.isfinite(loss)
                    and all(row["passed"] for row in tensors.values()),
                    "applicable": True, "autocast_bf16": autocast,
                    "reference_loss": ref_loss, "fused_loss": loss, "tensors": tensors,
                    "unused_parameters": unused, "gdn_layers": len(layers),
                    "sequence_length": ids.shape[1], "batch_size": ids.shape[0],
                    "loop_k": loop_k, "activation_checkpointing": True,
                    "reference_scan": "sequential", "training_state_seed": 0,
                    "fused_cuda_exercised": device.type == "cuda" and HAS_FLA}
    finally:
        model.zero_grad(set_to_none=True)
        model.train(old_mode)
        model.gradient_checkpointing = old_checkpointing
        for layer, fused, chunk in layers:
            layer.allow_fused, layer.chunk_size = fused, chunk
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32


def finite_state(tree):
    """Check model/optimizer tensors without allocating a whole-parameter mask."""
    if isinstance(tree, torch.Tensor):
        flat = tree.detach().reshape(-1)
        return all(bool(torch.isfinite(flat[start:start + 1_048_576]).all())
                   for start in range(0, flat.numel(), 1_048_576))
    if isinstance(tree, dict):
        return all(finite_state(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(finite_state(value) for value in tree)
    if isinstance(tree, float):
        return math.isfinite(tree)
    return True


def measure_updates(trainer, *, warmup_steps=2, measured_steps=3):
    """Measure synchronized real steps; checkpoints disabled by the caller."""
    if trainer.cfg.checkpoint_every or trainer.step != 0 or warmup_steps < 1 or measured_steps < 1:
        raise ValueError("measurement requires a fresh trainer, no checkpoints and positive intervals")
    if trainer.cfg.total_steps < warmup_steps + measured_steps:
        raise ValueError("trainer schedule shorter than the measurement")
    device = trainer.device
    teacher_before = trainer.distillation.fingerprint()["state_sha256"] if trainer.distillation else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seconds = []
    for index in range(warmup_steps + measured_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        trainer.train(max_steps=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        if trainer.step != index + 1:
            raise RuntimeError("measurement did not complete the requested step")
        if index >= warmup_steps:
            seconds.append(duration)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    teacher_unchanged = None
    if trainer.distillation:
        teacher = trainer.distillation.teacher
        teacher_unchanged = (state_sha256(teacher) == teacher_before and not teacher.training
                             and all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
    all_finite = finite_state(trainer.model.state_dict()) and all(
        finite_state(opt.state_dict()) for opt in trainer.optimizers)
    return {"passed": all_finite and trainer.skipped_nonfinite == 0 and teacher_unchanged is not False,
            "model_optimizer_tensors_finite": all_finite,
            "skipped_nonfinite": trainer.skipped_nonfinite,
            "teacher_unchanged_and_no_gradients": teacher_unchanged,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps,
            "synchronized_seconds_per_step": seconds, "mean_seconds_per_step": sum(seconds) / len(seconds),
            "peak_cuda_allocated_bytes": peak, "tokens_seen": trainer.tokens_seen,
            "teacher_tokens_seen": trainer.tokens_seen if trainer.distillation else 0,
            "metrics": [asdict(item) for item in trainer.history],
            "training_contract": trainer.training_contract(),
            "scope": "real Trainer updates, data loading and teacher included; checkpoints/evaluation excluded; weights discarded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "initialization", "audit", "train", "data-audit", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--objective", choices=("ce", "kl"), required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--probe-seq-len", type=int, default=67,
                        help="reference/fused gradient comparison length; timing still uses --seq-len")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--loop-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--muon-lr", type=float, required=True)
    parser.add_argument("--adamw-lr", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gdn-fp32-diagnostic", action="store_true",
                        help="disable autocast inside GDN on this discarded gate instance only")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16",
                        help="precision policy being gated; float32 does not certify BF16")
    args = parser.parse_args()
    if args.gdn_fp32_diagnostic and args.precision == "float32":
        parser.error("the GDN-only diagnostic requires the outer BF16 policy")
    if args.out.exists():
        raise FileExistsError("preserve previous gate evidence")
    if args.seq_len < 67 or min(args.batch_size, args.grad_accum, args.loop_k, args.chunk_tokens) < 1:
        parser.error("sequence length must be >=67 and other sizes positive")
    if not 67 <= args.probe_seq_len <= args.seq_len:
        parser.error("--probe-seq-len must lie between 67 and --seq-len")
    if any(not math.isfinite(x) or x <= 0 for x in (args.muon_lr, args.adamw_lr)):
        parser.error("learning rates must be finite and positive")
    if args.objective == "ce" and (args.alpha != 0.5 or args.temperature != 1.0):
        parser.error("--alpha and --temperature only apply to the KL arm")
    settings = DistillationSettings(args.alpha, args.temperature, args.chunk_tokens)
    if not torch.cuda.is_available() or not HAS_FLA:
        raise RuntimeError("this gate requires CUDA and the pinned FLA kernel")
    if os.environ.get("TRITON_F32_DEFAULT") != "tf32x3":
        raise RuntimeError("use the validated TRITON_F32_DEFAULT=tf32x3 policy")
    torch.set_num_threads(2)
    teacher_dtype = recovery.recovery_precision(args.precision, "cuda")
    source_config, tokenizer = recovery.load_source(args.source)
    payload, initialization_sha = recovery.load_initialization(args.initialization, args.audit)
    cfg = recovery.recovery_config(payload["config"], args.loop_k, args.seq_len)
    if cfg.frontend.vocab_size != tokenizer.vocab_size:
        raise ValueError("student vocabulary differs from the donor")
    data_audit = json.loads(args.data_audit.read_bytes())
    train_sha = recovery.digest(args.train)
    if not data_audit.get("complete") or train_sha != data_audit["splits"]["train"]["sha256"]:
        raise ValueError("training corpus differs from its preparation audit")
    source = TokenisedSource(LocalTextSource("recovery", 1.0, [args.train]), tokenizer, max_epochs=4)
    # A separate loader supplies a fixed prefix without advancing the timed stream.
    probe_loader = StreamingLoader([source], seq_len=args.probe_seq_len, batch_size=1, seed=0, separator=None)
    ids = torch.tensor(next(iter(probe_loader.batches(1))), device="cuda")
    model = ProphetModel(cfg)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.cuda()
    diagnostic_layers = diagnostic_gdn_fp32(model) if args.gdn_fp32_diagnostic else 0
    report = {"complete": False, "passed": False, "protocol": "qwen-recovery-gpu-gate-v1",
              "initialization_sha256": initialization_sha, "config": cfg.to_dict(),
              "donor_config": source_config, "donor_weights_sha256": recovery.WEIGHTS_SHA256,
              "tokenizer": tokenizer.fingerprint(), "train_sha256": train_sha,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "transformers": version("transformers"),
              "device": torch.cuda.get_device_name(), "fla": version("fla-core"),
              "triton": version("triton"), "triton_f32_default": os.environ["TRITON_F32_DEFAULT"],
              "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "objective": args.objective, "numerical": {},
              "precision": args.precision,
              "numerical_gate_scope": "FP32 only" if args.precision == "float32" else "FP32 and BF16",
              "gdn_fp32_diagnostic": bool(diagnostic_layers),
              "diagnostic_gdn_layers": diagnostic_layers,
              "scope": "initialized training path and measured allocation; not cached decoding or quality"
              + ("; diagnostic FP32 GDN override, not production BF16 acceptance" if diagnostic_layers else "")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recovery.write_report(args.out, report)
    for autocast in ((False,) if args.precision == "float32" else (False, True)):
        label = "bf16" if autocast else "fp32"
        try:
            result = training_kernel_agreement(model, ids, loop_k=args.loop_k,
                                               autocast=autocast, chunk_tokens=args.chunk_tokens)
        except Exception as exc:
            report["error"] = {"stage": label, "exception": repr(exc)}
            recovery.write_report(args.out, report)
            raise
        report["numerical"][label] = result
        recovery.write_report(args.out, report)
        print("NUMERICAL_GATE", label, result["passed"], flush=True)
        if not result["passed"]:
            raise SystemExit(1)
    del ids, probe_loader
    gc.collect()
    torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM

    objective = None
    if args.objective == "kl":
        teacher = AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, trust_remote_code=False,
            dtype=teacher_dtype, attn_implementation="sdpa")
        objective = DistillationObjective(teacher, identity={"donor_sha256": recovery.WEIGHTS_SHA256},
                                          settings=settings)
    source = TokenisedSource(LocalTextSource("recovery", 1.0, [args.train]), tokenizer, max_epochs=4)
    loader = StreamingLoader([source], seq_len=args.seq_len, batch_size=args.batch_size,
                             seed=0, separator=None)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="prophet-recovery-gate-") as scratch:
        train_cfg = TrainConfig(total_steps=128, seq_len=args.seq_len, batch_size=args.batch_size,
                                grad_accum_steps=args.grad_accum, checkpoint_every=0, checkpoint_dir=scratch,
                                log_every=1, peak_lr_muon=args.muon_lr, peak_lr_adamw=args.adamw_lr,
                                mtp_weight=0, confidence_weight=0, z_loss_weight=0, ponder_weight=0,
                                loss_chunk_tokens=args.chunk_tokens, device="cuda",
                                dtype=args.precision, allow_tf32=args.precision != "float32")
        trainer = Trainer(model, loader, train_cfg, model_config=cfg, tokenizer=tokenizer,
                          distillation=objective)
        try:
            report["updates"] = measure_updates(trainer)
            report["complete"] = True
            report["passed"] = report["updates"]["passed"]
        except Exception as exc:
            report["error"] = repr(exc)
            recovery.write_report(args.out, report)
            raise
    recovery.write_report(args.out, report)
    print("RECOVERY_GPU_GATE", report["passed"], flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
