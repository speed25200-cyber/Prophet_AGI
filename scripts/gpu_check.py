#!/usr/bin/env python3
"""The first thing to run on the A100: does the kernel agree, and how fast are we?

    python scripts/gpu_check.py --config configs/prophet_mini.json --seq-len 4096 --batch-size 8

Three questions, in order, each answered with a number:

1. **Does the fused delta-rule kernel match the reference scan?** Outputs and the state
   it hands back, on this device. Everything downstream assumes yes; nothing before
   this script has ever checked. A mismatch is a hard stop.
2. **What does one step cost?** Forward + backward at the requested batch shape under
   bf16 autocast with activation checkpointing -- the trainer's real policy -- measured
   after warm-up, reported as tokens per second and as hours for the run's token budget.
3. **Does memory hold?** Peak allocated memory against what ``prophet.budget`` predicted,
   so the estimator is calibrated against reality on day one rather than trusted.

No checkpoint is written. Exit code 1 on a kernel mismatch, 2 without a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.budget import training_memory  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.modeling.layers import HAS_FLA, GatedDeltaNet  # noqa: E402
from prophet.modeling.model import ProphetCache, ProphetModel  # noqa: E402
from prophet.modeling.moe import apply_router_updates  # noqa: E402
from prophet.train.loss import compute_loss  # noqa: E402
from prophet.train.optim import build_optimizers  # noqa: E402


def _set_fused(model: torch.nn.Module, fused: bool) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m.allow_fused = fused
            n += 1
    return n


def kernel_agreement(cfg: ProphetConfig, *, seq_len: int = 256, seed: int = 0) -> float:
    """Compare against an FP32 reference regardless of the caller's TF32 policy."""
    matmul, convolution = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        return _kernel_agreement(cfg, seq_len=seq_len, seed=seed)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = convolution


def _kernel_agreement(cfg: ProphetConfig, *, seq_len: int, seed: int) -> float:
    """Max abs difference between fused and reference paths on logits and states."""
    torch.manual_seed(seed)
    model = ProphetModel(cfg).cuda().eval()
    ids = torch.randint(0, cfg.frontend.vocab_size, (2, seq_len), device="cuda")
    worst = 0.0
    with torch.no_grad():
        _set_fused(model, False)
        ref_cache = ProphetCache()
        ref = model(ids, cache=ref_cache, loop_k=3).logits.float()
        _set_fused(model, True)
        cache = ProphetCache()
        out = model(ids, cache=cache, loop_k=3).logits.float()
        worst = max(worst, float((ref - out).abs().max()))
        for key, slot in ref_cache.slots.items():
            state = getattr(slot, "state", None)
            if state is None:
                continue
            other = cache.slots[key].state
            if other.shape != state.shape:
                print(f"STATE LAYOUT MISMATCH at {key}: {tuple(other.shape)} vs {tuple(state.shape)}")
                return float("inf")
            worst = max(worst, float((state.float() - other.float()).abs().max()))
    return worst


def step_cost(cfg: ProphetConfig, *, batch_size: int, seq_len: int, steps: int = 5) -> tuple[float, float]:
    """(seconds per step, peak GiB), including actual optimiser state and updates."""
    torch.manual_seed(0)
    model = ProphetModel(cfg).cuda().train()
    model.gradient_checkpointing = True
    optimizers, _ = build_optimizers(model, mup_base_width=cfg.mup_base_width, d_model=cfg.d_model)
    torch.backends.cuda.matmul.allow_tf32 = True
    ids = torch.randint(0, cfg.frontend.vocab_size, (batch_size, seq_len), device="cuda")
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(ids, loop_k=cfg.recurrent.train_loop_max if cfg.recurrent.enabled else None)
        terms = compute_loss(out, ids, project=model._project)
        terms.total.backward()
        apply_router_updates(out.router_stats)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:  # warm-up excluded
            times.append(time.perf_counter() - start)
    return sum(times) / len(times), torch.cuda.max_memory_allocated() / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/prophet_mini.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tokens", type=float, default=16.1e9, help="run budget, for the hours estimate")
    ap.add_argument("--tolerance", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=5, help="measured steps after two warm-up steps")
    ap.add_argument("--json-output", type=Path, help="save hardware, revision and measurements")
    args = ap.parse_args()

    if args.batch_size < 1 or args.seq_len < 1 or args.steps < 1:
        ap.error("batch-size, seq-len and steps must be positive")

    if not torch.cuda.is_available():
        print("no CUDA device: nothing to check here", file=sys.stderr)
        return 2
    cfg = ProphetConfig.from_json(args.config)
    cfg.validate()
    print(f"device     {torch.cuda.get_device_name(0)}")
    print(f"config     {cfg.name}")
    print(f"fla        {'installed' if HAS_FLA else 'MISSING -- the reference scan will be used and a real run is refused'}")

    if not HAS_FLA:
        print("Install the gpu extra before this gate; a fallback is not fused-kernel evidence.", file=sys.stderr)
        return 3

    if HAS_FLA:
        worst = kernel_agreement(cfg)
        verdict = "OK" if worst <= args.tolerance else "MISMATCH"
        print(f"kernel     max |fused - reference| = {worst:.2e} ({verdict}, tolerance {args.tolerance:.0e})")
        if worst > args.tolerance:
            print("Stop here. Fix the layout contract in GatedDeltaNet.forward before training.", file=sys.stderr)
            return 1

    seconds, peak = step_cost(cfg, batch_size=args.batch_size, seq_len=args.seq_len, steps=args.steps)
    tokens_per_step = args.batch_size * args.seq_len
    tps = tokens_per_step / seconds
    hours = args.tokens / tps / 3600
    predicted = training_memory(cfg, batch_tokens=tokens_per_step)
    print(f"step       {seconds:.3f} s for {tokens_per_step} tokens at k={cfg.recurrent.train_loop_max}: {tps:,.0f} tok/s")
    print(f"budget     {args.tokens / 1e9:.1f}B tokens at this measured shape/depth = {hours:.1f} h "
          f"(excludes data loading, evaluation and checkpoints)")
    print(f"memory     peak {peak:.1f} GiB measured vs {predicted.total_gb:.1f} GiB predicted "
          f"({peak / max(predicted.total_gb, 1e-9):.2f}x)")
    if args.json_output:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  cwd=Path(__file__).resolve().parent.parent, check=True).stdout.strip()
        report = {
            "revision": revision, "config": args.config, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.version.split()[0], "fla_core": version("fla-core"),
            "triton": version("triton"), "fused_chunk_size": 32,
            "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT", "tf32"),
            "device_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "batch_size": args.batch_size, "seq_len": args.seq_len, "measured_steps": args.steps,
            "kernel_max_abs_error": worst, "seconds_per_step": seconds, "tokens_per_second": tps,
            "peak_allocated_gib": peak, "predicted_gib": predicted.total_gb,
            "projected_tokens": args.tokens, "projected_hours": hours,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
