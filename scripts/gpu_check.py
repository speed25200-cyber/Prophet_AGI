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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.budget import training_memory  # noqa: E402
from prophet.config import ProphetConfig  # noqa: E402
from prophet.modeling.layers import HAS_FLA, GatedDeltaNet  # noqa: E402
from prophet.modeling.model import ProphetCache, ProphetModel  # noqa: E402
from prophet.train.loss import compute_loss  # noqa: E402


def _set_fused(model: torch.nn.Module, fused: bool) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m.allow_fused = fused
            n += 1
    return n


def kernel_agreement(cfg: ProphetConfig, *, seq_len: int = 256) -> float:
    """Max abs difference between fused and reference paths on logits and states."""
    torch.manual_seed(0)
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
    """(seconds per step, peak GB) for forward + backward at the trainer's policy."""
    torch.manual_seed(0)
    model = ProphetModel(cfg).cuda().train()
    model.gradient_checkpointing = True
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
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:  # warm-up excluded
            times.append(time.perf_counter() - start)
    return sum(times) / len(times), torch.cuda.max_memory_allocated() / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/prophet_mini.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tokens", type=float, default=16.1e9, help="run budget, for the hours estimate")
    ap.add_argument("--tolerance", type=float, default=2e-3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: nothing to check here", file=sys.stderr)
        return 2
    cfg = ProphetConfig.from_json(args.config)
    cfg.validate()
    print(f"device     {torch.cuda.get_device_name(0)}")
    print(f"config     {cfg.name}")
    print(f"fla        {'installed' if HAS_FLA else 'MISSING -- the reference scan will be used and a real run is refused'}")

    if HAS_FLA:
        worst = kernel_agreement(cfg)
        verdict = "OK" if worst <= args.tolerance else "MISMATCH"
        print(f"kernel     max |fused - reference| = {worst:.2e} ({verdict}, tolerance {args.tolerance:.0e})")
        if worst > args.tolerance:
            print("Stop here. Fix the layout contract in GatedDeltaNet.forward before training.", file=sys.stderr)
            return 1

    seconds, peak = step_cost(cfg, batch_size=args.batch_size, seq_len=args.seq_len)
    tokens_per_step = args.batch_size * args.seq_len
    tps = tokens_per_step / seconds
    hours = args.tokens / tps / 3600
    predicted = training_memory(cfg, batch_tokens=tokens_per_step)
    print(f"step       {seconds:.3f} s for {tokens_per_step} tokens at k={cfg.recurrent.train_loop_max}: {tps:,.0f} tok/s")
    print(f"budget     {args.tokens / 1e9:.1f}B tokens at the *deepest* k = {hours:.1f} h "
          f"(E[k] is lower; treat this as the ceiling)")
    print(f"memory     peak {peak:.1f} GB measured vs {predicted.total_gb:.1f} GB predicted "
          f"({peak / max(predicted.total_gb, 1e-9):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
