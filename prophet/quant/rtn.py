"""Weight-only round-to-nearest quantization, per output channel, for the H5 measurement.

This is deliberately the simplest post-training quantizer there is: symmetric,
per-output-channel scales, nearest rounding, weights dequantized back to floating point
so the unchanged model code runs the quantized function. It exists to measure how the
quantization error of a looped core accumulates with depth for two core mixers, not to
be a deployment quantizer; anything smarter (GPTQ, LoopQ) would only lower both curves.

Scope: the ``nn.Linear`` modules inside the blocks (prelude, core, coda). Embeddings,
the tied language-model head, norms and auxiliary heads stay in floating point, so the
read-out is the same for every depth and the difference between depths is the core's.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

BLOCK_PREFIX = "sections."


def quantize_tensor_rtn(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row RTN; returns the dequantized weight and the integer codes."""
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    if weight.dim() != 2:
        raise ValueError("expected a 2-D weight (out, in)")
    qmax = 2 ** (bits - 1) - 1
    w = weight.detach().to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return (codes * scale).to(weight.dtype), codes.to(torch.int8)


def block_linear_modules(model: nn.Module, *, prefix: str = BLOCK_PREFIX) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.startswith(prefix)
    ]


def quantize_model_rtn(
    model: nn.Module, bits: int, *, prefix: str = BLOCK_PREFIX
) -> tuple[nn.Module, dict]:
    """A deep copy of ``model`` with its block linears fake-quantized, plus a report."""
    quantized = copy.deepcopy(model)
    names = block_linear_modules(quantized, prefix=prefix)
    if not names:
        raise ValueError(f"no linear modules under {prefix!r}")
    modules = dict(quantized.named_modules())
    params = 0
    squared_error = 0.0
    squared_norm = 0.0
    for name in names:
        linear = modules[name]
        original = linear.weight.detach().clone()
        dequantized, _ = quantize_tensor_rtn(original, bits)
        with torch.no_grad():
            linear.weight.copy_(dequantized)
        params += original.numel()
        squared_error += float(((dequantized.float() - original.float()) ** 2).sum())
        squared_norm += float((original.float() ** 2).sum())
    report = {
        "bits": bits,
        "scheme": "symmetric per-output-channel round-to-nearest, weight only",
        "scope_prefix": prefix,
        "quantized_linear_modules": len(names),
        "quantized_parameters": params,
        "relative_l2_error": math.sqrt(squared_error / squared_norm) if squared_norm else 0.0,
    }
    return quantized, report
