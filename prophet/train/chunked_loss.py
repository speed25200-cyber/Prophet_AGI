"""First-order CE and z-loss with vocabulary workspaces bounded by token chunks.

The model's logits remain materialised. Unlike ordinary autograd through separate
CE and logsumexp expressions, this operation retains no full-size FP32 vocabulary
buffers: backward recomputes row probabilities and combines both objectives before
returning one logits gradient. Higher-order differentiation is intentionally unsupported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable


class _TokenLosses(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, targets: Tensor, chunk_tokens: int):
        rows = logits.reshape(-1, logits.shape[-1])
        labels = targets.reshape(-1)
        ce = torch.empty(labels.shape, device=logits.device, dtype=torch.float32)
        log_partition = torch.empty_like(ce)
        for start in range(0, len(labels), chunk_tokens):
            end = min(start + chunk_tokens, len(labels))
            block = rows[start:end].float()
            ce[start:end] = F.cross_entropy(
                block, labels[start:end], reduction="none", ignore_index=-100,
            )
            log_partition[start:end] = torch.logsumexp(block, dim=-1)
        ctx.save_for_backward(logits, targets, log_partition)
        ctx.chunk_tokens = chunk_tokens
        ctx.set_materialize_grads(False)
        return ce.view_as(targets), log_partition.square().view_as(targets)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_ce: Tensor | None, grad_z: Tensor | None):
        logits, targets, log_partition = ctx.saved_tensors
        rows = logits.reshape(-1, logits.shape[-1])
        labels = targets.reshape(-1)
        grad_ce = None if grad_ce is None else grad_ce.reshape(-1)
        grad_z = None if grad_z is None else grad_z.reshape(-1)
        grad_logits = torch.empty_like(rows)
        for start in range(0, len(labels), ctx.chunk_tokens):
            end = min(start + ctx.chunk_tokens, len(labels))
            gold = labels[start:end]
            # log_softmax remains stable even for logits with a very large common
            # offset, where exp(logits - saved_logsumexp) would lose precision.
            probability = F.log_softmax(rows[start:end].float(), dim=-1).exp_()
            ce_weight = torch.zeros_like(log_partition[start:end])
            if grad_ce is not None:
                ce_weight = grad_ce[start:end].float().masked_fill(gold == -100, 0)
            coefficient = ce_weight
            if grad_z is not None:
                coefficient = coefficient + 2 * log_partition[start:end] * grad_z[start:end].float()
            probability.mul_(coefficient.unsqueeze(-1))
            probability.scatter_add_(1, gold.clamp_min(0).unsqueeze(-1), -ce_weight.unsqueeze(-1))
            grad_logits[start:end].copy_(probability)
        return grad_logits.view_as(logits), None, None


def token_losses(logits: Tensor, targets: Tensor, chunk_tokens: int) -> tuple[Tensor, Tensor]:
    """Per-position cross entropy (ignore=-100) and squared log-partition.

    Targets are already aligned with logits. Z-loss includes *all* positions,
    including ignored CE targets, matching the original training objective.
    Non-contiguous logits are supported but may require a contiguous copy.
    """
    if not isinstance(chunk_tokens, int) or isinstance(chunk_tokens, bool) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if logits.ndim < 2 or logits.shape[-1] < 1 or targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match the non-vocabulary dimensions of logits")
    if targets.dtype != torch.long or targets.device != logits.device:
        raise ValueError("targets must be long tensors on the logits device")
    return _TokenLosses.apply(logits, targets, chunk_tokens)


def shifted_token_losses(
    logits: Tensor, targets: Tensor, offset: int, chunk_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Shift within each sequence, without a full-vocabulary reshape/copy."""
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        raise ValueError("offset must be a positive integer")
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("shifted losses require (batch, sequence, vocabulary) logits")
    aligned = torch.full_like(targets, -100)
    if offset < targets.shape[1]:
        aligned[:, :-offset] = targets[:, offset:]
    ce, z = token_losses(logits, aligned, chunk_tokens)
    return ce[:, :max(0, logits.shape[1] - offset)], z
