"""Frozen-donor forward KL for controlled recovery experiments.

Only probability workspaces are chunked: both models' logits remain materialised.
The first-order backward recomputes probabilities and never differentiates the donor.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import once_differentiable


class _ForwardKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, temperature, chunk_tokens):
        rows = student.reshape(-1, student.shape[-1])
        donor = teacher.reshape_as(rows)
        losses = torch.empty(rows.shape[0], device=student.device, dtype=torch.float32)
        for start in range(0, len(rows), chunk_tokens):
            end = start + chunk_tokens
            log_q = F.log_softmax(donor[start:end].float() / temperature, dim=-1)
            log_p = F.log_softmax(rows[start:end].float() / temperature, dim=-1)
            losses[start:end] = (log_q.exp() * (log_q - log_p)).sum(-1) * temperature**2
        ctx.save_for_backward(student, teacher)
        ctx.temperature, ctx.chunk_tokens = temperature, chunk_tokens
        return losses.view(student.shape[:-1])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        student, teacher = ctx.saved_tensors
        rows = student.reshape(-1, student.shape[-1])
        donor = teacher.reshape_as(rows)
        weights = grad_loss.reshape(-1)
        gradient = torch.empty_like(rows)
        for start in range(0, len(rows), ctx.chunk_tokens):
            end = start + ctx.chunk_tokens
            p = F.softmax(rows[start:end].float() / ctx.temperature, dim=-1)
            q = F.softmax(donor[start:end].float() / ctx.temperature, dim=-1)
            gradient[start:end] = (p - q) * (ctx.temperature * weights[start:end, None])
        return gradient.view_as(student), None, None, None


def forward_kl(student: Tensor, teacher: Tensor, *, temperature=1.0, chunk_tokens=128):
    """Per-position ``T² KL(teacher/T || student/T)``; teacher is always detached."""
    if (student.shape != teacher.shape or student.ndim < 2 or student.shape[-1] < 1
            or student.device != teacher.device):
        raise ValueError("student and teacher logits must have matching shapes and devices")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("logits must be floating point")
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(chunk_tokens, int) or isinstance(chunk_tokens, bool) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    return _ForwardKL.apply(student, teacher.detach(), float(temperature), chunk_tokens)


def state_sha256(model: nn.Module) -> str:
    """Hash names, shapes, dtypes and actual tensor bytes, with bounded CPU copies."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        header = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":"))
        digest.update(header.encode() + b"\0")
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_048_576):
            block = flat[start:start + 1_048_576].contiguous().cpu().view(torch.uint8)
            digest.update(block.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DistillationSettings:
    alpha: float = 0.5
    temperature: float = 1.0
    chunk_tokens: int = 128

    def __post_init__(self):
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1]; omit the objective for CE only")
        if (isinstance(self.temperature, bool) or not math.isfinite(self.temperature)
                or self.temperature <= 0):
            raise ValueError("temperature must be finite and positive")
        if (not isinstance(self.chunk_tokens, int) or isinstance(self.chunk_tokens, bool)
                or self.chunk_tokens < 1):
            raise ValueError("chunk_tokens must be a positive integer")


class DistillationObjective:
    """A separately owned HF-style frozen teacher; never part of student checkpoints.

    Construct after loading the teacher's final weights and dtype. Identity must include
    its configuration, source revision and tokenizer policy. The actual weights are
    hashed too, so swapping the donor while resuming changes the training contract.
    """

    def __init__(self, teacher: nn.Module, *, identity: dict,
                 settings: DistillationSettings | None = None):
        self.teacher = teacher.eval().requires_grad_(False)
        self.settings = settings or DistillationSettings()
        self._identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        if not self._identity:
            raise ValueError("teacher identity must describe the source, config and tokenizer")
        self._state_sha256 = state_sha256(teacher)

    def fingerprint(self):
        return {"format": "frozen-forward-kl-v1", "identity": deepcopy(self._identity),
                "state_sha256": self._state_sha256, "alpha": self.settings.alpha,
                "temperature": self.settings.temperature, "chunk_tokens": self.settings.chunk_tokens,
                "alignment": "next-token; exclude final position; full vocabulary"}

    def loss(self, logits: Tensor, batch: Tensor, *, autocast_dtype=None) -> Tensor:
        if logits.ndim != 3 or logits.shape[:2] != batch.shape or batch.shape[1] < 2:
            raise ValueError("distillation requires matching (batch, sequence>=2) logits and tokens")
        self.teacher.eval()
        with torch.no_grad(), torch.autocast(
            device_type=batch.device.type, dtype=autocast_dtype or torch.bfloat16,
            enabled=batch.device.type == "cuda" and autocast_dtype is not None,
        ):
            donor_logits = self.teacher(input_ids=batch, use_cache=False).logits
        losses = forward_kl(logits, donor_logits, temperature=self.settings.temperature,
                            chunk_tokens=self.settings.chunk_tokens)
        return losses[:, :-1].mean()
