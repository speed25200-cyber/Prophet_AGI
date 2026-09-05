"""Persistent memory: the ledger.

Today's models cannot learn after deployment. Weights are frozen, the only memory is the
context window, and fine-tuning causes catastrophic forgetting. Track R03 found the
result the whole design rests on: **sparse memory updates lose 11% of prior knowledge
where full fine-tuning loses 89% and LoRA loses 71%, at equal new knowledge learned**.
Writing to a few slots is not merely cheaper than adjusting all the weights — it is the
only variant that does not destroy what was already there.

The ledger is a product-key memory layer whose values are rewritten by a **closed-form
local update**, not by gradient descent through the model:

.. math::
    m(x) = \\sum_i w_i V[i], \\qquad
    \\frac{\\partial m}{\\partial V[i]} = w_i I, \\qquad
    \\Delta V[i] = -\\eta\\, w_i \\,(m(x) - t)

Because the read is a weighted sum of value rows, the derivative of the output with
respect to a value row is just its weight. The optimal local step is therefore available
in closed form: two forward passes and a scatter-add, with **no gradient through the
backbone at any point**. That is what makes it runnable on a phone.

Product keys are what keep the read cheap. Addressing ``n`` slots directly costs ``n``
dot products; splitting the query in half against two codebooks of ``sqrt(n)`` sub-keys
costs ``2*sqrt(n)`` and reaches the same slots. At 65,536 slots that is 512 comparisons
instead of 65,536.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["ProductKeyMemory", "WriteStats", "LedgerConfig"]


@dataclass
class LedgerConfig:
    """Shape and update policy of a ledger."""

    dim: int
    memory_dim: int = 256
    n_slots: int = 65_536
    top_k: int = 32
    n_heads: int = 4
    """Independent query heads. More heads read more slots per token for the same
    codebook, which is the cheap way to raise capacity."""

    write_lr: float = 1.0
    """Fraction of the exact local step to take. Not a gradient-descent learning rate:
    the update below is the minimum-norm solution of the local least-squares problem, so
    ``1.0`` lands exactly on the target for an isolated token and anything less
    deliberately under-shoots."""
    trust_region: float = 4.0
    """Maximum L2 norm of any single slot's update. Without it one surprising example can
    overwrite a slot that thousands of earlier ones agreed on."""
    ewc_lambda: float = 0.5
    """Per-slot learning-rate decay with write count: often-written slots move less.
    A cheap stand-in for a Fisher penalty, and enough to stop the most-used slots from
    being churned by every new session."""
    decay: float = 1.0
    """Multiplicative decay applied to values on each write. Below 1.0 the ledger
    forgets slowly, which bounds drift over a long deployment."""
    max_writes: int | None = None
    """Lifetime cap on written tokens. Past it :meth:`ProductKeyMemory.write` refuses
    (``WriteStats.accepted`` is False) instead of applying: a bound on how much a
    runaway writer can churn."""


@dataclass
class WriteStats:
    slots_touched: int
    mean_update_norm: float
    clipped_fraction: float
    residual_before: float
    residual_after: float
    accepted: bool = True
    """False when the write was refused by ``max_writes``; nothing was applied."""

    @property
    def improvement(self) -> float:
        """Fraction of the residual the write removed. Near zero means the memory could
        not represent the target and the write was wasted."""
        if self.residual_before <= 0:
            return 0.0
        return 1.0 - self.residual_after / self.residual_before


class ProductKeyMemory(nn.Module):
    """A sparse, directly-writable associative store.

    Keys are **frozen after initialisation**. Only values move. This is deliberate: if
    keys drift, every previously written association silently points somewhere else, and
    the failure is invisible until an old fact comes back wrong.
    """

    def __init__(self, cfg: LedgerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        side = int(math.isqrt(cfg.n_slots))
        if side * side != cfg.n_slots:
            raise ValueError(
                f"n_slots={cfg.n_slots} must be a perfect square for product keys; "
                f"try {side ** 2} or {(side + 1) ** 2}"
            )
        self.side = side
        self.sub_dim = cfg.memory_dim // 2
        if cfg.memory_dim % 2 != 0:
            raise ValueError(f"memory_dim={cfg.memory_dim} must be even")
        if cfg.top_k > cfg.n_slots:
            raise ValueError(f"top_k={cfg.top_k} exceeds n_slots={cfg.n_slots}")

        # Addressing is **frozen**: a random projection and a non-affine normalisation,
        # registered as buffers so no optimiser can move them. The class promises that
        # keys never drift; a trainable query projection broke that promise from the
        # other side -- training moved every stored association's address.
        query = torch.empty(cfg.n_heads * cfg.memory_dim, cfg.dim)
        nn.init.normal_(query, std=cfg.dim**-0.5)
        self.register_buffer("query_weight", query, persistent=True)
        self.query_norm = nn.LayerNorm(cfg.memory_dim, elementwise_affine=False)

        # Frozen sub-key codebooks.
        keys = torch.randn(2, cfg.n_heads, side, self.sub_dim) / math.sqrt(self.sub_dim)
        self.register_buffer("sub_keys", keys, persistent=True)

        # The written state. A buffer, not a parameter: it is updated by the write rule,
        # never by an optimiser.
        self.register_buffer("values", torch.zeros(cfg.n_slots, cfg.dim), persistent=True)
        self.register_buffer("write_counts", torch.zeros(cfg.n_slots), persistent=True)
        self.register_buffer("tokens_written", torch.zeros((), dtype=torch.long), persistent=True)

    # -- reading -----------------------------------------------------------------------

    def address(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(indices, weights)`` of shape ``(tokens, heads * top_k)``.

        The product-key trick: score each half of the query against its own codebook of
        ``sqrt(n_slots)`` sub-keys, take the top candidates from each, and combine only
        those. The full slot table is never scored.
        """
        cfg = self.cfg
        flat = x.reshape(-1, cfg.dim)
        n = flat.shape[0]

        q = F.linear(flat, self.query_weight).view(n, cfg.n_heads, cfg.memory_dim)
        q = self.query_norm(q)
        q1, q2 = q.split(self.sub_dim, dim=-1)

        # (n, heads, side)
        s1 = torch.einsum("nhd,hsd->nhs", q1, self.sub_keys[0])
        s2 = torch.einsum("nhd,hsd->nhs", q2, self.sub_keys[1])

        k = min(cfg.top_k, self.side)
        top1, idx1 = s1.topk(k, dim=-1)
        top2, idx2 = s2.topk(k, dim=-1)

        # Combine the two halves: (n, heads, k, k)
        combined = top1.unsqueeze(-1) + top2.unsqueeze(-2)
        flat_scores = combined.view(n, cfg.n_heads, k * k)
        best, best_idx = flat_scores.topk(cfg.top_k, dim=-1)

        row = best_idx // k
        col = best_idx % k
        slot = (
            torch.gather(idx1, -1, row) * self.side + torch.gather(idx2, -1, col)
        )  # (n, heads, top_k)

        weights = F.softmax(best.float(), dim=-1).to(x.dtype)
        return slot.reshape(n, -1), weights.reshape(n, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Read the ledger. Shape-preserving: ``(batch, seq, dim)`` in and out."""
        indices, weights = self.address(x)
        gathered = self.values[indices]  # (tokens, heads*top_k, dim)
        out = (gathered * weights.unsqueeze(-1)).sum(1)
        # Heads are averaged rather than summed so the read magnitude is independent of
        # head count, which keeps the write step size comparable across configurations.
        out = out / self.cfg.n_heads
        return out.view(*x.shape[:-1], self.cfg.dim)

    # -- writing -----------------------------------------------------------------------

    @torch.no_grad()
    def write(self, x: Tensor, target: Tensor, *, lr: float | None = None) -> WriteStats:
        """Move the ledger's output at ``x`` toward ``target``.

        Closed form, no backpropagation: ``dV[i] = -lr * w_i * (m(x) - target)``, applied
        by scatter-add. Two forward passes and one scatter is the entire cost, which is
        why this can run on a device rather than only in a training job.
        """
        cfg = self.cfg
        lr = cfg.write_lr if lr is None else lr

        flat_x = x.reshape(-1, cfg.dim)
        flat_t = target.reshape(-1, cfg.dim)
        indices, weights = self.address(flat_x)

        current = (self.values[indices] * weights.unsqueeze(-1)).sum(1) / cfg.n_heads
        residual = current - flat_t                      # (tokens, dim)
        residual_before = residual.norm(dim=-1).mean().item()

        if cfg.max_writes is not None and int(self.tokens_written.item()) >= cfg.max_writes:
            return WriteStats(0, 0.0, 0.0, residual_before, residual_before, accepted=False)
        self.tokens_written += flat_x.shape[0]

        # The read is ``sum_i a_i V[i]`` with ``a_i = w_i / n_heads``, so moving the
        # output by ``-residual`` is an underdetermined linear system. Its minimum-norm
        # solution is ``dV[i] = -a_i / ||a||^2 * residual``: distributing the correction
        # across slots in proportion to how much each contributed. Dividing by ``||a||^2``
        # is what makes this the exact step rather than a scaled gradient -- without it
        # the write under-shoots by a factor of ``top_k``, and the ledger appears to
        # barely learn.
        a = weights / cfg.n_heads                        # (t, h*k)
        denom = a.pow(2).sum(-1, keepdim=True).clamp_min(1e-9)
        update = -lr * (a / denom).unsqueeze(-1) * residual.unsqueeze(1)  # (t, h*k, dim)

        # EWC-lite: slots written often move less. Without this, the slots that carry the
        # most agreed-upon knowledge are exactly the ones every new session churns.
        counts = self.write_counts[indices].unsqueeze(-1)
        update = update / (1.0 + cfg.ewc_lambda * counts.sqrt())

        # Trust region, per slot. One surprising example must not overwrite a slot that
        # thousands of earlier ones agreed on.
        norms = update.norm(dim=-1, keepdim=True)
        clipped = (norms > cfg.trust_region).float()
        scale = torch.clamp(cfg.trust_region / norms.clamp_min(1e-9), max=1.0)
        update = update * scale

        if cfg.decay < 1.0:
            self.values.mul_(cfg.decay)

        flat_idx = indices.reshape(-1)
        self.values.index_add_(0, flat_idx, update.reshape(-1, cfg.dim).to(self.values.dtype))
        self.write_counts.index_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=self.write_counts.dtype)
        )

        after = (self.values[indices] * weights.unsqueeze(-1)).sum(1) / cfg.n_heads
        residual_after = (after - flat_t).norm(dim=-1).mean().item()

        return WriteStats(
            slots_touched=int(flat_idx.unique().numel()),
            mean_update_norm=float(norms.mean().item()),
            clipped_fraction=float(clipped.mean().item()),
            residual_before=residual_before,
            residual_after=residual_after,
        )

    # -- housekeeping ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self) -> None:
        self.values.zero_()
        self.write_counts.zero_()
        self.tokens_written.zero_()

    def occupancy(self) -> dict[str, float]:
        """How much of the ledger is in use, and how evenly.

        A ledger where a handful of slots absorb every write has collapsed, and the symptom
        is not a bad loss curve -- it is a memory that gradually stops discriminating.
        """
        used = (self.write_counts > 0)
        n_used = int(used.sum().item())
        total_writes = float(self.write_counts.sum().item())
        share = self.write_counts / max(total_writes, 1.0)
        nonzero = share[share > 0]
        entropy = float(-(nonzero * nonzero.log()).sum().item()) if nonzero.numel() else 0.0
        return {
            "slots_used": float(n_used),
            "slots_total": float(self.cfg.n_slots),
            "occupancy": n_used / self.cfg.n_slots,
            "total_writes": total_writes,
            "write_entropy": entropy,
            "max_entropy": math.log(max(n_used, 1)),
            "value_norm": float(self.values.norm().item()),
        }

    def n_bytes(self, *, dtype_bytes: int = 2) -> int:
        """Deployed size of the ledger's state."""
        return self.cfg.n_slots * self.cfg.dim * dtype_bytes
