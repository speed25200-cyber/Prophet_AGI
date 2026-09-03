"""Sparse feed-forward layers.

Sparsity is how Prophet decouples knowledge capacity from per-token cost: total
parameters set what the model can know, active parameters set how fast it decodes on
bandwidth-bound consumer hardware.

Two choices here are deliberate and worth stating:

- **Fine-grained experts plus always-on shared experts.** Many small experts beat few
  large ones at equal active parameters, and the shared experts absorb what every token
  needs so the routed experts can specialise instead of each relearning the basics.
- **Loss-free load balancing.** Balance is maintained by nudging a per-expert routing
  bias rather than by an auxiliary loss. An auxiliary loss competes with the
  language-modelling objective for gradient; a bias on the routing logits does not
  touch the gradient at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prophet.modeling.layers import SwiGLU

__all__ = ["MoERouter", "SparseMoE", "RouterStats"]


class RouterStats:
    """Per-step routing diagnostics.

    Expert collapse is the characteristic MoE failure and it is invisible in the loss
    curve, so these are logged every step rather than computed on demand.
    """

    __slots__ = ("expert_counts", "max_share", "entropy", "aux_loss")

    def __init__(
        self,
        expert_counts: Tensor,
        max_share: float,
        entropy: float,
        aux_loss: Tensor,
    ) -> None:
        self.expert_counts = expert_counts
        self.max_share = max_share
        self.entropy = entropy
        self.aux_loss = aux_loss


class MoERouter(nn.Module):
    """Top-k token-choice router with loss-free bias balancing."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        *,
        bias_balancing: bool = True,
        bias_update_rate: float = 1e-3,
        z_loss_weight: float = 1e-3,
        load_balance_loss_weight: float = 0.0,
        router_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_balancing = bias_balancing
        self.bias_update_rate = bias_update_rate
        self.z_loss_weight = z_loss_weight
        self.load_balance_loss_weight = load_balance_loss_weight
        self.router_dtype = router_dtype

        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, std=dim**-0.5)
        # Not a parameter: updated by a rule, never by a gradient.
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, RouterStats]:
        """Route ``(tokens, dim)`` to experts.

        Returns ``(indices, weights, stats)`` where ``indices`` and ``weights`` are
        ``(tokens, top_k)``.
        """
        logits = F.linear(x.to(self.router_dtype), self.weight.to(self.router_dtype))
        scores = torch.sigmoid(logits)

        # The bias steers *selection* only. Combination weights come from the unbiased
        # scores, so balancing never distorts the values the layer actually mixes.
        selection = scores + self.expert_bias if self.bias_balancing else scores
        _, indices = torch.topk(selection, self.top_k, dim=-1)
        weights = torch.gather(scores, -1, indices)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).float()
        share = counts / counts.sum().clamp_min(1.0)

        aux = logits.new_zeros(())
        if self.z_loss_weight:
            # Keeps router logits from drifting to large magnitudes, which is the usual
            # precursor to instability in low precision.
            aux = aux + self.z_loss_weight * torch.logsumexp(logits, dim=-1).pow(2).mean()
        if self.load_balance_loss_weight:
            probs = scores.mean(0)
            aux = aux + self.load_balance_loss_weight * self.n_experts * (share * probs).sum()

        if self.bias_balancing and self.training:
            with torch.no_grad():
                target = 1.0 / self.n_experts
                # Under-used experts get a positive nudge, over-used a negative one.
                self.expert_bias -= self.bias_update_rate * torch.sign(share - target)

        entropy = -(share.clamp_min(1e-9) * share.clamp_min(1e-9).log()).sum()
        stats = RouterStats(counts, share.max().item(), entropy.item(), aux)
        return indices, weights.to(x.dtype), stats


class SparseMoE(nn.Module):
    """Mixture of fine-grained experts with always-on shared experts."""

    def __init__(
        self,
        dim: int,
        *,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        n_shared: int = 0,
        shared_hidden: int | None = None,
        bias_balancing: bool = True,
        bias_update_rate: float = 1e-3,
        z_loss_weight: float = 1e-3,
        load_balance_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = MoERouter(
            dim,
            n_experts,
            top_k,
            bias_balancing=bias_balancing,
            bias_update_rate=bias_update_rate,
            z_loss_weight=z_loss_weight,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        self.experts = nn.ModuleList(SwiGLU(dim, expert_hidden) for _ in range(n_experts))
        self.shared = (
            SwiGLU(dim, shared_hidden or expert_hidden * max(n_shared, 1))
            if n_shared
            else None
        )
        self.last_stats: RouterStats | None = None

    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        indices, weights, stats = self.router(flat)
        self.last_stats = stats

        out = torch.zeros_like(flat)
        # Gather-scatter per expert. Correct and simple; a grouped-GEMM kernel replaces
        # this loop for training throughput without changing semantics.
        for e, expert in enumerate(self.experts):
            hit = indices == e
            if not hit.any():
                continue
            token_idx, slot_idx = hit.nonzero(as_tuple=True)
            contribution = expert(flat[token_idx]) * weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, contribution.to(out.dtype))

        out = out.view(b, s, d)
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def aux_loss(self) -> Tensor | None:
        return None if self.last_stats is None else self.last_stats.aux_loss
