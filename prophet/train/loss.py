"""Training objectives.

The language-modelling loss is the main term; the rest are auxiliaries that each buy
something specific:

- **Multi-token prediction** densifies the signal (every position supervises several
  future tokens instead of one) and produces a draft head for speculative decoding at
  no extra inference cost.
- **z-loss** keeps the logit magnitudes bounded. It costs almost nothing and removes a
  class of low-precision instability that is otherwise diagnosed only after a run
  diverges.
- **Router auxiliaries** keep MoE experts from collapsing onto a few.
- **Confidence** trains the abstention signal from track R09.
- **Ponder** trains the halting head. Without it the head receives no gradient at all,
  since the halting distribution does not enter the logits — and a halting head that is
  never trained makes depth *look* input-dependent while being noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from prophet.modeling.model import ProphetOutput

__all__ = ["LossTerms", "compute_loss"]


@dataclass
class LossTerms:
    total: Tensor
    lm: Tensor
    mtp: Tensor | None = None
    z: Tensor | None = None
    router: Tensor | None = None
    confidence: Tensor | None = None
    ponder: Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def _shifted_cross_entropy(logits: Tensor, targets: Tensor, offset: int) -> Tensor:
    """Cross entropy predicting the token ``offset`` positions ahead.

    ``offset=1`` is ordinary next-token prediction; ``offset=2`` is what the first
    multi-token-prediction head learns.
    """
    if offset >= logits.shape[1]:
        return logits.new_zeros(())
    pred = logits[:, :-offset].reshape(-1, logits.shape[-1])
    gold = targets[:, offset:].reshape(-1)
    return F.cross_entropy(pred.float(), gold, ignore_index=-100)


def _geometric_prior(n_steps: int, target_steps: float, device, dtype) -> Tensor:
    """Truncated geometric distribution over stopping times.

    The prior is what stops the model from always thinking as long as it is allowed:
    without it the halting head learns to put all its mass on the last iteration, since
    more computation never hurts the language-modelling loss.
    """
    lam = 1.0 / max(target_steps, 1.0)
    steps = torch.arange(n_steps, device=device, dtype=dtype)
    prior = lam * (1.0 - lam) ** steps
    return prior / prior.sum()


def compute_loss(
    output: ProphetOutput,
    targets: Tensor,
    *,
    mtp_weight: float = 0.3,
    z_loss_weight: float = 1e-4,
    confidence_weight: float = 0.0,
    confidence_targets: Tensor | None = None,
    ponder_weight: float = 0.0,
    ponder_target_steps: float = 4.0,
    project: "Callable[[Tensor], Tensor] | None" = None,
) -> LossTerms:
    """Combine every training objective into one scalar.

    ``targets`` is the token sequence; shifting is handled here so callers cannot get the
    off-by-one wrong — a mistake that produces a plausible-looking loss curve for a model
    that has learned to copy its input.
    """
    lm = _shifted_cross_entropy(output.logits, targets, 1)
    total = lm
    metrics: dict[str, float] = {"loss/lm": lm.item()}

    mtp_loss: Tensor | None = None
    if output.mtp_logits:
        terms = [
            _shifted_cross_entropy(logits, targets, 2 + j)
            for j, logits in enumerate(output.mtp_logits)
        ]
        mtp_loss = torch.stack(terms).mean()
        total = total + mtp_weight * mtp_loss
        metrics["loss/mtp"] = mtp_loss.item()

    z: Tensor | None = None
    if z_loss_weight:
        # Penalise the log-partition function, which is what actually grows when logits
        # drift; squaring keeps it symmetric.
        z = torch.logsumexp(output.logits.float(), dim=-1).pow(2).mean()
        total = total + z_loss_weight * z
        metrics["loss/z"] = z.item()

    router: Tensor | None = None
    if output.aux_loss is not None:
        router = output.aux_loss
        total = total + router
        metrics["loss/router"] = router.item()
        if output.router_stats:
            metrics["router/max_share"] = max(s.max_share for s in output.router_stats)
            metrics["router/min_entropy"] = min(s.entropy for s in output.router_stats)

    conf: Tensor | None = None
    if confidence_weight and output.confidence is not None and confidence_targets is not None:
        conf = F.binary_cross_entropy_with_logits(
            output.confidence.float(), confidence_targets.float()
        )
        total = total + confidence_weight * conf
        metrics["loss/confidence"] = conf.item()

    ponder: Tensor | None = None
    if ponder_weight and output.halt_probs is not None:
        p = output.halt_probs.float()

        # Expected language-modelling loss over stopping times. Each candidate stopping
        # point is scored on its own read-out, so the halting head learns which
        # iterations were actually good enough to stop at -- not merely how many there
        # were. Requires ``project`` to turn per-step hidden states into logits.
        expected = lm.new_zeros(())
        if project is not None and output.hidden_per_step:
            for i, hidden in enumerate(output.hidden_per_step):
                step_loss = _shifted_cross_entropy(project(hidden), targets, 1)
                weight = p[..., i].mean()
                expected = expected + weight * step_loss

        prior = _geometric_prior(p.shape[-1], ponder_target_steps, p.device, p.dtype)
        kl = (p * ((p + 1e-9).log() - prior.log())).sum(-1).mean()

        ponder = expected + kl
        total = total + ponder_weight * ponder
        metrics["loss/ponder"] = float(ponder.item())
        metrics["loss/ponder_kl"] = float(kl.item())
        depth = output.expected_depth()
        if depth is not None:
            metrics["ponder/expected_depth"] = depth

    metrics["loss/total"] = total.item()
    metrics["ppl"] = float(torch.exp(lm.detach().clamp(max=20)).item())
    return LossTerms(
        total=total, lm=lm, mtp=mtp_loss, z=z, router=router, confidence=conf,
        ponder=ponder, metrics=metrics,
    )
