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

from prophet.modeling.action import ActionTargets
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
    action: Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def _shifted_cross_entropy(
    logits: Tensor, targets: Tensor, offset: int, *, per_token: bool = False
) -> Tensor:
    """Cross entropy predicting the token ``offset`` positions ahead.

    ``offset=1`` is ordinary next-token prediction; ``offset=2`` is what the first
    multi-token-prediction head learns. ``per_token`` returns the ``(batch, seq-offset)``
    matrix instead of the mean, which the ponder loss needs.
    """
    if offset >= logits.shape[1]:
        return logits.new_zeros(()) if not per_token else logits.new_zeros(logits.shape[:2])
    pred = logits[:, :-offset].reshape(-1, logits.shape[-1])
    gold = targets[:, offset:].reshape(-1)
    ce = F.cross_entropy(pred.float(), gold, ignore_index=-100, reduction="none")
    if per_token:
        return ce.view(logits.shape[0], logits.shape[1] - offset)
    mask = gold != -100
    return ce[mask].mean() if mask.any() else ce.sum() * 0.0


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
    action_targets: ActionTargets | None = None,
    sel_weight: float = 0.0,
    ptr_weight: float = 0.0,
    gate_weight: float = 0.0,
    jumped_lm_weight: float = 1.0,
) -> LossTerms:
    """Combine every training objective into one scalar.

    ``targets`` is the token sequence; shifting is handled here so callers cannot get the
    off-by-one wrong — a mistake that produces a plausible-looking loss curve for a model
    that has learned to copy its input.

    ``action_targets`` (track A3) adds the selection, pointer and gate terms and
    down-weights the LM loss on *jumped* tokens -- call syntax and names a typed runtime
    emits for the model -- to ``jumped_lm_weight``.
    """
    if action_targets is not None and jumped_lm_weight != 1.0:
        ce = _shifted_cross_entropy(output.logits, targets, 1, per_token=True)
        # The loss at position t predicts t+1, so a jumped *target* token is what gets
        # the small weight.
        jumped = action_targets.jumped[:, 1:].to(ce.dtype)
        weight = torch.where(jumped > 0, torch.full_like(ce, jumped_lm_weight), torch.ones_like(ce))
        weight = weight * (targets[:, 1:] != -100).to(ce.dtype)
        lm = (ce * weight).sum() / weight.sum().clamp_min(1.0)
    else:
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

        # Expected language-modelling loss over stopping times, **per token**. Each
        # candidate stopping point is scored on its own read-out, and the halting
        # probability at position t weights the loss at position t. The first version
        # multiplied two batch means -- mean(p_i) * mean(loss_i) -- whose gradient with
        # respect to p is the same number at every position, so the head could only
        # ever learn one constant distribution for the whole batch. Measured: one unique
        # gradient value across ten positions. Input-dependent depth was unlearnable by
        # construction, while ponder/expected_depth moved and looked learned.
        expected = lm.new_zeros(())
        if project is not None and output.hidden_per_step:
            for i, hidden in enumerate(output.hidden_per_step):
                ce_i = _shifted_cross_entropy(project(hidden), targets, 1, per_token=True)
                # p is (batch, seq, steps); the loss at position t predicts t+1.
                expected = expected + (p[:, : ce_i.shape[1], i] * ce_i).mean()

        prior = _geometric_prior(p.shape[-1], ponder_target_steps, p.device, p.dtype)
        kl = (p * ((p + 1e-9).log() - prior.log())).sum(-1).mean()

        ponder = expected + kl
        total = total + ponder_weight * ponder
        metrics["loss/ponder"] = float(ponder.item())
        metrics["loss/ponder_kl"] = float(kl.item())
        depth = output.expected_depth()
        if depth is not None:
            metrics["ponder/expected_depth"] = depth

    action: Tensor | None = None
    if action_targets is not None and output.sel_logits is not None:
        action = lm.new_zeros(())
        # Selection: one cross-entropy over [none, anchor_1..anchor_n] per decision.
        n_opt = output.sel_logits.shape[-1]
        sel_target = action_targets.selection
        sel_ok = (sel_target >= 0) & (sel_target < n_opt)
        sel_target = torch.where(sel_ok, sel_target, torch.full_like(sel_target, -100))
        if sel_ok.any():
            sel = F.cross_entropy(
                output.sel_logits.float().reshape(-1, n_opt), sel_target.reshape(-1),
                ignore_index=-100,
            )
            action = action + sel_weight * sel
            metrics["loss/sel"] = sel.item()
            correct = output.sel_logits.argmax(-1) == action_targets.selection
            metrics["action/sel_accuracy"] = float(correct[sel_ok].float().mean().item())
        # Pointers: start and end over the copy layer's keys. In a cache-free pass the
        # key index is the absolute position, which is what the targets hold.
        if output.copy_start is not None and (action_targets.copy_start >= 0).any():
            n_keys = output.copy_start.shape[-1]
            start_t = action_targets.copy_start.reshape(-1)
            end_t = action_targets.copy_end.reshape(-1)
            ptr = (
                F.cross_entropy(output.copy_start.float().reshape(-1, n_keys), start_t, ignore_index=-100)
                + F.cross_entropy(output.copy_end.float().reshape(-1, n_keys), end_t, ignore_index=-100)
            )
            action = action + ptr_weight * ptr
            metrics["loss/ptr"] = ptr.item()
        # Gate: is the value starting here verbatim in context?
        if output.copy_gate is not None and (action_targets.gate_target >= 0).any():
            pos = action_targets.gate_positions.clamp_min(0)
            logit = output.copy_gate.float().gather(1, pos)
            mask = action_targets.gate_target >= 0
            gate = F.binary_cross_entropy_with_logits(
                logit[mask], action_targets.gate_target[mask].float()
            )
            action = action + gate_weight * gate
            metrics["loss/gate"] = gate.item()
        total = total + action
        metrics["loss/action"] = float(action.item())

    metrics["loss/total"] = total.item()
    metrics["ppl"] = float(torch.exp(lm.detach().clamp(max=20)).item())
    return LossTerms(
        total=total, lm=lm, mtp=mtp_loss, z=z, router=router, confidence=conf,
        ponder=ponder, action=action, metrics=metrics,
    )
