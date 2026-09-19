"""Optimisers for Prophet.

Track R07's recipe: **Muon on every 2-D hidden matrix, AdamW on everything else**.

Muon orthogonalises the momentum before applying it. The intuition is that gradient
descent on a matrix repeatedly pushes in a few dominant directions; orthogonalising
spreads the update across the whole spectrum, so each step moves the matrix further in
directions it has not already learned.

Two facts from R07 shape how it is used here:

- The realistic speedup is **1.1-1.4x**, not the 2x sometimes quoted — measured against
  a *properly tuned* AdamW, and shrinking with scale. Worth having, not worth
  restructuring the project around.
- The bigger practical win is memory: **2 bytes/param of state versus AdamW's 4-8**. On
  a single 80GB A100 that is often what decides whether a configuration trains at all.

Muon applies only to 2-D matrices. Embeddings, the output head, norms, biases, gates and
the MoE router keep AdamW: they are not the kind of object orthogonalisation is defined
for, and routers in particular are numerically delicate.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn

__all__ = ["Muon", "newton_schulz_orthogonalise", "build_param_groups", "build_optimizers"]


@torch.no_grad()
def newton_schulz_orthogonalise(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal factor of ``G`` by a quintic Newton-Schulz iteration.

    The coefficients are the standard tuned quintic: they do not converge to the exact
    orthogonal factor, but they push every singular value into roughly [0.7, 1.3] in five
    steps, which is what the update needs and is far cheaper than an SVD.

    Runs in bfloat16 on purpose — the iteration is numerically forgiving, and this is a
    per-step cost on every matrix in the model.
    """
    if G.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(G.shape)}")

    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum with orthogonalised updates.

    Parameters
    ----------
    lr:
        Base learning rate. Muon's effective step is scaled by matrix shape (see
        ``rms_match``), so this is not directly comparable to an AdamW learning rate.
    momentum:
        Heavy-ball coefficient on the raw gradient.
    nesterov:
        Look ahead before orthogonalising. R07's recipe has this on.
    weight_decay:
        Decoupled, applied directly to the weights.
    ns_steps:
        Newton-Schulz iterations. Five is the standard trade-off.
    rms_match:
        Scale the update by ``0.2 * sqrt(max(rows, cols))`` so its RMS matches what
        AdamW would produce. Without it, a single learning rate cannot serve matrices of
        different shapes and the optimiser needs per-layer tuning.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        rms_match: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        super().__init__(
            list(params),
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                rms_match=rms_match,
            ),
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: D102
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon received a {p.ndim}-D parameter of shape {tuple(p.shape)}; "
                        "route 1-D and >2-D parameters to AdamW instead"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                buf.mul_(mu).add_(p.grad)
                g = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf

                update = newton_schulz_orthogonalise(g, steps=group["ns_steps"])
                scale = 0.2 * math.sqrt(max(p.shape)) if group["rms_match"] else 1.0

                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr * scale)

        return loss


# --------------------------------------------------------------------------------------
# Parameter routing
# --------------------------------------------------------------------------------------

#: Substrings identifying parameters that must never go to Muon, regardless of shape.
#: The router is the delicate one: orthogonalising its updates would fight the load
#: balancing that keeps experts from collapsing.
ADAMW_ONLY_PATTERNS: tuple[str, ...] = (
    "embed",
    "lm_head",
    "router",
    "expert_bias",
    "norm",
    "bias",
    "a_proj",
    "b_proj",
    "inv_freq",
)


def build_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split parameters into the Muon set and the AdamW set.

    Returns a dict with keys ``"muon"``, ``"adamw_decay"`` and ``"adamw_no_decay"``.
    Norms, biases and embeddings get no weight decay: decaying them shrinks the
    representation rather than regularising it.
    """
    muon: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        lowered = name.lower()
        forced_adamw = any(pat in lowered for pat in ADAMW_ONLY_PATTERNS)
        # A (1, d) or (d, 1) matrix -- the halting and confidence heads -- is a vector in
        # disguise. Newton-Schulz on it is just normalisation with a 0.2*sqrt(d) scale,
        # so it goes to AdamW with the other heads, as the docstring promises.
        if p.ndim == 2 and min(p.shape) == 1:
            forced_adamw = True

        if p.ndim == 2 and not forced_adamw:
            muon.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    return {"muon": muon, "adamw_decay": decay, "adamw_no_decay": no_decay}


def build_optimizers(
    model: nn.Module,
    *,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    mup_base_width: int | None = None,
    d_model: int | None = None,
) -> tuple[list[torch.optim.Optimizer], dict[str, int]]:
    """Build the optimiser pair and report how many parameters each got.

    When ``mup_base_width`` is set, the AdamW learning rate for hidden matrices is scaled
    by ``base_width / d_model`` — the muP rule that lets a learning rate tuned on a narrow
    proxy transfer to the full width without re-sweeping, which at our budget is the
    difference between one tuning run and a dozen.
    """
    groups = build_param_groups(model)

    if mup_base_width and d_model:
        ratio = mup_base_width / d_model
        muon_lr = muon_lr * ratio
        adamw_lr = adamw_lr * ratio

    optimizers: list[torch.optim.Optimizer] = []
    if groups["muon"]:
        optimizers.append(
            Muon(groups["muon"], lr=muon_lr, weight_decay=weight_decay)
        )
    adamw_groups = []
    if groups["adamw_decay"]:
        adamw_groups.append({"params": groups["adamw_decay"], "weight_decay": weight_decay})
    if groups["adamw_no_decay"]:
        adamw_groups.append({"params": groups["adamw_no_decay"], "weight_decay": 0.0})
    if adamw_groups:
        optimizers.append(torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=betas, eps=eps))

    counts = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    return optimizers, counts
