"""Self-verification: knowing when an answer can be trusted, and what that permits.

Track A4's result frames everything here. Below roughly 7B parameters a model checking
its *own* reasoning is not better than chance: verifier errors are coupled to generator
errors, and self-correction rounds lose points. Verification is easier than generation
in three cases only -- execution or lookup, comparison of several candidates, and a
separately trained verifier. So this module does not ask the model whether it was right.
It reads signals the trunk produces anyway, combines them with a small calibrated
scorer, and applies a policy whose shape is fixed by arithmetic:

    Think and retry in proportion to how cheaply the result can be checked: unbounded
    attempts behind an execution check, one extra attempt behind a learned check, none
    behind nothing -- then ask. Consolidate only what a program or three independent
    runs agreed on; a confidence head may gate acting, never remembering.

The last clause is load-bearing. A learned verifier at AUROC 0.80 admits about 30% wrong
answers into memory; a ledger written on its say-so caps the class's future accuracy
below what a deep pass already achieves. Hence :class:`Tier`: only ground truth writes.

Two of the signals -- disagreement between recurrence depths, and disagreement between
the multi-token-prediction head and the main head -- are specific to this architecture
and have no published error-prediction performance. :func:`auroc` exists so that the
first experiment (A4-0: does depth disagreement predict error?) is one function call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "Tier",
    "Signals",
    "Verdict",
    "VerifierConfig",
    "extract_signals",
    "depth_disagreement",
    "mtp_disagreement",
    "SignalScorer",
    "decide",
    "auroc",
]


class Tier(IntEnum):
    """How an answer was checked, which decides what it may be used for."""

    GROUND_TRUTH = 0
    """A program said yes: tests passed, a checker matched. Writes to the main ledger."""
    CONSENSUS = 1
    """At least three independent runs agreed. Quarantine; promoted by later agreement."""
    LEARNED = 2
    """A learned head said yes. May gate acting. Never consolidated."""
    UNVERIFIED = 3
    """Nothing checked it. Refused."""


@dataclass
class Signals:
    """Per-span features. Every field is cheap; ``execution`` comes from the tool layer."""

    confidence_logit: float = 0.0
    mean_entropy_bits: float = 0.0
    mean_margin: float = 0.0
    min_max_prob: float = 1.0
    depth_disagreement: float | None = None
    """Fraction of span tokens whose argmax differs between a shallow and a deep read-out.
    Architecture-specific; no published AUROC -- that is experiment A4-0."""
    depth_kl: float | None = None
    expected_depth: float | None = None
    mtp_disagreement: float | None = None
    execution: bool | None = None
    """True = passed, False = failed, None = no executable check available."""

    def vector(self) -> Tensor:
        """Fixed-order feature vector for the scorer. Missing signals become 0 with a
        presence flag so the scorer can learn 'unknown' separately from 'zero'."""
        vals = [
            self.confidence_logit,
            self.mean_entropy_bits,
            self.mean_margin,
            self.min_max_prob,
            self.depth_disagreement if self.depth_disagreement is not None else 0.0,
            1.0 if self.depth_disagreement is not None else 0.0,
            self.depth_kl if self.depth_kl is not None else 0.0,
            self.expected_depth if self.expected_depth is not None else 0.0,
            self.mtp_disagreement if self.mtp_disagreement is not None else 0.0,
            1.0 if self.mtp_disagreement is not None else 0.0,
        ]
        return torch.tensor(vals, dtype=torch.float32)


N_FEATURES = 10


@dataclass
class Verdict:
    p_correct: float
    tier: Tier
    decision: str
    """One of ``act``, ``retry_depth``, ``retry_sample``, ``ask``."""
    consolidate: bool
    signals: Signals
    reason: str = ""


@dataclass
class VerifierConfig:
    enabled: bool = True
    threshold: float = 0.5
    """Deployment threshold on P(correct): 0.0 on 0/1-scored evals, ~0.5 chat, 0.75 for
    factual claims, 0.9 for high-stakes actions (R09's table)."""
    depth_trigger: float = 0.15
    """Depth disagreement above which one retry at maximum depth is worth its cost --
    only if the accuracy-versus-depth sweep found a gain of at least ``depth_gain``."""
    depth_gain_points: float = 0.0
    """Measured accuracy gain of the deep pass over the shallow one, from the sweep.
    Zero until measured; below 5 the depth retry is disabled by the policy."""
    max_attempts_free: int = 8
    max_attempts_learned: int = 2
    verifier_cost_ratio: float = 0.0
    """rho = cost of checking / cost of generating, for this task class."""


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------


def _span_slice(t: Tensor, span: tuple[int, int] | None) -> Tensor:
    if span is None:
        return t
    return t[:, span[0]:span[1]]


@torch.no_grad()
def extract_signals(output, *, span: tuple[int, int] | None = None,
                    project=None, execution: bool | None = None) -> Signals:
    """Read every free signal off one :class:`ProphetOutput`.

    ``span`` selects the answer positions ``[start, end)``; ``project`` is the model's
    hidden-to-logits map, needed for the depth-disagreement signal when halting was on.
    """
    logits = _span_slice(output.logits, span).float()
    logp = F.log_softmax(logits, -1)
    probs = logp.exp()
    entropy_bits = float((-(probs * logp).sum(-1) / math.log(2)).mean().item())
    top2 = probs.topk(2, dim=-1).values
    margin = float((top2[..., 0] - top2[..., 1]).mean().item())
    min_max = float(top2[..., 0].min().item())

    conf = 0.0
    if getattr(output, "confidence", None) is not None:
        conf = float(_span_slice(output.confidence, span).mean().item())

    sig = Signals(confidence_logit=conf, mean_entropy_bits=entropy_bits,
                  mean_margin=margin, min_max_prob=min_max, execution=execution)

    if getattr(output, "halt_probs", None) is not None:
        sig.expected_depth = output.expected_depth()
    steps = getattr(output, "hidden_per_step", None)
    if steps and len(steps) >= 2 and project is not None:
        shallow = project(_span_slice(steps[min(1, len(steps) - 1)], span)).float()
        deep = logits
        sig.depth_disagreement, sig.depth_kl = _disagreement(shallow, deep)

    mtp = getattr(output, "mtp_logits", None)
    if mtp:
        sig.mtp_disagreement = mtp_disagreement(output.logits, mtp[0], span=span)
    return sig


def _disagreement(a: Tensor, b: Tensor) -> tuple[float, float]:
    frac = float((a.argmax(-1) != b.argmax(-1)).float().mean().item())
    kl = float(F.kl_div(F.log_softmax(a, -1), F.log_softmax(b, -1),
                        log_target=True, reduction="batchmean").item())
    return frac, kl


@torch.no_grad()
def depth_disagreement(model, input_ids: Tensor, *, shallow_k: int = 2, deep_k: int = 8,
                       span: tuple[int, int] | None = None) -> tuple[float, float]:
    """Run the same input at two depths and compare read-outs.

    Costs one extra forward; use :func:`extract_signals` instead when halting is on and
    the per-step read-outs are already there.
    """
    model.eval()
    a = _span_slice(model(input_ids, loop_k=shallow_k, return_mtp=False).logits, span).float()
    b = _span_slice(model(input_ids, loop_k=deep_k, return_mtp=False).logits, span).float()
    return _disagreement(a, b)


@torch.no_grad()
def mtp_disagreement(logits: Tensor, mtp_logits: Tensor, *,
                     span: tuple[int, int] | None = None) -> float:
    """KL between the first MTP head's prediction of token t+2 and the main head's
    prediction of the same token one position later."""
    main = logits[:, 1:].float()          # main head at position t+1 predicts t+2
    ahead = mtp_logits[:, :-1].float()    # MTP head at position t predicts t+2
    if span is not None:
        lo, hi = span
        main, ahead = main[:, max(lo - 1, 0):max(hi - 1, 0)], ahead[:, max(lo - 1, 0):max(hi - 1, 0)]
    if main.shape[1] == 0:
        return 0.0
    return float(F.kl_div(F.log_softmax(ahead, -1), F.log_softmax(main, -1),
                          log_target=True, reduction="batchmean").item())


# --------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------


class SignalScorer:
    """A calibrated logistic map from signals to P(correct).

    Tiny on purpose: ten weights and a bias. Fitted by Newton's method on
    program-produced labels only, with a post-hoc temperature. Until :meth:`fit` has run
    it uses a prior that trusts the confidence head and penalises entropy and
    disagreement -- a reasonable starting point, and labelled as such in ``fitted``.
    """

    def __init__(self) -> None:
        self.weight = torch.tensor(
            [1.0, -0.5, 1.0, 0.5, -2.0, 0.0, -0.5, 0.0, -0.5, 0.0], dtype=torch.float32
        )
        self.bias = torch.tensor(0.0)
        self.mean = torch.zeros(N_FEATURES)
        self.std = torch.ones(N_FEATURES)
        self.temperature = 1.0
        self.fitted = False

    def logit(self, sig: Signals) -> float:
        x = (sig.vector() - self.mean) / self.std
        return float((x @ self.weight + self.bias).item()) / self.temperature

    def prob(self, sig: Signals) -> float:
        if sig.execution is not None:
            # A program's verdict overrides every learned signal.
            return 0.98 if sig.execution else 0.02
        return float(torch.sigmoid(torch.tensor(self.logit(sig))).item())

    def fit(self, features: Tensor, labels: Tensor, *, steps: int = 25, l2: float = 1e-2) -> float:
        """Newton-fit the weights on ``(n, N_FEATURES)`` features and 0/1 labels.

        Returns the final mean negative log-likelihood. Standardises features first so
        the prior weights and the fitted ones live on the same scale.
        """
        x = features.float()
        y = labels.float()
        self.mean = x.mean(0)
        self.std = x.std(0).clamp_min(1e-6)
        z = (x - self.mean) / self.std
        zb = torch.cat([z, torch.ones(len(z), 1)], 1)
        w = torch.cat([self.weight, self.bias.view(1)])
        for _ in range(steps):
            p = torch.sigmoid(zb @ w)
            grad = zb.T @ (p - y) / len(y) + l2 * w
            hess = (zb.T * (p * (1 - p))) @ zb / len(y) + l2 * torch.eye(len(w))
            w = w - torch.linalg.solve(hess, grad)
        self.weight, self.bias = w[:-1], w[-1]
        self.fitted = True
        p = torch.sigmoid(zb @ w).clamp(1e-6, 1 - 1e-6)
        return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean().item())

    def calibrate_temperature(self, features: Tensor, labels: Tensor) -> float:
        """Post-hoc temperature on held-out data -- never fitted jointly."""
        z = ((features.float() - self.mean) / self.std) @ self.weight + self.bias
        y = labels.float()
        best_t, best_nll = 1.0, float("inf")
        for t in torch.linspace(0.25, 4.0, 31):
            p = torch.sigmoid(z / t).clamp(1e-6, 1 - 1e-6)
            nll = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean().item())
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        self.temperature = best_t
        return best_t


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


def decide(sig: Signals, p_correct: float, cfg: VerifierConfig, *,
           attempts: int, agreements: int = 0) -> Verdict:
    """The verifiability-aware rule. Pure function; the agent loop calls it."""
    if sig.execution is not None:
        tier = Tier.GROUND_TRUTH if sig.execution else Tier.UNVERIFIED
        if sig.execution:
            return Verdict(p_correct, tier, "act", True, sig, "execution passed")
        if attempts < cfg.max_attempts_free:
            return Verdict(p_correct, tier, "retry_sample", False, sig,
                           f"execution failed; attempt {attempts + 1} of {cfg.max_attempts_free}")
        return Verdict(p_correct, tier, "ask", False, sig, "execution kept failing")

    tier = Tier.CONSENSUS if agreements >= 3 else (
        Tier.LEARNED if p_correct >= cfg.threshold else Tier.UNVERIFIED
    )
    if p_correct >= cfg.threshold:
        # A head may gate acting, never remembering: LEARNED never consolidates.
        return Verdict(p_correct, tier, "act", tier == Tier.CONSENSUS, sig,
                       "above threshold" + (" with consensus" if tier == Tier.CONSENSUS else ""))

    d = sig.depth_disagreement
    if (d is not None and d >= cfg.depth_trigger and attempts == 0
            and cfg.depth_gain_points >= 5.0):
        return Verdict(p_correct, tier, "retry_depth", False, sig,
                       f"depth disagreement {d:.2f} >= {cfg.depth_trigger}")
    if cfg.verifier_cost_ratio <= 1.0 and attempts < cfg.max_attempts_learned:
        return Verdict(p_correct, tier, "retry_sample", False, sig,
                       f"below threshold; attempt {attempts + 1} of {cfg.max_attempts_learned}")
    return Verdict(p_correct, tier, "ask", False, sig, "out of cheap attempts")


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve by the rank statistic. ``labels`` are 1 = error.

    This is the number experiment A4-0 reports for each signal: keep a signal if its
    AUROC against 'the deep pass was wrong' is at least 0.70 and beats entropy by 0.05.
    """
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, l in labels_iter(labels) if l == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank-sum with tie handling.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rank_sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def labels_iter(labels: Sequence[int]):
    for i, l in enumerate(labels):
        yield i, int(l)
