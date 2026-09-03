"""Evaluation metrics.

The central choice, from track R11: **decide on bits-per-byte, not accuracy.**

Below roughly 500M parameters most multiple-choice benchmarks sit at chance. At 130M
parameters and 2.6B tokens — the scale of our ablations — ARC-Challenge, WinoGrande,
CommonsenseQA and MMLU are all pinned there. Choosing a data mixture on those scores is
choosing by coin flip.

Bits-per-byte moves monotonically from the first billion tokens and is comparable across
tokenizers, which matters because we may change ours. It is the ablation decision metric;
accuracy is reported but does not decide anything until the model is large enough for it
to mean something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "BPBResult",
    "bits_per_byte",
    "cross_entropy_nats",
    "multiple_choice_accuracy",
    "chance_level",
    "is_above_chance",
]


def cross_entropy_nats(logits: Tensor, targets: Tensor, *, ignore_index: int = -100) -> tuple[float, int]:
    """Summed cross-entropy in nats and the number of scored tokens.

    Returns a sum rather than a mean so results can be accumulated across batches of
    different sizes without weighting mistakes.
    """
    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
    flat_targets = targets[:, 1:].reshape(-1)
    mask = flat_targets != ignore_index
    if not mask.any():
        return 0.0, 0
    loss = F.cross_entropy(
        flat_logits[mask], flat_targets[mask], reduction="sum", ignore_index=ignore_index
    )
    return loss.item(), int(mask.sum().item())


@dataclass
class BPBResult:
    """Bits per byte, with the pieces kept so the number can be audited."""

    total_nats: float
    n_tokens: int
    n_bytes: int
    domain: str = ""

    @property
    def bits_per_byte(self) -> float:
        if self.n_bytes == 0:
            return float("nan")
        return self.total_nats / (self.n_bytes * math.log(2))

    @property
    def nats_per_token(self) -> float:
        return self.total_nats / max(self.n_tokens, 1)

    @property
    def perplexity(self) -> float:
        return math.exp(min(self.nats_per_token, 20.0))

    @property
    def bytes_per_token(self) -> float:
        """Tokenizer fertility. Falling BPB with rising fertility can mean the tokenizer
        changed rather than the model improving, so it is reported alongside."""
        return self.n_bytes / max(self.n_tokens, 1)


def bits_per_byte(total_nats: float, n_tokens: int, n_bytes: int, domain: str = "") -> BPBResult:
    """Normalise a loss by *bytes of raw text*, not tokens.

    This is what makes the number comparable across tokenizers: a tokenizer that packs
    more text per token gets a lower loss per token for free, and dividing by bytes
    removes exactly that advantage.
    """
    return BPBResult(total_nats=total_nats, n_tokens=n_tokens, n_bytes=n_bytes, domain=domain)


def multiple_choice_accuracy(
    scores: list[list[float]], gold: list[int], *, length_normalise: bool = True,
    lengths: list[list[int]] | None = None,
) -> dict[str, float]:
    """Accuracy over multiple-choice items, with and without length normalisation.

    Both are reported because they routinely disagree, and picking whichever is higher
    after the fact is how benchmark numbers become meaningless. Which one to report is
    fixed per task in the harness config, not chosen per run.
    """
    if len(scores) != len(gold):
        raise ValueError(f"{len(scores)} score rows against {len(gold)} labels")

    raw_correct = 0
    norm_correct = 0
    for i, row in enumerate(scores):
        raw_correct += int(max(range(len(row)), key=lambda j: row[j]) == gold[i])
        if length_normalise and lengths is not None:
            normed = [s / max(lengths[i][j], 1) for j, s in enumerate(row)]
            norm_correct += int(max(range(len(normed)), key=lambda j: normed[j]) == gold[i])

    n = len(gold)
    out = {"acc": raw_correct / n, "n": float(n)}
    if length_normalise and lengths is not None:
        out["acc_norm"] = norm_correct / n
    return out


def chance_level(n_choices: int) -> float:
    return 1.0 / max(n_choices, 1)


def is_above_chance(accuracy: float, n_items: int, n_choices: int, *, z: float = 2.0) -> bool:
    """Is this accuracy distinguishable from guessing?

    Most small-model benchmark scores are not, and reporting them as if they were is the
    most common way an ablation reaches a confident wrong conclusion. Uses a normal
    approximation to the binomial standard error at the chance rate.
    """
    p = chance_level(n_choices)
    if n_items <= 0:
        return False
    se = math.sqrt(p * (1 - p) / n_items)
    return accuracy > p + z * se
