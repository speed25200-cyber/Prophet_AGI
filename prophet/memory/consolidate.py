"""The consolidation pass: turning context into memory.

An idea borrowed from sleep, and the second tier of track R03's design. During a session
the model sees things it cannot keep: the context window ends and they are gone. The
consolidation pass runs offline, afterwards, and distils what the context contributed
into ledger slots, so a later session gets the benefit **without the context being
present**.

The mechanism is context distillation with a closed-form target. For a query the model
has seen both with and without its context:

.. math::
    h^+ = f(\\text{context} \\Vert \\text{query}), \\quad
    h^- = f(\\text{query}), \\quad
    t = m(h^-) + \\lambda (h^+ - h^-)

``h^+ - h^-`` is precisely what the context contributed. Asking the ledger to produce
that difference when addressed by the context-free state makes a later context-free query
behave as though the context were still there. The write itself is
:meth:`~prophet.memory.ledger.ProductKeyMemory.write` — two forward passes and a
scatter-add, with no gradient through the backbone at any point.

Replay is not optional here. Writing only new episodes drifts the ledger toward whatever
was learned most recently, which is the same catastrophic forgetting the design exists to
avoid, merely relocated from the weights into the memory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from prophet.memory.ledger import ProductKeyMemory

__all__ = ["Episode", "ConsolidationReport", "consolidate", "recall_error"]


@dataclass
class Episode:
    """One thing worth remembering: a query, and the context that explains it."""

    context: Tensor
    """Token ids, shape ``(1, n)``."""
    query: Tensor
    """Token ids, shape ``(1, m)``."""
    tag: str = ""


@dataclass
class ConsolidationReport:
    episodes: int
    passes: int
    residual_start: float
    residual_end: float
    slots_touched: int
    clipped_fraction: float
    occupancy: dict[str, float] = field(default_factory=dict)
    replayed: int = 0

    @property
    def improvement(self) -> float:
        if self.residual_start <= 0:
            return 0.0
        return 1.0 - self.residual_end / self.residual_start

    def summary(self) -> str:
        return (
            f"consolidated {self.episodes} episodes over {self.passes} passes "
            f"({self.replayed} replayed): residual {self.residual_start:.4f} -> "
            f"{self.residual_end:.4f} ({self.improvement:.0%} closed), "
            f"{self.slots_touched} slots touched, "
            f"{self.clipped_fraction:.1%} of updates hit the trust region"
        )


@torch.no_grad()
def _hidden_for(model: nn.Module, ids: Tensor, *, last_n: int) -> Tensor:
    """Final hidden states for the last ``last_n`` positions."""
    out = model(ids, return_mtp=False)
    return out.hidden[:, -last_n:, :]


@torch.no_grad()
def consolidate(
    model: nn.Module,
    ledger: ProductKeyMemory,
    episodes: Sequence[Episode],
    *,
    lam: float = 1.0,
    passes: int = 3,
    replay: Sequence[Episode] = (),
    replay_ratio: float = 0.25,
    lr: float | None = None,
    seed: int = 0,
    on_step: Callable[[int, float], None] | None = None,
) -> ConsolidationReport:
    """Write the contribution of each episode's context into the ledger.

    ``replay`` should hold previously consolidated episodes. A fraction of them is
    interleaved so that consolidating new material does not quietly displace old.
    """
    model.eval()
    rng = random.Random(seed)

    schedule: list[Episode] = []
    replayed = 0
    for _ in range(passes):
        batch = list(episodes)
        if replay and replay_ratio > 0:
            n_replay = max(1, int(len(batch) * replay_ratio))
            sample = [rng.choice(list(replay)) for _ in range(n_replay)]
            replayed += len(sample)
            batch += sample
        rng.shuffle(batch)
        schedule.extend(batch)

    residual_start = 0.0
    residual_end = 0.0
    slots: set[int] = set()
    clipped: list[float] = []

    for step, episode in enumerate(schedule):
        n_query = episode.query.shape[1]
        with_context = torch.cat([episode.context, episode.query], dim=1)

        h_plus = _hidden_for(model, with_context, last_n=n_query)
        h_minus = _hidden_for(model, episode.query, last_n=n_query)

        # Absolute target: what the ledger should output, not how far it should move.
        target = lam * (h_plus - h_minus)

        stats = ledger.write(h_minus, target, lr=lr)
        if step < len(episodes):
            residual_start += stats.residual_before / max(len(episodes), 1)
        residual_end = stats.residual_after
        clipped.append(stats.clipped_fraction)
        slots.add(stats.slots_touched)

        if on_step is not None:
            on_step(step, stats.residual_after)

    return ConsolidationReport(
        episodes=len(episodes),
        passes=passes,
        residual_start=residual_start,
        residual_end=residual_end,
        slots_touched=int(ledger.occupancy()["slots_used"]),
        clipped_fraction=sum(clipped) / max(len(clipped), 1),
        occupancy=ledger.occupancy(),
        replayed=replayed,
    )


@torch.no_grad()
def recall_error(
    model: nn.Module,
    ledger: ProductKeyMemory,
    episodes: Sequence[Episode],
    *,
    lam: float = 1.0,
) -> float:
    """How well the ledger reproduces the context's effect, with the context removed.

    Zero means a context-free query now behaves exactly as if the context were present;
    one means the ledger contributes nothing. This is the number that decides whether
    persistent memory works, and it is measured **after clearing the context**, which is
    the only measurement that distinguishes memory from a longer prompt.
    """
    model.eval()
    total = 0.0
    scale = 0.0
    for episode in episodes:
        n_query = episode.query.shape[1]
        with_context = torch.cat([episode.context, episode.query], dim=1)
        h_plus = _hidden_for(model, with_context, last_n=n_query)
        h_minus = _hidden_for(model, episode.query, last_n=n_query)

        wanted = lam * (h_plus - h_minus)
        got = ledger(h_minus)
        total += (got - wanted).pow(2).sum().item()
        scale += wanted.pow(2).sum().item()
    return (total / max(scale, 1e-9)) ** 0.5
