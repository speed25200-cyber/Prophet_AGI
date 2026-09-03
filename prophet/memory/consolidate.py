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


# --------------------------------------------------------------------------------------
# Depth consolidation: making expensive reasoning cheap
# --------------------------------------------------------------------------------------


@dataclass
class DepthEpisode:
    """A problem worth thinking hard about once."""

    tokens: Tensor
    """Token ids, shape ``(1, n)``."""
    tag: str = ""
    verified: bool = True
    """Whether the deep answer was checked. Unverified episodes are refused by default:
    a memory that confidently returns a wrong answer is worse than no memory, because the
    model stops recomputing."""


@torch.no_grad()
def consolidate_depth(
    model: nn.Module,
    ledger: ProductKeyMemory,
    episodes: Sequence[DepthEpisode],
    *,
    deep_k: int = 16,
    shallow_k: int = 2,
    lam: float = 1.0,
    passes: int = 3,
    replay: Sequence[DepthEpisode] = (),
    replay_ratio: float = 0.25,
    require_verified: bool = True,
    lr: float | None = None,
    seed: int = 0,
) -> ConsolidationReport:
    """Distil the result of deep recurrence into memory addressable by a shallow pass.

    Test-time compute is the industry's answer to hard problems, and it is thrown away the
    moment the response ends. A model that spent a long time on a problem yesterday knows
    nothing more about it today; every query starts from the same frozen weights. Humans
    get better at a class of problem by working on instances of it. Models do not.

    Prophet has the two pieces needed to change that: recurrence depth is a runtime dial,
    so we can deliberately spend a lot of compute on one problem, and the ledger takes a
    closed-form backprop-free write, so we can keep the result. This function is the link:

    .. math::
        h^{deep} = f_{k=16}(x), \\quad h^{shallow} = f_{k=2}(x), \\quad
        t = \\lambda (h^{deep} - h^{shallow})

    written into the ledger **addressed by the shallow state**. A later cheap pass then
    retrieves what the expensive pass computed. Structurally this is the same operation as
    :func:`consolidate`, which distils along the *context* axis; this one distils along the
    *depth* axis, using the same closed-form write.

    Two things decide whether it is worth anything, and neither is settled here:

    - **Verification.** Consolidating a wrong deep answer is worse than not consolidating,
      because the model stops recomputing it. ``require_verified`` refuses unchecked
      episodes; supplying the verifier is the caller's job and is the expensive part.
    - **Generalisation.** Storing the answer to one problem helps only with that problem.
      Whether the stored state transfers to *neighbouring* instances is what separates
      learning from memoisation, and it is measured by
      :func:`depth_transfer_error` on held-out instances, never on the consolidated ones.
    """
    usable = [e for e in episodes if e.verified or not require_verified]
    if require_verified and len(usable) < len(episodes):
        skipped = len(episodes) - len(usable)
        if not usable:
            raise ValueError(
                f"all {skipped} episodes are unverified and require_verified is set; "
                "consolidating unchecked answers makes the model confidently wrong and "
                "stops it recomputing"
            )

    model.eval()
    rng = random.Random(seed)

    schedule: list[DepthEpisode] = []
    replayed = 0
    for _ in range(passes):
        batch = list(usable)
        if replay and replay_ratio > 0:
            n_replay = max(1, int(len(batch) * replay_ratio))
            sample = [rng.choice(list(replay)) for _ in range(n_replay)]
            replayed += len(sample)
            batch += sample
        rng.shuffle(batch)
        schedule.extend(batch)

    residual_start = 0.0
    residual_end = 0.0
    clipped: list[float] = []

    for step, episode in enumerate(schedule):
        deep = model(episode.tokens, loop_k=deep_k, return_mtp=False).hidden
        shallow = model(episode.tokens, loop_k=shallow_k, return_mtp=False).hidden

        target = lam * (deep - shallow)
        stats = ledger.write(shallow, target, lr=lr)

        if step < len(usable):
            residual_start += stats.residual_before / max(len(usable), 1)
        residual_end = stats.residual_after
        clipped.append(stats.clipped_fraction)

    return ConsolidationReport(
        episodes=len(usable),
        passes=passes,
        residual_start=residual_start,
        residual_end=residual_end,
        slots_touched=int(ledger.occupancy()["slots_used"]),
        clipped_fraction=sum(clipped) / max(len(clipped), 1),
        occupancy=ledger.occupancy(),
        replayed=replayed,
    )


@torch.no_grad()
def depth_transfer_error(
    model: nn.Module,
    ledger: ProductKeyMemory,
    episodes: Sequence[DepthEpisode],
    *,
    deep_k: int = 16,
    shallow_k: int = 2,
    lam: float = 1.0,
) -> float:
    """How much of the deep pass a shallow pass plus memory recovers.

    Zero means a cheap pass now reproduces what the expensive one computed; one means the
    ledger contributes nothing. Run it on the **consolidated** episodes to measure recall,
    and on **held-out instances of the same class** to measure whether anything was
    learned rather than memorised. The second number is the one that matters: a ledger
    that only ever retrieves is a cache, not a skill.
    """
    model.eval()
    total = 0.0
    scale = 0.0
    for episode in episodes:
        deep = model(episode.tokens, loop_k=deep_k, return_mtp=False).hidden
        shallow = model(episode.tokens, loop_k=shallow_k, return_mtp=False).hidden

        wanted = lam * (deep - shallow)
        got = ledger(shallow)
        total += (got - wanted).pow(2).sum().item()
        scale += wanted.pow(2).sum().item()
    return (total / max(scale, 1e-9)) ** 0.5


@torch.no_grad()
def depth_agreement(
    model: nn.Module,
    ledger: ProductKeyMemory | None,
    episodes: Sequence[DepthEpisode],
    *,
    deep_k: int = 16,
    shallow_k: int = 2,
) -> float:
    """Fraction of positions where a shallow pass predicts the same token as a deep one.

    The end-to-end number. Bandwidth and residual measurements can improve without the
    model's actual output changing; this asks the only question a user would: does the
    cheap pass now answer like the expensive one? Pass ``ledger=None`` for the baseline.
    """
    model.eval()
    matches = 0
    total = 0
    for episode in episodes:
        deep_out = model(episode.tokens, loop_k=deep_k, return_mtp=False)
        shallow_out = model(episode.tokens, loop_k=shallow_k, return_mtp=False)

        hidden = shallow_out.hidden
        if ledger is not None:
            # No lambda here: it is already baked into what the ledger was trained to
            # output, and applying it twice would scale the correction by lambda squared.
            hidden = hidden + ledger(hidden)
        logits = model._project(hidden)

        matches += int((logits.argmax(-1) == deep_out.logits.argmax(-1)).sum().item())
        total += logits.shape[0] * logits.shape[1]
    return matches / max(total, 1)
