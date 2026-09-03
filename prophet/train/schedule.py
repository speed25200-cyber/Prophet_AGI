"""Learning-rate schedules.

Prophet uses **WSD** (warmup-stable-decay) rather than cosine, for a reason specific to
our situation. A cosine schedule bakes the total step count into every step: the run
length must be fixed in advance, and stopping early leaves the model at a high learning
rate, mid-descent and undertrained. WSD holds a constant rate through the middle and
decays only at the end, which gives three properties we need:

1. **The run length need not be known in advance.** Extend or shorten the plateau freely
   as the Colab budget turns out to be larger or smaller than hoped.
2. **Plateau checkpoints are reusable.** Several anneals can be branched from a single
   plateau checkpoint and souped together — the cheapest reliable point on the board,
   since the decay phase is a small fraction of total tokens but is where benchmark
   scores are actually made.
3. **Interruption is survivable.** A Colab session dying during the plateau costs the
   elapsed steps and nothing else.

The ``1-sqrt`` decay shape is the default: it spends longer at higher learning rates than
a linear ramp before dropping sharply, which empirically beats linear at equal length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = ["WSDSchedule", "CosineSchedule", "build_schedule"]

DecayShape = Literal["linear", "cosine", "one_minus_sqrt"]


@dataclass
class WSDSchedule:
    """Warmup, constant plateau, then decay.

    Fractions are of ``total_steps``; the plateau is whatever is left over.
    """

    peak_lr: float
    total_steps: int
    warmup_frac: float = 0.02
    decay_frac: float = 0.18
    final_lr_frac: float = 0.0
    """Final learning rate as a fraction of peak. Zero for a terminal run; leave it
    non-zero if training will be continued afterwards."""
    decay_shape: DecayShape = "one_minus_sqrt"
    min_warmup_steps: int = 100

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_frac + self.decay_frac > 1.0:
            raise ValueError(
                f"warmup_frac ({self.warmup_frac}) + decay_frac ({self.decay_frac}) "
                "exceeds 1.0, leaving no plateau"
            )
        self.warmup_steps = max(self.min_warmup_steps, int(self.warmup_frac * self.total_steps))
        self.warmup_steps = min(self.warmup_steps, self.total_steps // 2)
        self.decay_steps = int(self.decay_frac * self.total_steps)
        self.plateau_steps = self.total_steps - self.warmup_steps - self.decay_steps

    @property
    def decay_start(self) -> int:
        """Step at which decay begins — the branch point for anneal variants."""
        return self.warmup_steps + self.plateau_steps

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup from a non-zero floor: a literal zero first step wastes a
            # step and can leave optimiser state uninitialised.
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step < self.decay_start:
            return self.peak_lr

        progress = (step - self.decay_start) / max(self.decay_steps, 1)
        progress = min(max(progress, 0.0), 1.0)

        if self.decay_shape == "linear":
            factor = 1.0 - progress
        elif self.decay_shape == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # one_minus_sqrt
            factor = 1.0 - math.sqrt(progress)

        return self.peak_lr * (self.final_lr_frac + (1.0 - self.final_lr_frac) * factor)

    def phase_at(self, step: int) -> str:
        if step < self.warmup_steps:
            return "warmup"
        if step < self.decay_start:
            return "plateau"
        return "decay"

    def describe(self) -> str:
        return (
            f"WSD({self.peak_lr:.2e} peak, {self.warmup_steps} warmup / "
            f"{self.plateau_steps} plateau / {self.decay_steps} decay, "
            f"{self.decay_shape}, branch point at step {self.decay_start})"
        )


@dataclass
class CosineSchedule:
    """Cosine decay with warmup. Kept as the baseline WSD must beat, not as a default."""

    peak_lr: float
    total_steps: int
    warmup_frac: float = 0.02
    final_lr_frac: float = 0.1

    def __post_init__(self) -> None:
        self.warmup_steps = max(1, int(self.warmup_frac * self.total_steps))

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * (self.final_lr_frac + (1.0 - self.final_lr_frac) * factor)

    def phase_at(self, step: int) -> str:
        return "warmup" if step < self.warmup_steps else "decay"


def build_schedule(kind: str, **kwargs) -> WSDSchedule | CosineSchedule:
    if kind == "wsd":
        return WSDSchedule(**kwargs)
    if kind == "cosine":
        return CosineSchedule(**kwargs)
    raise ValueError(f"unknown schedule kind {kind!r}")
