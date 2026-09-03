"""Measuring the bandwidth of a model's reasoning channels.

Chain-of-thought is usually discussed as a latency cost. It is also, and more
fundamentally, an **information bottleneck**: at every reasoning step the model's entire
internal state is projected onto one symbol from a finite vocabulary, and everything that
does not survive that projection is gone.

The nominal arithmetic is stark. A residual stream of ``d_model = 2048`` in bfloat16
nominally carries ``2048 x 16 = 32,768`` bits. A token drawn from a 32,768-entry
vocabulary carries at most ``log2(32768) = 15`` bits. That is a nominal compression of
about **2,000:1** per reasoning step.

But nominal numbers are wrong in both directions, which is why this module measures
rather than asserts:

- The state does **not** carry 32,768 bits of usable information. Its covariance is
  dominated by a small number of directions, so its *effective* dimensionality is far
  below its nominal width. This module measures that with the participation ratio.
- The token does **not** carry 15 bits. It carries the entropy of the distribution it was
  sampled from, which for a confident model mid-reasoning is often under 1 bit.

Both corrections matter, and they push in the same direction: the real ratio is what an
experiment has to report. The point of this module is that the claim becomes falsifiable.

The second measurement here is the one that matters for Prophet: whether **recurrent
depth widens the channel**. If looping the core raises the effective dimensionality of
the state that reaches the next token, latent depth is doing work that verbalised depth
cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "effective_rank",
    "token_entropy_bits",
    "state_bits",
    "ChannelMeasurement",
    "measure_channels",
    "BITS_PER_EFFECTIVE_DIM",
]

#: Bits of usable resolution per effective dimension. Deliberately conservative: models
#: run acceptably at 4-bit weights and 8-bit activations, so treating a dimension as
#: carrying 4 distinguishable bits understates the state's capacity rather than
#: flattering the argument this module exists to test.
BITS_PER_EFFECTIVE_DIM = 4.0


def effective_rank(states: Tensor, *, method: str = "participation") -> float:
    """Effective dimensionality of a set of state vectors.

    ``states`` is ``(n, d)``. Nominal width ``d`` overstates capacity badly: the
    covariance spectrum of a residual stream is dominated by a handful of directions, so
    the number of directions actually carrying variance is what should enter any
    information estimate.

    ``"participation"`` uses the participation ratio ``(sum l)^2 / sum l^2``, which is the
    standard effective-dimension measure and is robust to the long tail of tiny
    eigenvalues. ``"entropy"`` uses ``exp(H)`` of the normalised spectrum, which weights
    that tail more heavily.
    """
    if states.ndim != 2:
        raise ValueError(f"expected (n, d) states, got {tuple(states.shape)}")
    if states.shape[0] < 2:
        return 1.0

    x = states.float()
    x = x - x.mean(0, keepdim=True)
    # Singular values of the centred matrix; their squares are covariance eigenvalues.
    sv = torch.linalg.svdvals(x)
    eig = sv.pow(2)
    total = eig.sum()
    if total <= 0:
        return 1.0

    if method == "entropy":
        p = eig / total
        p = p[p > 0]
        return float(torch.exp(-(p * p.log()).sum()).item())
    return float((total.pow(2) / eig.pow(2).sum()).item())


def token_entropy_bits(logits: Tensor) -> float:
    """Mean entropy, in bits, of the next-token distributions.

    This is the information a sampled token actually carries — not ``log2(vocab)``, which
    is only reached by a uniform distribution. A model mid-reasoning is often confident,
    so the real channel is narrower than the vocabulary suggests.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    entropy_nats = -(logp.exp() * logp).sum(-1)
    return float((entropy_nats / math.log(2)).mean().item())


def state_bits(states: Tensor, *, bits_per_dim: float = BITS_PER_EFFECTIVE_DIM) -> float:
    """Conservative estimate of the information in a state vector, in bits."""
    return effective_rank(states) * bits_per_dim


@dataclass
class ChannelMeasurement:
    """Bandwidth of the latent and verbalised channels at one recurrence depth."""

    loop_k: int
    n_states: int
    d_model: int
    vocab_size: int

    effective_dims: float
    state_channel_bits: float
    token_channel_bits: float
    nominal_state_bits: float
    nominal_token_bits: float

    extra: dict[str, float] = field(default_factory=dict)

    @property
    def measured_ratio(self) -> float:
        """How many times wider the latent channel is than the verbalised one."""
        return self.state_channel_bits / max(self.token_channel_bits, 1e-9)

    @property
    def nominal_ratio(self) -> float:
        return self.nominal_state_bits / max(self.nominal_token_bits, 1e-9)

    @property
    def dimension_utilisation(self) -> float:
        """Fraction of the residual stream's width that carries variance."""
        return self.effective_dims / self.d_model

    def row(self) -> str:
        return (
            f"| {self.loop_k} | {self.effective_dims:.1f} | "
            f"{self.dimension_utilisation:.1%} | {self.state_channel_bits:.0f} | "
            f"{self.token_channel_bits:.2f} | {self.measured_ratio:.0f}x |"
        )


@torch.no_grad()
def measure_channels(
    model,
    input_ids: Tensor,
    *,
    loop_k: int | None = None,
    bits_per_dim: float = BITS_PER_EFFECTIVE_DIM,
) -> ChannelMeasurement:
    """Measure both channels on one batch.

    The latent channel is the final hidden state; the verbalised channel is the token
    distribution it produces. Measuring them on the *same* forward pass is what makes the
    ratio meaningful — it is the compression applied at that exact point.
    """
    model.eval()
    out = model(input_ids, loop_k=loop_k, return_mtp=False)

    hidden = out.hidden.reshape(-1, out.hidden.shape[-1])
    logits = out.logits.reshape(-1, out.logits.shape[-1])

    dims = effective_rank(hidden)
    vocab = logits.shape[-1]
    d_model = hidden.shape[-1]

    return ChannelMeasurement(
        loop_k=out.loop_k,
        n_states=hidden.shape[0],
        d_model=d_model,
        vocab_size=vocab,
        effective_dims=dims,
        state_channel_bits=dims * bits_per_dim,
        token_channel_bits=token_entropy_bits(logits),
        nominal_state_bits=d_model * 16.0,
        nominal_token_bits=math.log2(vocab),
    )


def report(measurements: list[ChannelMeasurement]) -> str:
    """Markdown table of channel bandwidth against recurrence depth."""
    if not measurements:
        return "no measurements"

    first = measurements[0]
    lines = [
        "# Reasoning channel bandwidth",
        "",
        f"Model width {first.d_model}, vocabulary {first.vocab_size}. "
        f"Nominal ratio (state bf16 bits / log2 vocab): **{first.nominal_ratio:.0f}x**.",
        "",
        "State bits are the measured effective dimensionality times "
        f"{BITS_PER_EFFECTIVE_DIM:.0f} bits, a deliberately conservative figure. Token "
        "bits are the measured entropy of the next-token distribution, not log2(vocab).",
        "",
        "| Loop k | Effective dims | Width used | State bits | Token bits | Ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(m.row() for m in measurements)

    if len(measurements) > 1:
        first_m, last_m = measurements[0], measurements[-1]
        change = last_m.effective_dims / max(first_m.effective_dims, 1e-9)
        lines += [
            "",
            f"Effective dimensionality changes by **{change:.2f}x** between k="
            f"{first_m.loop_k} and k={last_m.loop_k}.",
            "",
            "That factor is the question this measurement exists to answer: if deeper "
            "recurrence raises the effective dimensionality of the state reaching the "
            "next token, latent depth is doing work that verbalised depth cannot. If it "
            "does not, looping is buying serial steps only, and the wide-channel argument "
            "for it is wrong.",
        ]
    return "\n".join(lines)
