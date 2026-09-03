"""Choosing Prophet's operating point under a fixed A100-hour budget.

The single most consequential decision in this project is not architectural — it is
picking (parameters, tokens) given a compute budget that is three orders of magnitude
below the competition. Getting this wrong wastes the entire budget on a model that is
either too big to train to convergence or too small to be interesting.

This module makes that decision explicit and quantitative. It is deliberately
dependency-free so it can be run and argued with before anything is installed.

Usage::

    python -m prophet.scaling --a100-hours 300
    python -m prophet.scaling --sweep
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ScalingLaw",
    "CHINCHILLA",
    "OperatingPoint",
    "compute_from_hours",
    "tokens_for",
    "chinchilla_optimal",
    "candidates",
    "plan_table",
]

A100_BF16_TFLOPS = 312.0
SECONDS_PER_HOUR = 3600.0


# --------------------------------------------------------------------------------------
# Loss model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingLaw:
    """A Hoffmann-style decomposition ``L(N, D) = E + A/N^alpha + B/D^beta``.

    The absolute loss values are only comparable within one tokenizer and corpus, so
    treat them as an *ordering* over configurations rather than a prediction. What the
    law is genuinely reliable for is the shape of the tradeoff: how much loss is paid
    for shrinking the model versus shortening training.
    """

    E: float = 1.69
    A: float = 406.4
    alpha: float = 0.34
    B: float = 410.7
    beta: float = 0.28
    name: str = "Chinchilla (Hoffmann et al. 2022)"

    def loss(self, n_params: float, n_tokens: float) -> float:
        return self.E + self.A / n_params**self.alpha + self.B / n_tokens**self.beta

    def optimal_ratio(self) -> float:
        """Tokens per parameter at the compute-optimal point (~20 for Chinchilla)."""
        return 20.0


CHINCHILLA = ScalingLaw()


# --------------------------------------------------------------------------------------
# Compute accounting
# --------------------------------------------------------------------------------------


def compute_from_hours(a100_hours: float, mfu: float = 0.35) -> float:
    """Usable training FLOPs from a wall-clock A100 budget."""
    return A100_BF16_TFLOPS * 1e12 * mfu * a100_hours * SECONDS_PER_HOUR


def tokens_for(n_active_params: float, flops: float, flops_per_param_token: float = 6.0) -> float:
    """Tokens affordable for a model with ``n_active_params`` active parameters."""
    return flops / (flops_per_param_token * n_active_params)


def chinchilla_optimal(flops: float, law: ScalingLaw = CHINCHILLA) -> tuple[float, float]:
    """The compute-optimal ``(params, tokens)`` split of a FLOP budget.

    Solving ``C = 6 N D`` under ``D = 20 N`` gives ``N = sqrt(C / 120)``.
    """
    n = (flops / (6.0 * law.optimal_ratio())) ** 0.5
    return n, law.optimal_ratio() * n


# --------------------------------------------------------------------------------------
# Operating points
# --------------------------------------------------------------------------------------


@dataclass
class OperatingPoint:
    n_params: float
    """Active (dense-equivalent) parameters."""
    n_tokens: float
    tokens_per_param: float
    flops: float
    predicted_loss: float
    a100_hours: float
    label: str = ""

    def inference_gb_int4(self) -> float:
        return self.n_params * 0.53 / 1024**3


def candidates(
    a100_hours: float,
    *,
    mfu: float = 0.35,
    law: ScalingLaw = CHINCHILLA,
    param_grid: tuple[float, ...] = (
        1.5e8, 2.5e8, 3.5e8, 5e8, 7e8, 1e9, 1.4e9, 2e9, 3e9, 4e9,
    ),
) -> list[OperatingPoint]:
    """All (params, tokens) points reachable with the budget, ordered by predicted loss.

    Every point spends the *whole* budget; they differ only in how it is split between
    model size and training length.
    """
    flops = compute_from_hours(a100_hours, mfu)
    out: list[OperatingPoint] = []
    for n in param_grid:
        d = tokens_for(n, flops)
        out.append(
            OperatingPoint(
                n_params=n,
                n_tokens=d,
                tokens_per_param=d / n,
                flops=flops,
                predicted_loss=law.loss(n, d),
                a100_hours=a100_hours,
            )
        )
    out.sort(key=lambda p: p.predicted_loss)
    if out:
        out[0].label = "compute-optimal"
    # Mark the point we would actually ship: over-training buys a smaller, faster model
    # at a small loss penalty, which is the right trade when inference memory is the
    # binding constraint (Sardana et al., arXiv 2401.00448).
    best = out[0].predicted_loss
    for p in out:
        if p.n_params < out[0].n_params and p.predicted_loss <= best + 0.05:
            p.label = "inference-optimal (<=0.05 loss penalty, smaller & faster)"
            break
    return out


def _fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.1f}"


def plan_table(a100_hours: float, *, mfu: float = 0.35) -> str:
    """Markdown report of the operating points reachable with a budget."""
    flops = compute_from_hours(a100_hours, mfu)
    n_opt, d_opt = chinchilla_optimal(flops)
    pts = candidates(a100_hours, mfu=mfu)

    lines = [
        f"## Budget: {a100_hours:.0f} A100-hours @ {mfu:.0%} MFU",
        "",
        f"Total training compute: **{flops:.2e} FLOPs**",
        f"Compute-optimal point: **{_fmt(n_opt)} params × {_fmt(d_opt)} tokens**",
        "",
        "| Active params | Tokens | Tok/param | Pred. loss | int4 size | Note |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for p in pts:
        lines.append(
            f"| {_fmt(p.n_params)} | {_fmt(p.n_tokens)} | {p.tokens_per_param:.0f} | "
            f"{p.predicted_loss:.3f} | {p.inference_gb_int4():.2f} GB | {p.label} |"
        )
    return "\n".join(lines)


REFERENCE_MODELS: dict[str, tuple[float, float]] = {
    # name: (params, training tokens) — public figures, for calibrating ambition.
    "SmolLM2-360M": (3.6e8, 4e12),
    "Qwen3-0.6B": (6e8, 3.6e13),
    "Llama-3.2-1B": (1.24e9, 9e12),
    "SmolLM2-1.7B": (1.7e9, 1.1e13),
    "Qwen3-1.7B": (1.7e9, 3.6e13),
    "Gemma-3-1B": (1.0e9, 2e12),
    "SmolLM3-3B": (3.1e9, 1.1e13),
    "Qwen3-4B": (4e9, 3.6e13),
}


def reference_table() -> str:
    """What the competition spent, so our own numbers are read in context."""
    lines = [
        "## What the competition spent",
        "",
        "| Model | Params | Train tokens | Tok/param | Train FLOPs | A100-hours @35% MFU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, (n, d) in REFERENCE_MODELS.items():
        flops = 6 * n * d
        hours = flops / (A100_BF16_TFLOPS * 1e12 * 0.35 * SECONDS_PER_HOUR)
        lines.append(
            f"| {name} | {_fmt(n)} | {_fmt(d)} | {d / n:.0f} | {flops:.2e} | {hours:,.0f} |"
        )
    return "\n".join(lines)


def gap_analysis(a100_hours: float, mfu: float = 0.35) -> str:
    """State the compute gap plainly, and name what it does and does not rule out."""
    ours = compute_from_hours(a100_hours, mfu)
    lines = [
        "## The gap, stated plainly",
        "",
        "| Model | Their FLOPs | Ours | Ratio |",
        "|---|---:|---:|---:|",
    ]
    for name, (n, d) in REFERENCE_MODELS.items():
        theirs = 6 * n * d
        lines.append(f"| {name} | {theirs:.2e} | {ours:.2e} | **{theirs / ours:,.0f}×** |")
    lines += [
        "",
        "Two conclusions follow, and both are load-bearing for the whole project:",
        "",
        "1. **Knowledge-heavy benchmarks are not winnable by pretraining.** MMLU-style",
        "   coverage is bought with tokens, and we are short by two to three orders of",
        "   magnitude. Attacking it head-on wastes the budget.",
        "2. **Capability-per-parameter and capability-per-byte are winnable.** Distillation",
        "   from open teachers, reasoning depth bought at inference time, and a data",
        "   mixture tuned for a small budget are all levers that do *not* scale with the",
        "   competitor's cluster size. That is where the budget goes.",
    ]
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prophet scaling planner")
    ap.add_argument("--a100-hours", type=float, default=300.0)
    ap.add_argument("--mfu", type=float, default=0.35)
    ap.add_argument("--sweep", action="store_true", help="show several budgets")
    args = ap.parse_args()

    print(reference_table())
    print()
    if args.sweep:
        for h in (100.0, 300.0, 1000.0, 3000.0):
            print(plan_table(h, mfu=args.mfu))
            print()
    else:
        print(plan_table(args.a100_hours, mfu=args.mfu))
        print()
    print(gap_analysis(args.a100_hours, args.mfu))


if __name__ == "__main__":
    _main()
