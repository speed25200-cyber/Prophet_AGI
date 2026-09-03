"""Compute allocation across research tracks.

Every track returned a budget request. Summed, they exceed the total budget several
times over — which is the normal outcome of asking twelve specialists what they need,
and is exactly why the allocation has to be an explicit artefact rather than an implicit
consequence of whatever gets run first.

This module states each request, ranks it, and produces an allocation that fits. It also
reports what was cut, because a plan that hides its omissions is not a plan.

Usage::

    python -m prophet.plan --a100-hours 300
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["Ask", "ASKS", "allocate", "plan_report"]

Kind = Literal["gate", "ablation", "production", "optional"]


@dataclass
class Ask:
    """One track's request for compute."""

    track: str
    name: str
    hours: float
    kind: Kind
    priority: int
    """1 is highest. Ties are broken by hours ascending, so cheap decisive work runs first."""
    rationale: str
    blocks: tuple[str, ...] = ()
    """Names of asks that are pointless if this one fails — a failed gate makes its
    dependants unnecessary, which is the whole reason to run gates first."""
    granted: float = field(default=0.0, init=False)

    @property
    def satisfied(self) -> bool:
        return self.granted >= self.hours - 1e-9


#: The requests, as returned by the twelve research tracks and trimmed to their
#: decision-relevant core. Hours are the tracks' own estimates.
ASKS: list[Ask] = [
    # --- gates: cheap experiments that can kill an expensive plan -------------------
    Ask("R04", "loop-vs-depth go/no-go", 24.0, "gate", 1,
        "Iso-FLOP comparison of looped depth against plain depth, plus depth "
        "generalisation. If looping does not beat equal-FLOP depth, the central "
        "architectural bet is dead and everything downstream changes.",
        blocks=("R04 depth ablations", "R03 two-tier memory")),
    Ask("R07", "optimiser bake-off (trimmed)", 14.0, "gate", 1,
        "Muon against a properly tuned AdamW. Break-even is a 1.058x speedup; below "
        "that the bake-off costs more than it saves and we keep AdamW.",),
    Ask("R01", "byte-frontend MFU probe", 2.0, "gate", 1,
        "Measure realised MFU of patch gather/scatter kernels. Below 0.18 the entire "
        "byte-level track is dead, and 2 hours settles it.",
        blocks=("R01 byte-frontend ablations",)),
    Ask("R09", "confidence-probe AUROC probe", 3.0, "gate", 2,
        "Every published confidence-probe result is at 7B or above. If AUROC at 0.6B "
        "is below 0.70 the head is dropped. No training required.",
        blocks=("R09 confidence head training",)),
    Ask("R02", "hybrid recall gate (MK-NIAH)", 8.0, "gate", 2,
        "Multi-key retrieval is where linear mixers collapse. Gate the interleave "
        "ratio on it before committing to the stack.",),

    # --- ablations that shape the production run ------------------------------------
    Ask("R06", "data mixture ablations", 20.0, "ablation", 2,
        "Mixture weights are the highest-leverage decision at our budget and the "
        "cheapest to test, at ~4 hours per arm on a 150M proxy."),
    Ask("R04", "depth ablations", 24.0, "ablation", 3,
        "Recurrence depth schedule, injection, truncation depth. Must run at >=350M: "
        "recursion underperforms vanilla at 135M and only wins from ~360M, so a 130M "
        "ablation would produce a false negative."),
    Ask("R02", "interleave and long-context ablations", 20.0, "ablation", 3,
        "Trimmed from the track's full 96-hour plan to the arms that change the stack."),
    Ask("R05", "MoE routing and upcycling", 16.0, "ablation", 3,
        "Router balance and the dense-to-sparse upcycling recipe."),
    Ask("R08", "quantisation ladder", 20.0, "ablation", 3,
        "Trimmed from 100-200 hours. Establishes how far the over-trained mini can be "
        "quantised before the on-device claim fails."),
    Ask("R03", "two-tier memory", 20.0, "ablation", 4,
        "Write, clear context, read — against a retrieval baseline at equal context "
        "budget. The differentiating bet, but unproven at any scale."),

    # --- production ------------------------------------------------------------------
    Ask("R06", "Prophet-mini pretraining", 90.0, "production", 1,
        "The dense 229M model. Trained from scratch, it is the honest existence proof "
        "of the architecture and the iPhone target."),
    Ask("R10", "post-training", 60.0, "production", 2,
        "Reasoning mid-training, dual-mode SFT, on-policy distillation, then RL. "
        "Trimmed from the track's 95-hour recipe. Distillation gets the compute; RL is "
        "polish -- on-policy distillation beat RL on every metric at a tenth the cost."),
    Ask("R02", "long-context extension", 12.0, "production", 3,
        "Costs 1.19x base FLOPs rather than the 7.9x a dense model would pay, because "
        "NoPE and bounded-state layers have nothing positional to relearn."),
    Ask("R11", "evaluation", 19.0, "production", 1,
        "Three-tier harness plus reproducing four competitor baselines in our own "
        "harness. Without it every other number is unverifiable."),

    # --- optional --------------------------------------------------------------------
    Ask("R12", "vision adapter", 26.0, "optional", 5,
        "Vision adds nothing to the benchmarks that decide our win condition. Only if "
        "the budget survives everything above."),
    Ask("R01", "byte-frontend retrofit", 36.0, "optional", 5,
        "Retrofit onto a finished checkpoint, gated on the MFU probe."),
    Ask("R09", "confidence head training", 20.0, "optional", 4,
        "Gated on the AUROC probe."),
]


def allocate(total_hours: float, *, reserve_frac: float = 0.10) -> tuple[list[Ask], dict]:
    """Greedily fund asks by (priority, cost) and report the shortfall.

    A reserve is held back on purpose. Multi-week runs on preemptible hardware lose time
    to reruns, and a plan with no slack is a plan that fails on its first bad week.
    """
    reserve = total_hours * reserve_frac
    available = total_hours - reserve

    asks = [Ask(**{k: v for k, v in a.__dict__.items() if k != "granted"}) for a in ASKS]
    ordered = sorted(asks, key=lambda a: (a.priority, a.hours))

    spent = 0.0
    for ask in ordered:
        if spent + ask.hours <= available:
            ask.granted = ask.hours
            spent += ask.hours

    unfunded = [a for a in asks if not a.satisfied]
    summary = {
        "total_hours": total_hours,
        "reserve_hours": reserve,
        "available_hours": available,
        "allocated_hours": spent,
        "unallocated_hours": available - spent,
        "requested_hours": sum(a.hours for a in asks),
        "oversubscription": sum(a.hours for a in asks) / max(available, 1),
        "funded": len(asks) - len(unfunded),
        "total_asks": len(asks),
    }
    return asks, summary


def plan_report(total_hours: float = 300.0, *, reserve_frac: float = 0.10) -> str:
    asks, s = allocate(total_hours, reserve_frac=reserve_frac)

    lines = [
        f"# Compute plan — {total_hours:.0f} A100-hours",
        "",
        f"Requested across all tracks: **{s['requested_hours']:.0f} h**. "
        f"Available after a {reserve_frac:.0%} reserve: **{s['available_hours']:.0f} h**. "
        f"Oversubscribed **{s['oversubscription']:.1f}x**.",
        "",
        f"Funded {s['funded']} of {s['total_asks']} requests, "
        f"{s['allocated_hours']:.0f} h allocated, "
        f"{s['unallocated_hours']:.0f} h unspent, "
        f"{s['reserve_hours']:.0f} h held in reserve for reruns and preemption losses.",
        "",
        "## Funded",
        "",
        "| Pri | Track | Work | Kind | Hours |",
        "|---:|---|---|---|---:|",
    ]
    for a in sorted([a for a in asks if a.satisfied], key=lambda a: (a.priority, -a.hours)):
        lines.append(f"| {a.priority} | {a.track} | {a.name} | {a.kind} | {a.hours:.0f} |")

    unfunded = [a for a in asks if not a.satisfied]
    lines += ["", "## Not funded", ""]
    if unfunded:
        lines += ["| Pri | Track | Work | Hours | Why it was cut |", "|---:|---|---|---:|---|"]
        for a in sorted(unfunded, key=lambda a: (a.priority, -a.hours)):
            lines.append(
                f"| {a.priority} | {a.track} | {a.name} | {a.hours:.0f} | "
                f"{'optional; below the funding line' if a.kind == 'optional' else 'ranked below the funding line'} |"
            )
    else:
        lines.append("Everything requested is funded.")

    gates = [a for a in asks if a.kind == "gate"]
    lines += [
        "",
        "## Gates run first",
        "",
        f"{sum(g.hours for g in gates):.0f} hours of gate experiments "
        f"({sum(g.hours for g in gates) / total_hours:.0%} of the budget) decide whether "
        "the expensive work is worth doing at all:",
        "",
    ]
    for g in sorted(gates, key=lambda a: a.priority):
        lines.append(f"- **{g.track} — {g.name}** ({g.hours:.0f} h). {g.rationale}")
        if g.blocks:
            lines.append(f"  - Failure cancels: {', '.join(g.blocks)}")
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prophet compute allocation")
    ap.add_argument("--a100-hours", type=float, default=300.0)
    ap.add_argument("--reserve", type=float, default=0.10)
    args = ap.parse_args()
    print(plan_report(args.a100_hours, reserve_frac=args.reserve))


if __name__ == "__main__":
    _main()
