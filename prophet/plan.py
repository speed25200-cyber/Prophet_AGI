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
    """1 is highest. Ties are broken by **declaration order**, not by cost.

    Cheapest-first looks efficient and is not: it funds three small requests ahead of the
    one large request the deliverable depends on, and the shortfall lands on whatever is
    biggest rather than on whatever is least important. Within a priority band the order
    in :data:`ASKS` is the intended order, so it is the order used.
    """
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
#:
#: Two entries carry an explicit project decision rather than a track's recommendation:
#: the mixed path (train the mini from scratch, convert a donor for the main model), and
#: the promotion of persistent memory to priority 2.
ASKS: list[Ask] = [
    # --- gates: cheap experiments that can kill an expensive plan -------------------
    Ask("R01", "byte-frontend MFU probe", 2.0, "gate", 1,
        "Measure realised MFU of patch gather/scatter kernels. Below 0.18 the entire "
        "byte-level track is dead, and 2 hours settles it.",
        blocks=("R01 byte-frontend retrofit",)),
    Ask("R07", "optimiser bake-off (trimmed)", 14.0, "gate", 1,
        "Muon against a properly tuned AdamW. Break-even is a 1.058x speedup; below "
        "that the bake-off costs more than it saves and we keep AdamW."),
    Ask("W4", "accuracy-versus-depth sweep", 1.0, "gate", 1,
        "Does accuracy actually rise with recurrence depth? One hour settles it. Our own "
        "R04 puts latent depth at ~1.8 GSM8K points against ~33 for verbalised CoT; if the "
        "sweep is flat there is nothing for depth consolidation to store and that track "
        "closes before it costs anything.",
        blocks=("W4 depth consolidation",)),
    Ask("W2", "multi-key recall versus state and depth", 1.0, "gate", 1,
        "The sharpest test of the bounded-state bet. R02 already measured 89.8% "
        "single-needle against 37.8% multi-needle; W2 adds that the effective "
        "linear-to-attention ratio degrades from 1:1 at k=1 to 8:1 at k=8, so raising the "
        "depth dial makes the known weakness worse. One hour shows whether the depth dial "
        "and the recall budget are the same dial pulling opposite ways."),
    Ask("R04", "loop-vs-depth go/no-go", 24.0, "gate", 1,
        "Iso-FLOP comparison of looped depth against plain depth, plus depth "
        "generalisation, at >=350M parameters. If looping does not beat equal-FLOP "
        "depth, the central architectural bet is dead and everything downstream changes.",
        blocks=("R04 depth ablations", "R03 two-tier memory")),

    # --- production: the deliverables themselves --------------------------------------
    Ask("R11", "evaluation", 19.0, "production", 1,
        "Three-tier harness plus reproducing four competitor baselines in our own "
        "harness. Without it every other number is unverifiable."),
    Ask("R06", "Prophet-mini pretraining", 85.0, "production", 1,
        "The dense 229M model, trained from random initialisation. It is the honest "
        "existence proof of the architecture and the iPhone target, and the half of the "
        "mixed path that owes nothing to anyone else's pretraining."),
    Ask("R02", "Prophet-main donor conversion", 30.0, "production", 1,
        "Convert an open Apache-2.0 donor into the Prophet architecture and train to "
        "recover. At ~89% parameter coverage this is recovery, not pretraining, which is "
        "why it costs a fraction of what the from-scratch mini needs and is the only way "
        "the compute arithmetic permits a competitive main model."),
    Ask("R10", "post-training", 45.0, "production", 1,
        "Reasoning mid-training, dual-mode SFT, then on-policy distillation. Priority 1 "
        "because a base model is not a deliverable. Trimmed from the track's 95 hours by "
        "dropping the RL polish stage on R10's own evidence: on-policy distillation beat "
        "RL on every metric at 1,800 GPU-hours against 17,920."),

    # --- second tier: gates and ablations that shape the production runs ---------------
    Ask("R09", "confidence-probe AUROC probe", 3.0, "gate", 2,
        "Every published confidence-probe result is at 7B or above. If AUROC at 0.6B "
        "is below 0.70 the head is dropped. No training required.",
        blocks=("R09 confidence head training",)),
    Ask("R02", "hybrid recall gate (MK-NIAH)", 8.0, "gate", 2,
        "Multi-key retrieval is where linear mixers collapse. Gate the interleave "
        "ratio on it before committing to the stack."),
    Ask("R06", "data mixture ablations", 18.0, "ablation", 2,
        "Mixture weights are the highest-leverage decision at our budget and the cheapest "
        "to test, at ~4 hours per arm on a 150M proxy. Trimmed from 20 hours by one arm: "
        "persistent memory landed two hours short of the line, and one mixture arm is a "
        "smaller loss than the project's most differentiating bet."),
    Ask("R03", "two-tier memory", 20.0, "ablation", 2,
        "Write, clear context, read -- against a retrieval baseline at equal context "
        "budget. Unproven at any scale, and promoted to priority 2 by an explicit "
        "project decision: it is the one capability no competitor has, and the donor "
        "conversion frees the hours that fund it."),

    # --- third tier: fund from the reserve or from a cancelled gate --------------------
    Ask("W1", "halting: input-dependent depth", 12.0, "ablation", 3,
        "A constant k changes no complexity class, so the loop earns no asymptotic credit "
        "at all -- only input-dependent depth does. The mechanism is implemented and "
        "tested; this trains it and measures whether learned depth beats a fixed dial at "
        "equal average compute. Ranked below persistent memory by explicit project "
        "decision: the loop's constant-factor benefit survives without halting, so this "
        "buys the asymptotic argument rather than the working model. First in line if a "
        "gate frees hours."),
    Ask("R04", "depth ablations", 24.0, "ablation", 3,
        "Recurrence depth schedule, injection, truncation depth. Must run at >=350M: "
        "recursion underperforms vanilla at 135M and only wins from ~360M, so a 130M "
        "ablation would produce a false negative."),
    Ask("R08", "quantisation ladder", 20.0, "ablation", 3,
        "Trimmed from 100-200 hours. Establishes how far the over-trained mini can be "
        "quantised before the on-device claim fails."),
    Ask("R02", "interleave and long-context ablations", 20.0, "ablation", 3,
        "Trimmed from the track's full 96-hour plan to the arms that change the stack."),
    Ask("R05", "MoE routing and upcycling", 16.0, "ablation", 3,
        "Router balance and the dense-to-sparse upcycling recipe."),
    Ask("R02", "long-context extension", 12.0, "production", 3,
        "Costs 1.19x base FLOPs rather than the 7.9x a dense model would pay, because "
        "NoPE and bounded-state layers have nothing positional to relearn."),
    Ask("A2", "per-token depth ceilings versus one depth per sequence", 4.0, "ablation", 3,
        "The agent loop reads observations at depth 1 and thinks deep on one cache, "
        "which is only defined for a model trained with per-token ceilings "
        "(recurrent.token_depth). Mechanically exact and tested; what is unknown is the "
        "quality cost. Two matched 100M runs at ~1B tokens, ~2 hours each. If BPB "
        "degrades by more than 1% the loop runs in its fixed-depth regime instead.",
        blocks=("A2 agentic training recipe",)),
    Ask("A4", "depth-disagreement AUROC probe", 1.0, "ablation", 3,
        "Does disagreement between a shallow and a deep pass predict error? Inference "
        "only, on the mini checkpoint, over the tier-1 suite. Below 0.65 the signal is "
        "dropped from the verifier's feature vector."),
    Ask("A2", "agentic training recipe", 67.0, "production", 3,
        "Tool-use SFT with omission and null-action negatives, then on-policy "
        "distillation on executable tasks with the quarantine's promoted episodes as a "
        "replay stream. The track's own estimate. Unfunded at 300 hours: it would "
        "displace persistent memory, which an explicit project decision ranks first.",
        blocks=()),

    # --- optional ---------------------------------------------------------------------
    Ask("W4", "depth consolidation", 14.0, "optional", 4,
        "Gated on the accuracy-versus-depth sweep. Also needs a learned addressing key: "
        "probed as built, same-class and different-class slot overlap are at chance, so "
        "the ledger memorises the consolidated instance and nothing near it."),
    Ask("R09", "confidence head training", 20.0, "optional", 4,
        "Gated on the AUROC probe."),
    Ask("R12", "vision adapter", 26.0, "optional", 5,
        "Vision adds nothing to the benchmarks that decide our win condition. Only if "
        "the budget survives everything above."),
    Ask("R01", "byte-frontend retrofit", 36.0, "optional", 5,
        "Retrofit onto a finished checkpoint, gated on the MFU probe."),
]


def allocate(total_hours: float, *, reserve_frac: float = 0.10) -> tuple[list[Ask], dict]:
    """Greedily fund asks by (priority, cost) and report the shortfall.

    A reserve is held back on purpose. Multi-week runs on preemptible hardware lose time
    to reruns, and a plan with no slack is a plan that fails on its first bad week.
    """
    reserve = total_hours * reserve_frac
    available = total_hours - reserve

    asks = [Ask(**{k: v for k, v in a.__dict__.items() if k != "granted"}) for a in ASKS]
    ordered = sorted(enumerate(asks), key=lambda pair: (pair[1].priority, pair[0]))
    ordered = [ask for _, ask in ordered]

    # Strict order, no backfill. Skipping an item that does not fit and funding a cheaper,
    # lower-priority one behind it is the knapsack answer, not the planning answer: it
    # trades the thing the plan depends on for two things it does not. The allocator
    # therefore stops at the first item that does not fit and reports the remainder as
    # slack, which joins the reserve.
    spent = 0.0
    for ask in ordered:
        if spent + ask.hours > available:
            break
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
        f"{s['unallocated_hours']:.0f} h unspent (added to the reserve), "
        f"{s['reserve_hours']:.0f} h held for reruns and preemption losses.",
        "",
        "Allocation is in strict priority order with no backfill: an item that does not "
        "fit stops the line rather than being skipped so that cheaper work behind it can "
        "squeeze in.",
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
                f"{'optional; below the funding line' if a.kind == 'optional' else 'below the funding line'} |"
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
