"""The three-tier evaluation harness.

Track R11's design, and the reason for each tier:

- **Tier-0 (smoke, <5 min, every checkpoint).** Is the model still producing coherent
  output? Catches a divergence in minutes rather than at the end of a day.
- **Tier-1 (ablation, ~16 min, every few days).** The early-signal suite. Decides
  ablations, on held-out bits-per-byte rather than accuracy.
- **Tier-2 (release, hours, four times in the project).** The full suite, with three
  benchmarks held in reserve and never run before the pre-registered final evaluation.

Reserving benchmarks is a deliberate constraint. Seventeen planned ablations offer ample
opportunity to overfit the scoreboard without noticing, and the only defence is not
looking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn

from prophet.eval.metrics import BPBResult, bits_per_byte, cross_entropy_nats, is_above_chance

__all__ = ["EvalTask", "EvalResult", "EvalSuite", "TIER0", "TIER1", "RESERVED", "run_suite"]


@dataclass
class EvalTask:
    """One evaluation task.

    ``min_params`` and ``min_tokens`` record the scale below which the task carries no
    signal. The harness runs it anyway and marks the result as uninformative, which is
    more useful than silently omitting it — a missing row looks like an oversight, a
    flagged row looks like what it is.
    """

    name: str
    kind: str
    """``"bpb"`` or ``"multiple_choice"``."""
    n_choices: int = 0
    min_params: float = 0.0
    min_tokens: float = 0.0
    decides_ablations: bool = False
    reserved: bool = False
    """Held back from every run before the pre-registered final evaluation."""
    notes: str = ""


#: Tier-0: is the model alive.
TIER0: list[EvalTask] = [
    EvalTask("held_out_bpb", "bpb", decides_ablations=True,
             notes="Loss on data the model has never seen. Any divergence shows here first."),
]

#: Tier-1: the early-signal ablation suite. BPB decides; accuracy is context.
TIER1: list[EvalTask] = [
    EvalTask("bpb_web", "bpb", decides_ablations=True),
    EvalTask("bpb_code", "bpb", decides_ablations=True),
    EvalTask("bpb_math", "bpb", decides_ablations=True),
    EvalTask("bpb_reference", "bpb", decides_ablations=True),
    EvalTask("lambada_openai", "multiple_choice", n_choices=0, min_params=1e8,
             notes="Generative; above chance early, unlike most of the suite."),
    EvalTask("sciq", "multiple_choice", n_choices=4, min_params=1e8),
    EvalTask("piqa", "multiple_choice", n_choices=2, min_params=1e8),
    EvalTask("arc_easy", "multiple_choice", n_choices=4, min_params=1.5e8),
    EvalTask("hellaswag", "multiple_choice", n_choices=4, min_params=2e8),
    EvalTask("social_iqa", "multiple_choice", n_choices=3, min_params=2e8),
    EvalTask("openbookqa", "multiple_choice", n_choices=4, min_params=3e8,
             notes="Noisy at our scale; read with caution."),
    # Below: at chance for accuracy at ablation scale, so scored by BPB only.
    EvalTask("arc_challenge", "bpb", min_params=5e8),
    EvalTask("commonsense_qa", "bpb", min_params=5e8),
    EvalTask("winogrande", "bpb", min_params=5e8),
    EvalTask("mmlu_continuation", "bpb", min_params=5e8),
]

#: Never run before the final pre-registered evaluation.
RESERVED: list[EvalTask] = [
    EvalTask("mmlu_pro", "multiple_choice", n_choices=10, min_params=1e9, reserved=True),
    EvalTask("gpqa_diamond", "multiple_choice", n_choices=4, min_params=1e9, reserved=True),
    EvalTask("arena_hard_v2", "multiple_choice", min_params=1e9, reserved=True),
]

#: Excluded from ablations entirely, with the reason recorded so the decision is not
#: relitigated every time someone notices they are missing.
EXCLUDED: dict[str, str] = {
    "truthfulqa": "inverse scaling — an improving model scores worse, so it misleads ablations",
    "boolq": "dominated by the majority-class prior; measures calibration to a prior, not ability",
    "gsm8k": "floors at 0.0 at ablation scale, so it carries no information",
    "humaneval": "floors at 0.0 at ablation scale, so it carries no information",
}


@dataclass
class EvalResult:
    task: str
    value: float
    metric: str
    informative: bool
    seconds: float
    detail: dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass
class EvalSuite:
    name: str
    tasks: list[EvalTask]
    results: list[EvalResult] = field(default_factory=list)

    def report(self, *, model_params: float = 0.0) -> str:
        lines = [
            f"# {self.name}",
            "",
            "| Task | Metric | Value | Informative | Decides ablations |",
            "|---|---|---:|---|---|",
        ]
        for r in self.results:
            task = next((t for t in self.tasks if t.name == r.task), None)
            decides = "yes" if task and task.decides_ablations else ""
            lines.append(
                f"| {r.task} | {r.metric} | {r.value:.4f} | "
                f"{'yes' if r.informative else 'NO — at or below chance'} | {decides} |"
            )
        uninformative = [r for r in self.results if not r.informative]
        if uninformative:
            lines += [
                "",
                f"{len(uninformative)} of {len(self.results)} tasks are not informative at "
                f"{model_params / 1e6:.0f}M parameters. They are reported rather than "
                "hidden, but they must not be used to decide anything.",
            ]
        return "\n".join(lines)

    def decision_metric(self) -> float | None:
        """Mean bits-per-byte over the tasks that are allowed to decide ablations."""
        deciding = {t.name for t in self.tasks if t.decides_ablations}
        values = [r.value for r in self.results if r.task in deciding and r.metric == "bpb"]
        return sum(values) / len(values) if values else None


@torch.no_grad()
def evaluate_bpb(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, int]],
    *,
    domain: str = "",
    device: torch.device | str = "cpu",
) -> BPBResult:
    """Bits-per-byte over a stream of ``(token_ids, byte_count)`` batches."""
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    for tokens, n_bytes in batches:
        tokens = tokens.to(device)
        output = model(tokens, return_mtp=False)
        logits = output.logits if hasattr(output, "logits") else output
        nats, counted = cross_entropy_nats(logits, tokens)
        total_nats += nats
        total_tokens += counted
        total_bytes += n_bytes
    return bits_per_byte(total_nats, total_tokens, total_bytes, domain=domain)


def run_suite(
    suite_tasks: Sequence[EvalTask],
    runners: dict[str, Callable[[], tuple[float, str, dict[str, float]]]],
    *,
    model_params: float,
    model_tokens: float = 0.0,
    name: str = "eval",
    skip_reserved: bool = True,
) -> EvalSuite:
    """Run the tasks that have a runner, marking under-scale results as uninformative.

    ``skip_reserved`` defaults to True and should only be turned off for the final
    pre-registered evaluation.
    """
    suite = EvalSuite(name=name, tasks=list(suite_tasks))
    for task in suite_tasks:
        if task.reserved and skip_reserved:
            continue
        runner = runners.get(task.name)
        if runner is None:
            continue

        start = time.time()
        value, metric, detail = runner()
        elapsed = time.time() - start

        informative = True
        note = ""
        if model_params and model_params < task.min_params:
            informative = False
            note = f"model is {model_params / 1e6:.0f}M, task needs {task.min_params / 1e6:.0f}M"
        elif model_tokens and model_tokens < task.min_tokens:
            informative = False
            note = "trained on too few tokens for this task to de-noise"
        elif task.kind == "multiple_choice" and task.n_choices > 1 and metric.startswith("acc"):
            n_items = int(detail.get("n", 0))
            if n_items and not is_above_chance(value, n_items, task.n_choices):
                informative = False
                note = f"not distinguishable from {1 / task.n_choices:.1%} chance"

        suite.results.append(
            EvalResult(task.name, value, metric, informative, elapsed, detail, note)
        )
    return suite
