"""Shipped, reviewed data recipes."""

from __future__ import annotations

from typing import Any

from prophet.data.mixture import Mixture, Phase, Source

__all__ = ["prophet_v1_mixture"]

B = 1e9


def _source(
    name: str,
    hf_id: str,
    domain: str,
    weight: float,
    available_tokens: float | None,
    license: str,
    *,
    config: str | None = None,
    filters: dict[str, Any] | None = None,
    notes: str = "",
) -> Source:
    return Source(
        name=name,
        hf_id=hf_id,
        domain=domain,
        weight=weight,
        available_tokens=available_tokens,
        license=license,
        config=config,
        filters=dict(filters or {}),
        notes=notes,
    )


def prophet_v1_mixture(total_tokens: float = 40 * B) -> Mixture:
    """Return the three-phase R06 recipe, rescaled to ``total_tokens``.

    Only the absolute budget changes.  Phase and source proportions remain identical so
    that a probe run and a full run exercise the same research decision.
    """

    nvidia = "NVIDIA Open Data (permissive)"
    odc = "ODC-By-1.0"
    apache = "Apache-2.0"

    phase_a = Phase(
        name="A-stable",
        weight=0.70,
        lr_schedule="warmup_then_constant",
        context_len=4096,
        purpose=(
            "Build the world model. Broadest mixture, highest token volume, constant "
            "peak learning rate. No instruction data at all — it is deliberately saved "
            "for the phases where recency makes it count."
        ),
        sources=[
            _source(
                "nemotron-cc-v2-hq",
                "nvidia/Nemotron-CC-v2",
                "web",
                0.22,
                1000 * B,
                nvidia,
                notes="R06 D2: ranks above DCLM and FineWeb-Edu on downstream MMLU.",
            ),
            _source(
                "fineweb-edu",
                "HuggingFaceFW/fineweb-edu",
                "web",
                0.18,
                1300 * B,
                odc,
                filters={"edu_score_min": 3},
                notes="Kept for decorrelation from the Nemotron pipeline.",
            ),
            _source(
                "dclm-baseline",
                "mlfoundations/dclm-baseline-1.0",
                "web",
                0.14,
                2600 * B,
                "CC-BY-4.0",
            ),
            _source(
                "nemotron-cc-synthetic",
                "nvidia/Nemotron-CC-v2",
                "synthetic",
                0.08,
                800 * B,
                nvidia,
                config="synthetic",
                notes="Diverse-QA / Distill / Extract-Knowledge.",
            ),
            _source(
                "stack-edu",
                "HuggingFaceTB/stack-edu",
                "code",
                0.09,
                125 * B,
                odc,
            ),
            _source(
                "nemotron-code",
                "nvidia/Nemotron-Pretraining-Code-v2",
                "code",
                0.05,
                None,
                nvidia,
            ),
            _source(
                "nemotron-cc-math",
                "nvidia/Nemotron-CC-Math-v1",
                "math",
                0.06,
                52 * B,
                nvidia,
                config="4plus",
            ),
            _source(
                "megamath-finemath",
                "LLM360/MegaMath",
                "math",
                0.03,
                25 * B,
                odc,
            ),
            _source(
                "cosmopedia-v2",
                "HuggingFaceTB/smollm-corpus",
                "synthetic",
                0.04,
                28 * B,
                apache,
                config="cosmopedia-v2",
            ),
            _source(
                "curated-reference",
                "allenai/dolma3",
                "reference",
                0.05,
                None,
                odc,
                notes="Wikipedia / StackExchange / books / science. Config IDs unverified.",
            ),
            _source(
                "proof-pile-2",
                "EleutherAI/proof-pile-2",
                "reference",
                0.03,
                55 * B,
                "REVIEW: mixed (per-subset)",
            ),
            _source(
                "fineweb2-hq",
                "epfml/FineWeb2-HQ",
                "multilingual",
                0.03,
                1000 * B,
                odc,
                notes="Top 6 languages only; capped per R06 D5.",
            ),
        ],
    )

    phase_b = Phase(
        name="B-midtrain",
        weight=0.20,
        lr_schedule="hold_then_slow_decay",
        context_len=16384,
        purpose=(
            "Capability injection: math, code and reasoning format, while the model is "
            "still plastic. Context is extended 4k -> 16k over the last third of the phase."
        ),
        sources=[
            _source(
                "nemotron-cc-math-up",
                "nvidia/Nemotron-CC-Math-v1",
                "math",
                0.14,
                52 * B,
                nvidia,
                config="4plus",
            ),
            _source(
                "megamath-finemath",
                "LLM360/MegaMath",
                "math",
                0.08,
                25 * B,
                odc,
            ),
            _source(
                "stack-edu-anneal",
                "HuggingFaceTB/stack-edu",
                "code",
                0.16,
                125 * B,
                odc,
                filters={"edu_score_min": 4},
            ),
            _source(
                "nemotron-cc-v2-hq",
                "nvidia/Nemotron-CC-v2",
                "web",
                0.16,
                1000 * B,
                nvidia,
            ),
            _source(
                "fineweb-edu-4",
                "HuggingFaceFW/fineweb-edu",
                "web",
                0.10,
                350 * B,
                odc,
                filters={"edu_score_min": 4},
            ),
            _source(
                "nemotron-diverse-qa",
                "nvidia/Nemotron-CC-v2",
                "synthetic",
                0.10,
                400 * B,
                nvidia,
                config="diverse-qa",
            ),
            _source(
                "cosmopedia-v2",
                "HuggingFaceTB/smollm-corpus",
                "synthetic",
                0.06,
                28 * B,
                apache,
                config="cosmopedia-v2",
            ),
            _source(
                "curated-reference",
                "allenai/dolma3",
                "reference",
                0.05,
                None,
                odc,
            ),
            _source(
                "instruction-as-documents",
                "HuggingFaceTB/smoltalk2",
                "instruction",
                0.08,
                None,
                apache,
                notes="Rendered as pretraining documents, not chat turns.",
            ),
            _source(
                "long-documents",
                "EleutherAI/proof-pile-2",
                "long_context",
                0.05,
                55 * B,
                "REVIEW: mixed (per-subset)",
                filters={"min_tokens": 16384},
            ),
            _source(
                "fineweb2-hq",
                "epfml/FineWeb2-HQ",
                "multilingual",
                0.02,
                1000 * B,
                odc,
            ),
        ],
    )

    phase_c = Phase(
        name="C-anneal",
        weight=0.10,
        lr_schedule="linear_to_zero",
        context_len=32768,
        purpose=(
            "Learning rate decays to zero on the highest-quality data available. Zero "
            "low-quality web. Run three times from the same phase-B checkpoint with "
            "different orderings and soup the results — nearly free relative to the "
            "phase cost, and reliably worth a point."
        ),
        sources=[
            _source(
                "openthoughts3",
                "open-thoughts/OpenThoughts3-1.2M",
                "instruction",
                0.18,
                2 * B,
                apache,
                notes="Long chain-of-thought traces: maths, science, code.",
            ),
            _source(
                "nemotron-cc-math",
                "nvidia/Nemotron-CC-Math-v1",
                "math",
                0.16,
                52 * B,
                nvidia,
                config="4plus",
            ),
            _source(
                "opc-annealing",
                "OpenCoder-LLM/opc-annealing-corpus",
                "code",
                0.14,
                None,
                "MIT",
            ),
            _source(
                "instruction-constraints",
                "HuggingFaceTB/smoltalk2",
                "instruction",
                0.14,
                None,
                apache,
                notes="Drives IFEval. Plus a bespoke constraint-following set.",
            ),
            _source(
                "fineweb-edu-top",
                "HuggingFaceFW/fineweb-edu",
                "web",
                0.12,
                350 * B,
                odc,
                filters={"edu_score_min": 4},
            ),
            _source(
                "nemotron-diverse-qa",
                "nvidia/Nemotron-CC-v2",
                "synthetic",
                0.10,
                400 * B,
                nvidia,
                config="diverse-qa",
            ),
            _source(
                "curated-reference",
                "allenai/dolma3",
                "reference",
                0.06,
                None,
                odc,
            ),
            _source(
                "cosmopedia-v2",
                "HuggingFaceTB/smollm-corpus",
                "synthetic",
                0.06,
                28 * B,
                apache,
                config="cosmopedia-v2",
            ),
            _source(
                "long-context-32k",
                "EleutherAI/proof-pile-2",
                "long_context",
                0.04,
                55 * B,
                "REVIEW: mixed (per-subset)",
                filters={"min_tokens": 32768},
            ),
        ],
    )

    return Mixture(
        name="prophet-v1",
        total_tokens=float(total_tokens),
        description=(
            "Three-phase WSD mixture from track R06, proportions preserved and token "
            "counts scaled to the measured single-A100 budget."
        ),
        phases=[phase_a, phase_b, phase_c],
    )
