"""Planning a donor-to-Prophet conversion.

The conversion has to answer three questions, and this module answers them explicitly so
the answers can be inspected before any weights move:

1. **Which Prophet config does this donor imply?** Attention slots are shaped to match the
   donor (head_dim, kv-head count, width) so those weights transfer without reshaping.
2. **Which donor layer initialises which Prophet block?** The prelude and coda take the
   donor's first and last layers. The weight-shared core is initialised from the donor's
   *middle* layers, which have to be collapsed from many into few — the standard
   recursive-transformer initialisation.
3. **What cannot be transferred, and how is it initialised instead?** Gated-delta layers
   have no counterpart in the donor. Their projections are seeded from the attention
   layer they replace, which is a heuristic, not a theorem, and is labelled as such in the
   report.

The planner never touches tensors. It produces a plan and a coverage estimate, so a bad
conversion is visible before a single gigabyte is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.convert.donors import DonorSpec

__all__ = [
    "BlockSource",
    "ConversionPlan",
    "CoreInit",
    "plan_conversion",
    "prophet_config_for_donor",
]

CoreInit = Literal["average", "stride", "first"]


@dataclass
class BlockSource:
    """Where one Prophet block's weights come from."""

    section: str
    index: int
    mixer_kind: str
    donor_layers: tuple[int, ...]
    """Donor layers combined to initialise this block. More than one means averaging."""
    transferable: bool
    """False when the mixer has no donor counterpart and must be seeded heuristically."""
    note: str = ""


@dataclass
class ConversionPlan:
    donor: DonorSpec
    target: ProphetConfig
    blocks: list[BlockSource]
    core_init: CoreInit
    vocab_strategy: str
    warnings: list[str] = field(default_factory=list)

    @property
    def transferable_blocks(self) -> int:
        return sum(1 for b in self.blocks if b.transferable)

    def coverage(self) -> dict[str, float]:
        """Share of Prophet's parameters that come from the donor rather than fresh init.

        This is the number that predicts how much recovery training a conversion needs. A
        conversion at 40% coverage is closer to pretraining than to conversion, and should
        be recognised as such before the budget is committed rather than after.

        Counts are taken from the real per-component breakdown, not estimated: the FFN of
        every block transfers because its shape is unchanged, while a gated-delta mixer
        has no donor counterpart and only gets a heuristic seed.
        """
        from prophet.budget import _attention_params, _ffn_params, count_parameters

        params = count_parameters(self.target)
        total = float(params.total)

        transferred = float(params.embedding) if self.vocab_strategy == "keep_donor" else 0.0

        attn_params = _attention_params(self.target)
        ffn_resident, _ = _ffn_params(self.target, is_moe=False)
        for i, block in enumerate(self.blocks):
            if block.donor_layers:
                # The channel-mixing half is shape-identical, so it always transfers.
                transferred += ffn_resident
            if block.transferable:
                transferred += attn_params

        return {
            "transferred_params": transferred,
            "total_params": total,
            "coverage": min(transferred / max(total, 1.0), 1.0),
            "seeded_blocks": float(sum(1 for b in self.blocks if not b.transferable)),
        }

    def report(self) -> str:
        cov = self.coverage()
        lines = [
            f"# Conversion plan — {self.donor.name} -> {self.target.name}",
            "",
            f"Donor: {self.donor.n_layers} layers, d={self.donor.d_model}, "
            f"{self.donor.n_heads}q/{self.donor.n_kv_heads}kv heads of {self.donor.head_dim}, "
            f"vocab {self.donor.vocab_size}, licence {self.donor.license}",
            f"Target: {self.target.parameterised_depth()} parameterised blocks, "
            f"effective depth {self.target.effective_depth()}, "
            f"core initialised by {self.core_init}",
            f"Vocabulary: {self.vocab_strategy}",
            "",
            f"**Parameter coverage: {cov['coverage']:.0%}** "
            f"({cov['transferred_params'] / 1e6:.0f}M of "
            f"{cov['total_params'] / 1e6:.0f}M come from the donor)",
            "",
            "| Block | Mixer | Donor layers | Transfers | Note |",
            "|---|---|---|---|---|",
        ]
        for b in self.blocks:
            layers = ", ".join(str(i) for i in b.donor_layers) or "—"
            lines.append(
                f"| {b.section}[{b.index}] | {b.mixer_kind} | {layers} | "
                f"{'yes' if b.transferable else 'seeded'} | {b.note} |"
            )
        if self.warnings:
            lines += ["", "## Warnings", ""]
            lines += [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


def prophet_config_for_donor(
    donor: DonorSpec,
    *,
    prelude_layers: int = 4,
    core_layers: int = 4,
    coda_layers: int = 4,
    loop_k: int = 4,
    keep_donor_vocab: bool = True,
    prophet_vocab_size: int = 32_768,
    name: str | None = None,
) -> ProphetConfig:
    """Build a Prophet config whose attention slots match the donor's shapes.

    Matching ``head_dim``, ``n_kv_heads`` and ``d_model`` is what makes the attention
    weights transfer by direct copy instead of by interpolation, and direct copy is the
    difference between a conversion that recovers in a few billion tokens and one that
    does not recover at all.
    """
    return ProphetConfig(
        name=name or f"prophet-from-{donor.hf_id.split('/')[-1].lower()}",
        d_model=donor.d_model,
        frontend=FrontendConfig(
            mode="bpe",
            vocab_size=donor.vocab_size if keep_donor_vocab else prophet_vocab_size,
            tie_word_embeddings=donor.tie_word_embeddings,
        ),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"],
            n_heads=donor.n_heads,
            n_kv_heads=donor.n_kv_heads,
            head_dim=donor.head_dim,
            qk_norm=True,
            sliding_window=2048,
            attention_sink_tokens=1,
            linear_heads=max(donor.d_model // donor.head_dim, 1),
            linear_head_dim=donor.head_dim,
            linear_expand=2.0,
            rope_theta=donor.rope_theta,
            nope_layers=(1,),
        ),
        # The donor computed x + f(x) in every block. A forward-time residual multiplier
        # would turn a "direct copy" into x + 0.1*f(x): the same tensors, a different
        # function. Off for conversion, and the init-time scaling it replaced does not
        # apply to copied weights either.
        residual_scaling=False,
        recurrent=RecurrentCoreConfig(
            enabled=True,
            prelude_layers=prelude_layers,
            core_layers=core_layers,
            coda_layers=coda_layers,
            core_pattern=["gdn"],
            default_loop_k=loop_k,
            train_loop_min=1,
            train_loop_max=max(2 * loop_k, 2),
            truncated_backprop_steps=3,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=3.0 * donor.ffn_hidden / (2 * donor.d_model)),
        heads=HeadsConfig(n_multi_token_predict=1, confidence_head=True),
    )


def _group_middle_layers(
    donor_layers: range, n_groups: int, mode: CoreInit
) -> list[tuple[int, ...]]:
    """Collapse the donor's middle layers into ``n_groups`` initialisation sources."""
    layers = list(donor_layers)
    if not layers or n_groups <= 0:
        return [() for _ in range(max(n_groups, 0))]

    if mode == "first":
        return [(layers[i],) if i < len(layers) else () for i in range(n_groups)]
    if mode == "stride":
        step = max(len(layers) // n_groups, 1)
        return [(layers[min(i * step, len(layers) - 1)],) for i in range(n_groups)]

    # "average": contiguous groups, each averaged. This is the recursive-transformer
    # initialisation -- consecutive layers of a trained transformer compute similar
    # updates, so their mean is a reasonable starting point for a block that will be
    # applied repeatedly.
    out: list[tuple[int, ...]] = []
    for i in range(n_groups):
        start = (i * len(layers)) // n_groups
        end = ((i + 1) * len(layers)) // n_groups
        out.append(tuple(layers[start:end]) or (layers[min(start, len(layers) - 1)],))
    return out


def plan_conversion(
    donor: DonorSpec,
    target: ProphetConfig,
    *,
    core_init: CoreInit = "average",
    keep_donor_vocab: bool = True,
) -> ConversionPlan:
    """Decide which donor layer initialises which Prophet block."""
    target.validate()
    layout = target.section_layout()
    r = target.recurrent

    n_prelude = sum(1 for s, _, _ in layout if s == "prelude")
    n_core = sum(1 for s, _, _ in layout if s == "core")
    n_coda = sum(1 for s, _, _ in layout if s == "coda")

    warnings: list[str] = []
    if n_prelude + n_coda > donor.n_layers:
        warnings.append(
            f"prelude+coda ({n_prelude + n_coda}) exceeds the donor's "
            f"{donor.n_layers} layers; some blocks will be initialised fresh"
        )
    if target.d_model != donor.d_model:
        warnings.append(
            f"width mismatch: target d_model={target.d_model} against donor "
            f"{donor.d_model}. Nothing transfers by direct copy at a different width."
        )
    if target.head_dim != donor.head_dim:
        warnings.append(
            f"head_dim mismatch ({target.head_dim} against {donor.head_dim}); "
            "attention weights cannot be copied without reshaping"
        )
    if not donor.verified:
        warnings.append(
            f"{donor.name}'s architecture figures are unverified — run "
            "scripts/verify_donors.py against the Hub before downloading weights"
        )

    middle_start = min(n_prelude, donor.n_layers)
    middle_end = max(donor.n_layers - n_coda, middle_start)
    core_groups = _group_middle_layers(range(middle_start, middle_end), n_core, core_init)

    blocks: list[BlockSource] = []
    core_seen = 0
    for section, index, kind in layout:
        if section == "prelude":
            src = (index,) if index < donor.n_layers else ()
            note = "direct copy" if src else "no donor layer available"
        elif section == "coda":
            donor_index = donor.n_layers - n_coda + index
            src = (donor_index,) if 0 <= donor_index < donor.n_layers else ()
            note = "direct copy" if src else "no donor layer available"
        else:
            src = core_groups[core_seen] if core_seen < len(core_groups) else ()
            core_seen += 1
            note = (
                f"{core_init} of {len(src)} donor layers"
                if len(src) > 1
                else f"{core_init} from one donor layer"
            )

        is_attention = kind in ("full_attn", "swa")
        if not is_attention and src:
            note += "; gated-delta projections seeded from attention (heuristic)"

        blocks.append(
            BlockSource(
                section=section,
                index=index,
                mixer_kind=kind,
                donor_layers=src,
                transferable=bool(src) and is_attention,
                note=note,
            )
        )

    return ConversionPlan(
        donor=donor,
        target=target,
        blocks=blocks,
        core_init=core_init,
        vocab_strategy="keep_donor" if keep_donor_vocab else "reproject_to_prophet_tok",
        warnings=warnings,
    )
