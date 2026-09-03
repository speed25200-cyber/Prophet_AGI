"""Moving donor weights into a Prophet model.

Three kinds of transfer happen here, and the report distinguishes them because they carry
very different confidence:

- **Direct copy.** Same shape, same role: embeddings, norms, FFN matrices, and the
  attention projections of prelude and coda blocks. The Prophet config is built to match
  the donor's ``head_dim`` and ``n_kv_heads`` precisely so this path applies as widely as
  possible.
- **Averaged copy.** The weight-shared core is initialised from several donor middle
  layers at once. Consecutive layers of a trained transformer compute similar updates, so
  their mean is a defensible starting point for a block that will be applied repeatedly —
  but it is an initialisation, not an equivalence.
- **Heuristic seed.** Gated-delta layers have no donor counterpart. Their query and key
  projections take the donor's attention projections (both map the residual stream into a
  space where a dot product means similarity, so the correspondence is real), the value
  path is widened by the expansion factor, and the output projection places the donor's
  weights in the first half with **zeros in the second**, so the widened capacity starts
  inert and the layer's initial function is as close to the donor's attention as a
  bounded-state mixer can be.

Nothing here claims the converted model works. It claims the conversion is a better
starting point than random initialisation, which is what the recovery training then has
to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from prophet.convert.plan import ConversionPlan

__all__ = ["TransferReport", "convert_state_dict", "DONOR_KEYS"]

#: Donor parameter naming. Llama and Qwen3 share this layout; a donor that does not can
#: supply its own mapping.
DONOR_KEYS: dict[str, str] = {
    "embed": "model.embed_tokens.weight",
    "final_norm": "model.norm.weight",
    "lm_head": "lm_head.weight",
    "norm1": "model.layers.{i}.input_layernorm.weight",
    "norm2": "model.layers.{i}.post_attention_layernorm.weight",
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "q_norm": "model.layers.{i}.self_attn.q_norm.weight",
    "k_norm": "model.layers.{i}.self_attn.k_norm.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}


@dataclass
class TransferReport:
    copied: list[str] = field(default_factory=list)
    averaged: list[str] = field(default_factory=list)
    seeded: list[str] = field(default_factory=list)
    fresh: list[str] = field(default_factory=list)
    """Target parameters with no donor origin at all."""
    skipped: list[str] = field(default_factory=list)
    """Donor parameters with no target slot."""
    mismatched: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n_from_donor = len(self.copied) + len(self.averaged) + len(self.seeded)
        total = n_from_donor + len(self.fresh)
        lines = [
            "# Weight transfer report",
            "",
            f"| Origin | Tensors |",
            f"|---|---:|",
            f"| direct copy | {len(self.copied)} |",
            f"| averaged from several donor layers | {len(self.averaged)} |",
            f"| heuristically seeded from attention | {len(self.seeded)} |",
            f"| freshly initialised | {len(self.fresh)} |",
            "",
            f"{n_from_donor} of {total} target tensors ({n_from_donor / max(total, 1):.0%}) "
            "have a donor origin.",
        ]
        if self.mismatched:
            lines += ["", "## Shape mismatches (left at fresh initialisation)", ""]
            lines += [f"- {m}" for m in self.mismatched[:20]]
            if len(self.mismatched) > 20:
                lines.append(f"- ... and {len(self.mismatched) - 20} more")
        return "\n".join(lines)


def _get(donor: Mapping[str, Tensor], key: str, layer: int | None = None) -> Tensor | None:
    name = DONOR_KEYS.get(key, key)
    if layer is not None:
        name = name.format(i=layer)
    return donor.get(name)


def _mean_of(donor: Mapping[str, Tensor], key: str, layers: tuple[int, ...]) -> Tensor | None:
    tensors = [t for t in (_get(donor, key, i) for i in layers) if t is not None]
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0].clone()
    return torch.stack([t.float() for t in tensors]).mean(0).to(tensors[0].dtype)


def _expand_kv(weight: Tensor, *, n_kv_heads: int, n_heads: int, head_dim: int) -> Tensor:
    """Repeat grouped-query KV heads up to the full query-head count.

    This is exactly what GQA attention does at runtime, so the expanded matrix computes
    the same function the donor computed — it is a re-materialisation, not an
    approximation.
    """
    if n_kv_heads == n_heads:
        return weight.clone()
    reps = n_heads // n_kv_heads
    out_features, in_features = weight.shape
    view = weight.view(n_kv_heads, head_dim, in_features)
    return view.repeat_interleave(reps, dim=0).reshape(n_heads * head_dim, in_features)


def _seed_gdn_from_attention(
    target: dict[str, Tensor],
    prefix: str,
    donor: Mapping[str, Tensor],
    layers: tuple[int, ...],
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    report: TransferReport,
) -> None:
    """Seed a gated-delta mixer from the attention layer(s) it replaces."""
    q = _mean_of(donor, "q_proj", layers)
    k = _mean_of(donor, "k_proj", layers)
    v = _mean_of(donor, "v_proj", layers)
    o = _mean_of(donor, "o_proj", layers)

    def assign(name: str, value: Tensor) -> bool:
        full = f"{prefix}.{name}"
        current = target.get(full)
        if current is None:
            return False
        if current.shape != value.shape:
            report.mismatched.append(
                f"{full}: target {tuple(current.shape)} against donor-derived "
                f"{tuple(value.shape)}"
            )
            return False
        target[full] = value.to(current.dtype)
        report.seeded.append(full)
        return True

    if q is not None:
        assign("mixer.q_proj.weight", q.clone())
    if k is not None:
        assign(
            "mixer.k_proj.weight",
            _expand_kv(k, n_kv_heads=n_kv_heads, n_heads=n_heads, head_dim=head_dim),
        )
    if v is not None:
        expanded = _expand_kv(v, n_kv_heads=n_kv_heads, n_heads=n_heads, head_dim=head_dim)
        current = target.get(f"{prefix}.mixer.v_proj.weight")
        if current is not None:
            factor = current.shape[0] // expanded.shape[0]
            if factor >= 1 and current.shape[0] == expanded.shape[0] * factor:
                # Tile across the expansion, halving so the summed magnitude is preserved.
                assign(
                    "mixer.v_proj.weight",
                    expanded.repeat(factor, 1) / factor,
                )
    if o is not None:
        current = target.get(f"{prefix}.mixer.o_proj.weight")
        if current is not None and current.shape[1] >= o.shape[1]:
            widened = torch.zeros_like(current)
            widened[:, : o.shape[1]] = o.to(current.dtype)
            # Zeros in the widened half: the extra capacity starts inert, so the layer's
            # initial function stays as close to the donor's attention as possible.
            assign("mixer.o_proj.weight", widened)


def convert_state_dict(
    donor_state: Mapping[str, Tensor],
    plan: ConversionPlan,
    target_state: dict[str, Tensor],
) -> tuple[dict[str, Tensor], TransferReport]:
    """Write donor weights into a freshly initialised Prophet state dict.

    ``target_state`` is modified in place and also returned. Parameters with no donor
    origin keep their fresh initialisation, which is the correct behaviour: a wrongly
    shaped copy is far worse than a clean random start.
    """
    report = TransferReport()
    cfg = plan.target
    donor = plan.donor

    def copy_into(target_name: str, value: Tensor | None, bucket: list[str]) -> None:
        if value is None:
            return
        current = target_state.get(target_name)
        if current is None:
            return
        if current.shape != value.shape:
            report.mismatched.append(
                f"{target_name}: target {tuple(current.shape)} against donor "
                f"{tuple(value.shape)}"
            )
            return
        target_state[target_name] = value.to(current.dtype).clone()
        bucket.append(target_name)

    if plan.vocab_strategy == "keep_donor":
        copy_into("embed.weight", _get(donor_state, "embed"), report.copied)
        if cfg.frontend.tie_word_embeddings:
            # Tied weights need the *same* tensor under both keys. A state dict taken
            # from a live model happens to satisfy this by aliasing, but that is an
            # accident of reference semantics: copy or serialise the dict first and the
            # stale lm_head entry silently overwrites the donor embedding on load. The
            # model still runs, and is quietly ruined.
            if "lm_head.weight" in target_state and "embed.weight" in target_state:
                target_state["lm_head.weight"] = target_state["embed.weight"]
                report.copied.append("lm_head.weight (tied to embed)")
        else:
            copy_into("lm_head.weight", _get(donor_state, "lm_head"), report.copied)
    copy_into("norm_out.weight", _get(donor_state, "final_norm"), report.copied)

    for block in plan.blocks:
        prefix = f"sections.{block.section}.{block.index}"
        layers = block.donor_layers
        if not layers:
            continue
        bucket = report.copied if len(layers) == 1 else report.averaged

        copy_into(f"{prefix}.norm1.weight", _mean_of(donor_state, "norm1", layers), bucket)
        copy_into(f"{prefix}.norm2.weight", _mean_of(donor_state, "norm2", layers), bucket)
        for name in ("gate_proj", "up_proj", "down_proj"):
            copy_into(f"{prefix}.ffn.{name}.weight", _mean_of(donor_state, name, layers), bucket)

        if block.mixer_kind in ("full_attn", "swa"):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                copy_into(
                    f"{prefix}.mixer.{name}.weight", _mean_of(donor_state, name, layers), bucket
                )
            for name in ("q_norm", "k_norm"):
                copy_into(
                    f"{prefix}.mixer.{name}.weight", _mean_of(donor_state, name, layers), bucket
                )
        else:
            _seed_gdn_from_attention(
                target_state, prefix, donor_state, layers,
                n_heads=donor.n_heads, n_kv_heads=donor.n_kv_heads, head_dim=donor.head_dim,
                report=report,
            )

    touched = set(report.copied) | set(report.averaged) | set(report.seeded)
    report.fresh = sorted(k for k in target_state if k not in touched)

    donor_used = {
        DONOR_KEYS[k].format(i=i)
        for b in plan.blocks
        for i in b.donor_layers
        for k in DONOR_KEYS
        if "{i}" in DONOR_KEYS[k]
    } | {DONOR_KEYS["embed"], DONOR_KEYS["final_norm"], DONOR_KEYS["lm_head"]}
    report.skipped = sorted(k for k in donor_state if k not in donor_used)

    return target_state, report
