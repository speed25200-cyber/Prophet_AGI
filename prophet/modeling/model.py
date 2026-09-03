"""The Prophet model.

The structural idea, and the project's central bet, is that **depth is a runtime dial**.
Instead of ``n_layers`` distinct blocks, the trunk is:

    prelude  ->  [ shared core ] x k  ->  coda

The core owns one set of weights and is applied ``k`` times. Effective depth is
``prelude + core * k + coda`` while memory only pays for ``prelude + core + coda``.
``k`` is chosen per request, so one set of weights serves an iPhone at ``k=2`` and an
RTX 5090 at ``k=8`` — which is what makes a single model span the three hardware
targets rather than requiring three separately trained models.

Two details make this affordable rather than merely elegant:

- The core is **recurrent, not attentive**, by default. Looping attention would need a
  separate KV cache per iteration; looping a bounded-state mixer needs a few kilobytes.
- Backpropagation is **truncated** to the last few iterations, so training a deep loop
  costs the activation memory of a shallow one.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prophet.config import ProphetConfig
from prophet.modeling.layers import (
    AttentionCache,
    CausalSelfAttention,
    GatedDeltaNet,
    RecurrentState,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    build_mixer,
)
from prophet.modeling.moe import SparseMoE

__all__ = ["ProphetBlock", "ProphetCache", "ProphetOutput", "ProphetModel"]


def _swiglu_hidden(d_model: int, mult: float) -> int:
    """Mirror of ``prophet.budget._swiglu_hidden`` so accounting matches reality."""
    hidden = int(2 * mult * d_model / 3)
    return max(128, ((hidden + 127) // 128) * 128)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


@dataclass
class ProphetCache:
    """All per-layer state for incremental decoding.

    Slots are keyed by ``(section, block, iteration)`` because a looped core block sees
    different inputs on each pass and therefore needs its own state each time round.
    """

    slots: dict[tuple[str, int, int], AttentionCache | RecurrentState] = field(
        default_factory=dict
    )
    position: int = 0
    """Number of tokens consumed, which is what positional encoding must use — the
    retained buffer length is not the position once window eviction starts."""

    def get(
        self, section: str, block: int, iteration: int, kind: str
    ) -> AttentionCache | RecurrentState:
        key = (section, block, iteration)
        slot = self.slots.get(key)
        if slot is None:
            slot = RecurrentState() if kind in ("gdn", "mamba2") else AttentionCache()
            self.slots[key] = slot
        return slot

    def n_bytes(self) -> int:
        return sum(s.n_bytes() for s in self.slots.values())

    def summary(self) -> dict[str, Any]:
        attn = sum(s.n_bytes() for s in self.slots.values() if isinstance(s, AttentionCache))
        rec = sum(s.n_bytes() for s in self.slots.values() if isinstance(s, RecurrentState))
        return {
            "slots": len(self.slots),
            "position": self.position,
            "attention_bytes": attn,
            "recurrent_bytes": rec,
            "total_bytes": attn + rec,
        }


# --------------------------------------------------------------------------------------
# Block
# --------------------------------------------------------------------------------------


class ProphetBlock(nn.Module):
    """Pre-norm block: ``x + mix(norm(x))`` then ``x + ffn(norm(x))``."""

    def __init__(
        self,
        cfg: ProphetConfig,
        *,
        kind: str,
        is_moe: bool,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.residual_scale = residual_scale

        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = build_mixer(kind, cfg, layer_index=0)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)

        if is_moe:
            f = cfg.ffn
            self.ffn: nn.Module = SparseMoE(
                cfg.d_model,
                n_experts=f.n_experts,
                top_k=f.n_experts_per_token,
                expert_hidden=_swiglu_hidden(cfg.d_model, f.expert_hidden_mult),
                n_shared=f.n_shared_experts,
                bias_balancing=f.router_bias_balancing,
                bias_update_rate=f.router_bias_update_rate,
                z_loss_weight=f.router_z_loss_weight,
                load_balance_loss_weight=f.load_balance_loss_weight,
            )
        else:
            self.ffn = SwiGLU(cfg.d_model, _swiglu_hidden(cfg.d_model, cfg.ffn.hidden_mult))

    def forward(
        self,
        x: Tensor,
        *,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        cache: AttentionCache | RecurrentState | None = None,
    ) -> Tensor:
        if self.mixer is not None:
            h = self.norm1(x)
            if isinstance(self.mixer, GatedDeltaNet):
                mixed = self.mixer(h, state=cache)  # type: ignore[arg-type]
            else:
                mixed = self.mixer(h, cos=cos, sin=sin, cache=cache)  # type: ignore[arg-type]
            x = x + self.residual_scale * mixed
        return x + self.residual_scale * self.ffn(self.norm2(x))


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


@dataclass
class ProphetOutput:
    logits: Tensor
    hidden: Tensor
    loop_k: int
    mtp_logits: list[Tensor] = field(default_factory=list)
    """Logits for tokens t+2..t+n from the multi-token-prediction heads. These densify
    the training signal and double as a speculative-decoding draft at inference."""
    confidence: Tensor | None = None
    """Per-position logit of 'the answer here is correct', driving abstention."""
    aux_loss: Tensor | None = None
    router_stats: list[Any] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class ProphetModel(nn.Module):
    """Prophet trunk with a weight-shared, runtime-tunable recurrent core."""

    def __init__(self, cfg: ProphetConfig) -> None:
        super().__init__()
        cfg.validate()
        if cfg.frontend.mode != "bpe":
            raise NotImplementedError(
                f"frontend mode {cfg.frontend.mode!r} is specified in the config schema but "
                "not yet implemented; use 'bpe' until the R01 ablation selects a design"
            )
        self.cfg = cfg
        d = cfg.d_model

        self.embed = nn.Embedding(cfg.frontend.vocab_size, d)
        self.modality_embed = (
            nn.Embedding(cfg.modality.n_modalities, d)
            if cfg.modality.modality_embeddings and cfg.modality.n_modalities > 1
            else None
        )

        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            theta=cfg.mixer.rope_theta,
            scaling=cfg.mixer.rope_scaling,
            scaling_factor=cfg.mixer.rope_scaling_factor,
            position_dims=cfg.modality.position_dims,
        )

        layout = cfg.section_layout()
        # 1/sqrt(2 * depth) keeps residual-stream variance bounded. The depth that
        # matters is the *effective* one: a heavily looped core adds just as much
        # variance as distinct layers would.
        depth_for_scale = cfg.effective_depth(cfg.recurrent.train_loop_max)
        scale = (2.0 * max(depth_for_scale, 1)) ** -0.5 if cfg.residual_scaling else 1.0

        sections: dict[str, nn.ModuleList] = {}
        for name in ("prelude", "core", "coda", "trunk"):
            blocks = [
                ProphetBlock(
                    cfg,
                    kind=kind,
                    is_moe=cfg.layer_is_moe(self._global_index(layout, sec, idx)),
                    residual_scale=scale,
                )
                for sec, idx, kind in layout
                if sec == name
            ]
            if blocks:
                sections[name] = nn.ModuleList(blocks)
        self.sections = nn.ModuleDict(sections)

        self.norm_out = RMSNorm(d, cfg.norm_eps)
        self.lm_head = nn.Linear(d, cfg.frontend.vocab_size, bias=False)
        if cfg.frontend.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.mtp_heads = nn.ModuleList(
            ProphetBlock(cfg, kind="swa", is_moe=False, residual_scale=scale)
            for _ in range(cfg.heads.n_multi_token_predict)
        )
        self.confidence_head = (
            nn.Sequential(RMSNorm(d, cfg.norm_eps), nn.Linear(d, 1))
            if cfg.heads.confidence_head
            else None
        )

        self.apply(self._init_weights)

    # -- setup -------------------------------------------------------------------------

    @staticmethod
    def _global_index(layout: list[tuple[str, int, str]], section: str, idx: int) -> int:
        for g, (sec, i, _) in enumerate(layout):
            if sec == section and i == idx:
                return g
        return 0

    def _init_weights(self, module: nn.Module) -> None:
        cfg = self.cfg
        std = cfg.init_std
        if cfg.mup_base_width is not None:
            # muP: scale hidden-matrix init by 1/sqrt(width ratio) so that the learning
            # rate found on a narrow proxy transfers to the full width without a sweep.
            std = std * (cfg.mup_base_width / cfg.d_model) ** 0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None and not getattr(module, "_prophet_keep_bias", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings are indexed, not multiplied, so they keep the unscaled init.
            nn.init.normal_(module.weight, mean=0.0, std=cfg.init_std)

    # -- forward -----------------------------------------------------------------------

    def sample_loop_k(self, generator: torch.Generator | None = None) -> int:
        """Draw a training recurrence depth.

        Depth is randomised per step so the model stays usable at *any* ``k``, and so
        expected gradient cost stays bounded. Log-uniform is the default because the
        interesting differences are between 1, 2, 4 and 8 rather than between 7 and 8.
        """
        r = self.cfg.recurrent
        lo, hi = r.train_loop_min, r.train_loop_max
        if lo == hi:
            return lo
        if r.train_loop_dist == "uniform":
            return int(torch.randint(lo, hi + 1, (1,), generator=generator).item())
        if r.train_loop_dist == "poisson":
            k = int(torch.poisson(torch.tensor(float(r.train_loop_poisson_lambda))).item())
            return max(lo, min(hi, k))
        u = torch.rand(1, generator=generator).item()
        return int(round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))))

    def forward(
        self,
        input_ids: Tensor,
        *,
        positions: Tensor | None = None,
        modality_ids: Tensor | None = None,
        cache: ProphetCache | None = None,
        loop_k: int | None = None,
        return_mtp: bool = True,
    ) -> ProphetOutput:
        cfg = self.cfg
        b, s = input_ids.shape
        offset = cache.position if cache is not None else 0

        if positions is None:
            positions = torch.arange(offset, offset + s, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(b, s)
            if cfg.modality.position_dims > 1:
                pad = positions.new_zeros(b, s, cfg.modality.position_dims - 1)
                positions = torch.cat([positions.unsqueeze(-1), pad], dim=-1)
        cos, sin = self.rotary(positions)

        x = self.embed(input_ids)
        if self.modality_embed is not None and modality_ids is not None:
            x = x + self.modality_embed(modality_ids)

        aux_terms: list[Tensor] = []
        router_stats: list[Any] = []

        def run(section: str, iteration: int, h: Tensor) -> Tensor:
            for idx, block in enumerate(self.sections[section]):
                slot = (
                    cache.get(section, idx, iteration, block.kind)
                    if cache is not None
                    else None
                )
                h = block(h, cos=cos, sin=sin, cache=slot)
                if isinstance(block.ffn, SparseMoE) and block.ffn.last_stats is not None:
                    router_stats.append(block.ffn.last_stats)
                    aux_terms.append(block.ffn.last_stats.aux_loss)
            return h

        if not cfg.recurrent.enabled:
            k = 1
            x = run("trunk", 0, x)
        else:
            r = cfg.recurrent
            k = loop_k if loop_k is not None else (
                self.sample_loop_k() if self.training else r.default_loop_k
            )
            x = run("prelude", 0, x)
            injected = x

            # Training randomises the starting state so the loop learns to be
            # init-independent; inference must be deterministic (see
            # ``RecurrentCoreConfig.eval_state_init``).
            init_mode = r.state_init if self.training else r.eval_state_init
            if init_mode == "randn":
                h = torch.randn_like(x) * cfg.init_std
            elif init_mode == "prelude":
                h = x
            else:
                h = torch.zeros_like(x)

            # Backprop only through the trailing iterations: this is what keeps the
            # activation memory of a k=8 loop equal to that of a shallow stack.
            first_grad_iter = max(0, k - r.truncated_backprop_steps)
            for i in range(k):
                grad_on = i >= first_grad_iter
                ctx = contextlib.nullcontext() if grad_on else torch.no_grad()
                with ctx:
                    step_in = h + injected if r.inject_input_each_step else h
                    h = run("core", i, step_in)
                if not grad_on:
                    h = h.detach()
            x = run("coda", 0, h)

        hidden = self.norm_out(x)
        logits = self._project(hidden)

        mtp_logits: list[Tensor] = []
        if return_mtp and len(self.mtp_heads):
            for j, head in enumerate(self.mtp_heads):
                slot = cache.get("mtp", j, 0, head.kind) if cache is not None else None
                mtp_logits.append(self._project(self.norm_out(head(x, cos=cos, sin=sin, cache=slot))))

        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head(x).squeeze(-1)

        if cache is not None:
            cache.position = offset + s

        aux = torch.stack(aux_terms).sum() if aux_terms else None
        return ProphetOutput(
            logits=logits,
            hidden=hidden,
            loop_k=k,
            mtp_logits=mtp_logits,
            confidence=confidence,
            aux_loss=aux,
            router_stats=router_stats,
        )

    def _project(self, hidden: Tensor) -> Tensor:
        logits = self.lm_head(hidden)
        if self.cfg.logit_soft_cap is not None:
            cap = self.cfg.logit_soft_cap
            logits = cap * torch.tanh(logits / cap)
        return logits

    # -- convenience -------------------------------------------------------------------

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 32,
        loop_k: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Greedy or sampled decoding. Deliberately minimal — a correctness reference,
        not a serving path."""
        self.eval()
        cache = ProphetCache()
        out = self.forward(input_ids, cache=cache, loop_k=loop_k, return_mtp=False)
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            out = self.forward(nxt, cache=cache, loop_k=loop_k, return_mtp=False)
        return generated
