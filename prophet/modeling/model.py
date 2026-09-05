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
import copy
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
from prophet.modeling.moe import SparseMoE, apply_router_updates

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
    loop_k: int | None = None
    """Depth ceiling of this cache.

    Without ``recurrent.token_depth`` it is exact and may only shrink: a core slot for
    iteration i is valid only if iteration i ran on every previous token, so a deeper
    call later would read states that never saw the earlier tokens, while a shallower
    call (explicit, or a halting early-exit) just retires the deeper slots. With
    ``token_depth`` the invariant is per token -- iteration i saw every earlier token
    whose ceiling exceeds i -- and this field only records the deepest slot in use."""

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
        layer_index: int = 0,
        section: str = "trunk",
    ) -> None:
        super().__init__()
        self.kind = kind
        self.gradient_checkpointing = False

        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = build_mixer(kind, cfg, layer_index=layer_index, section=section)
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
        # Recompute activations in backward when asked. A cached call is stateful and
        # cannot be replayed, so checkpointing only applies to the cache-free path.
        if (
            self.gradient_checkpointing
            and self.training
            and cache is None
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, cos, sin, use_reentrant=False
            )
        return self._forward(x, cos, sin, cache)

    def _forward(self, x, cos=None, sin=None, cache=None) -> Tensor:
        # Residual branches are added unscaled. The 1/sqrt(2*depth) factor the config
        # names is applied to the *initialisation* of the output projections, where the
        # docstring always said it was; as a forward-time multiplier it attenuated every
        # branch by ~0.11 forever and turned a converted donor's x + f(x) into
        # x + 0.1 f(x) -- the same tensors computing a different function.
        if self.mixer is not None:
            h = self.norm1(x)
            if isinstance(self.mixer, GatedDeltaNet):
                mixed = self.mixer(h, state=cache)  # type: ignore[arg-type]
            else:
                mixed = self.mixer(h, cos=cos, sin=sin, cache=cache)  # type: ignore[arg-type]
            x = x + mixed
        return x + self.ffn(self.norm2(x))


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

    halt_probs: Tensor | None = None
    """``(batch, seq, steps)`` probability that reasoning stopped at each iteration.

    Present only when halting is enabled. Looping a constant number of times leaves the
    model's depth bounded by a constant and therefore changes no complexity class; only
    depth that *depends on the input* buys anything asymptotically, and this distribution
    is what makes it depend on the input."""
    hidden_per_step: list[Tensor] | None = None
    """Coda-applied, **normalised** hidden state after each iteration -- the same space as
    ``hidden`` -- so the ponder loss can project it straight through the LM head. The
    first version stored the pre-norm coda output (rms 0.1 against 1.0 after norm) and
    scored every stopping point on near-uniform logits."""

    def expected_depth(self) -> float | None:
        """Mean number of iterations actually used, weighted by the halting distribution."""
        if self.halt_probs is None:
            return None
        steps = torch.arange(
            1, self.halt_probs.shape[-1] + 1, device=self.halt_probs.device,
            dtype=self.halt_probs.dtype,
        )
        return float((self.halt_probs * steps).sum(-1).mean().item())


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

        sections: dict[str, nn.ModuleList] = {}
        for name in ("prelude", "core", "coda", "trunk"):
            blocks = [
                ProphetBlock(
                    cfg,
                    kind=kind,
                    is_moe=cfg.layer_is_moe(self._global_index(layout, sec, idx)),
                    layer_index=idx,
                    section=sec,
                )
                for sec, idx, kind in layout
                if sec == name
            ]
            if blocks:
                sections[name] = nn.ModuleList(blocks)
        self.sections = nn.ModuleDict(sections)

        # Persistent memory (track R03). Attached at the trunk indices named in the
        # config, read as a residual addition. It is inert until written, so enabling it
        # cannot change behaviour before anything has been stored.
        # Keys: "output" for the single output-mounted ledger (the space that
        # prophet.memory.consolidate addresses), or "coda_<i>" for a per-block ledger. A
        # ledger keyed by a global index that the section-local read never matched was
        # a no-op that validated and budgeted parameters.
        self.ledgers = nn.ModuleDict()
        if cfg.memory.enabled and cfg.memory.kind == "product_key":
            from prophet.memory.ledger import LedgerConfig, ProductKeyMemory

            def _ledger() -> ProductKeyMemory:
                return ProductKeyMemory(
                    LedgerConfig(
                        dim=d,
                        memory_dim=cfg.memory.memory_dim,
                        n_slots=cfg.memory.n_slots,
                        write_lr=cfg.memory.write_lr,
                        decay=cfg.memory.decay,
                    )
                )

            if cfg.memory.mount == "output":
                self.ledgers["output"] = _ledger()
            else:
                for index in cfg.memory.layers:
                    # Underscore, not dot: nn.ModuleDict forbids '.' in a key.
                    self.ledgers[f"coda_{index}"] = _ledger()

        self.norm_out = RMSNorm(d, cfg.norm_eps)
        self.lm_head = nn.Linear(d, cfg.frontend.vocab_size, bias=False)
        if cfg.frontend.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.mtp_heads = nn.ModuleList(
            ProphetBlock(cfg, kind="swa", is_moe=False)
            for _ in range(cfg.heads.n_multi_token_predict)
        )
        self.confidence_head = (
            nn.Sequential(RMSNorm(d, cfg.norm_eps), nn.Linear(d, 1))
            if cfg.heads.confidence_head
            else None
        )

        # Learned halting. A single scalar per position per iteration: "is this enough
        # thinking?". Cheap to add, and it is the only mechanism that makes recurrence
        # depth a function of the input rather than a constant chosen by the caller.
        self.halt_head = (
            nn.Sequential(RMSNorm(d, cfg.norm_eps), nn.Linear(d, 1))
            if cfg.recurrent.enabled and cfg.recurrent.halting == "ponder"
            else None
        )

        self.apply(self._init_weights)

        if cfg.residual_scaling:
            # 1/sqrt(2 * depth) on the output projections at init keeps residual-stream
            # variance bounded. The depth that matters is the *effective* one: a
            # heavily looped core adds as much variance as distinct layers would.
            depth_for_scale = cfg.effective_depth(cfg.recurrent.train_loop_max)
            scale = (2.0 * max(depth_for_scale, 1)) ** -0.5
            with torch.no_grad():
                for name, module in self.named_modules():
                    if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in (
                        "o_proj", "down_proj"
                    ):
                        module.weight.mul_(scale)

    @property
    def gradient_checkpointing(self) -> bool:
        return any(getattr(b, "gradient_checkpointing", False) for s in self.sections.values() for b in s)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        for section in self.sections.values():
            for block in section:
                block.gradient_checkpointing = bool(value)
        for head in self.mtp_heads:
            head.gradient_checkpointing = bool(value)

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
            k = int(torch.poisson(
                torch.tensor(float(r.train_loop_poisson_lambda)), generator=generator
            ).item())
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
        halt_threshold: float | None = None,
        generator: torch.Generator | None = None,
        token_depth: Tensor | None = None,
    ) -> ProphetOutput:
        """Run the model.

        ``token_depth`` (``(batch, seq)`` long) gives each token its own recurrence
        ceiling; it requires ``recurrent.token_depth`` and is what the trainer passes
        when that switch is on. At iteration ``i`` the core runs on the compacted
        subsequence of tokens whose ceiling exceeds ``i``; the others keep the hidden
        state they exited with, and neither the recurrent state nor the causal
        convolution of the deeper iterations ever sees them -- which is exactly what an
        incremental decode that ran them shallow would have produced.
        """
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
        halt_logits: list[Tensor] = []
        hidden_per_step: list[Tensor] = []

        def run(
            section: str, iteration: int, h: Tensor, *, use_cache: bool = True,
            probe: bool = False, token_mask: Tensor | None = None,
        ) -> Tensor:
            """Apply one section.

            ``probe`` marks a halting probe pass: it reads the cache through a shallow
            copy of each slot -- so the coda sees its real context -- and discards the
            copy, and it suppresses MoE stats and bias updates. The first version ran
            probes cache-free, so at decode time the halting decision was made by a coda
            that saw only the current token. ``token_mask`` marks the real rows of a
            compacted core pass; padding rows are kept out of the router statistics.
            """
            for idx, block in enumerate(self.sections[section]):
                slot = None
                if cache is not None and use_cache:
                    real = cache.get(section, idx, iteration, block.kind)
                    slot = copy.copy(real) if probe else real
                moe = block.ffn if isinstance(block.ffn, SparseMoE) else None
                if moe is not None:
                    moe.probe_mode = probe
                    moe.token_mask = token_mask
                h = block(h, cos=cos, sin=sin, cache=slot)
                if moe is not None:
                    moe.probe_mode = False
                    moe.token_mask = None
                    if not probe and moe.last_stats is not None:
                        router_stats.append(moe.last_stats)
                        aux_terms.append(moe.last_stats.aux_loss)
                key = f"{section}_{idx}"
                if section == "coda" and key in self.ledgers:
                    # Residual read: the ledger contributes nothing until written.
                    h = h + self.ledgers[key](h)
            return h

        if not cfg.recurrent.enabled:
            if token_depth is not None:
                raise ValueError("token_depth needs a recurrent core")
            k = 1
            x = run("trunk", 0, x)
        else:
            r = cfg.recurrent
            if token_depth is not None:
                if not r.token_depth:
                    raise ValueError(
                        "token_depth was given but recurrent.token_depth is off: a "
                        "model trained at one depth per sequence has no defined "
                        "behaviour for a depth that varies within it"
                    )
                if tuple(token_depth.shape) != (b, s):
                    raise ValueError(f"token_depth must be shaped {(b, s)}, got {tuple(token_depth.shape)}")
                token_depth = token_depth.to(device=input_ids.device, dtype=torch.long)
                deepest = int(token_depth.max().item()) if s else 1
                if int(token_depth.min().item()) < 1:
                    raise ValueError("every token needs a ceiling of at least 1")
                k = loop_k if loop_k is not None else deepest
                if deepest > k:
                    raise ValueError(f"token_depth reaches {deepest}, above loop_k={k}")
            else:
                k = loop_k if loop_k is not None else (
                    self.sample_loop_k(generator)
                    if self.training
                    else (
                        cache.loop_k
                        if cache is not None and cache.loop_k is not None
                        else r.default_loop_k
                    )
                )
            if cache is not None:
                if r.token_depth:
                    # Per-token invariant: any depth is defined. Record the deepest slot.
                    cache.loop_k = max(cache.loop_k or 0, k)
                elif cache.loop_k is None:
                    cache.loop_k = k
                elif k > cache.loop_k:
                    raise ValueError(
                        f"this cache's depth ceiling is {cache.loop_k}; a call at "
                        f"loop_k={k} would read core states that never saw the earlier "
                        "tokens. Without recurrent.token_depth a cache's depth can only "
                        "shrink."
                    )
                else:
                    cache.loop_k = k  # shallower is exact: the deeper slots retire
            x = run("prelude", 0, x)
            injected = x

            # Training randomises the starting state so the loop learns to be
            # init-independent; inference must be deterministic (see
            # ``RecurrentCoreConfig.eval_state_init``).
            init_mode = r.state_init if self.training else r.eval_state_init
            if init_mode == "randn":
                noise = (
                    torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
                    if generator is not None else torch.randn_like(x)
                )
                h = noise * cfg.init_std
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
                    if token_depth is None:
                        h = run("core", i, step_in)
                    else:
                        h = self._run_core_compacted(
                            run, i, step_in, h, token_depth > i, cached=cache is not None
                        )
                if not grad_on:
                    h = h.detach()

                if self.halt_head is not None:
                    # Each candidate stopping point needs a real read-out to be scored
                    # against, so the coda is applied per iteration as a *probe*: it
                    # reads the cache through disposable copies of the slots (so it sees
                    # its context) and never writes to them (so it cannot append the same
                    # positions k times). The real, cached coda runs once below.
                    step_out = self.norm_out(run("coda", 0, h, probe=True))
                    hidden_per_step.append(step_out)
                    logit = self.halt_head(step_out).squeeze(-1)
                    if token_depth is not None:
                        # A ceiling is a forced stop: all remaining mass halts at the
                        # token's last iteration. Earlier iterations keep the learned
                        # logit, so the head still learns where stopping *sooner* would
                        # have been fine.
                        last = token_depth - 1
                        logit = torch.where(
                            last == i, logit.new_full((), 30.0),
                            torch.where(last > i, logit, logit.new_full((), -30.0)),
                        )
                    halt_logits.append(logit)

                    if not self.training and halt_threshold is not None:
                        survived = torch.stack(
                            [1 - torch.sigmoid(l) for l in halt_logits]
                        ).prod(dim=0)
                        # Per sequence and per position: stop only when *every* one has
                        # crossed the threshold. Conservative on purpose -- a batch mean
                        # let one confident sequence cut off another's thinking.
                        if bool(((1.0 - survived) >= halt_threshold).all()):
                            k = i + 1
                            if cache is not None and not r.token_depth:
                                cache.loop_k = k  # pin: later steps must not go deeper
                            break

            x = run("coda", 0, h)

        hidden = self.norm_out(x)
        # Output-mounted memory: read on the normalised, ledger-free hidden state and
        # added only for the projection. ``hidden`` stays ledger-free, which is what
        # ``prophet.memory.consolidate`` addresses and targets.
        read = hidden + self.ledgers["output"](hidden) if "output" in self.ledgers else hidden
        logits = self._project(read)

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

        halt_probs = None
        if halt_logits:
            # PonderNet stopping distribution: halt at step i with probability lambda_i,
            # having survived every earlier step.
            lams = torch.sigmoid(torch.stack(halt_logits, dim=-1))
            survive = torch.cumprod(1 - lams, dim=-1)
            shifted = torch.cat([torch.ones_like(survive[..., :1]), survive[..., :-1]], -1)
            halt_probs = lams * shifted
            # The tail mass has to go somewhere: assign it to the last iteration, which
            # is what actually happens when the loop runs out.
            halt_probs[..., -1] = halt_probs[..., -1] + (1.0 - halt_probs.sum(-1))

        aux = torch.stack(aux_terms).sum() if aux_terms else None
        return ProphetOutput(
            logits=logits,
            hidden=hidden,
            loop_k=k,  # iterations actually run, after any halting exit
            mtp_logits=mtp_logits,
            confidence=confidence,
            aux_loss=aux,
            router_stats=router_stats,
            halt_probs=halt_probs,
            hidden_per_step=hidden_per_step or None,
        )

    @staticmethod
    def _run_core_compacted(
        run, iteration: int, step_in: Tensor, h: Tensor, active: Tensor, *, cached: bool
    ) -> Tensor:
        """One core iteration over only the tokens whose ceiling exceeds ``iteration``.

        Active tokens are gathered left-aligned per row, run as a shorter sequence, and
        scattered back; inactive tokens keep ``h``. Rows with fewer active tokens are
        padded at the *end*, which is exact in a cache-free pass -- the core is causal
        and its final state is discarded -- and would poison a cached recurrent state,
        so a cached call insists on equal counts (in practice: one sequence at a time).
        """
        if bool(active.all()):
            return run("core", iteration, step_in)
        if not bool(active.any()):
            return h
        b, s, d = step_in.shape
        counts = active.sum(dim=1)
        length = int(counts.max().item())
        if cached and b > 1 and not bool((counts == length).all()):
            raise ValueError(
                "per-token depth on a cache needs every row to have the same number of "
                "active tokens at each iteration; padding would enter the recurrent "
                "state. Decode such rows one sequence at a time."
            )
        # Stable sort on "inactive" puts active positions first, in order.
        order = torch.argsort((~active).to(torch.int8), dim=1, stable=True)[:, :length]
        index = order.unsqueeze(-1).expand(b, length, d)
        gathered_in = torch.gather(step_in, 1, index)
        gathered_h = torch.gather(h, 1, index)
        valid = torch.arange(length, device=active.device).unsqueeze(0) < counts.unsqueeze(1)
        out = run("core", iteration, gathered_in, token_mask=valid)
        # Padding rows write their own ``h`` back to the inactive positions they borrowed.
        src = torch.where(valid.unsqueeze(-1), out, gathered_h)
        return h.scatter(1, index, src)

    def _project(self, hidden: Tensor) -> Tensor:
        logits = self.lm_head(hidden)
        if self.cfg.logit_soft_cap is not None:
            cap = self.cfg.logit_soft_cap
            logits = cap * torch.tanh(logits / cap)
        return logits

    # -- convenience -------------------------------------------------------------------

    @staticmethod
    def apply_router_updates(output: ProphetOutput) -> int:
        """Apply the loss-free balancing steps recorded during ``output``'s forward.

        Call it after ``backward()``: the routers must not move between a checkpointed
        forward and its recompute. Returns the number of routers updated.
        """
        return apply_router_updates(output.router_stats)

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
