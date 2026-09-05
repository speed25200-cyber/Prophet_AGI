"""Configuration schema for Prophet models.

Every architectural bet in `docs/00_PROBLEM_LANDSCAPE.md` is exposed here as an
explicit, serialisable switch. Nothing is hard-wired: an unvalidated idea must
always be reducible to a baseline by changing configuration alone (CLAUDE.md
rule 3). The defaults below describe the *baseline* — a conventional dense
transformer — so that every ablation measures a delta against something known.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_origin, get_type_hints

__all__ = [
    "FrontendConfig",
    "MixerConfig",
    "RecurrentCoreConfig",
    "FeedForwardConfig",
    "MemoryConfig",
    "HeadsConfig",
    "ModalityConfig",
    "ProphetConfig",
]


# --------------------------------------------------------------------------------------
# 1. Frontend — how raw input becomes model-dimension vectors (track R01)
# --------------------------------------------------------------------------------------


@dataclass
class FrontendConfig:
    """Input representation.

    ``mode`` selects the bet:

    - ``"bpe"``: conventional subword embedding table. The baseline, and the fallback
      if dynamic patching fails its ablation.
    - ``"byte_patch"``: byte-level input grouped into variable-length patches by a
      small entropy model, with a local encoder/decoder around the main trunk. Removes
      the vocabulary parameter tax and the character-blindness class of failures.
    - ``"hybrid"``: a small subword vocabulary with byte fallback, patched on top.
    """

    mode: Literal["bpe", "byte_patch", "hybrid"] = "bpe"

    # --- "bpe" / "hybrid" ---
    vocab_size: int = 49152
    tie_word_embeddings: bool = True

    # --- "byte_patch" / "hybrid" ---
    byte_vocab_size: int = 260  # 256 bytes + BOS/EOS/PAD/MASK
    patch_target_bytes: float = 4.5
    """Average bytes per patch the entropy router is calibrated to produce."""
    patch_max_bytes: int = 16
    patch_entropy_threshold: float | None = None
    """Absolute entropy threshold in nats; ``None`` calibrates it to hit ``patch_target_bytes``."""
    local_encoder_layers: int = 2
    local_decoder_layers: int = 3
    local_dim: int = 512
    local_heads: int = 8
    local_window: int = 128
    """Byte-level attention window inside the local encoder/decoder."""
    hash_ngram_sizes: tuple[int, ...] = (3, 4, 5, 6)
    hash_ngram_buckets: int = 32768
    """Hash-embedding buckets *per n-gram size*, giving byte n-gram context cheaply.

    Cost is ``len(hash_ngram_sizes) * hash_ngram_buckets * local_dim`` parameters, which
    is easy to make accidentally enormous — the published byte-level models size these
    tables for 8B-parameter trunks. At our scale the tables must stay a small fraction of
    the model, so :func:`prophet.budget.allocation_warnings` flags configurations where
    the frontend eats the parameter budget the trunk needs."""


# --------------------------------------------------------------------------------------
# 2. Sequence mixing — the attention / recurrence stack (track R02)
# --------------------------------------------------------------------------------------

MixerKind = Literal["full_attn", "swa", "gdn", "mamba2", "identity"]


@dataclass
class MixerConfig:
    """Per-layer sequence mixer.

    The stack is described by ``pattern``, a list of mixer kinds cycled over the
    trunk's depth. ``["gdn", "gdn", "gdn", "full_attn"]`` produces the 3:1 hybrid
    interleave that shipped in several 2025 hybrid models: bounded-state recurrence
    carries most layers, periodic full attention restores exact recall.
    """

    pattern: list[MixerKind] = field(default_factory=lambda: ["full_attn"])

    # --- softmax attention (``full_attn`` / ``swa``) ---
    n_heads: int = 16
    n_kv_heads: int = 4
    """Grouped-query attention: ``n_heads // n_kv_heads`` queries share each KV head."""
    head_dim: int | None = None
    """Defaults to ``d_model // n_heads``. Kept explicit so it can be pinned to a
    hardware-friendly value (64 or 128) independently of model width."""
    qk_norm: bool = True
    """RMSNorm on queries and keys. Cheap, and it removes the activation outliers that
    make small models quantise badly (track R08)."""
    sliding_window: int = 4096
    attention_sink_tokens: int = 4
    """Always-attended prefix tokens; prevents the softmax-sink collapse that breaks
    windowed attention at long context."""
    kv_compression: Literal["none", "mla"] = "none"
    kv_lora_rank: int = 512
    """Latent dimension when ``kv_compression == "mla"``."""

    # --- gated linear recurrence (``gdn`` / ``mamba2``) ---
    linear_heads: int = 8
    linear_head_dim: int = 128
    linear_expand: float = 2.0
    conv_kernel: int = 4
    """Short causal depthwise convolution before the recurrence; standard in this family
    and worth several points of local-pattern quality."""
    linear_beta_max: float = 2.0
    """Upper bound of the delta-rule write strength.

    At 1.0 every eigenvalue of the state transition is strictly positive, and a product of
    such transitions provably cannot express parity or other sign-flipping problems. At
    2.0 the transition may reflect, and those problems come back within reach. This costs
    one multiplication and is the difference between chance and 0.9+ on length-generalised
    parity, so 1.0 is kept only as an ablation arm."""

    # --- positions ---
    rope_theta: float = 500_000.0
    rope_scaling: Literal["none", "yarn", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    nope_layers: tuple[int, ...] = ()
    """Indices of attention layers that get *no* positional encoding. A small number of
    NoPE layers markedly improves length extrapolation in hybrid stacks."""


# --------------------------------------------------------------------------------------
# 3. Recurrent core — depth as a runtime dial (track R04)
# --------------------------------------------------------------------------------------


@dataclass
class RecurrentCoreConfig:
    """Weight-shared recurrent trunk.

    The central bet: instead of ``n_layers`` distinct blocks, use a prelude, a shared
    core of ``core_layers`` blocks applied ``k`` times, and a coda. Effective depth is
    ``prelude + core_layers * k + coda`` while parameter count only pays for
    ``prelude + core_layers + coda``. ``k`` is chosen *at inference time*, which is what
    lets one set of weights serve an iPhone and an RTX 5090.
    """

    enabled: bool = False
    prelude_layers: int = 2
    core_layers: int = 4
    coda_layers: int = 2

    train_loop_min: int = 1
    train_loop_max: int = 8
    train_loop_dist: Literal["uniform", "log_uniform", "poisson"] = "log_uniform"
    """Recurrence depth is sampled per micro-batch during training so that the model is
    usable at any depth, and so that gradient cost stays bounded."""
    train_loop_poisson_lambda: float = 4.0

    default_loop_k: int = 4
    """Inference default when the caller does not specify a depth."""

    core_pattern: list[MixerKind] | None = None
    """Mixer pattern for the looped core, overriding ``mixer.pattern``.

    This exists because of a hard memory result. Every loop iteration of an
    *attention* layer needs its own KV cache — iteration *i* asks different questions
    than iteration *j* — so looping attention multiplies KV memory by ``k``. Looping a
    bounded-state recurrent layer multiplies a few kilobytes instead. The default
    Prophet stack therefore puts attention in the prelude and coda, where it is applied
    exactly once, and keeps the looped core purely recurrent: full-attention recall and
    cheap deep recurrence at the same time.
    """
    prelude_pattern: list[MixerKind] | None = None
    coda_pattern: list[MixerKind] | None = None

    inject_input_each_step: bool = True
    """Re-add the prelude output at every iteration. Without it, deep recurrence drifts
    away from the input and the loop stops being conditioned on the prompt."""
    state_init: Literal["zeros", "randn", "prelude"] = "randn"
    """Initial core state **during training**. Random init is a regulariser: it forces the
    loop to converge to the same answer from any starting point, which is what makes the
    depth dial safe to move at inference."""
    eval_state_init: Literal["zeros", "randn", "prelude"] = "zeros"
    """Initial core state at inference. Must be deterministic, or the same prompt gives
    different answers on every call and incremental decoding stops matching a full
    forward pass — a bug that produces fluent, plausible, wrong output and is therefore
    very hard to notice without an equivalence test."""

    truncated_backprop_steps: int = 4
    """Backpropagate through at most this many trailing iterations; earlier ones run
    under ``no_grad``. This is what makes deep loops affordable on one A100."""

    halting: Literal["none", "ponder", "entropy"] = "none"
    halting_loss_weight: float = 0.01
    halting_target_steps: float = 4.0

    token_depth: bool = False
    """Train with a per-token depth *ceiling* so that inference may vary depth per call
    and per token on one cache.

    Without this, a cache's depth is fixed for its lifetime (it may only shrink by
    halting): iteration ``i``'s core state must have seen every earlier token, or it
    reads a state that skipped part of the context. With it, the invariant becomes
    "iteration ``i`` saw every earlier token whose ceiling exceeds ``i``": at each
    iteration the core runs on the compacted subsequence of tokens still active, and the
    model is trained that way, so a tool observation ingested at depth 1 followed by a
    think span at depth 8 is in-distribution rather than undefined. This is what the
    agent loop's "ingest cheap, think deep" schedule needs (docs/08_AGENT.md). Unvalidated:
    an ablation against constant-depth training is required before it carries a run."""
    ingest_depth: int = 1
    """Ceiling applied, when ``token_depth`` is on, to tool-observation spans (from
    ``<|tool|>`` to the next control token) and to the random shallow spans below."""
    token_depth_random_spans: float = 0.25
    """Probability that a training sequence gets one random contiguous span at
    ``ingest_depth``. Pretraining text has no ``<|tool|>`` spans, and the model must
    still meet the shallow-then-deep transition there or the agent loop's first
    observation is its first sight of one. A guess, pending the ablation above."""


# --------------------------------------------------------------------------------------
# 4. Feed-forward / sparsity (track R05)
# --------------------------------------------------------------------------------------


@dataclass
class FeedForwardConfig:
    """Channel mixing, dense or sparse."""

    kind: Literal["dense", "moe"] = "dense"
    activation: Literal["swiglu", "geglu", "relu2"] = "swiglu"
    hidden_mult: float = 4.0
    """Dense FFN inner dimension as a multiple of ``d_model`` (before the 2/3 SwiGLU
    correction, which is applied automatically)."""

    # --- MoE ---
    n_experts: int = 64
    n_experts_per_token: int = 6
    n_shared_experts: int = 2
    """Always-on experts. They absorb the knowledge every token needs, letting the routed
    experts specialise instead of all relearning the same basics."""
    expert_hidden_mult: float = 0.5
    """Fine-grained experts: many small ones beat few large ones at equal active params."""
    moe_layer_stride: int = 1
    """Place an MoE layer every ``stride`` blocks; 1 means all of them."""
    moe_first_dense_layers: int = 1
    """Keep the first N blocks dense — routing on barely-formed representations is unstable."""
    router_bias_balancing: bool = True
    """Loss-free load balancing by an updated per-expert routing bias, rather than an
    auxiliary loss that fights the language-modelling objective."""
    router_bias_update_rate: float = 1e-3
    router_z_loss_weight: float = 1e-3
    load_balance_loss_weight: float = 0.0
    router_dtype: Literal["float32", "bfloat16"] = "float32"


# --------------------------------------------------------------------------------------
# 5. Persistent memory (track R03)
# --------------------------------------------------------------------------------------


@dataclass
class MemoryConfig:
    """Test-time-updatable memory: the anti-'frozen brain' bet.

    Tier 1 (``fast_weight``) is an in-layer associative memory whose contents are written
    during the forward pass and survive across a session. Tier 2 is an offline
    consolidation pass, run outside the model, that distils accumulated session memory
    into a sparse weight delta.
    """

    enabled: bool = False
    kind: Literal["none", "fast_weight", "product_key"] = "none"
    mount: Literal["output", "coda"] = "output"
    """Where the ledger reads and writes.

    ``"output"``: one ledger on the normalised final hidden state, added before the LM
    head. This is the space every function in ``prophet.memory.consolidate`` addresses
    and targets, so a ledger consolidated offline can be mounted and read by the model
    without translation. ``"coda"``: one ledger per index in ``layers``, reading each
    coda block's residual output -- a different space, unsupported by consolidation, and
    kept only as an ablation arm."""
    layers: tuple[int, ...] = ()
    """Coda block indices carrying a ledger when ``mount == "coda"``. Ignored otherwise.
    Validated against the coda's actual length, not the global depth: a ledger declared
    at a global index the model never reads was a no-op that budgeted parameters."""

    memory_dim: int = 512
    n_slots: int = 4096
    update_rule: Literal["delta", "hebbian", "surprise_gated"] = "delta"
    write_lr: float = 1.0
    """Fraction of the exact local write step. Aligned with ``LedgerConfig``: every
    number in ``docs/06_MEMORY.md`` was measured at 1.0, and the model once built its
    ledger at 0.01 -- one percent of the step the docs describe."""
    decay: float = 1.0
    """Forgetting factor per write; without it the memory saturates and stops discriminating."""
    surprise_threshold: float = 1.0
    """In ``surprise_gated`` mode, only write when the token's loss exceeds this — memory
    capacity is spent on what the weights did not already know."""

    persist_across_sessions: bool = True
    max_persisted_writes: int = 1_000_000


# --------------------------------------------------------------------------------------
# 6. Auxiliary heads (tracks R08, R09)
# --------------------------------------------------------------------------------------


@dataclass
class HeadsConfig:
    """Output heads beyond the language-modelling head."""

    n_multi_token_predict: int = 0
    """Extra heads predicting tokens t+2..t+n. Nearly free to train, they densify the
    training signal *and* provide a draft model for speculative decoding at inference."""
    mtp_loss_weight: float = 0.3

    confidence_head: bool = False
    """Predicts calibrated P(the answer just produced is correct), driving abstention and
    retrieval triggering rather than letting the model guess confidently."""
    confidence_loss_weight: float = 0.1


# --------------------------------------------------------------------------------------
# 7. Modality hooks (track R12)
# --------------------------------------------------------------------------------------


@dataclass
class ModalityConfig:
    """Reserved structure so non-text modalities can be added later without retraining
    the trunk. Costs almost nothing now; costs a full retrain if omitted now."""

    n_modalities: int = 1
    modality_embeddings: bool = True
    """A learned per-modality bias added to inputs, so modality is a first-class signal."""
    position_dims: int = 1
    """1 for text; 3 reserves (t, y, x) so 2-D position encodings work for images later."""
    bidirectional_spans: bool = True
    """Allow marked spans to attend bidirectionally — required for image patches, and
    useful for text infilling in the meantime."""
    adapter_mount_points: bool = True
    """Emit per-layer hooks where modality LoRA adapters can attach."""


# --------------------------------------------------------------------------------------
# 8. Top level
# --------------------------------------------------------------------------------------


@dataclass
class ProphetConfig:
    """Full model description."""

    name: str = "prophet-baseline"

    d_model: int = 1024
    n_layers: int = 16
    """Trunk depth when ``recurrent.enabled`` is False. Otherwise derived from the
    recurrent core (see :meth:`effective_depth`)."""
    norm_eps: float = 1e-5
    norm_kind: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    residual_scaling: bool = True
    """Scale residual branches by 1/sqrt(2 * depth) at init — keeps activation variance
    bounded in deep or heavily looped stacks."""
    logit_soft_cap: float | None = None
    z_loss_weight: float = 1e-4

    max_seq_len: int = 4096
    dropout: float = 0.0

    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    mixer: MixerConfig = field(default_factory=MixerConfig)
    recurrent: RecurrentCoreConfig = field(default_factory=RecurrentCoreConfig)
    ffn: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)

    init_std: float = 0.02
    mup_base_width: int | None = None
    """Enables muP scaling of init and learning rates against this reference width, so
    hyperparameters found on a small proxy transfer to the full model without a sweep."""

    # ---- derived quantities ----------------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.mixer.head_dim or (self.d_model // self.mixer.n_heads)

    def effective_depth(self, loop_k: int | None = None) -> int:
        """Number of block applications in a forward pass."""
        if not self.recurrent.enabled:
            return self.n_layers
        k = loop_k if loop_k is not None else self.recurrent.default_loop_k
        r = self.recurrent
        return r.prelude_layers + r.core_layers * k + r.coda_layers

    def parameterised_depth(self) -> int:
        """Number of blocks that own weights (what memory actually pays for)."""
        if not self.recurrent.enabled:
            return self.n_layers
        r = self.recurrent
        return r.prelude_layers + r.core_layers + r.coda_layers

    def layer_mixer(self, index: int, section: str = "trunk") -> MixerKind:
        """Mixer kind for a parameterised block.

        ``section`` is one of ``"trunk"``, ``"prelude"``, ``"core"`` or ``"coda"``; the
        recurrent sections may override the global pattern (see
        :attr:`RecurrentCoreConfig.core_pattern`).
        """
        pattern = self.mixer.pattern
        if self.recurrent.enabled:
            override = {
                "prelude": self.recurrent.prelude_pattern,
                "core": self.recurrent.core_pattern,
                "coda": self.recurrent.coda_pattern,
            }.get(section)
            if override:
                pattern = override
        if not pattern:
            raise ValueError("mixer pattern must not be empty")
        return pattern[index % len(pattern)]

    def layer_uses_rope(self, index: int, section: str = "trunk") -> bool:
        """Whether the attention block at this position gets rotary positions.

        ``nope_layers`` holds *pattern positions*, not absolute block indices, so that
        "every global layer is position-free" is one entry rather than one per block.
        With ``pattern=["swa", "full_attn"]`` and ``nope_layers=(1,)``, every block
        produced by pattern slot 1 -- each full-attention layer -- runs without RoPE. That
        is the R02 design: local layers keep positions, global layers extrapolate freely.
        """
        pattern = self.mixer.pattern
        if self.recurrent.enabled:
            override = {
                "prelude": self.recurrent.prelude_pattern,
                "core": self.recurrent.core_pattern,
                "coda": self.recurrent.coda_pattern,
            }.get(section)
            if override:
                pattern = override
        if not pattern:
            return True
        return (index % len(pattern)) not in self.mixer.nope_layers

    def section_layout(self) -> list[tuple[str, int, MixerKind]]:
        """The parameterised blocks in order, as ``(section, index_in_section, kind)``.

        This is the single source of truth for how blocks are laid out, shared by the
        model, the budget calculator and the tests.
        """
        if not self.recurrent.enabled:
            return [("trunk", i, self.layer_mixer(i)) for i in range(self.n_layers)]
        r = self.recurrent
        out: list[tuple[str, int, MixerKind]] = []
        for name, count in (
            ("prelude", r.prelude_layers),
            ("core", r.core_layers),
            ("coda", r.coda_layers),
        ):
            for i in range(count):
                out.append((name, i, self.layer_mixer(i, name)))
        return out

    def cache_slots(self, loop_k: int | None = None) -> list[tuple[str, int, int, MixerKind]]:
        """Every distinct cache a forward pass needs: ``(section, block, iteration, kind)``.

        A looped core block needs one cache *per iteration*, because its inputs differ
        each time round. This is why the default core is recurrent rather than attentive:
        the per-iteration cost is a fixed-size state, not a growing KV cache.
        """
        k = 1
        if self.recurrent.enabled:
            k = loop_k if loop_k is not None else self.recurrent.default_loop_k
        slots: list[tuple[str, int, int, MixerKind]] = []
        for section, idx, kind in self.section_layout():
            iterations = k if section == "core" else 1
            for it in range(iterations):
                slots.append((section, idx, it, kind))
        return slots

    def layer_is_moe(self, index: int) -> bool:
        if self.ffn.kind != "moe":
            return False
        if index < self.ffn.moe_first_dense_layers:
            return False
        return (index - self.ffn.moe_first_dense_layers) % self.ffn.moe_layer_stride == 0

    # ---- validation ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise on configurations that are structurally impossible.

        This runs before any expensive job so that a typo in a YAML costs a second
        rather than an A100-hour.
        """
        errors: list[str] = []

        if self.d_model % self.mixer.n_heads != 0 and self.mixer.head_dim is None:
            errors.append(
                f"d_model={self.d_model} is not divisible by n_heads={self.mixer.n_heads}; "
                "set mixer.head_dim explicitly"
            )
        if self.mixer.n_heads % self.mixer.n_kv_heads != 0:
            errors.append(
                f"n_heads={self.mixer.n_heads} must be divisible by "
                f"n_kv_heads={self.mixer.n_kv_heads}"
            )
        if not self.mixer.pattern:
            errors.append("mixer.pattern must contain at least one entry")

        if self.recurrent.enabled:
            r = self.recurrent
            if r.core_layers < 1:
                errors.append("recurrent.core_layers must be >= 1 when recurrence is enabled")
            if r.train_loop_min < 1 or r.train_loop_max < r.train_loop_min:
                errors.append(
                    f"require 1 <= train_loop_min ({r.train_loop_min}) "
                    f"<= train_loop_max ({r.train_loop_max})"
                )
            if r.truncated_backprop_steps < 1:
                errors.append("recurrent.truncated_backprop_steps must be >= 1")
            if r.token_depth:
                if not (1 <= r.ingest_depth <= r.train_loop_max):
                    errors.append(
                        f"recurrent.ingest_depth ({r.ingest_depth}) must lie in "
                        f"[1, train_loop_max={r.train_loop_max}]"
                    )
                if not (0.0 <= r.token_depth_random_spans <= 1.0):
                    errors.append("recurrent.token_depth_random_spans must be a probability")
                core = r.core_pattern or self.mixer.pattern
                if any(kind in ("full_attn", "swa") for kind in core):
                    errors.append(
                        "recurrent.token_depth compacts the active tokens at each "
                        "iteration, which a recurrent core supports and an attention "
                        "core does not (its positions would have to be gathered too)"
                    )

        if self.ffn.kind == "moe":
            f = self.ffn
            if f.n_experts_per_token > f.n_experts:
                errors.append(
                    f"n_experts_per_token ({f.n_experts_per_token}) exceeds "
                    f"n_experts ({f.n_experts})"
                )
            if f.n_experts_per_token < 1:
                errors.append("n_experts_per_token must be >= 1")

        if self.memory.enabled:
            if self.memory.kind == "none":
                errors.append("memory.enabled is True but memory.kind is 'none'")
            if self.memory.kind == "fast_weight":
                errors.append(
                    "memory.kind='fast_weight' is declared but not implemented: it "
                    "validated, was budgeted, and built nothing. Use 'product_key'."
                )
            if self.memory.mount == "coda":
                n_coda = (
                    self.recurrent.coda_layers if self.recurrent.enabled else self.n_layers
                )
                bad = [i for i in self.memory.layers if not 0 <= i < n_coda]
                if bad:
                    errors.append(
                        f"memory.layers {bad} are outside the coda ({n_coda} blocks); "
                        "indices are coda-local, not global"
                    )
                if not self.memory.layers:
                    errors.append("memory.mount='coda' needs at least one index in layers")

        if self.frontend.mode == "byte_patch" and self.frontend.patch_max_bytes < 1:
            errors.append("frontend.patch_max_bytes must be >= 1")

        if errors:
            raise ValueError(
                "Invalid ProphetConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def design_warnings(self) -> list[str]:
        """Flag configurations that contradict the architecture's own invariants.

        These are not structural errors -- every one of them trains perfectly well and
        produces a plausible loss curve. They are silent design mistakes, which is worse:
        a run configured this way confounds every ablation built on top of it, and nothing
        in the metrics says so. This check exists because exactly such a mistake shipped
        in ``configs/prophet_500m_probe.json``, where the only full-attention layer sat
        inside the looped core.
        """
        out: list[str] = []
        layout = self.section_layout()

        if self.recurrent.enabled:
            core_attn = [
                f"core[{i}]" for sec, i, kind in layout
                if sec == "core" and kind in ("full_attn", "swa")
            ]
            if core_attn:
                k = self.recurrent.default_loop_k
                out.append(
                    f"attention inside the looped core ({', '.join(core_attn)}): its KV "
                    f"cache is duplicated per iteration, so memory scales with k (x{k} at "
                    "the default depth). This is decision D1's invariant; enable it only "
                    "as the deliberate A-KV ablation."
                )
            outer_attn = any(
                kind in ("full_attn", "swa")
                for sec, _, kind in layout
                if sec in ("prelude", "coda")
            )
            if not outer_attn:
                out.append(
                    "no attention in the prelude or coda: the model has no exact-recall "
                    "layer that runs at constant cost, which is what those sections are for"
                )

        if not any(kind == "full_attn" for _, _, kind in layout):
            out.append(
                "no full-attention layer anywhere: bounded-state mixers alone collapse on "
                "multi-key retrieval, and the fix is more global layers, not a wider window"
            )

        n_full = sum(1 for _, _, kind in layout if kind == "full_attn")
        if n_full and not self.mixer.nope_layers:
            out.append(
                "full-attention layers are present but nope_layers is empty: without a "
                "position-free global layer, length extrapolation has to be bought with a "
                "context-extension run instead of coming for free"
            )

        if self.mixer.linear_beta_max <= 1.0 and any(
            kind in ("gdn", "mamba2") for _, _, kind in layout
        ):
            out.append(
                "linear_beta_max <= 1.0 keeps every state-transition eigenvalue positive, "
                "which provably cannot express parity or other sign-flipping problems; "
                "2.0 costs one multiplication"
            )

        return out

    # ---- serialisation ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=list), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProphetConfig":
        return _build(cls, data)

    @classmethod
    def from_json(cls, path: str | Path) -> "ProphetConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Reconstruct a nested dataclass tree, tolerating missing keys (use the default).

    ``from __future__ import annotations`` turns field types into strings, so the real
    types are recovered with :func:`typing.get_type_hints` rather than read off
    ``field.type``.
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        elif isinstance(value, list) and _origin_is_tuple(ftype):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _origin_is_tuple(ftype: Any) -> bool:
    """True for ``tuple[...]`` annotations, which JSON round-trips as lists."""
    return get_origin(ftype) is tuple
