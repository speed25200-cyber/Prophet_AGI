"""Parameter, FLOP, memory and throughput accounting for Prophet configurations.

Every claim in this project has to survive arithmetic before it costs an A100-hour.
This module is that arithmetic: given a :class:`~prophet.config.ProphetConfig` it
reports what the model costs to store, to train, and to run on each target device.

It deliberately depends only on the standard library, so it runs anywhere — including
before a single dependency is installed.

Usage::

    python -m prophet.budget configs/prophet_1b3.json
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from prophet.config import ProphetConfig

__all__ = [
    "expected_train_loop_k",
    "block_passes_per_token",
    "Device",
    "DEVICES",
    "ParamBreakdown",
    "TrainingMemory",
    "InferenceProfile",
    "count_parameters",
    "training_memory",
    "inference_profile",
    "tokens_affordable",
    "report",
]


# --------------------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """A target device, described by the two numbers that actually bind.

    Decoding a single token is memory-bandwidth bound, not compute bound: the model
    must stream every active weight and the whole KV state per token. So ``bandwidth``
    sets the token-rate ceiling and ``memory_gb`` sets what fits at all.
    """

    name: str
    memory_gb: float
    """Usable memory, not nameplate capacity."""
    bandwidth_gb_s: float
    bf16_tflops: float
    """Dense BF16 matmul throughput, for prefill and training estimates."""
    notes: str = ""


DEVICES: dict[str, Device] = {
    "a100_80gb": Device(
        "A100 80GB SXM", memory_gb=79.0, bandwidth_gb_s=2039.0, bf16_tflops=312.0,
        notes="Ampere sm_80. Training target. No FP8 hardware, no FlashAttention-3.",
    ),
    "rtx5090": Device(
        "RTX 5090", memory_gb=30.0, bandwidth_gb_s=1792.0, bf16_tflops=210.0,
        notes="Blackwell sm_120, native FP4 tensor cores. ~2GB reserved for the OS/driver.",
    ),
    "mac_studio_ultra": Device(
        "Mac Studio (M-Ultra)", memory_gb=100.0, bandwidth_gb_s=800.0, bf16_tflops=55.0,
        notes="Unified memory; capacity is generous, bandwidth is the constraint.",
    ),
    "iphone17pro": Device(
        "iPhone 17 Pro", memory_gb=4.0, bandwidth_gb_s=90.0, bf16_tflops=10.0,
        notes="~8GB unified RAM; a foreground app realistically gets 3-5GB before jetsam.",
    ),
}

BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
    # Group-quantised 4-bit carries a scale (and often a zero-point) per group of 32-64
    # weights, so the real cost is above the nominal 0.5 bytes.
    "int4_g64": 0.5 + 2.0 / 64,
    "nvfp4": 0.5 + 1.0 / 16,
}

GIB = 1024.0**3


# --------------------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------------------


@dataclass
class ParamBreakdown:
    """Parameter counts, split into what memory pays for vs what each token pays for."""

    by_component: dict[str, int] = field(default_factory=dict)

    total: int = 0
    """Every parameter that must be resident in memory."""
    active_per_token: int = 0
    """Parameters read to produce one token, at the default recurrence depth. This is
    the number that sets decode speed."""
    embedding: int = 0
    non_embedding: int = 0

    def add(self, component: str, n: int) -> None:
        self.by_component[component] = self.by_component.get(component, 0) + n


def _swiglu_hidden(d_model: int, mult: float) -> int:
    """SwiGLU inner width, with the conventional 2/3 correction and rounding to 128.

    SwiGLU uses three matrices instead of two, so the inner width is scaled by 2/3 to
    keep the parameter count comparable to a plain 4x FFN.
    """
    hidden = int(2 * mult * d_model / 3)
    return max(128, ((hidden + 127) // 128) * 128)


def _attention_params(cfg: ProphetConfig) -> int:
    d, hd = cfg.d_model, cfg.head_dim
    m = cfg.mixer
    q = d * m.n_heads * hd
    k = d * m.n_kv_heads * hd
    v = d * m.n_kv_heads * hd
    o = m.n_heads * hd * d
    if m.kv_compression == "mla":
        # Down-project to a shared latent, then up-project per head. Cuts the KV cache by
        # roughly an order of magnitude at a small parameter cost.
        r = m.kv_lora_rank
        k = v = 0
        kv = d * r + r * (m.n_kv_heads * hd) * 2
        return q + kv + o + (2 * hd if m.qk_norm else 0)
    return q + k + v + o + (2 * hd if m.qk_norm else 0)


def _linear_mixer_params(cfg: ProphetConfig) -> int:
    """Gated DeltaNet / Mamba-2 style bounded-state recurrence."""
    d = cfg.d_model
    m = cfg.mixer
    inner = m.linear_heads * m.linear_head_dim
    v_inner = int(inner * m.linear_expand)
    qk = 2 * d * inner
    v = d * v_inner
    gates = 2 * d * m.linear_heads  # decay + write strength, one scalar per head
    conv = (2 * inner + v_inner) * m.conv_kernel
    out = v_inner * d
    return qk + v + gates + conv + out


def _ffn_params(cfg: ProphetConfig, is_moe: bool) -> tuple[int, int]:
    """Return ``(resident, active_per_token)`` parameters for one channel-mixing block."""
    d = cfg.d_model
    f = cfg.ffn
    if not is_moe:
        h = _swiglu_hidden(d, f.hidden_mult)
        n = 3 * d * h
        return n, n

    eh = _swiglu_hidden(d, f.expert_hidden_mult)
    per_expert = 3 * d * eh
    router = d * f.n_experts
    resident = per_expert * (f.n_experts + f.n_shared_experts) + router
    active = per_expert * (f.n_experts_per_token + f.n_shared_experts) + router
    return resident, active


def _memory_params(cfg: ProphetConfig) -> int:
    mem = cfg.memory
    if not mem.enabled or mem.kind == "none":
        return 0
    d = cfg.d_model
    if mem.kind == "fast_weight":
        # Query/key/value projections into the memory space plus the read-out.
        return 3 * d * mem.memory_dim + mem.memory_dim * d
    # product_key: two half-key codebooks addressing n_slots values.
    n_sqrt = int(mem.n_slots**0.5) + 1
    return d * mem.memory_dim + 2 * n_sqrt * (mem.memory_dim // 2) + mem.n_slots * d


def _frontend_params(cfg: ProphetConfig) -> tuple[int, int]:
    """Return ``(resident, active_per_token)`` for the input/output frontend."""
    fe = cfg.frontend
    d = cfg.d_model

    if fe.mode == "bpe":
        emb = fe.vocab_size * d
        head = 0 if fe.tie_word_embeddings else fe.vocab_size * d
        return emb + head, emb + head

    # Byte-level: a 260-entry embedding costs nothing, but we pay for hash n-gram
    # embeddings and for the local encoder/decoder transformers around the trunk.
    ld = fe.local_dim
    byte_emb = fe.byte_vocab_size * ld
    ngram_emb = len(fe.hash_ngram_sizes) * fe.hash_ngram_buckets * ld

    def _local_block(dim: int, heads: int) -> int:
        head_dim = dim // heads
        attn = 4 * dim * dim + 2 * head_dim
        ffn = 3 * dim * _swiglu_hidden(dim, 4.0)
        return attn + ffn

    enc = fe.local_encoder_layers * _local_block(ld, fe.local_heads)
    dec = fe.local_decoder_layers * _local_block(ld, fe.local_heads)
    # Cross-attention pooling bytes -> patch, and the projections between local and
    # trunk widths.
    cross = 2 * (4 * ld * ld) + 2 * ld * d
    out_head = ld * fe.byte_vocab_size

    resident = byte_emb + ngram_emb + enc + dec + cross + out_head
    # Hash n-gram tables are a lookup: only a handful of rows are touched per byte.
    active = resident - ngram_emb + len(fe.hash_ngram_sizes) * ld
    if fe.mode == "hybrid":
        resident += fe.vocab_size * d
        active += fe.vocab_size * d
    return resident, active


def count_parameters(cfg: ProphetConfig, loop_k: int | None = None) -> ParamBreakdown:
    """Full parameter accounting for a configuration.

    ``active_per_token`` counts a weight-shared recurrent block **once**, because memory
    stores it once — but the FLOPs to apply it scale with the loop depth. See
    :func:`inference_profile`, which separates the two.
    """
    cfg.validate()
    out = ParamBreakdown()
    d = cfg.d_model

    fe_resident, fe_active = _frontend_params(cfg)
    out.add("frontend/embeddings", fe_resident)
    out.embedding = fe_resident

    attn_p = _attention_params(cfg)
    lin_p = _linear_mixer_params(cfg)

    # Walk the section-aware layout, never ``layer_mixer(i)`` over a flat depth: the
    # looped core overrides the global pattern, and reading the global pattern here once
    # counted four recurrent core blocks as four attention blocks -- 31M parameters short
    # on the shipped probe config, and wrong in the direction that flatters memory.
    trunk_resident = 0
    trunk_active = 0
    for i, (_section, _idx, kind) in enumerate(cfg.section_layout()):
        mixer = 0 if kind == "identity" else (lin_p if kind in ("gdn", "mamba2") else attn_p)
        out.add(f"mixer/{kind}", mixer)
        ff_res, ff_act = _ffn_params(cfg, cfg.layer_is_moe(i))
        out.add("ffn/moe" if cfg.layer_is_moe(i) else "ffn/dense", ff_res)
        mem = _memory_params(cfg) if i in cfg.memory.layers else 0
        if mem:
            out.add("memory", mem)
        norms = 2 * d
        out.add("norms", norms)
        trunk_resident += mixer + ff_res + mem + norms
        trunk_active += mixer + ff_act + mem + norms

    heads = 0
    if cfg.heads.n_multi_token_predict:
        # Each extra prediction head is one transformer block plus a shared output
        # projection; cheap to train, and it doubles as a speculative-decoding draft.
        per_head = _attention_params(cfg) + 3 * d * _swiglu_hidden(d, cfg.ffn.hidden_mult)
        heads += cfg.heads.n_multi_token_predict * per_head
    if cfg.heads.confidence_head:
        heads += 2 * d + 1  # RMSNorm gain + Linear(d, 1)
    if cfg.recurrent.enabled and cfg.recurrent.halting == "ponder":
        heads += 2 * d + 1  # the halting head has the same shape
    if cfg.heads.action_head:
        # Track A3: a shared norm, two selection projections plus the null key, two copy
        # queries against an existing attention layer's keys, and the gate.
        dk, hd = cfg.heads.action_dk, cfg.head_dim
        heads += d + 2 * d * dk + dk + 2 * d * hd + (d + 1)
    if heads:
        out.add("aux_heads", heads)

    if cfg.frontend.mode == "bpe" and not cfg.frontend.tie_word_embeddings:
        pass  # already counted in the frontend

    out.total = fe_resident + trunk_resident + heads + d
    # Aux heads are not run during ordinary decoding (MTP heads only matter when
    # speculating), so they are excluded from the per-token active count.
    out.active_per_token = fe_active + trunk_active + d
    out.non_embedding = out.total - out.embedding
    return out


# --------------------------------------------------------------------------------------
# Training memory
# --------------------------------------------------------------------------------------


@dataclass
class TrainingMemory:
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    total_gb: float
    fits: bool
    device: str
    detail: dict[str, float] = field(default_factory=dict)


def _optimizer_bytes_per_param(cfg: ProphetConfig) -> float:
    """Optimiser state per parameter for the trainer as it actually exists.

    Muon keeps one fp32 momentum buffer (4 bytes) on the 2-D hidden matrices; AdamW keeps
    two fp32 moments (8 bytes) on embeddings, norms, biases and heads. Weighted by the
    parameter split, not assumed: the gate in scripts/train.py once said "fits" at 2
    bytes/param for an 8-bit optimiser that nothing implements.
    """
    p = count_parameters(cfg)
    adamw_params = p.embedding + p.by_component.get("norms", 0) + p.by_component.get("aux_heads", 0)
    muon_params = max(p.total - adamw_params, 0)
    return (muon_params * 4.0 + adamw_params * 8.0) / max(p.total, 1)


def training_memory(
    cfg: ProphetConfig,
    *,
    device: str = "a100_80gb",
    batch_tokens: int = 16384,
    param_dtype: str = "fp32",
    grad_dtype: str = "fp32",
    optimizer_bytes_per_param: float | None = None,
    master_weights: bool = False,
    activation_checkpointing: bool = True,
    loop_k: int | None = None,
) -> TrainingMemory:
    """Estimate peak training memory for one micro-batch.

    Defaults describe the trainer **as implemented**: fp32 parameters and gradients
    (bf16 autocast keeps the params fp32 -- they *are* the master copy, so no separate
    master is added), optimiser bytes from the real Muon/AdamW split, activation
    checkpointing on. Pass other values to explore, never to make a run "fit".

    Activation cost is estimated per *effective* block pass that participates in
    backprop, including the extra coda passes learned halting runs -- this is exactly
    what ``recurrent.truncated_backprop_steps`` exists to bound.
    """
    dev = DEVICES[device]
    p = count_parameters(cfg, loop_k=loop_k)
    n = p.total
    if optimizer_bytes_per_param is None:
        optimizer_bytes_per_param = _optimizer_bytes_per_param(cfg)

    weights = n * BYTES_PER_PARAM[param_dtype]
    if master_weights and param_dtype != "fp32":
        weights += n * 4.0
    grads = n * BYTES_PER_PARAM[grad_dtype]
    optim = n * optimizer_bytes_per_param

    # Activations. With checkpointing we keep one tensor per block boundary and
    # recompute inside; without it, roughly a dozen d_model-sized tensors per block.
    if cfg.recurrent.enabled:
        r = cfg.recurrent
        backprop_steps = min(r.truncated_backprop_steps, loop_k or r.default_loop_k)
        act_layers = r.prelude_layers + r.core_layers * backprop_steps + r.coda_layers
        if r.halting == "ponder":
            # Every probe coda pass inside the backprop window keeps its activations.
            act_layers += r.coda_layers * backprop_steps
    else:
        act_layers = cfg.n_layers
    act_layers += cfg.heads.n_multi_token_predict

    per_layer_elems = 2.0 if activation_checkpointing else 14.0
    act = batch_tokens * cfg.d_model * per_layer_elems * act_layers * 2.0  # bf16

    # Attention scores are materialised only if a fused kernel is unavailable; assume
    # FlashAttention-style kernels, whose extra memory is O(seq) not O(seq^2).
    act += batch_tokens * cfg.d_model * 4 * 2.0

    if cfg.ffn.kind == "moe":
        # Permutation buffers for grouped-GEMM expert dispatch.
        act += batch_tokens * cfg.d_model * cfg.ffn.n_experts_per_token * 2.0 * 2.0

    total = (weights + grads + optim + act) / GIB
    return TrainingMemory(
        weights_gb=weights / GIB,
        gradients_gb=grads / GIB,
        optimizer_gb=optim / GIB,
        activations_gb=act / GIB,
        total_gb=total,
        fits=total <= dev.memory_gb,
        device=dev.name,
        detail={
            "total_params": float(n),
            "activation_layers": float(act_layers),
            "device_memory_gb": dev.memory_gb,
            "headroom_gb": dev.memory_gb - total,
        },
    )


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


@dataclass
class InferenceProfile:
    device: str
    weights_gb: float
    kv_state_gb: float
    total_gb: float
    fits: bool
    decode_tok_s: float
    """Bandwidth-bound ceiling: an upper bound no kernel can beat, typically reached at
    50-70% in practice."""
    prefill_tok_s: float
    flops_per_token: float
    context_len: int
    loop_k: int


def _kv_bytes_per_token(cfg: ProphetConfig, kv_dtype: str, context_len: int) -> float:
    """Bytes of per-token state, averaged over the parameterised depth.

    Full-attention layers grow linearly with context. Windowed layers saturate at the
    window size. Recurrent layers are *constant* — that asymmetry is the whole reason
    for a hybrid stack.
    """
    b = BYTES_PER_PARAM[kv_dtype]
    m = cfg.mixer
    hd = cfg.head_dim
    total = 0.0
    # Section-aware for the same reason as count_parameters: a recurrent core block
    # holds a fixed-size state, and counting it as attention would make the cache
    # appear to grow with context where it does not.
    for _section, _idx, kind in cfg.section_layout():
        if kind == "full_attn":
            if m.kv_compression == "mla":
                total += m.kv_lora_rank * b
            else:
                total += 2 * m.n_kv_heads * hd * b
        elif kind == "swa":
            per_tok = (
                m.kv_lora_rank * b
                if m.kv_compression == "mla"
                else 2 * m.n_kv_heads * hd * b
            )
            # Amortise a fixed-size window over the full context.
            total += per_tok * min(1.0, m.sliding_window / max(context_len, 1))
        elif kind in ("gdn", "mamba2"):
            state = m.linear_heads * m.linear_head_dim * int(
                m.linear_head_dim * m.linear_expand
            )
            total += state * b / max(context_len, 1)
    return total


def inference_profile(
    cfg: ProphetConfig,
    *,
    device: str = "rtx5090",
    weight_dtype: str = "int4_g64",
    kv_dtype: str = "int8",
    context_len: int = 32768,
    loop_k: int | None = None,
    reserve_gb: float = 0.5,
) -> InferenceProfile:
    """Memory footprint and bandwidth-bound token rate on a target device."""
    dev = DEVICES[device]
    p = count_parameters(cfg)
    k = loop_k if loop_k is not None else (
        cfg.recurrent.default_loop_k if cfg.recurrent.enabled else 1
    )

    # Embeddings and norms are usually kept at higher precision: they are a small share
    # of the bytes and the first thing to break under aggressive quantisation.
    quantisable = max(p.non_embedding - 0, 0)
    weights_b = quantisable * BYTES_PER_PARAM[weight_dtype] + p.embedding * BYTES_PER_PARAM["int8"]

    kv_b = _kv_bytes_per_token(cfg, kv_dtype, context_len) * context_len
    total_gb = (weights_b + kv_b) / GIB + reserve_gb

    # Decoding streams the active weights once per token. A weight-shared recurrent core
    # is *read* k times, but stays resident once — so bandwidth cost scales with k while
    # capacity cost does not.
    core_share = 0.0
    if cfg.recurrent.enabled:
        r = cfg.recurrent
        core_share = r.core_layers / max(r.prelude_layers + r.core_layers + r.coda_layers, 1)
    active_read_b = p.active_per_token * BYTES_PER_PARAM[weight_dtype]
    active_read_b *= (1 - core_share) + core_share * k
    bytes_per_token = active_read_b + _kv_bytes_per_token(cfg, kv_dtype, context_len) * context_len

    decode_tok_s = dev.bandwidth_gb_s * 1e9 / max(bytes_per_token, 1.0)

    flops_per_token = 2.0 * p.active_per_token * ((1 - core_share) + core_share * k)
    flops_per_token += 2.0 * 2 * cfg.d_model * context_len * _n_full_attn_layers(cfg)
    prefill_tok_s = dev.bf16_tflops * 1e12 * 0.4 / max(flops_per_token, 1.0)

    return InferenceProfile(
        device=dev.name,
        weights_gb=weights_b / GIB,
        kv_state_gb=kv_b / GIB,
        total_gb=total_gb,
        fits=total_gb <= dev.memory_gb,
        decode_tok_s=decode_tok_s,
        prefill_tok_s=prefill_tok_s,
        flops_per_token=flops_per_token,
        context_len=context_len,
        loop_k=k,
    )


def _n_full_attn_layers(cfg: ProphetConfig) -> int:
    return sum(1 for _s, _i, kind in cfg.section_layout() if kind in ("full_attn", "swa"))


# --------------------------------------------------------------------------------------
# Training budget
# --------------------------------------------------------------------------------------


def expected_train_loop_k(cfg: ProphetConfig) -> float:
    """Mean recurrence depth under the *configured* sampling distribution.

    The plan once used ``(min + max) / 2`` = 4.5 for a log-uniform sampler whose true
    mean over integers 1..8 is 3.38 -- every token count downstream was off in the
    flattering direction. Computed by quadrature over the sampler's actual rule.
    """
    r = cfg.recurrent
    if not r.enabled:
        return 1.0
    lo, hi = r.train_loop_min, r.train_loop_max
    if lo == hi:
        return float(lo)
    if r.train_loop_dist == "uniform":
        return (lo + hi) / 2.0
    if r.train_loop_dist == "poisson":
        lam = r.train_loop_poisson_lambda
        # E[clamp(Poisson(lam), lo, hi)] by direct summation.
        total, mass = 0.0, 0.0
        for n in range(0, 200):
            pmf = math.exp(-lam) * lam**n / math.factorial(n)
            total += pmf * min(max(n, lo), hi)
            mass += pmf
        return total / max(mass, 1e-12)
    # log-uniform, rounded to an integer, exactly as ProphetModel.sample_loop_k does.
    n = 20000
    acc = 0.0
    for i in range(n):
        u = (i + 0.5) / n
        acc += round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return acc / n


def block_passes_per_token(cfg: ProphetConfig, loop_k: float) -> float:
    """Parameterised-block applications per token at depth ``loop_k``.

    Counts what actually runs, including the passes the plan used to ignore: with
    learned halting the coda runs once per core iteration to score each candidate
    stopping point, and each multi-token-prediction head is one more block.
    """
    if not cfg.recurrent.enabled:
        passes = float(cfg.n_layers)
    else:
        r = cfg.recurrent
        passes = r.prelude_layers + r.core_layers * loop_k + r.coda_layers
        if r.halting == "ponder":
            passes += r.coda_layers * loop_k
    passes += cfg.heads.n_multi_token_predict
    return passes


def tokens_affordable(
    cfg: ProphetConfig,
    *,
    a100_hours: float = 300.0,
    mfu: float = 0.35,
    loop_k_train: float | None = None,
) -> dict[str, float]:
    """How many training tokens a given A100-hour budget buys.

    Uses the standard ``6N`` estimate: forward and backward together cost roughly six
    FLOPs per active parameter per token, with N scaled by the block passes that
    actually run per token.
    """
    dev = DEVICES["a100_80gb"]
    p = count_parameters(cfg)
    k = loop_k_train if loop_k_train is not None else expected_train_loop_k(cfg)

    base_passes = block_passes_per_token(cfg, 1.0) if not cfg.recurrent.enabled else (
        cfg.recurrent.prelude_layers + cfg.recurrent.core_layers + cfg.recurrent.coda_layers
    )
    effective_active = p.active_per_token * block_passes_per_token(cfg, k) / max(base_passes, 1)

    flops_per_token = 6.0 * effective_active
    total_flops = dev.bf16_tflops * 1e12 * mfu * a100_hours * 3600.0
    tokens = total_flops / flops_per_token
    return {
        "a100_hours": a100_hours,
        "mfu": mfu,
        "avg_loop_k": k,
        "block_passes_per_token": block_passes_per_token(cfg, k),
        "effective_active_params": effective_active,
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "tokens": tokens,
        "tokens_per_param": tokens / max(p.total, 1),
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def report(cfg: ProphetConfig, *, a100_hours: float = 300.0) -> str:
    """Human-readable budget report for a configuration."""
    p = count_parameters(cfg)
    lines: list[str] = []
    add = lines.append

    add(f"# Budget report — {cfg.name}")
    add("")
    add(f"d_model={cfg.d_model}  parameterised_depth={cfg.parameterised_depth()}  "
        f"effective_depth={cfg.effective_depth()}  frontend={cfg.frontend.mode}  "
        f"ffn={cfg.ffn.kind}")
    add("")
    add("## Parameters")
    add("")
    add("| Component | Params | Share |")
    add("|---|---:|---:|")
    for comp, n in sorted(p.by_component.items(), key=lambda kv: -kv[1]):
        add(f"| {comp} | {_fmt(n)} | {100 * n / p.total:.1f}% |")
    add(f"| **total (resident)** | **{_fmt(p.total)}** | 100% |")
    add(f"| **active / token** | **{_fmt(p.active_per_token)}** | "
        f"{100 * p.active_per_token / p.total:.1f}% |")
    add(f"| embedding share | {_fmt(p.embedding)} | {100 * p.embedding / p.total:.1f}% |")
    add("")

    add("## Training on one A100 80GB")
    add("")
    add("| Batch tokens | Optimiser | Weights | Grads | Optim | Acts | Total | Fits |")
    add("|---:|---|---:|---:|---:|---:|---:|---|")
    real_opt = _optimizer_bytes_per_param(cfg)
    for bt in (8192, 16384, 32768):
        for opt_name, opt_bytes in ((f"Muon+AdamW ({real_opt:.1f} B/param)", real_opt),
                                    ("8-bit (not implemented)", 2.0)):
            tm = training_memory(cfg, batch_tokens=bt, optimizer_bytes_per_param=opt_bytes)
            add(
                f"| {bt} | {opt_name} | {tm.weights_gb:.1f} | {tm.gradients_gb:.1f} | "
                f"{tm.optimizer_gb:.1f} | {tm.activations_gb:.1f} | {tm.total_gb:.1f} GB | "
                f"{'yes' if tm.fits else 'NO'} |"
            )
    add("")

    tb = tokens_affordable(cfg, a100_hours=a100_hours)
    add(f"Token budget at {a100_hours:.0f} A100-hours and {tb['mfu']:.0%} MFU: "
        f"**{_fmt(tb['tokens'])} tokens** "
        f"({tb['tokens_per_param']:.0f} tokens/param, avg loop k={tb['avg_loop_k']:.1f}).")
    add("")

    add("## Inference")
    add("")
    add("| Device | Precision | Context | k | Weights | KV/state | Total | Fits | Decode ceiling |")
    add("|---|---|---:|---:|---:|---:|---:|---|---:|")
    targets = [
        ("rtx5090", "int4_g64", 32768, None),
        ("rtx5090", "int4_g64", 131072, None),
        ("mac_studio_ultra", "int4_g64", 131072, None),
        ("iphone17pro", "int4_g64", 8192, 2),
        ("iphone17pro", "int4_g64", 32768, 2),
    ]
    for dev, dt, ctx, k in targets:
        ip = inference_profile(cfg, device=dev, weight_dtype=dt, context_len=ctx, loop_k=k)
        add(
            f"| {ip.device} | {dt} | {ctx} | {ip.loop_k} | {ip.weights_gb:.2f} | "
            f"{ip.kv_state_gb:.2f} | {ip.total_gb:.2f} GB | "
            f"{'yes' if ip.fits else 'NO'} | {ip.decode_tok_s:.0f} tok/s |"
        )
    add("")
    add("Decode figures are bandwidth-bound ceilings (bytes streamed per token divided "
        "into device bandwidth). Real kernels reach roughly 50-70% of these.")

    warnings = allocation_warnings(cfg)
    if warnings:
        add("")
        add("## Allocation warnings")
        add("")
        for w in warnings:
            add(f"- {w}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Allocation sanity checks
# --------------------------------------------------------------------------------------

#: Share of total parameters above which a component is considered misallocated.
#: Parameters spent on lookup tables are parameters not spent on computation, and at our
#: scale that trade is almost always wrong.
ALLOCATION_LIMITS: dict[str, float] = {
    "frontend/embeddings": 0.20,
    "norms": 0.01,
    "aux_heads": 0.10,
}


def allocation_warnings(cfg: ProphetConfig) -> list[str]:
    """Flag configurations that spend the parameter budget in the wrong place.

    This exists because the failure is silent: a model with 25% of its parameters in an
    embedding or hash table trains perfectly well and is simply worse than it should be,
    and nothing in the loss curve says so.
    """
    p = count_parameters(cfg)
    out: list[str] = []
    for comp, limit in ALLOCATION_LIMITS.items():
        n = p.by_component.get(comp, 0)
        share = n / max(p.total, 1)
        if share > limit:
            out.append(
                f"{comp} holds {_fmt(n)} params ({share:.1%} of the model), above the "
                f"{limit:.0%} guideline — these parameters do no computation."
            )

    if cfg.frontend.mode in ("byte_patch", "hybrid"):
        fe = cfg.frontend
        ngram = len(fe.hash_ngram_sizes) * fe.hash_ngram_buckets * fe.local_dim
        if ngram > 0.10 * p.total:
            out.append(
                f"hash n-gram tables alone are {_fmt(ngram)} params "
                f"({ngram / p.total:.1%}); reduce hash_ngram_buckets, drop n-gram sizes, "
                f"or reduce local_dim."
            )

    if p.active_per_token > 0.9 * p.total and cfg.ffn.kind == "moe":
        out.append(
            "MoE is configured but almost every parameter is active per token; "
            "increase n_experts or reduce n_experts_per_token."
        )

    ratio = p.total / max(p.active_per_token, 1)
    if cfg.ffn.kind == "moe" and ratio < 2.0:
        out.append(
            f"sparsity ratio is only {ratio:.1f}x — MoE adds routing complexity and "
            "training instability for little capacity gain below ~4x."
        )

    out.extend(cfg.design_warnings())
    return out


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prophet budget calculator")
    ap.add_argument("config", nargs="?", help="path to a ProphetConfig JSON file")
    ap.add_argument("--a100-hours", type=float, default=300.0)
    args = ap.parse_args()

    cfg = ProphetConfig.from_json(args.config) if args.config else ProphetConfig()
    print(report(cfg, a100_hours=args.a100_hours))


if __name__ == "__main__":
    _main()
