# R02 — The attention bottleneck: sequence mixing, long context, and KV cache

**Track:** R02 · **Status:** decision-ready · **Date:** 2026-09-03
**Scope:** what mixes tokens in Prophet, how long the context can be, and what it costs in bytes and bandwidth on an A100 (train), a 5090 / Mac Studio (serve), and an iPhone 17 Pro (Prophet-mini).

**Verification note.** `arxiv.org`, `huggingface.co`, `alphaxiv.org` and most paper mirrors are blocked by
this session's egress policy; `github.com` and `raw.githubusercontent.com` are reachable. Numbers below come
from (a) search-engine extraction of paper abstracts and result tables, (b) GitHub sources and READMEs read
directly this session (`fla`, `flash-attention`, `mamba-ssm`, `NVlabs/GatedDeltaNet-2`, `Kimi-Linear`, and
HF `transformers` config files for Qwen3-Next / Gemma-3 / Falcon-H1 / Nemotron-H / GraniteMoeHybrid), and
(c) first-principles arithmetic done here. arXiv IDs marked `†` are cited from prior knowledge and were not
re-verified this session — verify before quoting externally. Every FLOP/byte/hour figure I derived myself is
reproducible from the formulas given inline.

> **TL;DR recommendation.** A **3:1 linear-to-attention hybrid** with a period-8 layer pattern
> `[GDN, GDN, SWA-2048(RoPE), GDN, GDN, GDN, FULL(NoPE), GDN]`, i.e. **75 % Gated DeltaNet,
> 12.5 % sliding-window attention, 12.5 % global full attention**, GQA with `n_kv = 2, d_head = 128`,
> output-gated attention with QK-RMSNorm and a learned sink, and **no positional encoding at all on the
> global layers**. This puts **4 KiB/token** of KV cache on Prophet-main (vs 32 KiB/token for a dense
> 32-layer equivalent) — 512 MiB at 128 k, 4 GiB at 1 M — and makes the whole model *positionally
> length-agnostic*, so the long-context extension phase is a ~500 M-token, ~30-A100-hour job instead of
> a rewrite. Details and the arithmetic are below.

---

## 1. Problem statement

### 1.1 Three different bottlenecks, three different regimes

Transformer inference has three costs that scale differently and bind on different hardware:

| Cost | Scaling | Binds when |
|---|---|---|
| **Prefill compute** | `O(L_attn · d_attn · N²)` for attention, `O(N · P_active)` for the rest | long prompts, weak compute (phones), batch-1 latency |
| **KV cache capacity** | `O(N · L_attn · n_kv · d_head)` — linear in context, linear in *attention layers* | high batch × long context, or any long context on a small-memory device |
| **Decode bandwidth** | bytes/step = `active weight bytes + KV bytes + recurrent-state bytes` | *always*, at batch 1, on every device we care about |

The decisive fact is the **roofline ridge point** — the arithmetic intensity above which a kernel becomes
compute-bound. Batch-1 decoding has an arithmetic intensity of ~2 FLOP/byte (one multiply-accumulate per
weight element read). Ridge points:

| Device | Peak dense math | Memory BW | Ridge (FLOP/byte) | Batch-1 decode intensity | Verdict |
|---|---|---|---|---|---|
| **A100 80 GB** (train) | 312 TFLOP/s BF16 | 2039 GB/s | 153 | ~2 | 76× bandwidth-bound |
| **RTX 5090** (32 GB GDDR7, sm_120) | 209.5 TFLOP/s BF16 (FP32 acc); ~419 FP8; ~838–1676 NVFP4 | **1792 GB/s** | 117 (BF16) / 468 (NVFP4) | ~2–4 | 30–230× bandwidth-bound |
| **Mac Studio M3 Ultra** | ~57 TFLOP/s FP16 (80-core GPU, no tensor cores) | **819 GB/s** | ~70 | ~2–4 | ~20–35× bandwidth-bound |
| **iPhone 17 Pro / A19 Pro** | ~2.6–3.5 TFLOP/s FP16 GPU; ~35–40 TOPS class ANE | **75.8 GB/s** peak (LPDDR5X-9600, 12 GB) | ~40 | ~2–4 | ~10–20× bandwidth-bound |

**Conclusion 1: decoding is bandwidth-bound on all four devices, without exception.** The only levers that
matter for tokens/s are (a) fewer active weight bytes (MoE + 4-bit weights — R03/R04's job) and (b) fewer
KV bytes read per step (this track's job).

### 1.2 The 5090 arithmetic (32 GB)

Take a dense-attention reference: 8 B total / 1.2 B active MoE, `d_model = 2048`, 32 layers, GQA
`n_kv = 2`, `d_head = 128`. KV per token per attention layer = `2 (K,V) × n_kv × d_head` = 512 elements
= **1024 B at BF16**.

| Config | KV B/token | KV @128 k | KV @1 M | Weights (NVFP4) | Total @128 k, bs=1 |
|---|---|---|---|---|---|
| Dense: 32 attention layers | 32 768 | **4.0 GiB** | 32 GiB (**OOM**) | ~4.8 GB | 9.4 GB |
| Hybrid: 4 global + 4 SWA(2048) | **4 096** (+8 MiB fixed) | **512 MiB** | 4.0 GiB | ~4.8 GB | 5.9 GB |

So on a 5090 at **batch 1**, capacity is *not* the binding constraint even for the dense model at 128 k.
Capacity binds in two places:

* **Batch serving.** Dense @128 k: 4 GiB/sequence → batch 6 fills the card. Hybrid: 0.5 GiB/sequence →
  batch ~45. That is the difference between "toy" and "usable local server".
* **Beyond 128 k.** Dense dies at ~700 k tokens. The hybrid holds **4 M tokens of KV in 16 GiB** (BF16) or
  **8 M in 16 GiB** (FP8) — memory-feasible even if compute makes it slow.

And it binds on *bandwidth* at batch 1 right now: at 128 k the dense model reads 4.0 GiB of KV **per decoded
token**, which is 5.6× the ~0.73 GB of active weight bytes. KV read, not weight read, is the dominant term.
The hybrid inverts that ratio (0.54 GB KV vs 0.73 GB weights).

Prefill compute on the 5090 (fwd only; `2·P_active·N` for params, `2·L_attn·d_attn·N²` for causal attention):

| Config | 128 k prefill FLOPs | @ 73.5 TFLOP/s effective BF16 | @ ~210 TFLOP/s effective NVFP4 |
|---|---|---|---|
| Dense (32 attn layers) | 2.56 PFLOP | 35 s | 12 s |
| Hybrid (4 global attn layers) | **0.60 PFLOP** | **8.2 s** | **2.9 s** |

The attention-vs-FFN crossover for the hybrid is at **N ≈ 36 600 tokens** (`2·4·2048·N = 2·1.2e9`): below
~36 k the model is FFN/MoE-bound, above it attention-bound. For the dense variant the crossover is at
**4 600 tokens** — i.e. a dense model is attention-bound over essentially its whole useful range.

### 1.3 The iPhone arithmetic

The actual iPhone 17 Pro ships **12 GB LPDDR5X-9600** with ~75.8 GB/s peak. With
`com.apple.developer.kernel.increased-memory-limit` a foreground app can hold more than the default cap,
but `CLAUDE.md` fixes the planning budget at the conservative **3–5 GB** of real application memory, and
that is the number used throughout this report. Even at 3 GB the hybrid mini has ~10× headroom at 32 k
(see the table below), so **bandwidth, not capacity, is the scarce resource** — 75.8 GB/s is **24× less**
than a 5090 and **27× less** than an A100.

Prophet-mini reference: 450 M dense, `d_model = 1024`, 24 layers, MQA `n_kv = 1, d_head = 128`.
KV per token per attention layer = 256 elements = **512 B BF16**.

| Config | KV B/token | @32 k | @128 k | Weights (4-bit) | Total @32 k |
|---|---|---|---|---|---|
| Dense: 24 attention layers | 12 288 | 384 MiB | 1.5 GiB | ~250 MB | ~640 MB |
| Hybrid: 3 global + 3 SWA(1024) | **1 536** (+1.5 MiB fixed) | **48 MiB** | 192 MiB | ~250 MB | **~310 MB** |

Decode (assume 60 % of peak BW achieved = 45 GB/s):

* Hybrid @32 k: `0.25 GB weights + 0.048 GB KV + 0.018 GB state = 0.32 GB/step` → 7.1 ms → **141 tok/s
  roofline**, realistically **70–100 tok/s**.
* Dense @32 k: `0.25 + 0.384 = 0.63 GB/step` → 14 ms → 71 roofline, realistically **35–50 tok/s**.

**Prefill is the genuine binding constraint on iPhone.** A 32 k prefill for the hybrid mini is
`32768 × 0.9 GFLOP + 2·3·1024·32768² = 3.6e13` = **36 TFLOP**. At ~1.5 TFLOP/s effective on the A19 Pro
GPU that is **~24 s**; the ANE could plausibly cut it to 3–5 s if the graph is expressible there (see §6).
The dense variant is 2.5× worse. This is why the mini model must be small, must have few global attention
layers, and must **persist its state across turns** — which the hybrid makes cheap: the entire resumable
state for a 32 k conversation is `48 MiB KV + 1.5 MiB SWA + 9 MiB GDN state ≈ 59 MiB`.

### 1.4 Mac Studio

M3 Ultra: 819 GB/s, 96–512 GB unified. Capacity is a non-issue. At 128 k, batch 1, the hybrid reads
1.32 GB/step → 1.61 ms → **620 tok/s roofline**, realistically ~280 tok/s. The interesting Mac constraint
is that **Apple GPUs have no tensor cores before M5**, so the ridge point is only ~70 FLOP/byte and prefill
throughput is ~4× worse than a 5090; long-prompt latency, not memory, is what the Mac is bad at. Same
conclusion as the phone: minimise `L_attn`.

### 1.5 The training-budget bottleneck (the one nobody wants to say out loud)

At 300 A100-hours and 35–40 % MFU we have **~4×10¹⁹ FLOP**. For a 1.2 B-active model that is
`C / 6N ≈ 5.6 B tokens`. Chinchilla-optimal for 1.2 B active is ~24 B tokens; Qwen3-1.7B saw ~36 T.
**We cannot out-pretrain the baselines from scratch.** This changes the architecture decision:

> Prophet's sequence mixer must be **convertible from an existing open-weights transformer**, not only
> trainable from scratch.

The 2026 literature makes this cheap and is directly on-track for R02:
* **HALO / HypeNet** (arXiv **2601.22156**) distils a pretrained Transformer into an RNN-attention hybrid
  with **2.3 B tokens**, down from the 10 B+ that earlier methods (MOHAWK, Mamba-in-the-Llama, LoLCATs)
  needed.
* **MiniCPM-SALA** (arXiv **2602.11761**) converts a pretrained Transformer to a sparse+linear hybrid at
  **~75 % lower cost than from-scratch**.
* **Taylor-Calibrate** (arXiv **2606.16429**) gives a principled initialisation for hybrid-linear
  distillation, further shortening the conversion.

A 2.3 B-token conversion of a ~1.7 B-parameter donor costs `6 × 1.7e9 × 2.3e9 = 2.3e19 FLOP` ≈ **175
A100-hours** (≈230 h with a teacher forward for logit distillation). A 0.5 B donor costs **~50 A100-hours**.
Both fit the budget; from-scratch pretraining to competitive quality does not. **Design the architecture so
the attention layers are byte-compatible with a donor's GQA attention** (same `n_kv`, same `d_head`, same
RoPE convention on the layers that keep RoPE) so that 25 % of the layers are simply *copied*.

---

## 2. State of the art

### 2.1 Sequence mixers

`d_model = 2048` throughout; "state" is per layer per sequence, BF16, and is the quantity that must be read
and written on **every decoded token**.

| Mechanism | Recurrence / state | State @d=2048 | Recall (MQAR / RULER) | Throughput note | Paper |
|---|---|---|---|---|---|
| **Softmax attention (GQA `n_kv=2`)** | KV cache, grows with N | 1024 B **per token** | perfect (ceiling) | FA2 is *faster* than linear kernels below ~3–4 k | 2307.08691 † |
| **Sliding-window attention (w=2048)** | KV cache, capped at w | 2 MiB (capped) | perfect **inside** the window, zero outside | O(N·w); free at decode | 2004.05150 † |
| **Linear attention / RetNet** | `S ← γS + kᵀv`, fixed decay | `n_h·d_k·d_v` | poor — fails MQAR at moderate KV-pair counts | very fast | 2307.08621 |
| **Based** | Taylor feature map + tiny sliding window | ~ (d_k²/2)·d_v (large) | better than pure linear, below attention | fast | 2402.18668 |
| **Hyena / long conv** | implicit long convolution, no state | 0 (but needs FFT over N) | weak on associative recall | fast prefill, awkward decode | 2302.10866 † |
| **GLA (Gated Linear Attention)** | data-dependent **matrix-valued** decay | `n_h·d_k·d_v` = 1 MiB | mid; benefits a lot from hybridisation | `chunk_gla` 1.77 ms @ (1,8192,96,128) | 2312.06635 |
| **HGRN-2** | hierarchically gated, state-expanded | ~1 MiB | **surprisingly strong in hybrids** (2507.06457) | fast | 2404.07904 |
| **Mamba-2 / SSD** | scalar-decay SSM, `S ← a_t S + B_tᵀ x_t` | `n_h·d_head·d_state` ≈ 1 MiB | mid; loses to delta-rule variants on recall | mature CUDA kernels (`mamba-ssm`) | 2405.21060 |
| **Mamba-3** | MIMO + trapezoidal discretisation | ~1 MiB | ≈ GDN-class; below GDN-2 | in `fla` (2026) | 2603.15569 |
| **DeltaNet** | delta rule `S ← (I − β kkᵀ)S + β kvᵀ` | 1 MiB | best-in-class *state tracking*, weaker LM ppl | chunkwise WY parallel form | 2406.06484 |
| **DeltaProduct** | n_h Householder products per step | 1 MiB (n_h× compute) | better state tracking than DeltaNet | n_h× slower | 2502.10297 |
| **Gated DeltaNet (GDN)** | gated delta rule: **scalar** decay `α_t` × delta update | `n_v·d_k·d_v` = 1 MiB (expand_v=2) | **strong**; the reference choice in hybrids | `chunk_gdn` **1.265 ms** vs FA2 3.753 ms @ (1,8192,96,128) | **2412.06464** |
| **KDA (Kimi Delta Attention)** | GDN with **channel-wise (per-key-dim) decay**, `g = −exp(A)·softplus(f(x))` | 0.5 MiB (expand_v=1) | > GDN at equal state; shipped at 48 B scale | `chunk_kda` in `fla` | **2510.26692** |
| **Gated DeltaNet-2 (GDN-2)** | **decouples erase and write**: `S ← (I − k(b⊙k)ᵀ)D S + k(w⊙v)ᵀ` | 1 MiB | **SoTA recurrent recall** (see numbers below) | Triton, fused gate-aware backward | **2605.22791** |
| **RWKV-7 "Goose"** | generalised DPLR delta rule, vector gates, in-context LR | 0.25 MiB (head_dim 64) | strong; recognises all regular languages (beyond TC⁰) | mature; CPU/mobile ports exist | **2503.14456** |
| **xLSTM / mLSTM** | matrix memory + exponential gating with normaliser | ~1 MiB | mid-to-strong | TFLA tiled kernels (2503.14376) | 2405.04517 † |
| **TTT layers** | inner-loop gradient descent on an MLP "state" | ≫ 1 MiB (MLP weights) | good, but expensive per token | slow; poor kernel maturity | 2407.04620 † |
| **Titans** | neural long-term memory with surprise-based writes | large | good; complex | limited kernels | 2501.00663 † |
| **Log-linear attention** | Fenwick-tree hierarchy of `O(log N)` states | `log₂N × 1 MiB` (≈17 MiB @128 k) | between linear and full | `fla` has kernels | 2506.04761 |
| **MesaNet / Comba / PaTH / Parallax / Preconditioned GDN** | 2025–26 refinements (mesa-optimisation, preconditioning, path-dependent PE) | ~1 MiB | ≥ GDN on paper, thin ablation record | all in `fla` | 2506.05233, 2506.02475, 2505.16381, 2605.29157, 2604.21100 |

**The state-size↔recall wall.** *Zoology* (arXiv **2312.04927**) established that MQAR accuracy is a
function of **state bytes at generation time**, essentially independent of which gated-convolution family
you use; *Based* (**2402.18668**) traced the resulting throughput-recall Pareto frontier. Practical
consequence: shrinking `d_k` to save state is the fastest way to destroy recall, and any linear mixer with
a state smaller than ~256 KiB/layer at `d=2048` will fail multi-key retrieval regardless of gating
sophistication.

**Hard numbers on where linear-only fails** (GDN-2 paper, 1.3 B params / 100 B FineWeb-Edu tokens):

| Task | GDN-2 (recurrent-only) | Mamba-3 |
|---|---|---|
| Wikitext PPL | **15.90** | 16.45 |
| LAMBADA PPL | **11.41** | 11.66 |
| Commonsense avg acc | **53.11** | 52.39 |
| RULER **S-NIAH-3 @2 k** (single needle) | **89.8 %** | 72.4 % |
| RULER **MK-NIAH-1 @4 k** (multi-key) | **37.8 %** | 18.0 % |
| Real-world retrieval avg — **recurrent** | 29.88 | 26–28 |
| Real-world retrieval avg — **hybrid** | **42.28** | 39–40 |

Read that table twice. The best 2026 linear mixer, at 1.3 B, gets **89.8 % on single-needle @2 k and 37.8 %
on multi-key @4 k**. Hybridisation moves the retrieval average from **29.9 → 42.3 (+41 %)**. Linear layers
alone are not an option for a model that must beat Qwen3-4B on RULER.

### 2.2 Hybrids that actually shipped — exact interleave ratios

| Model | Linear : attention | Exact pattern | Attention type | Positional | Why |
|---|---|---|---|---|---|
| **Jamba** (2403.19887) | **7:1** | 8-layer block: 1 attn + 7 Mamba, MoE every 2nd | MHA | RoPE | fits 256 k on one 80 GB GPU |
| **Samba** (2406.07522) | **1:1** | `Mamba → SWA(2048) → Mamba → SWA…` | **SWA only, no global attention** | RoPE in SWA | 3.8 B/3.2 T; trained @4 k, **extrapolates to 1 M** ppl zero-shot |
| **Zamba / Zamba2** (2405.16712 †, 2411.15242 †) | Mamba backbone + **1–2 *weight-shared* global attention blocks** re-applied in ABAB order | shared-weight attention ≈ every 6 layers | MHA over `[x; embed]` | RoPE | attention quality at near-zero parameter cost |
| **Nemotron-H** (2504.03624) | **~11.5:1** | 8 B: **4 attention of 52 layers (~8 %)**, evenly dispersed; 56 B: 10 of 118; rest = even split Mamba-2 / FFN | GQA | RoPE | up to **3× faster inference** at parity with Qwen-2.5/Llama-3.1 |
| **Hunyuan-TurboS** (2505.15431) | **~8:1** | 128 layers = **57 Mamba2 + 7 attention + 64 FFN**, blocks `AMF` / `MF` | GQA | RoPE | 560 B-A56 B, 256 k, first industry-deployed large Mamba |
| **MiniMax-01** (2501.08313) | **7:1** | every 8 layers: 7 lightning-attention + 1 softmax | MHA/softmax | — | 456 B-A45.9 B; train 1 M, **extrapolate 4 M** |
| **Qwen3-Next-80B-A3B** | **3:1** | `full_attention_interval = 4` → `[lin, lin, lin, full] × 12`, **48 layers** | **gated attention**: GQA 16 q / 2 kv, `head_dim=256`, sigmoid output gate, zero-centred QK-RMSNorm, **`partial_rotary_factor = 0.25`** | partial RoPE (¼ of head dims) | throughput at 256 k |
| **Kimi Linear 48B-A3B** (2510.26692) | **3:1** | 3 × KDA then 1 × **MLA** | MLA, **NoPE on the full-attention layers** | **NoPE** | −75 % KV, **up to 6× decode throughput @1 M**; 128 k score 84.3 at 3.98× speedup; *beats* full MLA at parity |
| **IBM Granite 4.0** | **9:1** | Mamba-2 : transformer 9:1 | GQA | **NoPE everywhere** | −70 % memory for long-context / multi-session; trained on samples to **512 k**, validated to 128 k |
| **Falcon-H1** (2507.22448) | **parallel / intra-layer** | attention heads **and** Mamba-2 heads in the *same* block, independently sized; channel allocation ≈ **2:1:5** (attn:SSM:MLP) | GQA 32 q / 8 kv | RoPE | decouples the two head budgets |
| **MiniCPM-SALA** (2602.11761) | **3:1**, but the "1" is **sparse**, not full | 75 % Lightning Attention + 25 % **InfLLM-v2 sparse attention**, layer-selection algorithm | sparse attention | **HyPE** hybrid positional encoding | 9 B; −75 % conversion cost |

**The convergent answer is 3:1 to 9:1, with the frontier settling on 3:1 when quality matters.**
Two systematic studies confirm it directly:

* **A Systematic Analysis of Hybrid Linear Attention** (arXiv **2507.06457**) — 72 models trained and
  released: 36 at 340 M/20 B tokens and 36 at 1.3 B/100 B tokens, 6 linear variants × 5 hybridisation
  ratios. Findings: (i) **"superior standalone linear models do not necessarily excel in hybrids"**;
  (ii) selective gating, hierarchical recurrence and controlled forgetting are the properties that matter;
  (iii) explicit recommendation: **HGRN-2 or GatedDeltaNet at a linear:full ratio between 3:1 and 6:1**
  reaches Transformer-level recall.
* **Hybrid Architectures for Language Models: Systematic Analysis and Design Insights** (NVIDIA, arXiv
  **2510.04800**) — inter-layer (sequential) vs intra-layer (parallel/head-wise) fusion across LM quality,
  long context, scaling, and train/infer efficiency, with per-strategy design recipes.
* **Rethinking the Role of Efficient Attention in Hybrid Architectures** (arXiv **2606.15378**, Tsinghua/
  OpenBMB) — three results we must design around: (i) **long-range retrieval is carried almost entirely by
  the full-attention layers**; the efficient module mainly shapes the optimisation trajectory; (ii) hybrids
  with different efficient modules **converge to similar long-context quality given enough training** —
  the module choice buys *speed of emergence*, not a different ceiling; (iii) **"Large-Window Laziness"**:
  *larger* SWA windows **delay** the formation of retrieval heads.

That last point is a concrete design constraint: **keep the sliding window small (1–2 k), not 8 k.**

### 2.3 KV-cache reduction

| Technique | Mechanism | Reduction | Cost / caveat | Paper |
|---|---|---|---|---|
| **MQA / GQA** | share K,V across query heads | `n_h/n_kv`× (8× at 16 q / 2 kv) | free; universal | 1911.02150 †, 2305.13245 † |
| **MLA** (DeepSeek) | low-rank joint KV latent + decoupled RoPE key | DeepSeek-V3: **70 KB/token vs 192–328 KB/token** for GQA models (2.7–4.7×); 4–14 % of MHA | extra up-projection FLOPs at decode; needs absorbed-matmul tricks; **no MLX/Metal kernel** | 2405.04434, 2412.19437 |
| **SWA + attention sinks (StreamingLLM)** | window + keep first tokens as sink | O(1) per layer | loses everything outside the window; **22.2× speedup** vs sliding recompute; stable to 4 M tokens | **2309.17453** |
| **NoPE/RoPE + SWA (RNoPE-SWA)** | 3 SWA-RoPE : 1 full-NoPE | 4× fewer global-KV layers | see §2.4 | **2501.18795** |
| **YOCO** | self-decoder produces **one** global KV set, cross-decoder cross-attends | ~L× (single cache) | asymmetric architecture; **prefill can early-exit**; near-perfect 1 M needle | **2405.05254** |
| **Cross-layer KV sharing (CLA, MLKV)** | adjacent layers share one KV cache | 2–4× | small quality cost; composes with GQA | 2405.12981 †, 2406.09297 † |
| **xKV** | cross-layer SVD on aligned singular vectors | ~4–8× | post-hoc, inference-only | 2503.18893 |
| **KV quantisation (KIVI)** | **key per-channel, value per-token**, INT2 + FP16 residual window | 8× vs FP16 | ≤2 % accuracy drop on Llama/Mistral; needs the recent-token FP window or GSM8K collapses | **2402.02750** |
| **KVQuant** | non-uniform + ~1 % outlier channels kept in FP16 | 8× | sparse-format overhead | 2401.18079 † |
| **Newer KV quant (2025–26)** | RotateKV, PM-KVQ (2505.18610), KVarN (2606.03458), KITTY (2511.18643), XQuant rematerialisation (2508.10395) | 8–16× | 2-bit is now roughly safe *with* rotation/normalisation | see refs |
| **Eviction: H2O / SnapKV / PyramidKV** | drop low-attention tokens; pyramid budget by depth | 2–10× | **lossy and query-dependent** — catastrophic for needle tasks if the needle is evicted before the question arrives | 2306.14048 †, 2404.14469 †, 2406.02069 † |
| **Sparse attention (NSA, MoBA, DSA, InfLLM-v2)** | learned top-k block selection; cache stays, *reads* shrink | prefill/decode FLOPs, not bytes | trains natively sparse; strong 2025–26 direction | 2502.11089, 2502.13189, 2602.11761 |

**Ranking for Prophet.** The layer-count lever (`32 → 4` global attention layers) is an **8× reduction and
is free** — it costs no extra kernels, no quality-dependent heuristics, and it composes multiplicatively
with everything else. GQA gives another 8×. FP8 KV gives 2× more, INT4 4×. MLA would give ~3× on top of
GQA but only applies to 4 layers (0.54 GB → ~0.18 GB at 128 k: a 28 % decode-step improvement) and costs us
the entire Apple deployment path. **Eviction methods are disqualified** for a model whose selling point is
retrieval.

### 2.4 Positional schemes and free length generalisation

* **RoPE** (2104.09864 †) is the default but decays badly past the trained length; **Position Interpolation**
  (2306.15595 †), **NTK/YaRN** (2309.00071 †) and **LongRoPE** (2402.13753 †) rescale it, all requiring a
  continued-pretraining phase at the target length.
* **ALiBi** (2108.12409 †) extrapolates but is effectively a soft sliding window — it destroys true
  long-range retrieval.
* **NoPE** (2305.19466 †) — a *causal* decoder can encode position implicitly; with no RoPE there is nothing
  to extrapolate. In a hybrid this is essentially free, because **the recurrent layers already impose
  order**. This is exactly why **Granite 4.0 removed positional encodings entirely** and why **Kimi Linear
  uses NoPE on its full-attention layers**.
* **RNoPE-SWA** (Cohere, arXiv **2501.18795**) interleaves RoPE-SWA layers with NoPE-full layers. Reported
  effect: retrieval **96.1 → 74.8 from 8 k to 256 k**, where baseline RoPE collapses; avoids both RoPE's
  "recency collapse" and NoPE's "attention dispersion".
* **HyPE** (MiniCPM-SALA, 2602.11761) and the **HALO** positional scheme (2601.22156) are 2026 refinements
  of the same idea, both aimed explicitly at extreme-length generalisation.

**Empirical evidence that hybrids get length generalisation nearly free:** Samba pretrained at **4 k**
extrapolates to **1 M** in perplexity zero-shot; MiniMax-01 trains at 1 M and extrapolates to **4 M**;
Granite 4.0 trains on samples to 512 k with no positional encoding at all. This is the single most important
fact for our budget: **it means we can pretrain short and extend cheaply.**

---

## 3. What actually transfers to our scale

Prophet is 8–12 B total / 1–1.5 B active, trained on ≤10 B tokens of our own compute (or converted from a
donor with ~2.3 B tokens). Most frontier findings were established at 1.3 B–80 B on 100 B–20 T tokens. What
survives the transfer:

**Transfers cleanly:**

1. **The 3:1 ratio.** Validated at 340 M and 1.3 B in a controlled 72-model sweep (2507.06457) and shipped
   at 48 B (Kimi Linear) and 80 B (Qwen3-Next). It is one of the few architecture constants that is
   *scale-stable* in the literature. Use 3:1.
2. **Gated DeltaNet as the linear mixer.** Explicitly recommended for hybrids at both 340 M and 1.3 B
   (2507.06457), and it is the mixer with the best kernel and runtime ecosystem (`fla`, HF transformers,
   vLLM, SGLang, MLX community ports, llama.cpp work).
3. **NoPE on the global layers.** Granite 4.0 does this at 3 B and 7 B; Kimi Linear at 48 B. It is a
   *simplification*, not an addition, so it is low-risk at small scale.
4. **KV-cache math.** Bytes/token is arithmetic; it does not care about scale.
5. **Small sliding windows.** "Large-window laziness" (2606.15378) is a *training-dynamics* effect, which
   matters *more* at our tiny token budget, not less: we cannot afford to delay retrieval-head formation.

**Transfers with caveats:**

6. **Linear-mixer micro-innovations (KDA, GDN-2, MesaNet, Comba, PaTH, Preconditioned GDN).** GDN-2's
   channel-wise erase/write is genuinely better on paper (SoTA at 1.3 B/100 B). But 2606.15378's finding —
   different efficient modules converge to similar long-context quality given enough training, and mainly
   differ in *how fast* the capability emerges — cuts both ways at our budget: because we train for very
   few tokens, **faster emergence is worth more to us than to a frontier lab**. That is an argument *for*
   the newest mixer. Against it: no MLX/Metal path, no llama.cpp path, thinner kernel testing.
   → Ship GDN; ablate KDA and GDN-2 (§7 A1); adopt only on a clear win *and* a written Metal kernel plan.
7. **MoE + hybrid interaction.** Every shipped 3:1 hybrid at our sparsity (Qwen3-Next, Kimi Linear) is also
   MoE, so the combination is validated — but at 512 experts and 3 B active, not 1.2 B. Coordinate with R03.

**Does NOT transfer / actively harmful at our scale:**

8. **MLA.** DeepSeek's motivation is 128-head serving at extreme batch; at 4 global layers it saves 0.36 GB
   of a 1.3 GB decode step while costing an exotic attention op with no Metal, no CoreML, and no
   community mobile kernel. **Skip.**
9. **Learned sparse attention (NSA/DSA/InfLLM-v2).** These need to be trained natively sparse, which means
   spending our scarce tokens teaching an indexer. Revisit only if 128 k *prefill* latency becomes the
   product blocker (§6).
10. **YOCO.** Elegant (one global KV set, early-exit prefill, near-perfect 1 M needle) but it is an
    asymmetric decoder-decoder — it forecloses converting from a standard donor checkpoint, which §1.5
    says is our only path to competitive quality.
11. **Pure-linear or ≥15:1 ratios.** 37.8 % multi-key NIAH at 4 k (§2.1) ends this discussion.
12. **KV eviction (H2O/SnapKV/PyramidKV).** Directly antagonistic to the retrieval quality we are buying.

**The uncomfortable transfer note:** from the `fla` benchmark table (GB200, CUDA 12.9, torch 2.9),
`chunk_gdn` vs FlashAttention-2 forward:

| B, T, H, D | chunk_gdn | flash_attn | winner |
|---|---|---|---|
| 8, 1024, 8, 64 | 0.631 ms | **0.157 ms** | FA2 **4.0×** |
| 4, 2048, 16, 128 | 0.753 ms | **0.346 ms** | FA2 **2.2×** |
| 4, 4096, 64, 128 | **1.581 ms** | 2.560 ms | GDN 1.6× |
| 1, 8192, 96, 128 | **1.265 ms** | 3.753 ms | GDN **3.0×** |
| 2, 16384, 16, 128 | **1.029 ms** | 5.035 ms | GDN **4.9×** |

**Crossover ≈ 3–4 k tokens.** At our 4 k pretraining length the hybrid is *not* a training speedup — it is
roughly neutral to slightly slower than a pure-attention model. **The hybrid buys inference and long-context
training, not short-context pretraining.** Budget accordingly and do not promise a pretraining speedup.

---

## 4. Recommendation for Prophet

### 4.1 The layer stack (commit to this)

**Prophet-main — 8–12 B total / ~1.2 B active, `d_model = 2048`, `n_layers = 32`, period 8:**

```
i % 8 :   0     1     2            3     4     5     6            7
        GDN   GDN   SWA-2048     GDN   GDN   GDN   FULL-NoPE    GDN
                    (RoPE θ=10k)                   (global)
```

| Slot | Layers (0-indexed) | Count | Share |
|---|---|---|---|
| Gated DeltaNet | 0,1,3,4,5,7, 8,9,11,12,13,15, 16,17,19,20,21,23, 24,25,27,28,29,31 | **24** | 75 % |
| Sliding-window attention, w = 2048, RoPE θ = 10 000 | **2, 10, 18, 26** | **4** | 12.5 % |
| Full attention, **NoPE**, global | **6, 14, 22, 30** | **4** | 12.5 % |

Rationale for the exact placement: layers 0–1 are GDN so the network establishes positional structure
*before* the first NoPE attention layer; the last layer is GDN (Nemotron-H disperses attention evenly and
avoids the first/last positions); global attention sits at 3/8 depth intervals so retrieval information is
re-injected four times.

**Attention slots (both SWA and FULL) — identical shapes so they are donor-compatible:**

| Parameter | Value | Why |
|---|---|---|
| `n_heads` (query) | 16 | `16 × 128 = 2048 = d_model` |
| `n_kv_heads` | **2** (GQA 8:1) | 8× KV reduction, the standard safe point |
| `head_dim` | **128** | matches Llama-3.2/Qwen3 donors |
| QK norm | **zero-centred RMSNorm** on q and k (`out·(1+w)`) | Qwen3-Next; stabilises training at low token budget |
| output gate | `attn_out * sigmoid(W_g x)` | Qwen3-Next "gated attention"; suppresses the massive-activation/sink pathology |
| attention sink | **1 learned K/V sink pair per layer** (`n_sinks=1`) | StreamingLLM (2309.17453); makes SWA and NoPE layers stable under streaming/extrapolation |
| positional | SWA: RoPE θ=10 000 (partial, `rotary_factor = 0.5`). FULL: **NoPE** | RNoPE-SWA (2501.18795); Granite 4.0 |
| logit scale | `1/sqrt(d_h)` × `(1 + c·log(N/N_train))⁺`, `c` learned per layer, init 0 | counters NoPE softmax dispersion at long N (YaRN's `mscale` idea, applied to NoPE) |
| window | **2048** | small on purpose — "large-window laziness" (2606.15378) |

**Gated DeltaNet slots (Qwen3-Next flavour):**

| Parameter | Value | State consequence |
|---|---|---|
| `linear_num_k_heads` | 16 (`16 × 128 = 2048 = d_model`) | — |
| `linear_num_v_heads` | 32 (`32 × 128 = 4096 = 2·d_model`, `expand_v = 2`) | state `= 32·128·128 = 524 288` el |
| `head_dim` (k and v) | **128** | ≥ the Zoology recall floor |
| short conv | depthwise Conv1d, **kernel 4**, on q, k, v | standard; also gives the model a local-shift primitive |
| gates | `β_t = σ(W_b x)` (write strength), `α_t = exp(−exp(A_log)·softplus(W_a x + dt_bias))` (decay) | per-head scalar decay (GDN). KDA upgrade = per-**key-dim** decay |
| output | gated RMSNorm `norm(o, z)` with `z = W_z x` | as in `fla` |
| **State per layer** | **1 MiB BF16** | ×24 = **24 MiB per sequence** — constant in N |

**KV compression scheme (inference):**

1. **Structural (free, 8×):** 4 global-KV layers instead of 32.
2. **GQA 8:1 (free, 8×):** `n_kv = 2`.
3. **FP8 e4m3 KV** on the 4 global layers with per-(token, head) scales, plus a **KIVI-style BF16 residual
   window of the most recent 128 tokens** (2402.02750). 2× more, ~0 quality cost.
4. **Optional INT4 g64** behind a flag for >1 M contexts (ablate, §7 A5).
5. **Not used:** MLA, cross-layer sharing, eviction. (CLA across the 4 global layers is the one reserve
   lever: 2× more, at the cost of an ablation.)

**Prophet-mini — ~450 M dense, `d_model = 1024`, `n_layers = 24`, same period-8 pattern (3 repeats):**

| Slot | Layers | Count | Config |
|---|---|---|---|
| Gated DeltaNet | 18 layers | 75 % | 8 k-heads × 128 (=1024), 16 v-heads × 128 (=2048), conv 4 → **512 KiB state/layer** |
| SWA, w = **1024**, RoPE | 2, 10, 18 | 3 | 8 q-heads × 128, **`n_kv = 1` (MQA)** |
| Full, **NoPE** | 6, 14, 22 | 3 | same shapes |

`n_kv = 1` on the mini is deliberate: it halves the per-step KV read on the device where bandwidth is
scarcest, and at 3 global layers the absolute quality risk is small.

### 4.2 PyTorch sketch of the hybrid block

```python
# prophet/modeling/hybrid_block.py  — sketch, not final
from dataclasses import dataclass
from typing import Literal, Optional
import math, torch, torch.nn as nn, torch.nn.functional as F

from fla.layers import GatedDeltaNet          # Triton chunkwise kernel (chunk_gated_delta_rule)
# from fla.layers import KimiDeltaAttention   # drop-in: channel-wise decay (arXiv 2510.26692)
# from fla.layers import GatedDeltaNet2       # drop-in: decoupled erase/write (arXiv 2605.22791)

LayerKind = Literal["gdn", "swa", "full"]


@dataclass
class ProphetCfg:
    d_model: int = 2048
    n_layers: int = 32
    period: int = 8
    swa_slot: int = 2                 # i % period == 2  -> sliding window + RoPE
    full_slot: int = 6                # i % period == 6  -> full attention + NoPE
    # ---- attention slots (identical shapes so a donor checkpoint can be copied in) ----
    n_heads: int = 16
    n_kv_heads: int = 2               # GQA 8:1
    head_dim: int = 128
    window: int = 2048
    rope_theta: float = 10_000.0
    rotary_factor: float = 0.5        # partial RoPE on SWA layers only
    n_sinks: int = 1                  # learned K/V sink pair (StreamingLLM)
    train_ctx: int = 4096             # reference length for the NoPE logit-scale term
    # ---- linear slots (Gated DeltaNet, Qwen3-Next flavour) ----
    gdn_k_heads: int = 16
    gdn_v_heads: int = 32             # expand_v = 2
    gdn_head_dim: int = 128
    gdn_conv: int = 4


def layer_kind(i: int, cfg: ProphetCfg) -> LayerKind:
    r = i % cfg.period
    if r == cfg.swa_slot:
        return "swa"
    if r == cfg.full_slot:
        return "full"
    return "gdn"


class RMSNormZeroCentered(nn.Module):
    """Qwen3-Next style: weight initialised at 0, applied as (1 + w)."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + self.w.float())).type_as(x)


class GatedAttention(nn.Module):
    """One attention slot. `kind='swa'` -> window + partial RoPE. `kind='full'` -> global + NoPE."""

    def __init__(self, cfg: ProphetCfg, kind: LayerKind):
        super().__init__()
        self.cfg, self.kind = cfg, kind
        d, h, kv, dh = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.h, self.kv, self.dh = h, kv, dh
        self.q_proj = nn.Linear(d, h * dh, bias=False)
        self.k_proj = nn.Linear(d, kv * dh, bias=False)
        self.v_proj = nn.Linear(d, kv * dh, bias=False)
        self.g_proj = nn.Linear(d, h * dh, bias=False)          # sigmoid output gate
        self.o_proj = nn.Linear(h * dh, d, bias=False)
        self.q_norm = RMSNormZeroCentered(dh)
        self.k_norm = RMSNormZeroCentered(dh)
        # learned attention sink: a virtual K/V pair every query can attend to
        self.k_sink = nn.Parameter(torch.zeros(cfg.n_sinks, kv, dh))
        self.v_sink = nn.Parameter(torch.zeros(cfg.n_sinks, kv, dh))
        # NoPE layers: learned length-compensation on the logits (init 0 == no-op)
        self.len_scale = nn.Parameter(torch.zeros(())) if kind == "full" else None
        self.rotary_dim = int(dh * cfg.rotary_factor) if kind == "swa" else 0

    def forward(self, x, rope=None, cache=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.h, self.dh))
        k = self.k_norm(self.k_proj(x).view(B, T, self.kv, self.dh))
        v = self.v_proj(x).view(B, T, self.kv, self.dh)

        if self.kind == "swa" and rope is not None:          # partial RoPE, SWA layers only
            r = self.rotary_dim
            cos, sin = rope
            q_r, q_p = q[..., :r], q[..., r:]
            k_r, k_p = k[..., :r], k[..., r:]
            q = torch.cat([apply_rope(q_r, cos, sin), q_p], dim=-1)
            k = torch.cat([apply_rope(k_r, cos, sin), k_p], dim=-1)
        # kind == "full": NoPE — nothing applied. Nothing to extrapolate.

        if cache is not None:
            k, v = cache.update(k, v, window=self.cfg.window if self.kind == "swa" else None)

        # prepend the learned sink so window/NoPE layers keep a stable softmax denominator
        ks = self.k_sink.to(k.dtype).unsqueeze(0).expand(B, -1, -1, -1)
        vs = self.v_sink.to(v.dtype).unsqueeze(0).expand(B, -1, -1, -1)
        k = torch.cat([ks, k], dim=1)
        v = torch.cat([vs, v], dim=1)

        scale = 1.0 / math.sqrt(self.dh)
        if self.len_scale is not None:                       # NoPE dispersion compensation
            n = max(k.shape[1], 1)
            scale = scale * (1.0 + self.len_scale * math.log(max(n / self.cfg.train_ctx, 1.0)))

        # A100: flash_attn_func(..., window_size=(w-1, 0)) or flex_attention.
        # sm_120 (5090): prefer torch SDPA (cuDNN/mem-efficient) or flex_attention — see 4.3.
        o = sdpa_grouped(q, k, v, causal=True, scale=scale,
                         window=self.cfg.window if self.kind == "swa" else None,
                         n_sinks=self.cfg.n_sinks)
        o = o.reshape(B, T, self.h * self.dh) * torch.sigmoid(self.g_proj(x))
        return self.o_proj(o)


class ProphetBlock(nn.Module):
    """Pre-norm block. `ffn` is the MoE / SwiGLU module owned by R03."""

    def __init__(self, cfg: ProphetCfg, layer_idx: int, ffn: nn.Module):
        super().__init__()
        self.kind = layer_kind(layer_idx, cfg)
        self.norm1 = RMSNormZeroCentered(cfg.d_model)
        self.norm2 = RMSNormZeroCentered(cfg.d_model)
        if self.kind == "gdn":
            self.mixer = GatedDeltaNet(
                hidden_size=cfg.d_model,
                num_heads=cfg.gdn_v_heads,          # v-heads
                head_dim=cfg.gdn_head_dim,
                expand_v=2.0,                       # 16 k-heads -> 32 v-heads
                conv_size=cfg.gdn_conv,
                use_gate=True,
                mode="chunk",                       # "fused_recurrent" at decode
                layer_idx=layer_idx,
            )
        else:
            self.mixer = GatedAttention(cfg, self.kind)
        self.ffn = ffn

    def forward(self, x, rope=None, cache=None):
        h = self.norm1(x)
        h = self.mixer(h) if self.kind == "gdn" else self.mixer(h, rope=rope, cache=cache)
        x = x + h
        return x + self.ffn(self.norm2(x))
```

Notes on the sketch: `sdpa_grouped` and `apply_rope` are thin wrappers (GQA repeat-interleave + the chosen
attention backend). The sink is implemented as a prepended learned K/V pair rather than as a softmax-
denominator bias precisely so it works with **any** attention kernel, including FlashAttention, cuDNN SDPA,
FlexAttention and MLX — no custom kernel required.

### 4.3 Mapping onto the existing config schema

`configs/prophet_500m_probe.json` already carries a `mixer` block whose fields cover this recommendation.
The probe config's current pattern is `["gdn","gdn","gdn","full_attn"]` — a 3:1 ratio, which this report
endorses — but it uses no sliding-window slot, a 4096 window where it is used, RoPE on every attention
layer, and `nope_layers: []`. Proposed deltas (all reversible per rule 3 of `CLAUDE.md`):

| Field | Probe today | R02 proposal (main / mini) | Why |
|---|---|---|---|
| `mixer.pattern` | `[gdn, gdn, gdn, full_attn]` | `[gdn, gdn, swa_attn, gdn, gdn, gdn, full_attn, gdn]` | same 3:1 linear:attention, but splits the attention slots into 1 windowed + 1 global; puts GDN first and last |
| `mixer.n_heads` / `n_kv_heads` | 12 / 3 | **16 / 2** (main), **8 / 1** (mini) | 8:1 GQA on main; MQA on mini halves the phone's per-step KV read |
| `mixer.head_dim` | 128 | 128 (unchanged) | donor-compatible with Llama-3.2 / Qwen3 |
| `mixer.sliding_window` | 4096 | **2048** (main) / **1024** (mini) | "large-window laziness" (2606.15378): big windows *delay* retrieval-head formation |
| `mixer.nope_layers` | `[]` | **the `full_attn` layers** (6, 14, 22, 30 at 32 layers) | free length extrapolation; Granite 4.0 / Kimi Linear precedent |
| `mixer.rope_theta` | 500 000 | **10 000**, applied only to `swa_attn` layers, `rotary_factor = 0.5` | a 2 k window never needs a long-period RoPE; θ=500 k is for global RoPE layers we no longer have |
| `mixer.rope_scaling` | `none` | stays `none` — **permanently** | nothing positional to rescale; this is the point of the design |
| `mixer.attention_sink_tokens` | 4 | 1 learned K/V sink pair per attention layer | 4 also fine; 1 is enough and cheaper. Keep as a knob |
| `mixer.linear_heads` / `linear_head_dim` / `linear_expand` | 8 / 128 / 2.0 | **16 k-heads / 128 / 2.0** at `d_model=2048` (→ 32 v-heads) | keeps `n_k · d_k = d_model`; 1 MiB state/layer |
| `mixer.kv_compression` | `none` | `fp8` at inference (+128-token BF16 residual window); `none` during training | KIVI-style; 2× on top of the structural 8× |
| `mixer.kv_lora_rank` | 512 | unused — **MLA is not recommended** (§3, item 8) | 0.36 GB saved at 128 k, at the cost of the Apple deployment path |
| new: `mixer.qk_norm_zero_centered` | — | `true` | Qwen3-Next; stabilises training at a very low token budget |
| new: `mixer.attn_output_gate` | — | `true` | Qwen3-Next gated attention; suppresses massive activations |
| new: `mixer.nope_length_scale` | — | `true`, init 0 (no-op) | compensates NoPE softmax dispersion past the trained length |

Note that `mixer.pattern` is already a repeated period, so the only structural change needed is a
**period-8 pattern with two distinct attention kinds** — i.e. `layers.py` needs a third mixer kind
(`swa_attn`) that is the same module as `full_attn` with `window` set and `nope` cleared. Nothing else in
the schema has to move.

### 4.4 Kernel reality check

| Component | A100 (sm_80) — training | RTX 5090 (sm_120) — inference | Mac / iPhone |
|---|---|---|---|
| **Gated DeltaNet** | `flash-linear-attention` `chunk_gated_delta_rule` — **pure Triton**, works on sm_80 | **pure Triton → works on sm_120** with Triton ≥3.3 / CUDA ≥12.8. This is the main reason to prefer `fla` over `mamba-ssm` | MLX: Qwen3-Next-class GDN paths exist in community MLX ports; `mlx-lm` upstream support is landing. **Best-supported linear mixer on Apple.** |
| **Mamba-2** | `mamba-ssm` hand-written CUDA (sm_80 fine) | hand-written CUDA — needs a rebuild/patch for sm_120; historically the first thing to break on a new arch | weakest Apple story |
| **Full / SWA attention (train)** | **FlashAttention-2** (`window_size=(w-1,0)`); Ampere is FA2's primary target | — | — |
| **Full / SWA attention (infer)** | — | ⚠️ **flash-attn 2.x has no official sm_120 wheels**; building with `FLASH_ATTN_CUDA_ARCHS=120` **segfaults nvcc on the backward kernels** (Dao-AILab/flash-attention #2361, #1638). Community wheels exist (CUDA ≥12.8 only — cu126 wheels contain no Blackwell kernels). **FA3 = Hopper only. FA4 = Hopper + *datacenter* Blackwell (sm_100), NOT sm_120** — GeForce Blackwell lacks the tensor-memory hardware FA4 assumes. **Plan: torch SDPA (cuDNN backend) or `flex_attention` as the primary 5090 path**; treat flash-attn as an optimisation. | MLX `scaled_dot_product_attention` (Metal) supports GQA + causal; sliding window via mask |
| **NVFP4 weights** | ❌ A100 is Ampere — no FP8, no FP4. **Train in BF16.** | ✅ 5th-gen tensor cores, native NVFP4 | ❌ use 4-bit affine/MXFP4 in MLX/CoreML |
| **CoreML / ANE** | — | — | ⚠️ **The GDN recurrence has no CoreML scan primitive.** Decode-step GDN is expressible (elementwise + rank-1 update). **Chunkwise prefill must be unrolled at a fixed chunk length or run on the GPU.** See §6 R7. |

**Decision:** `flash-linear-attention` (Triton) for every linear layer, `torch.nn.functional.
scaled_dot_product_attention` / `flex_attention` as the portable attention path, FlashAttention-2 only where
it is known-good (A100 training). Avoid `mamba-ssm` entirely — it buys nothing over GDN and costs us
Blackwell and Apple portability.

---

## 5. Compute & memory budget

### 5.1 Bytes per token / per sequence

**Prophet-main** (`d=2048`, 32 layers, 4 global + 4 SWA(2048) + 24 GDN, GQA `n_kv=2`, `d_head=128`):

| Quantity | Formula | BF16 | FP8 | INT4 g64 |
|---|---|---|---|---|
| Global KV per token | `4 layers × 2 × 2 × 128` = 2048 el | **4096 B** | 2048 B | ~1088 B |
| SWA KV (fixed) | `4 × 2048 × 2 × 2 × 128` | 8 MiB | 4 MiB | 2.2 MiB |
| GDN state (fixed, per sequence) | `24 × 32 × 128 × 128` | **24 MiB** | 12 MiB | — (keep BF16) |

| Context | KV BF16 | KV FP8 | KV INT4 | *Dense-32-layer baseline (BF16)* |
|---|---|---|---|---|
| 4 k | 16 MiB | 8 MiB | 4.3 MiB | 128 MiB |
| 32 k | 128 MiB | 64 MiB | 34 MiB | 1.0 GiB |
| **128 k** | **512 MiB** | **256 MiB** | 136 MiB | **4.0 GiB** |
| 512 k | 2.0 GiB | 1.0 GiB | 0.53 GiB | 16 GiB |
| **1 M** | **4.0 GiB** | 2.0 GiB | 1.06 GiB | 32 GiB (**OOM on 32 GB**) |
| 4 M | 16 GiB | 8 GiB | 4.2 GiB | — |

**RTX 5090 (32 GB) totals**, NVFP4 weights ≈ 4.8 GB, activations ≈ 0.5–0.6 GB:

| Scenario | Total | Fits? |
|---|---|---|
| 128 k, batch 1, BF16 KV | 5.9 GB | ✅ (26 GB spare) |
| 128 k, **batch 32**, BF16 KV | 5.4 + 16.0 + 0.77 (state) = **22.2 GB** | ✅ |
| 1 M, batch 1, BF16 KV | 9.4 GB | ✅ |
| **4 M, batch 1, BF16 KV** | **21.4 GB** | ✅ (memory-feasible; compute-slow) |
| 8 M, batch 1, FP8 KV | 21.4 GB | ✅ |
| *dense baseline, 128 k, batch 8* | 4.8 + 32 GiB | ❌ OOM |

**Prophet-mini** (`d=1024`, 24 layers, 3 global + 3 SWA(1024) + 18 GDN, MQA `n_kv=1`, `d_head=128`):

| Quantity | BF16 | INT8 |
|---|---|---|
| Global KV per token | **1536 B** | 768 B |
| SWA KV (fixed) | 1.5 MiB | 0.75 MiB |
| GDN state (fixed) | **9 MiB** | — |
| KV @ 8 k / 32 k / 128 k | 12 / 48 / 192 MiB | 6 / 24 / 96 MiB |
| **Total resident @32 k** (4-bit weights ≈ 250 MB) | **~310 MB** | ~285 MB |
| **Total resident @128 k** | **~455 MB** | ~355 MB |
| *dense-24-layer baseline @32 k* | ~640 MB | — |

**A 32 k iPhone context costs 59 MiB of cache. A 128 k iPhone context costs 203 MiB.** Both are trivial
inside the 3–5 GB planning budget from `CLAUDE.md` (a ~10× margin at 32 k). The constraint on the
phone is prefill time, not memory (§5.3).

### 5.2 Decode throughput

Per-step bytes and roofline (realistic = 40–55 % of roofline for batch-1 MoE decode):

| Device | Context | Weight bytes | KV bytes | State | Total | Roofline | Realistic |
|---|---|---|---|---|---|---|---|
| **5090** (1792 GB/s, NVFP4) | 4 k | 0.73 GB | 0.017 | 0.05 | 0.80 GB | 2240 tok/s | **~1000–1300** |
| **5090** | 128 k, BF16 KV | 0.73 | 0.54 | 0.05 | 1.32 GB | 1360 | **~600–750** |
| **5090** | 128 k, FP8 KV | 0.73 | 0.27 | 0.05 | 1.05 GB | 1710 | ~750–940 |
| **5090** | 1 M, FP8 KV | 0.73 | 2.1 | 0.05 | 2.9 GB | 620 | ~280–340 |
| *5090, dense-attn variant* | 128 k | 0.73 | **4.3** | — | 5.1 GB | 351 | ~160–190 |
| **M3 Ultra** (819 GB/s, 4-bit) | 128 k | 0.73 | 0.54 | 0.05 | 1.32 GB | 620 | ~250–320 |
| **A19 Pro** (75.8 GB/s peak, ~45 effective), **mini** | 32 k | 0.25 | 0.048 | 0.018 | 0.32 GB | 141 | **~70–100** |
| *A19 Pro, dense-attn mini* | 32 k | 0.25 | 0.384 | — | 0.63 GB | 71 | ~35–50 |

**Headline: ~4× decode speedup at 128 k on the 5090 and ~2× on iPhone at 32 k, purely from the layer
pattern** — before any weight quantisation, MoE sparsity, or speculative decoding.

### 5.3 Prefill

FLOPs/token = `2·P_active` (params) + `2·L_full·d_attn·N` (global attention, causal) + `2·L_swa·d_attn·w`
(windowed) + `~4·n_v·d_k·d_v·L_gdn` (GDN chunkwise).

| Config | @4 k | @32 k | @128 k |
|---|---|---|---|
| Prophet-main FLOPs/token (fwd) | 2.47 GFLOP | 2.94 GFLOP | 4.65 GFLOP |
| *dense-32-attn baseline* | 3.0 | 6.7 | 19.5 |
| **128 k full prefill** | — | — | **0.60 PFLOP** (dense: 2.56 PFLOP) |
| 5090 @ 73.5 TFLOP/s eff. BF16 | — | 1.3 s | **8.2 s** (dense: 35 s) |
| 5090 @ ~210 TFLOP/s eff. NVFP4 | — | 0.45 s | **2.9 s** |
| M3 Ultra @ ~20 TFLOP/s eff. | — | 4.7 s | 30 s |
| **iPhone / mini** @32 k, ~1.5 TFLOP/s GPU | — | **~24 s** | — |
| iPhone / mini @32 k, if ANE-resident (~10 TFLOP/s eff.) | — | ~3.6 s | — |

### 5.4 Training budget consequences

Training FLOPs/token (fwd+bwd, `3×` fwd) at sequence length `N`:

| Phase | `N` | FLOPs/token | vs 4 k | 500 M tokens | 2 B tokens |
|---|---|---|---|---|---|
| Base pretrain | 4 k | 7.4 GFLOP | 1.00× | — | — |
| Long-ctx extension | 32 k | 8.8 GFLOP | **1.19×** | **~9 A100-h** | ~34 A100-h |
| Long-ctx extension | 128 k | 14.0 GFLOP | **1.89×** | ~14 A100-h | ~57 A100-h |
| *dense-32-attn @128 k* | 128 k | 58.7 GFLOP | *7.9×* | *~60 A100-h* | *~240 A100-h* |

*(A100 at 312 TFLOP/s × 35 % MFU = 109 TFLOP/s.)*

**This is the second headline: the hybrid makes the long-context extension phase ~4× cheaper in FLOPs, and
because NoPE+SWA+GDN have nothing positional to re-learn, it is also ~4× cheaper in *tokens*.** Budget
**500 M–1 B tokens at 32 k, ~10–20 A100-hours**, then validate to 128 k. Compare: a RoPE-only dense model
needs a YaRN/LongRoPE rescale plus billions of tokens at the target length.

**Optimizer memory warning (single A100 80 GB):** an 8 B-total MoE at BF16 params + BF16 grads + 8-bit Adam
states ≈ `8e9 × (2+2+2) = 48 GB`, leaving ~30 GB for activations and the GDN chunk workspace. Feasible but
tight; plan for 8-bit optimizer states, activation checkpointing on the GDN and MoE layers, and a
sequence length of 4 k during base pretraining. A 40 GB Colab A100 will **not** fit an 8 B-total MoE —
the mini model and all ablations must be the fallback plan.

---

## 6. Risks & failure modes

**R1 — Multi-key retrieval degradation (highest severity).** The measured failure mode of linear mixers is
not single-needle but **multi-key** retrieval: GDN-2 recurrent-only scores **89.8 % S-NIAH-3 @2 k but
37.8 % MK-NIAH-1 @4 k**. With 4 global layers we expect near-attention quality (hybrid retrieval avg
42.28 vs 29.88 recurrent), but 4 layers is the *minimum* that anyone has shipped as a fraction.
*Detection:* RULER MK-NIAH-2/3 and VT at 4 k/16 k/64 k on every ablation, not just S-NIAH.
*Mitigation ladder:* (a) raise global layers 4 → 6 (period 8 → period 5-ish, ratio 3:1 → 2.2:1);
(b) widen the SWA layers to 4 k; (c) worst case, convert 2 of the SWA layers to global.

**R2 — "Large-window laziness" (2606.15378).** Larger SWA windows *delay* retrieval-head formation. At our
token budget a delayed capability is an absent capability. *Mitigation:* window fixed at 2048 (1024 for
mini); if RULER is weak, the fix is **more global layers, not a bigger window** — a counter-intuitive
direction that we must not get wrong under pressure.

**R3 — NoPE dispersion at extreme length.** Softmax over a NoPE layer flattens as `N` grows. RNoPE-SWA
still degrades 96.1 → 74.8 from 8 k to 256 k. *Mitigation:* the learned `len_scale` logit term in §4.2
(init 0, so it is a no-op unless training uses it), the learned attention sink, and an explicit ablation
(§7 A3) against an all-RoPE and a partial-RoPE control.

**R4 — "Best standalone ≠ best in hybrid" (2507.06457).** Do not select the linear mixer on standalone
perplexity. *Mitigation:* every mixer ablation is run **inside the 3:1 hybrid**, never standalone.

**R5 — Kernel risk on Blackwell.** flash-attn 2.x cannot currently be built for sm_120 (nvcc segfaults on
backward kernels); FA4 does not target sm_120 at all. *Mitigation:* the 5090 path is torch SDPA /
FlexAttention from day one; flash-attn is an optional accelerant, never a dependency. `fla` being pure
Triton means the *linear* 75 % of the model is arch-portable by construction.

**R6 — Numerical stability of chunkwise GDN in BF16.** The delta rule accumulates `(I − βkkᵀ)` products
within a chunk; at chunk 64–128 and BF16 this can drift. *Mitigation:* FP32 accumulation in the chunk state
(fla default), monitor state norms during training, and keep a `fused_recurrent` reference path for
numerical cross-checks.

**R7 — Apple deployment of the recurrence.** CoreML has no scan primitive. Decode-step GDN is fine (rank-1
update + elementwise), but **chunkwise prefill must be unrolled at a fixed chunk size or run on the GPU**,
which is exactly the phase that is already the iPhone bottleneck (§5.3, ~24 s for 32 k on GPU).
*Mitigation:* (a) split the graph — FFN and attention on ANE, GDN mixers on GPU via MLX/Metal;
(b) unroll the chunk scan at chunk=64 into a static CoreML graph and measure;
(c) product-level: persist the 59 MiB session state so a long prompt is prefilled **once**, ever.
*This is the biggest engineering unknown in the plan and should be de-risked with a spike before we commit
the mini model's architecture.*

**R8 — Undertrained-model confound.** At ≤10 B tokens, *every* architecture looks bad at long context, and
ablation differences may be dominated by noise. 2606.15378 explicitly warns that hybrids converge to
similar long-context quality **given enough training** — we will never have "enough". *Mitigation:* run
ablations at matched token counts, report seed variance (2 seeds on the top-2 candidates), and treat
long-context results from the 130 M runs as *directional only*; the real long-context decision is made
after the donor conversion.

**R9 — Donor-conversion mismatch.** If we later convert from a Llama-3.2/Qwen3 donor, the attention slots
must match the donor's `n_kv`, `head_dim` and RoPE convention. Our NoPE global layers **do not** match a
RoPE donor. *Mitigation:* initialise the global layers from the donor *with* RoPE and anneal the RoPE
amplitude to zero over the first ~200 M conversion tokens (or keep RoPE on the global layers in the
converted variant and accept a YaRN extension). **Ablate this explicitly (§7 A7) before committing.**

**R10 — Non-uniform layers complicate everything downstream.** Quantisation calibration, speculative
decoding (see 2605.01106), KV paging (2605.22416) and any tensor-parallel plan all assume homogeneous
layers. We are single-GPU, so this is a tooling annoyance rather than a blocker, but it will cost
engineering time in the serving stack.

**R11 — Prefill remains quadratic.** 8.2 s for a 128 k prefill on a 5090 is acceptable; 30 s on a Mac is
not great. *Reserve lever:* replace the 4 global layers with **learned sparse attention** (NSA 2502.11089 /
MoBA 2502.13189 / InfLLM-v2 as in MiniCPM-SALA) in a v2. Deliberately deferred — it costs training tokens
we do not have.

---

## 7. Ablation plan

**Unit run.** 130 M non-embedding params, `d_model = 768`, 16 layers, period 8 (`12 GDN + 2 SWA + 2 FULL`),
`n_heads = 12 × d_head 64`, `n_kv = 2`, GDN `6 k-heads × 128 / 12 v-heads × 128`, seq 4096, 2 B tokens,
FineWeb-Edu. Cost: `6 × 1.3e8 × 2e9 = 1.56e18 FLOP` → at 109 TFLOP/s ≈ **4.0 A100-hours**. ✅ under 6 h.
**Screening runs** (MQAR only, synthetic): 50 M params, 1 B tokens → **~0.4 A100-h**.

**Fixed evaluation battery** for every run:
* Perplexity: Wikitext-103, LAMBADA.
* Commonsense: ARC-e/c, HellaSwag, PIQA, WinoGrande (0-shot).
* **MQAR** (Zoology protocol, 2312.04927): vocab 8 192, sequence lengths {256, 512, 1024, 2048}, KV pairs
  {4, 8, 16, 32, 64}. Report the accuracy-vs-(sequence × #pairs) grid, not a single number.
* **RULER** (2404.06654 †) at 4 k / 8 k / 16 k / 32 k: **S-NIAH-1/2/3, MK-NIAH-1/2, MV-NIAH, VT, CWE**.
  MK-NIAH is the decision metric.
* Passkey / needle-in-a-haystack sweep at 2×, 4×, 8×, 16× the training length (extrapolation check).

| # | Question | Arms | Runs | A100-h |
|---|---|---|---|---|
| **A0** | Sanity: does the 3:1 hybrid match a full-attention control at 4 k? | dense-attn control; 3:1 hybrid | 2 | 8 |
| **A1** | **Which linear mixer inside the 3:1 hybrid?** | GDN; **KDA**; **GDN-2**; Mamba-2; GLA; HGRN-2 | 6 | 24 |
| **A2** | **Ratio sweep** (linear:attention) | 1:1, 3:1, 7:1, 15:1, pure-linear | 5 (1 shared with A1) | 16 |
| **A3** | **Positional scheme** | all-RoPE full; **RNoPE-SWA (proposed)**; all-NoPE; partial-RoPE(0.25) everywhere | 4 (1 shared) | 12 |
| **A4** | **State size vs recall** | `expand_v ∈ {1, 2}` × `d_k ∈ {64, 128}` — MQAR-screening at 50 M first, then 2 full runs | 4 screen + 2 full | 10 |
| **A5** | **KV compression** (post-hoc, no training) | BF16 / FP8-e4m3 / INT4-g64 / +KIVI residual window 128 / CLA across the 2 global layers | 0 train | ~1 |
| **A6** | **Length extension recipe** | from the A0/A1 winner: continue at 32 k for {100 M, 300 M, 1 B} tokens; measure RULER @4–64 k | 3 (short) | 9 |
| **A7** | **Donor-conversion viability** (R9) | replace 12/16 layers of a small open donor with GDN; RoPE-anneal vs keep-RoPE on global layers; 300 M tokens | 2 | 8 |
| **A8** | **SWA window** | w ∈ {512, 1024, 2048, 4096} — tests "large-window laziness" directly | 4 (2 shared) | 8 |
| | | | **~28 runs** | **~96 A100-h** |

**Sequencing.** A0 → A1 (gate: pick the mixer) → A2+A3+A8 in parallel (gate: lock the pattern) →
A4 → A5 (free) → A6 → A7. **Stop-loss:** if A2's 3:1 arm does not reach ≥95 % of the dense control on
MQAR@2048/32-pairs and ≥90 % on RULER MK-NIAH@8 k, escalate to 2:1 before spending anything on A6/A7.

**Cheap wins to run first (≤1 h each):** A5 (no training at all), the A4 MQAR screens, and a
kernel-throughput matrix (`chunk_gdn` vs SDPA vs FA2 at T ∈ {2 k, 4 k, 8 k, 32 k} on the actual A100) to
confirm the ~3–4 k crossover on *our* hardware before we commit the pretraining sequence length.

---

## 8. References

*Verification status: IDs without a marker were confirmed against the source during this research session
(paper listings, repository metadata, or search results). IDs marked **†** are cited from prior knowledge
and should be spot-checked before publication.*

**Linear / recurrent sequence mixers**
- ABC — arXiv:2110.02488
- RetNet: Retentive Network — arXiv:2307.08621
- GLA: Gated Linear Attention — arXiv:2312.06635
- Zoology: Measuring and Improving Recall in Efficient Language Models — arXiv:2312.04927
- Based: Simple Linear Attention Balances the Recall-Throughput Tradeoff — arXiv:2402.18668
- Rebased — arXiv:2402.10644
- HGRN2: Gated Linear RNNs with State Expansion — arXiv:2404.07904
- RWKV-6 "Eagle/Finch" — arXiv:2404.05892
- Parallelizing Linear Transformers with the Delta Rule (DeltaNet) — arXiv:2406.06484
- Mamba-2 / SSD: Transformers are SSMs — arXiv:2405.21060
- Mamba — arXiv:2312.00752 †
- LightNet — arXiv:2405.21022
- GSA: Gated Slot Attention — arXiv:2409.07146
- Rodimus* — arXiv:2410.06577
- **Gated DeltaNet: Improving Mamba2 with Delta Rule — arXiv:2412.06464**
- DeltaProduct — arXiv:2502.10297
- Forgetting Transformer (FoX) — arXiv:2503.02130
- **RWKV-7 "Goose" with Expressive Dynamic State Evolution — arXiv:2503.14456**
- PaTH Attention — arXiv:2505.16381
- DeltaFormer — arXiv:2505.19488
- Comba — arXiv:2506.02475
- Log-Linear Attention — arXiv:2506.04761
- MesaNet — arXiv:2506.05233
- Scaling Linear Attention with Sparse State Expansion — arXiv:2507.16577
- **Kimi Linear / KDA: An Expressive, Efficient Attention Architecture — arXiv:2510.26692**
- Mamba-3 — arXiv:2603.15569
- Preconditioned Gated DeltaNet / Preconditioned KDA — arXiv:2604.21100
- **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention — arXiv:2605.22791**
- Parallax — arXiv:2605.29157
- xLSTM: Extended Long Short-Term Memory — arXiv:2405.04517 †
- Tiled Flash Linear Attention (TFLA / mLSTM kernels) — arXiv:2503.14376
- Hyena Hierarchy — arXiv:2302.10866 †
- Learning to (Learn at Test Time) — TTT layers — arXiv:2407.04620 †
- Titans: Learning to Memorize at Test Time — arXiv:2501.00663 †

**Hybrid architectures**
- Jamba: A Hybrid Transformer-Mamba Language Model — arXiv:2403.19887
- Samba: Simple Hybrid State Space Models — arXiv:2406.07522
- Zamba — arXiv:2405.16712 †; Zamba2 Suite Technical Report — arXiv:2411.15242 †
- An Empirical Study of Mamba-based Language Models — arXiv:2406.07887
- MiniMax-01: Scaling Foundation Models with Lightning Attention — arXiv:2501.08313
- MiniMax-M1 — arXiv:2506.13585
- **Nemotron-H: Accurate and Efficient Hybrid Mamba-Transformer Models — arXiv:2504.03624**
- Hunyuan-TurboS: Mamba-Transformer Synergy — arXiv:2505.15431
- Falcon-H1: Hybrid-Head Language Models — arXiv:2507.22448
- **A Systematic Analysis of Hybrid Linear Attention — arXiv:2507.06457**
- Speed Always Wins: A Survey on Efficient Architectures for LLMs — arXiv:2508.09834
- **Hybrid Architectures for Language Models: Systematic Analysis and Design Insights (NVIDIA) — arXiv:2510.04800**
- Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention — arXiv:2510.20787
- Understanding In-Context Learning Beyond Transformers — arXiv:2510.23006
- NVIDIA Nemotron 3 — arXiv:2512.20856
- **Hybrid Linear Attention Done Right (HALO / HypeNet) — arXiv:2601.22156**
- **MiniCPM-SALA: Hybridizing Sparse and Linear Attention — arXiv:2602.11761**
- Component Ablation for Efficient Hybrid LM Architectures — arXiv:2603.22473
- S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models — arXiv:2604.01168
- Component-Aware Self-Speculative Decoding in Hybrid Language Models — arXiv:2605.01106
- Asymmetric Virtual Memory Paging for Hybrid Mamba-Transformer Inference — arXiv:2605.22416
- **Rethinking the Role of Efficient Attention in Hybrid Architectures — arXiv:2606.15378**
- Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation — arXiv:2606.16429
- Morphing into Hybrid Attention Models — arXiv:2606.30562
- IBM Granite 4.0 (9:1 Mamba-2 : Transformer, NoPE) — IBM release, Oct 2025, Apache-2.0
- Qwen3-Next-80B-A3B (3:1 Gated DeltaNet : Gated Attention) — Qwen release; Qwen3 Technical Report arXiv:2505.09388 †

**Distillation / architecture conversion**
- MOHAWK: Transformers to SSMs — arXiv:2408.10189 †
- The Mamba in the Llama — arXiv:2408.15237 †
- LoLCATs: On Low-Rank Linearizing of LLMs — arXiv:2410.10254 †

**KV cache, attention efficiency, sparsity**
- MQA: Fast Transformer Decoding — arXiv:1911.02150 †
- GQA — arXiv:2305.13245 †
- Longformer (sliding window) — arXiv:2004.05150 †
- FlashAttention-2 — arXiv:2307.08691 †
- **StreamingLLM: Efficient Streaming LMs with Attention Sinks — arXiv:2309.17453**
- H2O — arXiv:2306.14048 †; SnapKV — arXiv:2404.14469 †; PyramidKV — arXiv:2406.02069 †
- **KIVI: Tuning-Free Asymmetric 2-bit KV Quantization — arXiv:2402.02750**
- KVQuant — arXiv:2401.18079 †
- **MLA / DeepSeek-V2 — arXiv:2405.04434**; DeepSeek-V3 Technical Report — arXiv:2412.19437
- Cross-Layer Attention (CLA) — arXiv:2405.12981 †; MLKV — arXiv:2406.09297 †
- **YOCO: You Only Cache Once — arXiv:2405.05254**
- A Survey on LLM Acceleration based on KV Cache Management — arXiv:2412.19442
- More Tokens, Lower Precision: token-precision trade-off in KV compression — arXiv:2412.12706
- xKV: Cross-Layer KV Compression via Aligned Singular Vectors — arXiv:2503.18893
- NSA: Native Sparse Attention — arXiv:2502.11089; MoBA — arXiv:2502.13189; MoM — arXiv:2502.13685
- PM-KVQ — arXiv:2505.18610; XQuant — arXiv:2508.10395; KITTY (2-bit KV) — arXiv:2511.18643; KVarN — arXiv:2606.03458
- You Only Index Once: Cross-Layer Sparse Attention — arXiv:2606.06467
- MiniMax Sparse Attention — arXiv:2606.13392
- Understanding Bottlenecks for Serving LLM Inference With KV Offloading — arXiv:2601.19910
- Attention Sink in Transformers: A Survey — arXiv:2604.10098

**Positional encoding and length generalization**
- RoPE / RoFormer — arXiv:2104.09864 †
- ALiBi — arXiv:2108.12409 †
- Position Interpolation — arXiv:2306.15595 †
- YaRN — arXiv:2309.00071 †
- LongRoPE — arXiv:2402.13753 †
- The Impact of Positional Encoding on Length Generalization (NoPE) — arXiv:2305.19466 †
- Length Generalization of Causal Transformers without Position Encoding — arXiv:2404.12224
- **Rope to Nope and Back Again (RNoPE-SWA, Cohere) — arXiv:2501.18795**
- Extrapolation by Association — arXiv:2506.09251
- Expansion Span: Fading Memory and Retrieval in Hybrid SSMs — arXiv:2412.13328

**Evaluation**
- RULER: What's the Real Context Size of Your Long-Context LMs? — arXiv:2404.06654 †
- Gemma 3 Technical Report (5:1 local:global, w=1024, local θ=10 k / global θ=1 M) — arXiv:2503.19786 †

**Hardware, kernels, numerics**
- `flash-linear-attention` — github.com/fla-org/flash-linear-attention (Triton; GDN, GDN-2, KDA, RWKV-7, GLA, Mamba-2/3, NSA, MoBA, log-linear, …)
- `flash-attention` — github.com/Dao-AILab/flash-attention; sm_120 build failures: issues #1638, #2168, #2361
- `mamba-ssm` — github.com/state-spaces/mamba (CUDA 11.6+, Linux; no documented sm_120 support)
- Kimi-Linear — github.com/MoonshotAI/Kimi-Linear (`fla-core >= 0.4.0`, torch >= 2.6)
- GatedDeltaNet-2 — github.com/NVlabs/GatedDeltaNet-2
- Pretraining LLMs with NVFP4 — arXiv:2509.25149
- The Anatomy of a Triton Attention Kernel — arXiv:2511.11581
- Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs — arXiv:2604.23466
- RTX 5090: 32 GB GDDR7, 512-bit, **1792 GB/s**, 21 760 CUDA cores, 680 5th-gen Tensor cores, 209.5 TFLOP/s BF16 (FP32 acc), GB202 / TSMC 4NP, 575 W
- Mac Studio M3 Ultra: **819 GB/s**, 96–512 GB unified, 80-core GPU, 32-core ANE. M4 Max: 410/546 GB/s
- A19 Pro / iPhone 17 Pro: **12 GB LPDDR5X-9600**, **~75.8 GB/s**, 16-core Neural Engine; app memory beyond
  the default cap requires `com.apple.developer.kernel.increased-memory-limit`
