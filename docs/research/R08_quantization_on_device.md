# R08 — Extreme Quantization and On-Device Inference

**Track owner:** R08 · **Status:** decision-ready draft · **Date:** 2026-09-03

> This track decides whether "Prophet runs on one RTX 5090 / a Mac Studio / an iPhone" is a
> real claim or marketing. Everything here is meant to feed **back** into the architecture
> before pretraining starts, not to be bolted on afterwards.

### Provenance legend

Every non-obvious number is tagged:

| Tag | Meaning |
|-----|---------|
| **[V]** | Verified in this session from a live web source (URL in §8) |
| **[K]** | From model prior knowledge (cutoff May 2026) — **must be re-verified before it is load-bearing** |
| **[C]** | Computed in this document; the arithmetic is shown so you can check it |

A note on method: this session's web-search budget was exhausted partway through; several
numbers that would normally be pulled from the primary paper are tagged **[K]**. Section 7
lists which of them are load-bearing enough to re-verify.

---

## 1. Problem statement

### 1.1 Reference configurations used throughout this report

Nothing below is meaningful without a concrete model, so I fix one. These are *budgeting*
configs — adjust and the arithmetic scales linearly.

**Prophet-9B-A1.35B (main)** — chosen to satisfy the quantization constraints derived in §4:

| Field | Value | Why this value |
|---|---|---|
| `d_model` | 2048 = 2¹¹ | Power of two → native fast Walsh–Hadamard transform (FWHT), no padding |
| `n_layers` | 28 | |
| `n_q_heads` / `n_kv_heads` | 16 / 4 (GQA) | GQA not MLA — see §4.4 |
| `head_dim` | 128 = 2⁷ | FWHT-friendly; also a clean ANE channel multiple |
| Routed experts | 48/layer, top-4 | |
| Expert `d_ff` | 1024 = 2¹⁰ | FWHT-friendly (online Hadamard before `down_proj`) |
| Shared expert `d_ff` | 2048 | 1 shared expert always on |
| Attention pattern | 3:1 SWA(4096) : global | 7 global layers, 21 sliding-window |
| `vocab` | 131 072 = 2¹⁷, tied | Shared with mini for distillation |
| MTP heads | 2 (predict t+2, t+3) | §4.6 — the single highest-leverage free addition |

Parameter accounting **[C]**:

| Block | Params | Share |
|---|---:|---:|
| Routed experts (48 × 3 × 2048 × 1024 × 28) | 8 455.7 M | 89.6 % |
| Shared experts (3 × 2048 × 2048 × 28) | 352.3 M | 3.7 % |
| Attention q/k/v/o (10.49 M × 28) | 293.6 M | 3.1 % |
| Embedding / lm_head (131072 × 2048, tied) | 268.4 M | 2.8 % |
| MTP heads (2 × 31.5 M) | 62.9 M | 0.7 % |
| Router (2048 × 48 × 28) + norms | 3.0 M | 0.03 % |
| **Total** | **9 435.9 M** | |
| **Active / token** (attn + shared + 4 experts + router) × 28 | **1 353 M** | |
| Active incl. lm_head read | 1 622 M | |

**The single most important structural fact for this track: 89.6 % of the parameters are
routed experts.** Expert precision *is* the memory budget. Everything else is rounding error,
which means everything else can afford to stay at high precision — and should.

**Prophet-mini-0.5B (iPhone)** — dense, ANE-shaped:

| Field | Value |
|---|---|
| `d_model` / `n_layers` | 1024 / 30 (deep-and-thin, per MobileLLM 2402.14905 **[K]**) |
| heads | 16 Q / 4 KV, `head_dim` 64 (ANE prefers 64-channel multiples) |
| `d_ff` | 3072 = 12 × 256 → Hadamard-constructible as H₁₂ ⊗ H₂₅₆ |
| vocab | 131 072 tied (shared tokenizer with main, for distillation) |
| Params | 362 M body + 134 M embedding = **496 M** |
| MTP heads | 2 (dense model → full speculative benefit, see §4.6) |

### 1.2 Device budgets

| | **Colab A100 80GB** (train) | **RTX 5090** (infer A) | **Mac Studio M3 Ultra** (infer B) | **iPhone 17 Pro** (infer C) |
|---|---|---|---|---|
| Compute arch | GA100, **sm_80** | GB202, **sm_120** | M3 Ultra GPU | A19 Pro (GPU + ANE) |
| Memory | 80 GB HBM2e | 32 GB GDDR7 | 96–512 GB unified | 12 GB unified **[K]** (brief assumed 8) |
| Bandwidth | ~1.94–2.04 TB/s **[K]** | **1792 GB/s** (512-bit × 28 Gbps) **[C]** | **819 GB/s** **[K]** | **≈ 75–90 GB/s** **[C]**, derived below |
| Low-precision HW | BF16/TF32/INT8 only. **No FP8, no FP4** | 5th-gen TC: **native NVFP4 & MXFP4**; NVIDIA quotes 3352 "AI TOPS" (FP4, 2:4 sparse) → ~1676 TFLOPS dense FP4 **[C]** | No FP4/FP8 tensor path; FP16/INT8/INT4 weight-only | ANE: FP16/INT8, INT4 palettization; GPU: FP16 + INT4 weight-only |
| App memory ceiling | 80 GB | 32 GB minus display/OS ≈ 30 GB usable | effectively unbounded | **~3–4.5 GB** with `com.apple.developer.kernel.increased-memory-limit` **[K]**; self-impose **1.5 GB** |

**Deriving the iPhone bandwidth [C].** Apple does not publish A19 Pro memory bandwidth. But
MLX-Swift runs Qwen 3.5 2B at 4-bit at **61 tok/s decode on iPhone 17 Pro** **[V]**. A 2B model
at ~4.5 effective bits reads ≈ 1.2 GB per decode step, so the *achieved* bandwidth is
61 × 1.2 ≈ **73 GB/s**. Since MLX reaches roughly 70–85 % of peak on dense decode, peak is
≈ 85–105 GB/s and *usable* is 73–90 GB/s. This measured-lower-bound approach is more
trustworthy than any spec sheet guess, and it is the number the iPhone row of the deployment
matrix is built on.

### 1.3 The three arithmetic walls

1. **5090 capacity wall.** 32 GB. BF16 weights alone are 18.9 GB **[C]**, leaving ~11 GB for
   KV + activations + CUDA context. That "works" but wastes the card. At NVFP4 the weights are
   **5.4 GB** (§5), which turns the 5090 from *barely fits* into *1M-token context or batch 32*.
   Quantization is not an optimization here; it is the difference between a demo and a product.

2. **Bandwidth wall (all three targets).** Single-stream decode is bandwidth-bound, not
   compute-bound. Tokens/s ≈ (achieved BW) / (bytes read per token). Halving weight bytes
   roughly doubles decode speed. This is why weight precision dominates the tok/s claim and
   why activation precision barely matters for decode (it matters for prefill).

3. **iPhone wall.** A 3–5 GB app budget with jetsam killing anything greedy. Evidence from
   this session **[V]**: MLX-Swift held **3 010 MB** on an iPhone 17 Pro and survived; Apple's
   own Core AI framework **was jetsam-killed** attempting Gemma-4-E2B at a 2048-token context
   and had to fall back to a 192-token variant. So ~3 GB is survivable but ~3 GB is also near
   the edge. Budget **≤ 1.5 GB** total process footprint.

**A fourth wall that is easy to miss — the lm_head.** For a model with only 1.35 B active
params, the tied 268 M embedding/output matrix is read *every* decode step. At INT8 that is
0.268 GB of the ~1.11 GB per-token traffic — **24 % of decode bandwidth spent on the output
projection** **[C]**. Sparse MoE makes the vocabulary head a first-class bandwidth citizen.
See §4.7 for what to do about it.

---

## 2. State of the art

### 2.1 Post-training quantization (weights only)

| Method | arXiv | Bits | Mechanism | Where it lands |
|---|---|---|---|---|
| GPTQ | 2210.17323 **[K]** | 3–4 | Layerwise OBQ w/ Hessian, Cholesky | Baseline; weak below 4-bit |
| AWQ | 2306.00978 **[K]** | 4 | Activation-aware per-channel scaling, protect 1 % salient channels | The pragmatic 4-bit default; fast |
| SqueezeLLM | 2306.07629 **[K]** | 3–4 | Sensitivity-weighted k-means + sparse outlier split | Non-uniform codebook |
| OmniQuant | 2308.13137 **[K]** | 2–4 | Learnable clipping + equivalent transform | Better 3-bit |
| QuIP# | 2402.04396 **[K]** | 2–4 | Randomized Hadamard incoherence + E8 lattice VQ + fine-tune | First credible 2-bit |
| AQLM | 2401.06118 **[K]** | 2–3 | Additive multi-codebook VQ | Strong 2-bit, slow to quantize |
| **QTIP** | **2406.11235** **[V]** | 2–4 | **Trellis-coded quantization** — stateful decoder decouples codebook size from bitrate, enabling very high VQ dimension | **Current PTQ SOTA.** "Even at 3 and 4 bits, where QuIP# and AQLM are close to lossless, QTIP roughly halves the perplexity gap" **[V]** |
| EXL3 (turboderp) | — | 2–8 | QTIP variant, procedural codebook + tail-biting trellis; practical runtime | Near-FP16 at 6–8 bpw, competitive at 3 bpw on Llama-3.1/Mistral **[V]** |
| HQQ | (no arXiv) **[K]** | 2–4 | Calibration-free, ℓₚ<1 robust optimization of zero-point | Seconds to quantize; no calibration set |
| GLVQ | 2510.20984 **[V]** | low-bit | Learned grouped lattice VQ | 2025 successor line |

**Throughput reality check [V]** (QTIP repo, batch 1, RTX 6000 Ada): Llama-2-7B FP16 55.9 tok/s;
AQLM 2-bit 81.5; QuIP# 2-bit 186; QTIP 2-bit 188. Two lessons: (a) trellis/lattice decode is
*not* slow when done right, (b) 2-bit gives ~3.4× decode speedup over FP16 — real, but far
from the 4× the bit-count suggests, because dequantization costs cycles.

**The critical caveat: every one of these tables is dominated by 7B–70B models.** QTIP's own
README **[V]** publishes throughput only for 2-7B and 2-70B. EXL3's docs show small-model
(Llama-3.2-1B) results only as *graphs*, and explicitly warn that "accounting for quantization
of the output layer can make a huge difference in practice, **especially for smaller
models**" **[V]**. This is the gap §3 is about.

### 2.2 PTQ with activations (W4A4 and friends)

| Method | arXiv | Config | Result |
|---|---|---|---|
| SmoothQuant | 2211.10438 **[K]** | W8A8 | Migrate activation outliers into weights; lossless-ish at 8-bit |
| Atom | 2310.19102 **[K]** | W4A4 + outlier channels in INT8 | Mixed-precision escape hatch |
| QuaRot | 2404.00456 **[K]** | W4A4KV4 | **Randomized Hadamard rotation** makes activations incoherent → outliers vanish |
| SpinQuant | 2405.16406 **[V]** | W4A4KV4 | Learned (Cayley-optimized) rotations instead of QuaRot's fixed ones |
| QServe / QoQ | 2405.04532 **[K]** | W4A8KV4 | Engineering-first: W4A8 is the throughput sweet spot on Ampere/Hopper |
| ReSpinQuant | 2604.11080 **[V]** | W4A4 | 2026: subspace residual rotation approximation |
| Outlier smoothing w/ closed-form rotations | 2511.22316 **[V]** | W4A4 | 2026: closed-form instead of learned |

Measured, from this session **[V]**:

| Model | Scheme | Metric | FP16 | SpinQuant | Gap |
|---|---|---|---|---|---|
| LLaMA-3-8B | W4A4KV4 | MMLU | 69.6 | 65.2 | **−4.4 pts** |
| LLaMA-2-7B | W4A4KV4 | avg zero-shot | 66.9 | 64.0 | **−2.9 pts** |
| LLaMA-3.2-3B | W4A4 | WikiText2 PPL / acc | — | 9.06 / 58.84 % | "surpasses full-precision LLaMA-3.2-**1B**" **[V]** |

Read that last row carefully. It is the most honest statement in the W4A4 literature: a
**3B model quantized to W4A4 is roughly as good as a 1B model in FP16**. W4A4 costs you
roughly a factor of ~2–3× in effective parameter count at this scale. That is fine if you are
memory-bound (you get 4× the weights in), and fatal if you are already small.

**Note also:** SpinQuant's own README **[V]** publishes tables only for LLaMA-2 7B/13B/70B and
LLaMA-3 8B/70B — the 1B/3B checkpoints are released but the numbers are not in the README.
The literature's small-model evidence is genuinely thin.

### 2.3 The 4-bit float formats (Blackwell)

| | **NVFP4** | **MXFP4** |
|---|---|---|
| Element | E2M1 (values ±{0, .5, 1, 1.5, 2, 3, 4, 6}) | E2M1 |
| Block size | **16** | 32 |
| Block scale | **FP8 E4M3** (has mantissa bits) | FP8 **E8M0** (power-of-two only) |
| Second-level scale | FP32 per-tensor | none |
| **Effective bits/weight** | 4 + 8/16 = **4.50** **[C]** | 4 + 8/32 = **4.25** **[C]** |
| Hardware | RTX 5090 (sm_120), B100/B200 (sm_100) | Same + AMD MI355 |

**[V]** "Unlike MXFP4, which is limited to power-of-two scale factors (E8M0) and prone to high
rounding errors, NVFP4 uses higher-precision E4M3 scale factors with additional mantissa bits."
"NVFP4 generally achieves lower perplexity and higher task scores than standard MXFP4 for the
same model and calibration dataset." Block size below 16 shows diminishing returns; B=16 costs
4.5078 bits/input and is the practical design point **[V]**.

**Measured PTQ degradation [V]:** DeepSeek-R1-0528 FP8 → NVFP4 stayed within 1 % on MMLU-Pro,
GPQA-Diamond and LiveCodeBench, and was *identical* on SciCode and Math-500. That is a
**671B** model. Do not generalize it downward (§3).

2026 follow-ups found this session **[V]**: *Four Over Six: More Accurate NVFP4 Quantization
with Adaptive Block Scaling* (2512.02010), *ScaleSweep: NVFP4 PTQ via Block Scale
Initialization* (2606.07618), *MixFP4: Adaptive FP4/INT4 Block Representations* (2605.31035),
and — directly on point for us — *Characterizing the Impact of NVFP4 Quantization for
Low-Power Edge AI Deployment* (2606.06527). The existence of four separate 2026 papers
improving NVFP4 *block scaling* tells you that naive NVFP4 PTQ has a real accuracy gap worth
chasing, i.e. the DeepSeek-R1 "1 % loss" headline is not the typical case.

### 2.4 Quantization-aware training and native low precision

| Work | arXiv | Claim |
|---|---|---|
| LLM-QAT | 2305.17888 **[K]** | Data-free QAT via self-generated data; 4-bit weights+activations+KV |
| BitNet b1.58 | **2402.17764** **[V]** | Ternary {−1,0,+1} weights. "At 3B, matches FP16 LLaMA in perplexity and zero-shot accuracy, with 3.55× less GPU memory and 2.71× faster" **[V]** |
| BitNet a4.8 | 2411.04965 **[V]** | 4-bit activations for 1-bit LLMs via hybrid quantization + sparsification of intermediates (8-bit) |
| BitNet v2 | 2504.18415 **[V]** | **H-BitLinear**: online Hadamard inside the linear layer reshapes activations to near-Gaussian → native 4-bit activations. 3B: PPL 9.85 / avg acc 55.43 **[V]** |
| BitNet b1.58 2B4T | **2504.12285** **[V]** | First open native 1-bit LLM at 2B scale, **4T tokens**; "on par with leading open-weight full-precision LLMs of similar size" **[V]** |
| bitnet.cpp | — **[V]** | 1.37–5.07× speedup on ARM (55–70 % energy cut); 2.37–6.17× on x86 (72–82 % energy cut). GPU kernels exist **[V]** |
| EfficientQAT | 2407.11062 **[K]** | Block-wise training of all params, then end-to-end scale training; 2-bit 70B in ~41 GPU-hours |
| Gemma-3 QAT | 2503.19786 **[V]** | **~5 000 QAT steps** with the BF16 checkpoint's probabilities as targets; **perplexity drop reduced by 54 %** at Q4_0 **[V]** |
| NVFP4 pretraining | **2509.25149** **[V]** | **12B params / 10T tokens in NVFP4** — loss curve and downstream accuracy match an FP8 baseline. Two-level scaling (FP8 E4M3 per-16 block × FP32 per-tensor) **[V]** |
| FP4 All the Way | 2505.19115 **[V]** | Fully quantized FP4 training |
| Nemotron 3 / Model Optimizer | 2512.20856 **[V]** | 2026: NVFP4 checkpoints shipped as first-class; **QAD (quantization-aware *distillation*) "recovers accuracy from aggressive NVFP4 quantization"** **[V]** |
| Normalized Architectures are Natively 4-Bit | 2605.06067 **[V]** | 2026: normalized (nGPT-style) architectures quantize to 4-bit *without* extra machinery |

The Gemma line is the cheapest, most directly reusable recipe: **a short QAT anneal with
logit distillation from your own BF16 checkpoint recovers about half the quantization loss for
~0.1 % of pretraining compute.** NVIDIA's 2026 QAD result says the same thing with a
different name.

### 2.5 The two scaling laws that govern this entire track

**(a) Scaling Laws for Precision — 2411.04330** (ICLR 2025) **[V]**

The paper's post-training-quantization degradation term has the shape **[K for constants,
V for structure]**:

```
delta_PTQ(N, D, P_post)  =  C_T · ( D^{gamma_D} / N^{gamma_N} ) · exp( − P_post / gamma_post )
```

with `N` = parameters, `D` = training tokens, `P_post` = post-training bit precision. The
three consequences, all verified in this session **[V]**:

1. "**Overtrained language models are more sensitive to post-training quantization.** The
   degradation introduced by PTQ increases as models are trained on more data, **eventually
   making additional pretraining data actively harmful**."
2. "Training in lower precision reduces the model's **effective parameter count**."
3. "The compute-optimal training precision is often around **7–8 bits**."

Consequence 1 is the headline warning of the whole track. Consequence 3 is a quiet argument
*against* pretraining in 4-bit even where hardware allows it.

**(b) Low-Bit Quantization Favors Undertrained LLMs — 2411.17691** **[V]**

Independently fits "quantization-induced degradation" (QiD) and finds it **increases with
training tokens and decreases with model size** — the same law, opposite framing. Its
extrapolation: a 7B model trained on 100T tokens would be severely damaged by 4-bit PTQ.

Also relevant: **Scaling Laws for Floating Point Quantization Training** (2501.02423) **[V]**,
which extends this to the exponent/mantissa split and block size.

**Direct empirical corroboration [K]:** *How Good Are Low-Bit Quantized LLaMA3 Models?*
(2404.14047) found Llama-3-8B (15T tokens, D/N ≈ 1875) degrades markedly more under low-bit
PTQ than Llama-2-7B (2T tokens, D/N ≈ 286) at identical settings, with W3 already painful and
W2 catastrophic without heavy machinery. Community experience with Qwen3's small models
(0.6B/1.7B/4B, trained on ~36T tokens → D/N in the tens of thousands) points the same way
**[K]**.

### 2.6 KV-cache quantization

| Method | arXiv | Scheme | Result |
|---|---|---|---|
| **KIVI** | **2402.02750** **[V]** | **K per-channel, V per-token**, asymmetric, 2-bit, with an FP16 residual window of the most recent tokens | 2.6× peak-memory cut, up to 4× batch, **2.35–3.47× throughput** **[V]** |
| KVQuant | 2401.18079 **[K]** | Per-channel K *pre-RoPE*, non-uniform datatype, dense-and-sparse outlier split | 3-bit near-lossless on Llama |
| Coupled Quantization | 2405.03917 **[V]** | Joint coding across channels — "KV cache is 1 bit per channel" |
| RotateKV | 2501.16383 **[V]** | Outlier-aware adaptive rotations for robust 2-bit KV |
| llama.cpp | — **[V]** | `--cache-type-k/v ∈ {f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1}` |

KIVI's core empirical finding is the design rule to internalize: **the Key cache has
per-channel outliers (some channels are systematically huge), the Value cache does not.**
Therefore K must be grouped along the channel axis and V along the token axis. Any KV
quantizer that uses one layout for both is leaving accuracy on the floor.

The other rule: **keep a small FP16 residual window** of the most recent tokens. Recent keys
are attended to most sharply, and they haven't been "averaged out" yet.

### 2.7 Decode acceleration

| Method | arXiv | Speedup | Cost |
|---|---|---|---|
| n-gram / prompt lookup | — | 1.1–2× on repetitive text | Zero training; in llama.cpp today **[V]** (`--spec-type ngram-simple`, `ngram-map-k`) |
| Medusa | 2401.10774 **[K]** | ~2–2.8× | Several MLP heads + tree attention; trained post-hoc |
| EAGLE-2 | 2406.16858 **[K]** | **4× vs vanilla (13B)** **[V]** | Draft head on features + dynamic draft tree |
| EAGLE-3 | 2503.01840 **[K]** | **5.6× vs vanilla (13B)**, 1.8× vs EAGLE-1 **[V]** | Multi-layer feature fusion + training-time test; "trainable within 1–2 days on 8× RTX 3090" **[V]** |
| Multi-token prediction | 2404.19737 **[K]** | up to 3× via self-speculation | **Trained *during* pretraining**; also improves quality, especially code |
| DeepSeek-V3 MTP | 2412.19437 **[K]** | ~1.8× TPS; reported 2nd-token acceptance ≈ 85–90 % | 1 extra transformer block per head; small % of pretrain cost |

The 2026 tooling has caught up: **exllamav3's converter has a default bitrate specifically for
MTP layers (4 bits)** **[V]**, and llama.cpp has grown a general `--spec-type` framework with
draft-model and n-gram backends **[V]**. MTP heads are now a *supported artifact type*, not a
research curiosity. If Prophet ships without them, every downstream runtime will be leaving
1.3–2× on the table.

### 2.8 The Apple stack (measured)

All numbers **[V]** from an independent reproducible benchmark harness.

**iPhone 17 Pro (A19 Pro), Gemma 4 E2B (~2B), decode:**

| Runtime | Quantization | tok/s | Charged memory | GSM8K |
|---|---|---:|---:|---:|
| LiteRT-LM | w4a8 QAT | **61.1** | **497 MB** | 86.0 % |
| Cactus | CQ4 | 50.0 | 632 MB | 3.0 % ⚠ |
| MLX-Swift | PTQ 4-bit | 49.1 | 3 010 MB | 84.0 % |
| Apple Core AI | int4 | 47.1 | 755 MB | 88.0 % |
| llama.cpp | Q4_K_M | 38.8 | 191 MB † | 76.0 % |

† llama.cpp mmaps weights; clean pages aren't charged to `phys_footprint`, hiding **2.9 GB**
of residency. **Do not trust `phys_footprint` for mmap'd runtimes** — jetsam eventually will.

Note the Cactus row: **50 tok/s and 3 % GSM8K.** A quantizer can preserve throughput and
destroy the model. This is why §7 insists on task metrics and KL divergence, never perplexity
alone.

**iPhone 17 Pro sustained throughput, 10 minutes continuous** **[V]** — the number nobody
publishes and the one that decides whether an app is usable:

| Runtime | Burst tok/s | Sustained tok/s | Retained | Power |
|---|---:|---:|---:|---:|
| **CoreML / ANE** | 33 | **22** | **67 %** | **~12.7 W** |
| LiteRT-LM | 56 | 27 | 48 % | — |
| MLX / GPU | 48 | 18 | 38 % | ~24.7 W |

**The ANE loses the burst benchmark and wins the product.** Roughly half the power and nearly
double the thermal retention. For anything that runs longer than a paragraph, ANE is the right
target on iPhone.

**Apple Silicon Mac, MLX-Swift 4-bit** (M4 Max, 546 GB/s) **[V]**:

| Model | TTFT | Decode tok/s | Peak mem |
|---|---:|---:|---:|
| Qwen 2.5 0.5B | 21 ms | 531.1 | 390 MB |
| Qwen 3.5 0.8B | 36 ms | 421.1 | 600 MB |
| Qwen 3.5 2B | 42 ms | 291.9 | 1 223 MB |
| Gemma 4 E2B | 68 ms | 185.4 | 2 829 MB |
| Gemma 4 E4B | 90 ms | 113.5 | 4 376 MB |

Also **[V]**: MLX-Swift 4-bit on M4 Max is the most *energy*-efficient at 0.090 J/token vs
llama.cpp Q4_K_M at 0.170 and CoreML/ANE at 0.48 (the ANE's efficiency advantage is a
phone-thermal effect, not a J/token effect on a plugged-in Mac).

**ANE architecture constraints [V/K].** Apple's `ml-ane-transformers` reports up to **10×
lower latency and 14× lower peak memory** versus a naive port **[V]**, and notes that even in
the optimized model, embedding-lookup ops fall back to CPU **[V]**. The four principles (from
Apple's "Deploying Transformers on the Apple Neural Engine") **[K]**: (1) use 4D
`(B, C, 1, S)` tensors, channels-first; (2) replace `nn.Linear` with `nn.Conv2d` 1×1;
(3) split attention into per-head chunked einsums rather than one big batched matmul;
(4) softmax along the channel axis. Practically, CoreML-LLM in 2026 achieves **99.9 % ANE
utilization** by splitting a 2B model into **4–5 INT8 chunks plus an mmap'd FP16 embedding
sidecar**, with a **2048-token context ceiling** **[V]**, and reports that further fusion
probes (SDPA fusion, native softmax) **yielded negative results** **[V]** — i.e. the ANE
ceiling is real and already close.

**ANE measured, iPhone 17 Pro** **[V]**: LFM2.5 350M → **52 tok/s**; Qwen3.5 0.8B → ~48 tok/s
(1.2 GB); Gemma 4 E2B → 34.2 tok/s (~1.7 GB); Qwen3.5 2B → ~27 tok/s.

**ExecuTorch on-device, Llama-3.2 (OnePlus 12, closest published proxy)** **[V]**:

| Model | Scheme | Decode | Prefill | Size |
|---|---|---:|---:|---:|
| 1B | BF16 | 19.2 | 60.3 | 2 358 MiB |
| 1B | **SpinQuant** | **50.2** | **260.5** | **1 083 MiB** |
| 1B | QAT+LoRA | 45.8 | 252.0 | 1 127 MiB |
| 3B | BF16 | 7.6 | 21.2 | 6 129 MiB |
| 3B | **SpinQuant** | **19.7** | **89.7** | **2 435 MiB** |
| 3B | QAT+LoRA | 18.5 | 88.8 | 2 529 MiB |

Note the **2.6× decode and 4.3× prefill speedup** from SpinQuant on the 1B — and that
QAT+LoRA lands within 10 % of SpinQuant on speed while being the more accuracy-preserving
recipe. Supported ExecuTorch recipes: SpinQuant, QAT+LoRA, 4-bit groupwise, 8da4w **[V]**.

**MLX quantization toolbox (2026)** **[V]**: `mlx_lm/quant/` ships `awq.py`, `gptq.py`,
`dwq.py`, `dynamic_quant.py`. **DWQ** (distilled weight quantization) is the interesting one:
it freezes the quantized weights and learns only the **scales and biases** of sub-8-bit affine
groups, against a **KL-divergence loss to the unquantized teacher's logits** (temperature 2.0),
defaults **4 bits / group 64**, 2048 calibration samples, batch 4, seq 1025, validating every
200 steps **[V]**. This is Gemma-style QAT-lite, already implemented, on the Mac. We should
use it directly.

### 2.9 Blackwell consumer (sm_120) reality check

This is where the "runs on a 5090" claim is most fragile, and this session found the specifics.

**[V]** sm_120 is *not* sm_100. From the community NVFP4 port: SM120 "is a hybrid architecture
that borrows from three generations but matches none of them." Concretely:

- **SM120 uses SM80-era `mma.sync` instructions, not SM100's `tcgen05.mma`.** Datacenter
  Blackwell kernels **fail to compile or crash** on a 5090.
- SM100 binaries **do not load** on SM120.
- GEMM tiles must fit in **99 KB** shared memory vs 228 KB on SM100.
- **Tensor Memory and UMMA descriptors do not exist** on SM120.

Getting NVFP4 running required **12 coordinated patches to vLLM + FlashInfer + CUTLASS**
(grouped-GEMM tile sizing, SM120 MXFP4 matmul kernels, FlashAttention rewritten on SM80 MMA,
JIT flags, FP4 quantization support) **[V]**. TensorRT-LLM needed **C++ runtime patches** to
`tllmRuntime.cpp` to work around an INT8-container/FP4-engine type mismatch and an allocation
bug — including a deliberate memory leak as a workaround **[V]**.

vLLM's own supported-hardware table, as of this session, **lists only Volta/Turing/Ampere/
Ada/Hopper and does not mention Blackwell SM120, NVFP4 or MXFP4 at all** **[V]**, with the
disclaimer to check the source tree instead.

**Measured on real consumer/prosumer Blackwell (single GPU, NVFP4)** **[V]**:

| Model | Hardware | tok/s | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B MoE | RTX PRO 6000 (sm_120) | **175** | vLLM + 12 patches |
| Gemma4-26B-A3B MoE | RTX PRO 6000 | **160** | |
| Qwen3.5-27B **dense** | RTX PRO 6000 | **57** | |
| 30B MoE | **RTX 5090 32GB** | **135** single / **158.4** @5 concurrent | TRT-LLM + patches; **24.1 GB VRAM**, TTFT ~15 ms, built at `--max_seq_len 4096` **[V]** |
| Llama 3.1 8B | RTX 5090 vs 4090 | +46 % throughput **[V]** | |
| Llama 3.x 8B Q4_K_M | RTX 5090, Ollama | ~142 tok/s **[V]** | |

⚠ One search result claimed **17 000 tok/s** single-stream for Qwen3-8B Q4_K_M on a 5090 with
speculative decoding. That is physically impossible — at ~4.9 GB per decode step it implies
83 TB/s of memory bandwidth, 46× the card's 1792 GB/s. **Discard it.** I flag it explicitly
because it is the top search hit for "RTX 5090 llama.cpp tok/s" and will otherwise contaminate
someone's slide.

**Derived efficiency constant [C].** Qwen3.6-35B-A3B at 175 tok/s: ~3B active × 4.5 bits
≈ 1.69 GB + head ≈ 1.9 GB/token → **332 GB/s achieved = 18.5 % of the 1792 GB/s peak.**
MoE decode at batch 1 is *badly* bandwidth-inefficient because top-k gathers are small,
scattered GEMVs. This 18.5 % figure is the anchor for every 5090 projection in §5, and it is
the most important single number in this report for setting honest expectations.

---

## 3. What actually transfers to our scale

### 3.1 Small models are the worst case, and the literature barely covers them

Everything in §2.1–2.3 that reports "near-lossless 4-bit" is measured on 7B–671B. The scaling
behavior is monotone and unfavorable:

- 2411.04330 **[V]**: δ_PTQ ∝ D^γ / N^γ. **Smaller N → larger degradation** at fixed D.
- 2411.17691 **[V]**: QiD **decreases with model size**, increases with tokens.
- SpinQuant **[V]**: W4A4 LLaMA-3.2-**3B** ≈ FP16 LLaMA-3.2-**1B**. A 3× effective-capacity tax.
- EXL3 **[V]**: output-layer quantization "makes a huge difference… **especially for smaller
  models**".

For Prophet the relevant N is arguably worse than the headline 9.4B, because at batch-1 decode
what carries the token is the **1.35 B active** subnetwork, and 4-bit noise is injected into
every one of its weights.

### 3.2 Are we actually over-trained? Do the compute arithmetic first.

The brief states we are over-training and therefore in the PTQ danger zone. **For the main
model, on a single A100, the arithmetic does not support that.** Let me show it, because the
conclusion flips a major design decision.

Training FLOPs for a sparse MoE ≈ 6 × N_active per token. With N_active ≈ 1.62 B (incl.
lm_head) plus ~8 % for the MTP heads: **≈ 10.5 GFLOP/token** **[C]**.

A100 80GB: 312 TFLOPS BF16 dense. A single-GPU MoE with expert gathers realistically achieves
**25–35 % MFU** → ~90 TFLOPS **[C]**.

```
tokens/s   = 90e12 / 10.5e9        ≈  8 570 tok/s
tokens/day (24h, no preemption)    ≈  740 M
tokens/day (8 effective h, Colab)  ≈  247 M
```

| Wall-clock | Tokens (8 h/day) | D/N_total | D/N_active | Regime |
|---|---:|---:|---:|---|
| 30 days | 7.4 B | 0.8 | 5 | Deeply under-trained |
| 90 days | 22 B | 2.4 | 16 | Under-trained |
| 180 days | 44 B | 4.7 | 33 | ~2× Chinchilla on active params |
| 365 days | 90 B | 9.6 | 67 | ~3× Chinchilla on active |

Compare: Llama-2-7B D/N ≈ 286; Llama-3-8B ≈ 1 875; Qwen3-1.7B ≈ 21 000 **[K]**.

**Conclusion for the main model: on a single Colab A100 we land at D/N ≈ 2–10, which is one to
three orders of magnitude *less* trained than the models where PTQ degradation has been
observed to bite.** Per 2411.17691, that puts Prophet-9B in the *favorable* PTQ regime. This
is good news that should not be thrown away by over-engineering.

**But the mini flips it.** Prophet-mini-0.5B has N_active = 0.5 B → ~3.0 GFLOP/token →
~30 000 tok/s → **~260 M tokens/day at 8 h**, so 100 B tokens is ~13 months, or ~4 months at
24/7, or trivially reachable if the mini is distilled from the main model on a shared corpus.
At 100 B tokens, **D/N = 200 — a genuinely over-trained 0.5B dense model**, in exactly the
regime the scaling laws warn about, deployed at the most aggressive precision (4-bit) on the
device with the least headroom.

> **The over-training warning in the brief is correct — but it applies to Prophet-mini, not
> Prophet-main.** Quantization risk is inverted relative to the intuition: the big MoE will
> quantize *easily*; the little phone model is the one that will break.

Two caveats that could still pull the main model into the danger zone:
- **Distillation counts.** If Prophet-9B is trained on teacher logits (dense supervision over
  131k classes rather than one-hot), the *effective* D per step is far larger than the token
  count. No published scaling law covers this. Treat a distilled 9B at 40 B tokens as if it
  were meaningfully more trained; measure it (§7, V3), don't assume.
- **Multi-epoch training.** If the corpus is small and we do 4–8 epochs, repeated tokens still
  push the model along the memorization axis that δ_PTQ tracks.

### 3.3 The MoE-specific hazard nobody's law covers

The scaling laws are fitted on dense models. For a sparse MoE the per-expert statistics are
extreme **[C]**:

```
tokens seen by one expert = D × top_k / n_experts = D × 4/48 = D/12
at D = 40 B:  3.33 B tokens  for  6.29 M expert params  →  530 tokens/param
```

So each *expert* sits at D/N ≈ 530 — 50–200× the model-level ratio, and squarely in the
Llama-2/Llama-3 range where PTQ degradation is measurable. The expert weights are 89.6 % of
the model and are the ones we most want at 4 bits.

I could not find a paper that settles this. The two readings:
- *Pessimistic:* experts are individually over-trained small matrices → the most PTQ-fragile
  part of the model, exactly where we apply the most aggressive precision.
- *Optimistic:* the law's N is total parameters because δ_PTQ tracks how densely information is
  packed into the whole network, and a sparse network packs less densely per weight.

Community experience mildly favors the pessimistic reading: Qwen3-30B-A3B at 4-bit is widely
reported to lose more than a dense 32B at 4-bit **[K]**. **Treat expert-level over-training as
an open risk (R7) with a cheap dedicated experiment (§7, V3b).**

### 3.4 The one genuinely favorable asymmetry: MoE lets us buy N almost for free

Training compute scales with **N_active**; δ_PTQ shrinks with **N_total**. In a sparse MoE
these are decoupled. Adding experts increases N_total (better quantization robustness, better
capacity) at **near-zero extra training FLOPs** — only memory and a little routing overhead.

```
48 experts → 96 experts:   N_total 9.4B → 17.9B    [C]
                            N_active unchanged at 1.35B
                            training FLOPs unchanged
                            NVFP4 weights 5.4 GB → 10.2 GB  (still fits 32 GB)
                            δ_PTQ falls by ~N^{-gamma}
```

**Recommendation to the architecture track: if you have a choice between more tokens and more
experts at equal training cost, choose more experts.** It is the only lever we have that
improves quantization robustness and quality simultaneously at fixed compute. The binding
constraint becomes the 5090's 32 GB, which caps us at roughly 40–45 B total params at 4.5 bpw
— we are nowhere near it at 9.4 B.

### 3.5 Numbers I would and would not bet on

| Claim | Confidence | Basis |
|---|---|---|
| NVFP4 weight-only on a 9B MoE loses < 1 % on standard benchmarks | **Medium-high** | 671B evidence **[V]** + our low D/N (§3.2) + QAT anneal |
| The same at 3 bits | **Low** | No small-model evidence; assume −2 to −5 pts |
| W4A4 (activations too) is free | **Low** | −2.9 to −4.4 pts measured at 7–8B **[V]**; worse at 1.35B active |
| INT4 KV with KIVI layout + FP16 window is ~free | **Medium-high** | Consistent across KIVI/KVQuant/RotateKV **[V]** |
| Prophet-mini-0.5B survives 4-bit PTQ | **Low** | Over-trained + small + tiny active. **Requires QAT.** |
| MTP gives 2× decode on the MoE | **Low — see §4.6** | The union-of-experts effect kills most of it |
| MTP gives ~2× on the dense mini | **Medium-high** | 2404.19737, DeepSeek-V3 **[K]** |

---

## 4. Recommendation for Prophet

### 4.0 Headline

> **Do not pretrain in low precision. Pretrain in BF16 with a quantization-aware
> *architecture* and a PTQ probe metric from step 0, then spend the last ~12 % of the token
> budget on a QAT anneal plus three short per-target specialization anneals.**

Why not native low-precision pretraining, given NVFP4 pretraining works at 12B/10T **[V]**?

1. **The A100 is sm_80: no FP8, no FP4 hardware.** Any low-precision pretraining is *simulated*
   (quantize-dequantize in BF16). We would pay a 15–40 % step-time penalty for **zero**
   throughput benefit — on a single GPU where compute is the binding constraint on the whole
   project.
2. 2411.04330 **[V]** says training in precision P **reduces effective parameter count**. We
   would be trading away capacity we cannot spare at 1.35 B active.
3. Its own fit puts **compute-optimal training precision at 7–8 bits** **[V]** — i.e. BF16 is
   already past the point of diminishing returns and 4-bit training is on the wrong side of it.
4. BitNet-style ternary is a different bet entirely: it requires from-scratch pretraining, a
   bespoke architecture, and it **forfeits the 5090's FP4 tensor cores** (ternary kernels are
   integer/LUT-based). Matching FP16 required 2B params × 4T tokens **[V]** — a token budget
   we do not have (§3.2).

### 4.1 Architecture constraints imposed by quantization (bake these in now)

These are cheap at design time and impossible to retrofit.

**C1 — All quantization-relevant dimensions are Hadamard-constructible.**
`d_model` 2048, `head_dim` 128, expert `d_ff` 1024, shared `d_ff` 2048, `vocab` 131072 — all
powers of two. Mini: `d_model` 1024, `head_dim` 64, `d_ff` **3072 = 12 × 256** (H₁₂ ⊗ H₂₅₆;
H₁₂ exists by Paley construction — a Hadamard matrix of order n requires n ∈ {1, 2} or
n ≡ 0 mod 4, which is why 2816 = 11 × 256 is *not* usable). This makes QuaRot/SpinQuant-style
online rotations a single fused FWHT kernel with zero padding on every target. Getting this
wrong costs a 20–40 % activation-rotation overhead forever.

**C2 — QK-norm on every attention layer.** RMSNorm applied to Q and K per head before RoPE.
Removes attention-logit blowup, which is the upstream cause of both massive residual
activations and KV outliers. Standard in Gemma-2/3, Qwen3, OLMo-2 **[K]**. Non-negotiable for
4-bit KV.

**C3 — Structurally suppress massive activations.** Per *Massive Activations in LLMs*
(2402.17762 **[K]**) a handful of residual dimensions run ~10⁴× larger than the rest, and they
exist because the model needs somewhere to "park" attention. Bake in the fixes rather than
rotating around them later:
- **Per-head learnable softmax sink logit** (an extra learned denominator term, à la
  off-by-one softmax / GPT-OSS sinks) so the model never needs a BOS outlier **[K]**.
- **Clipped softmax + gated attention** from *Quantizable Transformers* (2306.12929 **[K]**),
  whose entire thesis is "let attention heads do nothing" without producing outliers.
- **No biases anywhere** (attention projections, FFN, norms) — biases are pure outlier
  generators under per-tensor scaling.
- Corroborating 2026 evidence: *Normalized Architectures are Natively 4-Bit* (2605.06067
  **[V]**).

**C4 — Hadamard-before-`down_proj` in every FFN and expert.** SwiGLU's multiplicative gate
produces the widest dynamic range in the network at the `down_proj` input. An online FWHT
there (free because `d_ff` is a power of two, C1) is what makes 4-bit activations possible at
all — this is exactly BitNet v2's H-BitLinear result **[V]**.
*Alternative for the mini:* **ReLU²** instead of SwiGLU. Slightly worse quality/param, but
non-negative, naturally sparse activations that are far friendlier to INT8 on the ANE, and the
sparsity itself is a bandwidth win on a phone. **Recommend: SwiGLU + Hadamard for main,
ReLU² for mini.**

**C5 — Router stays in high precision, always.** The router is 3.0 M params (0.03 %). Keep the
weights BF16 and compute logits in **FP32**. A quantization-induced sign flip in a router logit
does not perturb an output slightly; it swaps in a completely different expert. Add a
**router-margin regularizer** during the QAT anneal (penalize top-1/top-k logit gaps below a
threshold) so routing decisions are robust to 4-bit weight noise. Track **routing agreement
rate** vs the BF16 model as a first-class eval metric (§7).

**C6 — Keep 8-bit floors on the two sensitive matrices.** The tied embedding/lm_head and the
attention projections are ~6 % of params but carry disproportionate sensitivity (EXL3 defaults
`head_bits=6` for exactly this reason **[V]**). Never share a quantization group across the
vocabulary axis of the head.

**C7 — MTP heads in the pretraining graph from step 0** (§4.6).

### 4.2 GQA, not MLA

The quantization verdict, offered to the architecture track as input:

| | GQA (4 KV heads) | MLA |
|---|---|---|
| KV bytes/token/layer | 1024 values | ~576 (latent) — better |
| Quantizes to 4-bit? | Yes, KIVI layout well-validated **[V]** | Latent is a dense low-rank bottleneck with high dynamic range and no per-head structure to exploit; the absorbed-matrix trick fights KV quantization |
| llama.cpp / MLX / CoreML support | Universal | Partial / immature **[K]** |
| Works with sliding-window hybrid | Yes | Awkward |

With 3:1 SWA hybrid + INT4 KV, GQA already gets the KV cache to **578 MB at 128 k context**
and **4.28 GB at 1 M** **[C]** (§5). MLA's compression advantage buys us nothing we need and
costs portability to two of our three targets. **Recommend GQA.** If more KV compression is
wanted, take it from fewer KV heads or 3-bit KV, not from MLA.

### 4.3 KV cache design

- **K: per-channel asymmetric INT4, group 64 along the channel axis. V: per-token asymmetric
  INT4.** (KIVI 2402.02750 **[V]** — the asymmetry is the whole point.)
- **FP16 residual window of the most recent 64–128 tokens**, unquantized. Cheap, recovers most
  of the loss.
- **Rotate K with an online Hadamard *after* RoPE** — rotating before RoPE breaks the rotary
  structure. Fold the V-side rotation offline into `W_v` and `W_o` so it is free at inference.
- **Only the 7 global-attention layers need long-range quantized KV.** The 21 SWA layers hold
  4096 tokens each; leave them at INT8, it costs 49.5 MB **[C]** and removes a whole class of
  bugs.
- Effective cost: 4 bits + (2 × 16 bits / 64) = **4.5 bits/value**.

### 4.4 Activation precision, per target

| Phase | 5090 | Mac | iPhone |
|---|---|---|---|
| Prefill (compute-bound) | **NVFP4 × NVFP4** on expert/FFN GEMMs (this is the only way to touch the 1676 TFLOPS FP4 path — NVFP4 GEMM needs *both* operands in FP4), FP8 attention | FP16 | FP16 / ANE FP16 |
| Decode (bandwidth-bound) | **W4A16**: activations BF16, weights NVFP4. Zero benefit to quantizing activations here | FP16 | FP16 |

The important consequence: **W4A4 is only needed for prefill.** Decode — where all our tok/s
claims live — needs only weight-only quantization. That materially de-risks the whole track,
because W4A4 is where the −3 to −4.4 pt losses live **[V]** and weight-only 4-bit is where the
"< 1 %" results live.

### 4.5 The QAT schedule

Total added compute: **~10–13 %** of the pretraining budget **[C]**.

| Phase | Token budget | What runs | Purpose |
|---|---|---|---|
| **P0 — Architecture** | — | C1–C7 above | Free; prevents 90 % of the pain |
| **P1 — BF16 pretrain + PTQ probe** | 0 → 85 % | Plain BF16. Every 2 000 steps, simulate NVFP4 / INT4-g64 / INT8 PTQ on a copy and log Δloss, ΔKL and routing agreement | **Turns the biggest unknown into a monitored dial.** Costs <0.5 % of compute. If δ_PTQ starts climbing, we see it months early and can stop adding tokens (per 2411.04330 **[V]**, more data can be net harmful) |
| **P2 — QAT anneal** | 85 → 97 % | Fake-quant with STE on expert + FFN + attention weights (NVFP4 sim), INT8 activations, INT4-sim KV; LR already on its decay tail; router-margin regularizer on | Makes weights *live in* the quantized manifold. This is the phase that buys robustness for the over-trained mini |
| **P3 — Per-target specialization** | 97 → 100 %, forked ×3 | Three ~1 % anneals with **logit distillation from the P2 BF16 teacher** (Gemma-3 recipe: ~5 000 steps, KL to teacher probabilities **[V]**): (a) NVFP4 for 5090, (b) INT4-affine-g64 for MLX/GGUF, (c) INT8/INT4-palettized for ANE | Ships three checkpoints, each matched to its runtime's exact numerics. Gemma-3 measured **54 % of the perplexity drop recovered** **[V]**; NVIDIA's QAD reports the same effect for NVFP4 **[V]** |

**Two A100 memory landmines [C]:**

1. Naive QAT keeps a BF16 master *and* a dequantized copy of every weight → +18.9 GB, which
   does not fit alongside 18.9 GB weights + 18.9 GB grads + 18.9 GB 8-bit-Adam states
   (≈ 56.7 GB before activations). **Implement fake-quant as a fused QDQ inside the matmul
   wrapper, materializing one tensor at a time into a reused workspace.**
2. **The A100 gets zero speedup from any of this** (sm_80 has no FP4/FP8). QAT is pure quality
   insurance costing 15–40 % step time during P2/P3, i.e. ~3 % of total project compute. Say
   this out loud in planning so nobody expects a training speedup.

### 4.6 MTP heads — build them in, but budget the speedup honestly

**Design.** Two DeepSeek-V3-style MTP heads predicting t+2 and t+3. Each: RMSNorm the main
model's final hidden `h_t` and the embedding of `x_{t+1}`, concatenate, project 2·d_model →
d_model, then **one transformer block** (attention + dense FFN, *no MoE* — MTP heads must not
route), then reuse the shared embedding and output head.

| | Value |
|---|---|
| Params per head | proj 8.39 M + attn 10.49 M + FFN 12.58 M = **31.5 M** **[C]** |
| Two heads | **62.9 M = 0.67 % of the model** **[C]** |
| Training overhead | ~5–8 % step time **[C]** |
| Quantized at | 4 bits — exllamav3 already defaults MTP layers to 4 bits **[V]** |

**Why during pretraining rather than as a post-hoc EAGLE head:**

1. **Free quality.** 2404.19737 **[K]** shows multi-token prediction improves the base model
   itself, especially on code and reasoning; DeepSeek-V3 **[K]** reports the same. We get a
   better model *and* a draft model from the same 0.67 %.
2. **Quantization-matched drafting — the non-obvious one.** Speculative decoding's acceptance
   rate depends on how well the draft distribution matches the target. If you quantize the
   target to NVFP4 and pair it with a draft model trained against the *FP16* target, the
   distributions drift apart and acceptance drops — you lose speedup exactly when you needed
   it. **MTP heads share the quantized backbone and are quantized in the same anneal, so draft
   and target absorb the same quantization error and stay matched.** An external EAGLE head
   would have to be retrained per precision per target device. This alone justifies the design.
3. **Blackwell at batch 1 is compute-idle.** Decode uses ~18.5 % of memory bandwidth **[V]**
   and a rounding error of the 1676 TFLOPS FP4 units. Verifying 4 candidate tokens costs
   almost nothing in FLOPs.

**Now the honest part — the union-of-experts effect [C].** For a *dense* model, verifying K
tokens reads the weights once and yields up to K tokens: a clean ~K× bandwidth win. For a
**sparse MoE**, verifying K tokens must load the **union** of the experts those K tokens route
to:

```
E[distinct experts over K tokens] = n_e · ( 1 − (1 − k/n_e)^K )
n_e = 48, k = 4:
   K=1 →  4.00 experts
   K=2 →  7.67
   K=3 → 11.04   (2.76× the weight traffic of a single token)
```

Per-step bytes at 4.5 bpw **[C]**:

| | Dense part (attn+shared+head) | Expert part | Total | Tokens | GB/token |
|---|---:|---:|---:|---:|---:|
| K=1 | 0.264 GB | 0.396 GB | 0.660 GB | 1 | 0.660 |
| K=3, all accepted | 0.264 GB | 1.093 GB | 1.357 GB | 3 | 0.452 |
| K=3, 2.2 accepted (realistic) | 0.264 GB | 1.093 GB | 1.357 GB | 2.2 | 0.617 |

**Speculative speedup on the MoE: 1.46× best case, ~1.07× at realistic acceptance** — versus
1.8–2.5× on a dense model of the same active size.

Implications, all actionable:
- **Budget 1.2–1.5× from MTP on Prophet-main, not 2×.** Do not put a 2× number on a slide.
- **The dense fraction is what speculation multiplies.** Raising the shared expert's `d_ff`
  from 1024 to 2048 (already in the reference config) moves 352 M params into the union-free
  bucket and directly improves speculative efficiency. There is a real design trade here
  between fine-grained sparsity (good for quality/FLOP) and speculative decoding (good for
  latency).
- **On Prophet-mini, which is dense, MTP delivers the full ~2×** — and the mini is the target
  where tok/s hurts most. **MTP is worth more on the phone than on the 5090.**
- Coarser MoE would help (top-2 of 8 → 1.30× at K=3 **[C]**) but costs quality per FLOP; not
  worth reorganizing the architecture for.
- Cheap mitigation worth prototyping: **expert-biased drafting** — during draft generation add
  a small bonus to continuations that reuse the current step's resident experts, trading a
  little acceptance rate for a lot of weight-traffic reuse.

Also enable **n-gram / prompt-lookup speculation** at inference. It is free, needs no training,
is already in llama.cpp (`--spec-type ngram-simple`, `ngram-map-k` **[V]**), and it composes
with MTP for repetitive/code/RAG workloads.

### 4.7 Two smaller findings worth acting on

**The lm_head is 24 % of decode bandwidth [C].** At INT8 the tied 268 M head is read every
step against ~1.11 GB total traffic. Options, in order of preference: (a) accept it —
correctness first; (b) INT6 with per-channel scales (0.20 GB, −7 % traffic) after verifying KL
divergence doesn't move; (c) a two-stage head (a cheap 4-bit "shortlist" projection to the top
~4 k candidate tokens, then a high-precision head restricted to those) — a real optimization
but a research project of its own. **Recommend (a) for v1, (b) after measurement.**

**Ship a GGUF path as the guaranteed fallback.** Given §2.9, a custom MoE architecture will not
be supported by vLLM or TensorRT-LLM on sm_120 without our own patches. `llama.cpp` with
`Q4_K_M` / `IQ4_XS` and `--cache-type-k q4_0 --cache-type-v q4_0` **[V]** runs on generic CUDA
on sm_120 today and gives us a working 5090 story that does not depend on NVFP4 kernels
landing. Treat NVFP4 as the *performance* path and GGUF as the *availability* path.

### 4.8 DEPLOYMENT MATRIX

| Component | **RTX 5090** (sm_120) | **Mac Studio M3 Ultra** | **iPhone 17 Pro** (mini, dense) |
|---|---|---|---|
| Model | Prophet-9B-A1.35B | Prophet-9B-A1.35B | Prophet-mini-0.5B |
| Runtime | vLLM/TRT-LLM + sm_120 NVFP4 patches → fallback llama.cpp GGUF | MLX (primary), llama.cpp Metal (fallback) | CoreML/ANE (primary), MLX-Swift (burst mode) |
| **Routed expert W** | **NVFP4** (E2M1, blk 16, FP8-E4M3 scale, FP32 tensor scale) — 4.50 bpw | **INT4 affine, group 32** — 5.00 bpw (or MXFP4) | n/a (dense) |
| **Shared expert / FFN W** | NVFP4 — 4.50 bpw | INT6 group 64 — 6.50 bpw | **INT8** (ANE) / INT4-g64 (GPU) |
| **Attention q/k/v/o W** | NVFP4 + online Hadamard — 4.50 bpw | INT6 group 64 | **INT8** |
| **Embedding + lm_head** (tied) | **INT8** per-channel | INT8 per-channel | INT8 (+ FP16 mmap sidecar, per CoreML-LLM **[V]**) |
| **Router** | **BF16 weights, FP32 logits** | FP16 / FP32 logits | n/a |
| **Norms, RoPE, softmax** | BF16 | FP16 | FP16 |
| **KV cache** | **INT4**: K per-channel g64, V per-token, + 64-token FP16 window; SWA layers INT8 | INT4 (long ctx) / INT8 (default) | **INT8**, 4 k context |
| **Activations — prefill** | **NVFP4 × NVFP4** on FFN/expert GEMMs; FP8 attention | FP16 | FP16 |
| **Activations — decode** | BF16 (W4A16) | FP16 | FP16 |
| **MTP heads** | NVFP4, 2 heads | INT4 | INT4 / INT8 |
| **Weights total** | **5.43 GB** **[C]** | **6.13 GB** **[C]** | **0.30 GB** (GPU) / **0.50 GB** (ANE INT8) **[C]** |
| **Expected decode** | **250–450 tok/s** (central 300); ×1.3 MTP → **~390** | **120–220 tok/s** (central 150) | **ANE 45–55 burst / 30–37 sustained**; MLX ~140 burst / ~53 sustained; ×2 MTP |

---

## 5. Memory & throughput budget

### 5.1 Weight memory, Prophet-9B-A1.35B [C]

| Block | Params | BF16 | FP8 | **NVFP4 plan (5090)** | **MLX plan (Mac)** |
|---|---:|---:|---:|---:|---:|
| Routed experts | 8 455.7 M | 16.91 GB | 8.46 GB | **4.756 GB** @4.5 bpw | 5.285 GB @5.0 bpw |
| Shared experts | 352.3 M | 0.70 | 0.35 | **0.198** @4.5 | 0.286 @6.5 |
| Attention | 293.6 M | 0.59 | 0.29 | **0.165** @4.5 | 0.239 @6.5 |
| MTP heads | 62.9 M | 0.13 | 0.06 | **0.035** @4.5 | 0.051 @6.5 |
| Embed / lm_head | 268.4 M | 0.54 | 0.27 | **0.268** @8 | 0.268 @8 |
| Router + norms | 3.0 M | 0.006 | — | **0.006** BF16 | 0.006 |
| **Total** | **9 435.9 M** | **18.87 GB** | **9.44 GB** | **5.428 GB** | **6.135 GB** |

**3.5× smaller than BF16.** The 5090 goes from "18.9 GB of weights in a 32 GB card" to
"5.4 GB of weights in a 32 GB card."

### 5.2 KV cache, Prophet-9B [C]

Per token per layer: 2 (K,V) × 4 KV heads × 128 = **1024 values**. Hybrid 3:1 → 7 global
layers + 21 SWA(4096) layers.

| Context | FP16 (all-global) | INT8 hybrid | **INT4 hybrid (plan)** |
|---:|---:|---:|---:|
| 8 k | 0.46 GB | 0.13 GB | **0.082 GB** |
| 32 k | 1.83 GB | 0.28 GB | **0.157 GB** |
| 128 k | 7.34 GB | 1.03 GB | **0.578 GB** |
| 1 M | 57.3 GB | 7.61 GB | **4.28 GB** |

*(INT4 = 4 bits + 2×FP16 scales per 64 = 4.5 bits = 0.5625 B/value.)*

**The hybrid SWA + INT4 KV combination is what makes a 1 M-token context fit on a consumer
card**: 5.43 GB weights + 4.28 GB KV + ~1.5 GB activations ≈ **11.2 GB of 32 GB.**

### 5.3 Full memory budget per target [C]

| | **RTX 5090** | **Mac Studio M3U (96 GB)** | **iPhone 17 Pro** |
|---|---:|---:|---:|
| Weights | 5.43 GB | 6.14 GB | 0.30 / 0.50 GB |
| KV @ working context | 0.58 GB (128 k) | 1.03 GB (128 k, INT8) | 0.063 GB (4 k, INT8) |
| Activations + workspace | ~1.2 GB | ~1.0 GB | ~0.15 GB |
| Runtime / CUDA ctx / CoreML | ~0.8 GB | ~0.5 GB | ~0.30 GB |
| **Total, batch 1** | **~8.0 GB / 32 GB** | **~8.7 GB / 96 GB** | **~0.8–1.0 GB** |
| Headroom | batch ≈ 32 @128 k, **or 1 M ctx @ bs 1** | batch ≈ 80, or multi-MB ctx | Under the 1.5 GB self-imposed cap; ~3× under the observed 3.0 GB jetsam-survival point **[V]** |

### 5.4 Throughput [C], with the derivation shown

**RTX 5090.** Bytes per decode token: active weights 1 353 M × 0.5625 B = 0.761 GB + lm_head
0.268 GB + KV read @8 k 0.082 GB = **1.11 GB/token**.

Anchor: the measured Qwen3.6-35B-A3B NVFP4 point (175 tok/s, ~1.9 GB/token) implies
**332 GB/s achieved = 18.5 % of 1792 GB/s peak** **[V/C]**.

```
conservative (18.5 % eff.):  332 / 1.11  =  299 tok/s
optimistic  (30 % eff.):     538 / 1.11  =  485 tok/s
```

| Metric | Conservative | Central | Optimistic |
|---|---:|---:|---:|
| Decode, bs=1 | 250 | **300** | 450 |
| Decode + MTP (×1.2–1.5) | 300 | **390** | 650 |
| Decode, bs=32 | — | 3 000–6 000 agg. | — |
| Prefill | 8 000 | **12 000** | 25 000 |
| TTFT @ 2 k prompt | 250 ms | **170 ms** | 80 ms |

*(Prefill is far below the FLOP ceiling — 2 × 1.62 GFLOP/token against ~419 TFLOPS of usable
FP4 would be ~129 k tok/s. It is kernel- and routing-bound, not FLOP-bound. Reference point:
the patched TRT-LLM 30B-MoE build achieves 15 ms TTFT **[V]**.)*

**Mac Studio M3 Ultra.** Bytes/token: 1 353 M × 0.625 B (5 bpw) = 0.846 + 0.268 + 0.10 =
**1.21 GB/token**. MLX MoE decode efficiency ≈ 20–27 % of 819 GB/s **[K/C]** → 164–221 GB/s.

| Metric | Conservative | Central | Optimistic |
|---|---:|---:|---:|
| Decode, bs=1 | 120 | **150** | 220 |
| Decode + MTP | 145 | **190** | 300 |
| Prefill | 600 | **1 100** | 2 000 |

*(Sanity check against a measured dense point: M4 Max at 546 GB/s runs 4-bit Gemma-4-E2B at
177.8 tok/s **[V]** ≈ 267 GB/s achieved ≈ 49 % of peak. MoE is roughly half as
bandwidth-efficient as dense, which is where the 20–27 % band comes from.)*

**iPhone 17 Pro, Prophet-mini-0.5B.** Bytes/token: 362 M × 0.5625 + 134 M × 0.75 =
0.204 + 0.100 = **0.304 GB/token** (GPU/4-bit path).

Two independent estimates:
- *Bandwidth scaling from measurement:* Qwen3.5-2B 4-bit → 61 tok/s at ~1.2 GB/token **[V]**;
  scaling by bytes gives 61 × (1.2/0.304) = 241 tok/s, discounted 0.6 for fixed per-step
  overhead at small sizes → **~145 tok/s**.
- *Cross-device scaling:* Qwen3.5-2B is 291.9 tok/s on M4 Max and 61 on iPhone 17 Pro **[V]**
  → iPhone ≈ 0.209 × M4 Max. Qwen3.5-0.8B is 421.1 on M4 Max **[V]** → ~88 tok/s on iPhone for
  a 0.8B; our 0.5B is smaller → **~110–150 tok/s**.

The two agree. Applying the measured sustained-retention factors **[V]**:

| Path | Burst | **Sustained (10 min)** | Power | With MTP (×~2, dense) |
|---|---:|---:|---:|---:|
| MLX / GPU, INT4 | 130–160 | **50–60** (38 % retention) | ~24.7 W | ~100 sustained |
| **CoreML / ANE, INT8** | 45–55 | **30–37** (67 % retention) | **~12.7 W** | **~60–70 sustained** |
| Prefill, ANE | 400–900 | — | — | — |

*(ANE anchors: LFM2.5-350M → 52 tok/s, Qwen3.5-0.8B → ~48 tok/s on iPhone 17 Pro **[V]**.)*

**Recommendation: ship ANE as the default on iPhone and MLX/GPU as an optional "turbo" mode.**
ANE wins on sustained throughput, halves power, and uses a third of the memory (INT8 CoreML at
~1.0 GB vs MLX at ~3.0 GB observed for a 2B **[V]**). Users notice the second paragraph, not
the first sentence.

### 5.5 Training budget on the A100 [C]

| Item | Value |
|---|---|
| FLOPs/token (fwd+bwd, incl. MTP) | ~10.5 GFLOP |
| A100 BF16 dense peak / assumed MFU | 312 TFLOPS / 25–35 % → ~90 TFLOPS |
| Throughput | ~8 570 tok/s |
| Tokens/day (8 effective h) | ~247 M |
| Memory: weights + grads (BF16) + AdamW-8bit states | 18.9 + 18.9 + 18.9 = 56.7 GB |
| + activations w/ full checkpointing | ~8–15 GB → **~65–72 GB of 80 GB** |
| QAT anneal overhead (P2/P3) | +15–40 % step time on 12 % of tokens ≈ **+3 % total** |

**It fits, barely.** Requires 8-bit optimizer states, gradient checkpointing, fused QDQ
(no second weight copy), and checkpoint-resumable training against Colab preemption.

---

## 6. Risks & failure modes

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | **Prophet-mini (over-trained, D/N≈200, 0.5B dense) degrades badly at 4-bit** — the highest-probability failure in this track (§3.2) | **High** | High — kills the iPhone claim | Mini gets the *most* QAT, not the least: full P2 anneal + P3 distillation. Ship INT8-on-ANE as the primary (INT8 is nearly free) and treat INT4 as the stretch. Run V3 ladder early |
| **R2** | **MoE routing collapse under quantization** — quantized router flips top-k, output diverges even though per-layer error looks small | Medium | High | FP32 router logits (C5); router-margin regularizer in P2; **routing-agreement-rate as a gating metric** (§7). Never quantize the router |
| **R3** | **sm_120 kernel immaturity.** NVFP4 on consumer Blackwell needs 12 community patches to vLLM/FlashInfer/CUTLASS and C++ patches to TRT-LLM **[V]**; vLLM's own docs don't list Blackwell **[V]**. Our custom MoE won't be supported anywhere out of the box | **High** | Medium | Two-path strategy: **GGUF/llama.cpp as the availability path** (works today on sm_120), NVFP4 as the performance path. Budget engineering time for one fused NVFP4 MoE decode kernel of our own |
| **R4** | **ANE cannot express a MoE** — no dynamic gather/control flow. Confirmed by design | Certain | — | Mini is dense by construction. Already handled |
| **R5** | **iOS jetsam kill.** Apple's own Core AI was killed at 2048-ctx on iPhone 17 Pro **[V]** | Medium | High (App Store rejection / crashes) | Hard 1.5 GB cap; request `com.apple.developer.kernel.increased-memory-limit`; INT8 ANE path; measure *residency*, not `phys_footprint` (llama.cpp's 191 MB hides 2.9 GB **[V]**) |
| **R6** | **MTP speculation underdelivers on the MoE** (union-of-experts, §4.6: 1.07–1.46× not 2×) | **Certain** — it's arithmetic | Medium (a broken promise, not a broken product) | Budget 1.2–1.5× in all external claims. Grow the shared expert. Prototype expert-biased drafting |
| **R7** | **Expert-level over-training** (each expert at ~530 tokens/param, §3.3) makes the 89.6 % of params we quantize hardest the most fragile | Medium | High | Dedicated ladder experiment V3b. Fallback: raise experts to 5 bpw (+1.06 GB, still fits) |
| **R8** | **More pretraining data makes the shipped model worse** (2411.04330 **[V]**) | Low at our token budget | High if it happens silently | The P1 PTQ probe makes this *observable*. If δ_PTQ starts rising, stop adding tokens and add experts instead (§3.4) |
| **R9** | **A100 gets zero speedup from QAT** (sm_80, no FP4/FP8) and QAT costs 15–40 % step time | Certain | Low (~3 % of total compute) | Confine QAT to the last 12 % of tokens; fused QDQ; be explicit in planning |
| **R10** | **W4A4 prefill loses accuracy** (−2.9 to −4.4 pts measured at 7–8B **[V]**, worse at 1.35B active) | Medium-high | Medium | W4A4 is prefill-only and optional. Default to FP8 activations for prefill; enable NVFP4×NVFP4 only if measured KL stays flat |
| **R11** | **The 4-bit lm_head trap.** EXL3 warns output-layer quantization matters most for small models **[V]** | Medium | Medium | INT8 floor on the tied head (C6). Never share groups across the vocab axis |
| **R12** | **A benchmark that looks fine and a model that is broken** — Cactus: 50 tok/s, **3 % GSM8K** **[V]** | Medium | Very high | Never accept perplexity alone. KL divergence + task suite + routing agreement, gated (§7) |
| **R13** | **Contaminated public benchmark numbers** (e.g. the 17 000 tok/s 5090 claim, §2.9) | High | Medium | Every external number gets a roofline sanity check before it enters a plan |
| **R14** | **Colab preemption during the QAT anneal** loses the most valuable 12 % of the run | Medium | Medium | Checkpoint every ~30 min including quantizer state; make P2/P3 resumable and idempotent |

---

## 7. Validation plan

The point of this section: **prove the on-device claims without owning a 5090, a Mac Studio, or
an iPhone 17 Pro** — then name exactly what must eventually be tested on real silicon.

### V1 — Bit-exact quantization emulator (week 1, A100 or CPU)

Implement simulated quantization that is bit-exact to each target format, not "round to 16
levels."

```python
E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6]          # ±, max = 6
def nvfp4(W, blk=16):                        # 4 + 8/16 = 4.50 bits/weight
    s_g   = W.abs().amax() / (6.0 * 448.0)   # 448 = max finite E4M3
    Wg    = (W / s_g).reshape(-1, blk)
    s_b   = to_e4m3(Wg.abs().amax(-1, keepdim=True) / 6.0)   # FP8 block scale
    return (round_to_e2m1(Wg / s_b) * s_b).reshape(W.shape) * s_g

def mxfp4(W, blk=32):                        # 4 + 8/32 = 4.25 bits/weight
    Wg  = W.reshape(-1, blk)
    s_b = 2.0 ** torch.floor(torch.log2(Wg.abs().amax(-1, keepdim=True) / 6.0))  # E8M0
    return (round_to_e2m1(Wg / s_b) * s_b).reshape(W.shape)
```

Plus INT4/INT6/INT8 affine group-32/64 (MLX and GGUF K-quant semantics) and INT4/INT8
palettization (CoreML). **Cross-validate against NVIDIA Model Optimizer's reference NVFP4
implementation** on random tensors — if our emulator disagrees with the vendor's on a single
tensor, every downstream conclusion is void.

### V2 — Metrics that actually detect breakage

Perplexity is not sufficient (R12). Gate every quantized checkpoint on:

| Metric | Threshold to ship |
|---|---|
| **KL(FP16 ‖ quantized)** on 2 M held-out tokens — the sensitive metric EXL3 uses | < 0.01 nats/token |
| Top-1 next-token agreement with FP16 | > 97 % |
| **Routing agreement rate** (fraction of (token, layer) pairs with identical top-4 expert sets) | > 95 % |
| MMLU / GSM8K / HumanEval / IFEval | within 1.0 pt of BF16 |
| Long-context needle-in-haystack at 128 k with INT4 KV | ≥ 95 % of BF16-KV score |
| Degenerate-output scan (repetition rate, empty-answer rate) | no regression |

Routing agreement is the metric nobody else reports and the one most likely to catch our
specific failure mode (R2). Add it to the eval harness on day one.

### V3 — The precision scaling ladder (the highest-value pre-flight experiment)

**Cost: ~100–200 A100-hours. Do this before committing the main run's token budget.**

**V3a — fit our own δ_PTQ law.** Train a ladder of tiny Prophets sharing the final
architecture: N ∈ {50 M, 120 M, 300 M} × D/N ∈ {20, 100, 400, 1000}. For each, measure the
PTQ gap at NVFP4 / INT4-g64 / INT3 / INT8 and fit

```
delta_PTQ  =  C_T · D^{gamma_D} / N^{gamma_N} · exp(−P_post / gamma_post)
```

Then extrapolate to (9.4 B, our planned D) and to (0.5 B, mini's D). **This converts the
central risk of the track from an argument into a measured, extrapolable number**, and it
directly answers "how many tokens can the mini absorb before 4-bit stops working?"

**V3b — the MoE expert-over-training question (R7, §3.3).** Same ladder, but sweep
`n_experts ∈ {8, 24, 48}` at fixed N_active and fixed D. If δ_PTQ tracks **N_total** we are
safe and should add experts (§3.4); if it tracks **tokens-per-expert-parameter** we must raise
expert precision to 5 bpw. **This is an open research question I could not resolve from the
literature, and it is cheap to settle ourselves.**

**V3c — QAT anneal ablation.** On the 300 M rung: no-QAT vs P2-only vs P2+P3-distillation.
Expect ~50 % gap recovery from P3 alone (Gemma-3's measured figure **[V]**). Confirms the
schedule pays for its 10–13 %.

### V4 — Continuous PTQ probe during the main run

Wire V1 + V2 into training as a callback (P1 in §4.5). Every 2 000 steps, log Δloss, ΔKL and
routing agreement for each target format to the training dashboard, next to the loss curve.
Cost < 0.5 % of compute. This is how R8 ("more data makes the shipped model worse") becomes a
line on a chart instead of a discovery made after the run.

### V5 — Hardware proxies (no target device required)

| Claim | How to validate without the device |
|---|---|
| 5090 memory footprint | Exact from the emulator — bytes are bytes. **No hardware needed; this claim is already proven.** |
| 5090 tok/s | Roofline calibrated against two independent published measurements on real sm_120: 175 tok/s Qwen3.6-35B-A3B NVFP4 and 135 tok/s 30B-MoE @24.1 GB **[V]**. Publish the roofline *with* its 18.5 %-efficiency assumption stated |
| NVFP4 numerical correctness | A100 runs the emulator; correctness is arch-independent. Only *speed* needs sm_120 |
| sm_120 kernel viability | Rent a 5090 or RTX PRO 6000 (same SM120) hourly — **~$0.3–0.7/hr [K]**, so a full day of kernel validation costs under $20. Do this once before publishing any tok/s number |
| Mac path | MLX is bandwidth-scalable: validate correctness and efficiency ratio on **any** M-series (an M1/M2 MacBook works), then scale by the 819/546 bandwidth ratio against the published M4 Max table **[V]** |
| iPhone ANE op coverage | **An iPhone 15 Pro or 16 Pro is sufficient.** CoreML compilation, ANE-vs-CPU op placement, chunking and jetsam headroom are all A-series-generic; A19 Pro is only ~1.3–1.6× faster **[K]** |
| Sustained/thermal behavior | Cannot be simulated. Must be measured (see V6) |

### V6 — What must eventually be tested on real hardware (non-negotiable)

1. **iPhone sustained throughput and thermals, 10-minute run.** The burst/sustained gap is
   38–67 % depending on runtime **[V]** and no emulator predicts it.
2. **iOS jetsam headroom under real memory pressure** (other apps running, camera warm). Apple's
   own framework got killed **[V]**; we will not learn our real ceiling from a simulator.
3. **CoreML op placement.** Whether a layer lands on ANE or silently falls back to CPU is
   decided by the on-device compiler and changes with every iOS release. Target 99 %+ ANE
   utilization **[V]** and re-verify per OS version.
4. **NVFP4 accumulate-order numerics on real sm_120 silicon.** FP4 GEMM accumulation order and
   the FP8 scale path are not bit-reproducible in emulation; small models can be sensitive.
5. **Long-context INT4 KV at 128 k–1 M on a real 32 GB card** — fragmentation and allocator
   behavior, not just arithmetic.
6. **End-to-end tok/s with our own MoE kernel**, since no published number covers our
   architecture.

**Gate:** no external claim of the form "Prophet runs at X tok/s on device Y" ships until it
has been measured on device Y. Until then, publish the roofline with its assumptions visible —
that is defensible; a projected number quoted as measured is not.

---

## 8. References

Verification note: entries marked **[V]** had their existence and/or content confirmed from a
live source during this session. Entries marked **[K]** come from prior knowledge (cutoff
May 2026); **the arXiv IDs for [K] entries should be spot-checked before they are cited
externally.**

### Scaling laws for precision (the core of §3)
- **[V]** Kumar, Ankner et al., *Scaling Laws for Precision*, arXiv **2411.04330** (ICLR 2025)
- **[V]** *Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens*, arXiv **2411.17691**
- **[V]** *Scaling Laws for Floating Point Quantization Training*, arXiv **2501.02423**

### Post-training quantization — weights
- **[K]** Frantar et al., *GPTQ*, arXiv **2210.17323**
- **[K]** Lin et al., *AWQ*, arXiv **2306.00978**
- **[K]** Kim et al., *SqueezeLLM*, arXiv **2306.07629**
- **[K]** Shao et al., *OmniQuant*, arXiv **2308.13137**
- **[K]** Tseng et al., *QuIP#*, arXiv **2402.04396**
- **[K]** Egiazarian et al., *AQLM*, arXiv **2401.06118**
- **[V]** Tseng et al., *QTIP: Quantization with Trellises and Incoherence Processing*, arXiv **2406.11235**
- **[V]** turboderp, *EXL3 / exllamav3* — github.com/turboderp-org/exllamav3
- **[K]** Badri & Shaji, *HQQ: Half-Quadratic Quantization* (Mobius Labs; no arXiv)
- **[V]** *Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression*, arXiv **2510.20984**

### Post-training quantization — activations & rotations
- **[K]** Xiao et al., *SmoothQuant*, arXiv **2211.10438**
- **[K]** Zhao et al., *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving*, arXiv **2310.19102**
- **[K]** Ashkboos et al., *QuaRot*, arXiv **2404.00456**
- **[V]** Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*, arXiv **2405.16406**
- **[K]** Lin et al., *QServe: W4A8KV4 Quantization*, arXiv **2405.04532**
- **[V]** *ReSpinQuant: Subspace Residual Rotation Approximation*, arXiv **2604.11080**
- **[V]** *Outlier Smoothing with Closed-Form Rotations for W4A4*, arXiv **2511.22316**
- **[V]** *ITQ3_S: Interleaved Ternary Quantization with Rotation-Domain Smoothing*, arXiv **2603.27914**

### FP4 formats, NVFP4 and low-precision training
- **[V]** NVIDIA, *Pretraining Large Language Models with NVFP4*, arXiv **2509.25149**
- **[V]** *FP4 All the Way: Fully Quantized Training of LLMs*, arXiv **2505.19115**
- **[V]** *Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling*, arXiv **2512.02010**
- **[V]** *ScaleSweep: Accurate NVFP4 PTQ of LLMs via Block Scale Initialization*, arXiv **2606.07618**
- **[V]** *MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations*, arXiv **2605.31035**
- **[V]** *Characterizing the Impact of NVFP4 Quantization for Low-Power Edge AI Deployment*, arXiv **2606.06527**
- **[V]** *The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training*, arXiv **2603.10444**
- **[V]** *Normalized Architectures are Natively 4-Bit*, arXiv **2605.06067**
- **[V]** NVIDIA, *Nemotron 3: Efficient and Open Intelligence*, arXiv **2512.20856**
- **[V]** NVIDIA TensorRT-Model-Optimizer — github.com/NVIDIA/TensorRT-Model-Optimizer

### QAT and native low-precision
- **[K]** Liu et al., *LLM-QAT*, arXiv **2305.17888**
- **[V]** Ma, Wang et al., *The Era of 1-bit LLMs: BitNet b1.58*, arXiv **2402.17764**
- **[V]** Wang, Ma, Wei, *BitNet a4.8: 4-bit Activations for 1-bit LLMs*, arXiv **2411.04965**
- **[V]** *BitNet v2: Native 4-bit Activations with Hadamard Transformation*, arXiv **2504.18415**
- **[V]** Ma et al., *BitNet b1.58 2B4T Technical Report*, arXiv **2504.12285**
- **[V]** microsoft/BitNet (bitnet.cpp) — github.com/microsoft/BitNet
- **[V]** *Litespark Inference for CPUs: SIMD Framework for Ternary LLMs*, arXiv **2605.06485**
- **[K]** Chen et al., *EfficientQAT*, arXiv **2407.11062**
- **[V]** Google DeepMind, *Gemma 3 Technical Report*, arXiv **2503.19786**; Gemma 3 QAT models blog
- **[V]** Google, *Gemma 4 with quantization-aware training* (2026)

### KV cache
- **[V]** Liu et al., *KIVI: Tuning-Free Asymmetric 2-bit Quantization for KV Cache*, arXiv **2402.02750** (ICML 2024)
- **[K]** Hooper et al., *KVQuant*, arXiv **2401.18079**
- **[V]** *KV Cache is 1 Bit Per Channel: Coupled Quantization*, arXiv **2405.03917**
- **[V]** *RotateKV: Accurate and Robust 2-Bit KV Cache Quantization via Outlier-Aware Adaptive Rotations*, arXiv **2501.16383**

### Outliers and quantization-friendly architecture
- **[K]** Bondarenko et al., *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing*, arXiv **2306.12929**
- **[K]** Sun et al., *Massive Activations in Large Language Models*, arXiv **2402.17762**
- **[K]** Xiao et al., *Efficient Streaming LLMs with Attention Sinks*, arXiv **2309.17453**
- **[K]** Liu et al., *MobileLLM: Optimizing Sub-billion Parameter LMs for On-Device Use*, arXiv **2402.14905**
- **[K]** *How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study*, arXiv **2404.14047**

### Speculative decoding and multi-token prediction
- **[K]** Cai et al., *Medusa*, arXiv **2401.10774**
- **[K]** Li et al., *EAGLE*, arXiv **2401.15077**; *EAGLE-2*, arXiv **2406.16858**; *EAGLE-3*, arXiv **2503.01840**
- **[V]** SafeAILab/EAGLE — github.com/SafeAILab/EAGLE (EAGLE-2 4×, EAGLE-3 5.6× on 13B)
- **[K]** Gloeckle et al., *Better & Faster LLMs via Multi-token Prediction*, arXiv **2404.19737**
- **[K]** DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv **2412.19437**

### Hardware, runtimes and measured deployments
- **[V]** lna-lab/blackwell-geforce-nvfp4-gemm — SM120 patches for vLLM + FlashInfer + CUTLASS; 175 tok/s Qwen3.6-35B-A3B NVFP4
- **[V]** JohnTDI-cpu/trtllm-nvfp4-blackwell-fix — TRT-LLM C++ patches; 135 tok/s 30B MoE, 24.1 GB on RTX 5090
- **[V]** *Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs*, arXiv **2601.09527**
- **[V]** ggml-org/llama.cpp — KV cache quant types, `--spec-type` speculative framework
- **[V]** vllm-project/vllm — quantization docs and supported-hardware table
- **[V]** ml-explore/mlx-lm — `mlx_lm/quant/{awq,gptq,dwq,dynamic_quant}.py`; DWQ = KL-distilled scale/bias learning, 4-bit group 64
- **[V]** pytorch/executorch — Llama 3.2 1B/3B SpinQuant & QAT+LoRA on-device benchmarks
- **[V]** apple/ml-ane-transformers — up to 10× latency, 14× peak-memory improvement on ANE
- **[K]** Apple ML Research, *Deploying Transformers on the Apple Neural Engine* (machinelearning.apple.com/research/neural-engine-transformers)
- **[V]** john-rocky/apple-silicon-llm-bench — iPhone 17 Pro / M4 Max cross-runtime decode, memory, energy, sustained-throughput tables
- **[V]** john-rocky/CoreML-LLM — ANE chunked stateful decoding; 99.9 % ANE utilization; 34.2 tok/s Gemma 4 E2B on iPhone 17 Pro
- **[V]** Cornell-RelaxML/qtip — batch-1 throughput table
- **[V]** facebookresearch/SpinQuant; jy-yuan/KIVI — reference implementations

---

## Appendix A — Decisions this report asks other tracks to make

| # | Decision | Owner | R08's position |
|---|---|---|---|
| A1 | GQA vs MLA | Architecture | **GQA.** MLA quantizes worse and is unsupported on 2 of 3 targets (§4.2) |
| A2 | Dimension choices | Architecture | All quantization-relevant dims **power-of-two or Hadamard-constructible** (C1) |
| A3 | QK-norm, no biases, softmax sink logits, clipped/gated attention | Architecture | **All four, from step 0** (C2, C3) |
| A4 | SwiGLU vs ReLU² | Architecture | SwiGLU + Hadamard-before-`down_proj` (main); **ReLU² for mini** (ANE + sparsity) (C4) |
| A5 | MTP heads in pretraining | Architecture + Training | **Yes, 2 heads, 0.67 % of params** — but budget 1.2–1.5× speedup on the MoE, not 2× (§4.6) |
| A6 | **More tokens vs more experts** at equal compute | Training + Data | **More experts.** Training FLOPs scale with N_active; δ_PTQ shrinks with N_total (§3.4) |
| A7 | Token budget ceiling | Training | Let the **P1 PTQ probe** decide, not a fixed number (§4.5, R8) |
| A8 | Mini's training budget | Training | The mini is the over-trained one and the PTQ-fragile one. **Its D/N is a quantization decision, not just a quality decision** (§3.2) |
| A9 | Distillation for the main model | Training | Flag: dense logit supervision may push us into the PTQ-fragile regime at far fewer tokens than the raw count suggests. Measure via V3 |
| A10 | Eval harness | Eval | Add **KL-to-FP16** and **routing agreement rate** as first-class gated metrics (§7 V2) |
