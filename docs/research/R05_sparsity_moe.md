# R05 — Sparsity: fitting 10B of knowledge into 32 GB (and 8 GB)

**Track owner:** R05 · **Status:** decision-ready · **Date:** 2026-09-03

**Verdict in one paragraph.** Sparse activation (MoE) is the correct architecture for Prophet and it is
essentially free at inference on all three targets. But **at 100–300 A100-hours the binding constraint is
not the architecture, it is the token budget (~10 B tokens)**, and 10 B tokens does not justify 8–12 B
total parameters — every successful open MoE was trained at 700–1200 tokens per *total* parameter; we
would be at 2. The concrete recommendation is therefore **Prophet-M v1 = 5.1 B total / 1.07 B active**
(64 fine-grained experts of width 512, top-8, 1 shared expert), trained **dense-first for ~40 % of the
budget then Drop-Upcycled** into the MoE, with a documented, cheap growth path to the 9.8 B / 128-expert
"v2" once more compute exists. And an unambiguous warning that appears again in §3: **beating Qwen3-1.7B
from a random init is arithmetically out of reach at this budget by a factor of ~470×.**

---

## 1. Problem statement

### 1.1 Why sparsity, stated precisely

Autoregressive decoding at batch size 1 issues one GEMV per weight matrix. Arithmetic intensity is ~2
FLOPs per byte read, versus ~300–500 FLOPs/byte needed to saturate an RTX 5090 or ~100 for an M-series
Ultra. Decoding is therefore **pure memory-bandwidth streaming**:

```
tok/s_ceiling  =  achievable_bandwidth  /  (bytes_read_per_token)
bytes_read_per_token  =  active_params × bytes_per_param   +   kv_cache_bytes(context)
```

Three consequences that drive every decision below:

1. **Only reducing *bytes read* helps decode.** FLOP-reduction techniques that still read all the weights
   — Mixture-of-Depths (2404.02258), intra-FFN activation sparsity without a loader that exploits it,
   speculative-free FLOP tricks — buy **zero** decode speedup at bs=1. MoE is the one technique that
   reduces bytes-read while keeping capacity. This is the whole thesis, and it holds.
2. **Total parameters are nearly free; active parameters are the price.** Weight *residency* costs VRAM
   (a one-time capacity check), weight *traffic* costs latency (a per-token cost). On a 32 GB 5090 a 10 B
   model at FP8 is 10 GB resident — comfortably under budget — while its per-token traffic is that of a
   1 B model.
3. **The KV cache competes with the weights for the same bandwidth.** For the config in §4, KV traffic is
   48 KB/token; at 32 K context that is 1.57 GB/token, i.e. **more than the 1.07 GB of active weights**.
   A sparse model with a fat KV cache is not fast. R05's recommendation is only valid alongside
   aggressive KV reduction (GQA 4 kv-heads minimum; see the attention track).

### 1.2 Bandwidth arithmetic for the three targets

Achievable bandwidth is 60–85 % of peak in practice; ceilings below use **peak** and are then discounted.

| Target | Memory | Peak BW | Notes |
|---|---|---|---|
| RTX 5090 | 32 GB GDDR7 | **1792 GB/s** | Blackwell, native NVFP4/MXFP4 tensor cores |
| Mac Studio M3 Ultra | up to 512 GB | **819 GB/s** | capacity is a non-issue, bandwidth is the wall |
| Mac Studio M4 Max | up to 128 GB | 546 GB/s | |
| iPhone 17 Pro (A19 Pro) | ~8 GB shared | **~60 GB/s** | app memory ceiling ≈ 3.5–5.5 GB with the increased-memory entitlement |

Ceiling tok/s for **1.07 B active parameters** (the recommended config), weights only:

| Precision | bytes/token | RTX 5090 | M3 Ultra | M4 Max | iPhone 17 Pro |
|---|---|---|---|---|---|
| bf16 | 2.15 GB | 836 | 382 | 255 | 28 |
| FP8 / int8 | 1.07 GB | 1672 | 764 | 510 | 56 |
| MXFP4/NVFP4 (≈0.53 B/param) | 0.57 GB | **3155** | 1442 | 961 | 106 |

Dense baselines for the same table:

| Model | active | bytes/token @Q4 | 5090 ceiling | M3 Ultra ceiling |
|---|---|---|---|---|
| Qwen3-1.7B | 1.7 B | 0.90 GB | 1991 | 910 |
| Llama-3.2-3B | 3.2 B | 1.70 GB | 1054 | 482 |
| Gemma-3-4B | 4.3 B | 2.28 GB | 786 | 359 |
| Qwen3-4B | 4.0 B | 2.12 GB | 845 | 386 |

**Reality discount.** Measured MoE engines currently realise only **24–30 % of the bandwidth ceiling** at
bs=1, because fine-grained experts turn one big GEMV into k small ones plus a gather/scatter:

* Qwen3-30B-A3B (3.3 B active) on an RTX 5090, llama.cpp Q4: **236 tok/s** measured vs 985 ceiling → 24 %.
* gpt-oss-20b (3.6 B active, MXFP4): ~250–300 tok/s reported on 5090 vs ~940 ceiling → 27–32 %.
* Qwen3.6-35B-A3B Q3 on a 3090 (936 GB/s): **120 tok/s** at 23 GB resident.

Dense models hit 50–65 % of ceiling. So the *honest* comparison for Prophet-M v1 on a 5090:

| | ceiling | realistic (llama.cpp-class, 27 %) | realistic (tuned fused-MoE + CUDA graphs, 45 %) |
|---|---|---|---|
| Prophet-M v1 @ MXFP4 | 3155 | **~850 tok/s** | ~1420 tok/s |
| Prophet-M v1 @ FP8 | 1672 | ~450 tok/s | ~750 tok/s |
| Llama-3.2-3B @ Q4 (dense, 55 %) | 1054 | ~580 tok/s | — |

i.e. Prophet gets ~1.5× the decode speed of a 3B dense model *and* carries 5.1 B parameters of knowledge.
Closing the 27 % → 45 % gap is worth as much as any architectural change and is listed as a work item.

### 1.3 Batch size erodes the advantage (know when it stops paying)

With E=64 routed experts and k=8, the expected number of *distinct* experts touched by a batch of B tokens
is `E·(1-(1-k/E)^B)`:

| batch | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| distinct experts read | 8 | 26 | 56 | 63 | 64 |

By batch 32 essentially all weights are read and the MoE degenerates to a dense model of 5.1 B parameters
with 1.07 B worth of FLOPs. MoE is a **single-user, low-batch** optimisation — which is exactly Prophet's
deployment story (one person, one 5090 / one Mac / one phone). It is the *wrong* optimisation for a
serving fleet, and this should be said out loud so nobody is surprised later.

---

## 2. State of the art

### 2.1 The MoE design space (real configurations)

| Model | arXiv | Total | Active | L | d_model | E routed | top-k | d_expert | d_exp/d_model | shared | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GShard | 2006.16668 | 600 B | — | — | — | 2048/layer | 2 | large | — | 0 | — |
| Switch-Base | 2101.03961 | 7 B | 0.6 B | 12 | 768 | 128 | 1 | 3072 | 4.0 | 0 | 500 B |
| Mixtral-8x7B | 2401.04088 | 46.7 B | 12.9 B | 32 | 4096 | 8 | 2 | 14336 | 3.5 | 0 | — |
| DeepSeekMoE-16B | 2401.06066 | 16.4 B | 2.8 B | 28 | 2048 | 64 | 6 | 1408 | 0.69 | 2 | 2 T |
| JetMoE-8B | 2404.07413 | 8 B | 2.2 B | 24 | 2048 | 8 | 2 | 5632 | 2.75 | 0 | 1.25 T |
| **OLMoE-1B-7B** † | 2409.02060 | 6.9 B | 1.3 B | 16 | 2048 | **64** | **8** | **1024** | 0.50 | **0** | 5 T |
| DeepSeek-V3 | 2412.19437 | 671 B | 37 B | 61 | 7168 | 256 | 8 | 2048 | 0.29 | 1 | 14.8 T |
| **Qwen3-30B-A3B** † | 2505.09388 | 30.5 B | 3.3 B | 48 | 2048 | **128** | **8** | **768** | 0.375 | 0 | 36 T |
| Qwen3-Next-80B-A3B | — | 80 B | 3.0 B | 48 | 2048 | 512 | 10 | 512 | 0.25 | 1 | ~15 T |
| **Ling-mini-2.0** † | GH: inclusionAI/Ling-V2 | **16.26 B** | **1.43 B** | 20 | 2048 | 256 | 8 | 512 | 0.25 | 1 | 20 T |
| gpt-oss-20b † | 2508.10925 | 21 B | 3.6 B | 24 | 2880 | 32 | 4 | 2880 | 1.0 | 0 | — |
| gpt-oss-120b † | 2508.10925 | 117 B | 5.1 B | 36 | 2880 | 128 | 4 | 2880 | 1.0 | 0 | — |
| Granite-4.0-h-tiny † | IBM model cards | 7 B | 1 B | — | — | 64 | 6 | — | — | 1 | — |

† = numbers verified during this research against the actual config file / repository README
(`allenai/OLMoE/configs/OLMoE-1B-7B-0924.yml`, `transformers/models/qwen3_moe/configuration_qwen3_moe.py`,
`transformers/models/granitemoehybrid/…`, `inclusionAI/Ling-V2` README, `openai/gpt-oss` README).
Unmarked rows are from the papers/model cards and were not re-verified (arxiv.org was unreachable from
this environment; see §8 note).

**Two clear trends.**
1. **Expert width is shrinking**: `d_expert/d_model` went 3.5 (Mixtral 2023) → 0.5 (OLMoE 2024) →
   0.375 (Qwen3 2025) → **0.25** (Ling 2.0, Qwen3-Next 2025-26). Fine-grained wins.
2. **Total/active ratio is growing**: 3.6× (Mixtral) → 5.3× (OLMoE) → 9.2× (Qwen3-30B) → **11.4×**
   (Ling-mini-2.0) → 27× (Qwen3-Next). But every one of these was trained on ≥2 T tokens.

### 2.2 Quality-per-active-parameter: what the ablations actually say

| Finding | Source | Number |
|---|---|---|
| Granularity: 32 experts (top-4) vs 8 experts (top-1) at iso-FLOP | OLMoE 2409.02060 | **≈ +10 %** on HellaSwag / MMLU |
| Granularity: 64 (top-8) vs 32 (top-4) | OLMoE | only **+1–2 %** (diminishing returns) |
| Shared expert vs none, at E=64/k=8 | OLMoE | **hurts** — reduces expert-combination count by ~90 % for no gain |
| Shared expert isolation + fine-grained segmentation | DeepSeekMoE 2401.06066 | DeepSeekMoE-2B ≈ GShard-2.9B with 1.5× fewer expert params |
| Sparse MoE vs dense at iso-token | DeepSeekMoE 2401.06066 | 16.4B/2.8B ≈ LLaMA2-7B at **39.6 %** of the compute |
| Efficiency leverage at 1/32 activation | Ling / 2507.17702 | Ling-mini-beta **0.85 B active ≡ 6.1 B dense** on the same 1 T tokens, **7× less compute** |
| Ling-mini-2.0 in production | GH README | 1.43 B active ⇒ "7× equivalent dense performance", 300+ tok/s on H20 |
| MoE beats dense at *strictly equal* total params + compute + data | 2506.12119 (ICLR'26) | yes, in an optimal activation-rate band (≈20 % for 7 B); 250+ MoE runs |
| Granularity has a **non-linear optimum** | 2507.17702 | EL is a power law in activation-ratio and compute; granularity has a clear optimal range, over-fine hurts |
| Fine-grained experts genuinely increase expressivity | 2505.06839 | theory + experiments |

**Interpretation for us.** Every headline MoE gain is measured at ≥1 T tokens. The 2507.17702 scaling
law is explicit that efficiency leverage is *itself* a power law in the compute budget — **at 1e20 FLOPs
the leverage is a fraction of the 7× measured at 1e23.** We should plan for **2–3×**, not 7×.

### 2.3 Routing and its pathologies

| Mechanism | Source | What it does / real coefficients |
|---|---|---|
| Token-choice top-k | GShard/Switch | Default. Needs a balancing mechanism or it collapses. |
| **Expert-choice** | 2202.09368 | Perfect balance by construction, ~2× faster convergence than top-1. **Non-causal**: an expert's choice depends on other tokens in the sequence → leaks future information and cannot be replicated at decode. **Rejected for Prophet.** |
| Auxiliary load-balance loss | 2101.03961 | `α·E·Σ f_i·p_i`. OLMoE **α=0.01** (verified from config). Qwen3-MoE **α=0.001** (verified). Granite **α=0.001** (verified). DeepSeek-V3 **α=1e-4**. |
| **Router z-loss** | ST-MoE 2202.08906 | `(logsumexp logits)²`; keeps router logits small → removes bf16 round-off instability. OLMoE **1e-3** (verified from config). Cheap, no quality cost. **Use it.** |
| **Aux-loss-free bias balancing** | DeepSeek-V3 2412.19437 | Per-expert bias `b_i` added to the score **for top-k selection only** (never to the gate value); after each step `b_i += γ·sign(mean_load − load_i)`. **γ = 1e-3**, decayed to 0 for the last ~3 % of tokens. Avoids the gradient interference of an aux loss → better specialisation. |
| **Global-batch vs micro-batch LBL** | 2501.11873 | Computing the balance loss over the *micro*-batch forces local uniformity and destroys domain specialisation. Accumulating expert counts over the **global batch** improves both perplexity and downstream. On a single GPU with grad accumulation this is a free ring buffer of counts — **a genuine single-GPU advantage**. |
| Softmax-then-topk vs topk-then-softmax | 2410.07524 | softmax-then-topk-then-renormalise is better (measured on upcycling). |
| Sigmoid scoring | DeepSeek-V3, Ling 2.0, Qwen3-Next | More stable with ≥128 experts; pairs naturally with the additive bias. |
| ReLU routing (dropless, learned sparsity) | ReMoE 2412.14711 | Fully differentiable, no top-k discontinuity. Interesting, unproven at scale. |
| Router saturation | OLMoE 2409.02060 | Routing decisions largely freeze **very early** in training. At 10 B tokens this is a first-class risk: a badly initialised router is a permanently badly routed model. |
| Expert collapse / dead experts | universal | With E=64, k=8 and a 32 K-token micro-batch each expert sees ~4096 tokens/micro-batch — enough for stable statistics. The danger at our scale is not statistical noise, it is **too many experts for too little data**. |

### 2.4 Alternative sparsity mechanisms — and the verdict for Prophet

| Approach | Source | Best number | Verdict for Prophet |
|---|---|---|---|
| **PEER / Mixture of a Million Experts** | 2407.04153 | C4 ppl **20.63** vs MoE 21.41 vs PKM 21.92 at 6e18 FLOPs; >1 M single-neuron experts via product keys | **No for v1.** Best-in-class isoFLOP curve but only validated at ≤1e19 FLOPs, and product-key retrieval over 10⁶ experts is gather-bound — terrible arithmetic intensity at bs=1, hopeless on ANE. Revisit as a *knowledge* layer. |
| **Memory Layers at Scale** | 2412.09764 (and 1907.05242) | Beats dense with **>2× the compute**, and beats MoE at **matched compute + params**, especially on factual QA; scaled to 128 B memory params / 1 T tokens / 8 B base | **Promising add-on.** Exactly targets our weakest axis (factual recall from a 10 B-token run). Sparse row updates make CPU-resident optimizer state natural. But it needs data to fill. Ablation A11. |
| Monet (monosemantic experts) | 2412.04139 | interpretability win, not a quality win | No |
| **Deja Vu / contextual sparsity** | 2310.17157 | 2× decode speedup by predicting hot FFN rows | Only for ReLU-family nets; SwiGLU models have low intrinsic sparsity. **MoE is the structured version of the same idea** — a router is a contextual-sparsity predictor you train instead of bolt on. |
| ProSparse | 2402.13516 | converts SwiGLU→ReLU, ~88 % activation sparsity on LLaMA-2-7B | Post-hoc; costs a conversion run we can't afford |
| **Q-Sparse** | 2407.10969 | top-K activation sparsity + STE; 40 % sparsity on q/k/v/o/up/down, 60 % on gate; 300 M–7 B scaling laws | ~2× at best, complicates training. **Cheap to fold in partially**: use **ReLU²** in experts (free, raises natural sparsity, no quality loss — ablation A9). |
| BlockFFN | 2507.08771 | **chunk-level** activation sparsity designed for end-side accelerators | **Directly relevant to the iPhone.** Chunk-level routing makes the graph static enough for ANE. |
| CoSMoEs | 2503.00245 | compact sparse MoE for on-device | Same family; the on-device answer is chunk/sequence routing, not token routing |
| **Mixture-of-Depths** | 2404.02258 | 12.5 % capacity, route every other block; **up to 1.5 % better ppl at isoFLOP**; a 220 M MoD model matched baseline while stepping **66 % faster** | **No for inference** (saves FLOPs, not bytes — zero decode gain). **Maybe for training**: 1.3–1.5× more tokens per A100-hour is real money at our budget. But MoD's top-k over the sequence is non-causal and needs an auxiliary predictor at sampling time. Optional, ablation A12, not on the critical path. |

### 2.5 Inference engineering: expert offloading and what it really costs

Relevant only if total params exceed device memory. **For Prophet at ≤10 B total this never happens on a
5090 or a Mac** — but the numbers below define the wall we are choosing to stay behind.

| System | arXiv | Setup | Measured |
|---|---|---|---|
| Mixtral-offloading | 2312.17238 | Mixtral-8x7B, LRU expert cache + mixed quant, RTX 3060 12 GB / Colab T4 | **2–4 tok/s** |
| **Fiddler** | 2402.07033 (ICLR'25) | *Unquantised* Mixtral-8x7B (>90 GB) on a single **24 GB** GPU. Key trick: move **activations to the CPU** (batch×4096 floats) instead of expert weights to the GPU (3×4096×14336 per expert) | **>3 tok/s**; **8.2–10.1×** vs Mixtral-offloading, **19.4–22.5×** vs DeepSpeed-MII |
| MoE-Infinity | 2401.14361 | request-level expert-activation tracing to prefetch | — |
| PowerInfer | 2312.12456 | hot/cold neuron split (ReLU models), GPU keeps hot neurons | OPT-30B on RTX 4090: **11.69 tok/s**, 11.7× llama.cpp |
| **llama.cpp `--n-cpu-moe` / `-ot`** | — | keeps attention + KV + non-expert weights on GPU, streams/computes expert FFNs on CPU | 120 B MoE Q3_K_XL on **RTX 3090: ~16 tok/s** @4 K depth; 35B-A3B IQ4_XS on an 8 GB laptop dGPU: **~27 tok/s**; 122 B FP8 with `--cpu-moe`: 7.1 tok/s using only **5.6 GB VRAM** |

**Design rule extracted:** offloading costs an order of magnitude. Size Prophet so that **FP8 weights fit
in 12 GB and MXFP4 in 6 GB**, guaranteeing full residency on a 5090 with room for a 128 K KV cache, and
guaranteeing that even a 16 GB laptop GPU runs it fully resident at 4 bits.

### 2.6 Does MoE work on the Apple Neural Engine / iPhone?

Short answer: **not as token-level MoE.** Core ML `MLProgram` compilation to ANE requires static shapes and
a restricted op set; a data-dependent `gather` that selects among a stack of expert weight tensors forces
a fallback to CPU/GPU, which loses the ANE's energy advantage and usually its speed advantage too. There
is no public example of a token-routed MoE running on ANE.

Two workable on-device patterns, both from 2025–26 work:
* **Chunk / sequence-level routing** (BlockFFN 2507.08771, CoSMoEs 2503.00245): choose the expert set once
  per chunk of N tokens so the compiled graph is static within a chunk. Costs quality, buys a static graph.
* **Dense small model.** Simplest, known-good, and at 8 GB the *total*-parameter footprint — not
  bandwidth — is the binding constraint anyway (see §4.4).

**Decision: Prophet-mini ships dense.** Chunk-MoE is an R&D item, not a v1 commitment.

### 2.7 Single-GPU MoE training mechanics

| Item | Reality |
|---|---|
| Naive HF `for expert in experts:` loop | Catastrophic at E≥64 — kernel-launch bound, ~10× slower. Never use. |
| **MegaBlocks** | 2211.15841. Block-sparse dMoE, no capacity factor, no dropped tokens. Up to **40 % faster end-to-end** than Tutel-based MoE, **2.4×** vs dense Megatron-LM at equal params. |
| **ScatterMoE** | 2403.08245. Triton, ~700 LoC, fuses the scatter/gather into the expert GEMMs — **no padding, no input copies**, so peak memory is lower than MegaBlocks. Reports ~38 % higher throughput than MegaBlocks in the authors' benchmark. **Best fit for a single A100.** |
| `torch._grouped_mm` / cuBLAS grouped GEMM | Requires **sm90+ (Hopper/Blackwell)**. Colab's A100 is **sm80** → *not available*. This is a concrete trap: the modern PyTorch MoE path will silently not apply. Plan on ScatterMoE/MegaBlocks Triton kernels. |
| Realistic MFU on one A100 | dense ~1 B model: **40–50 %**. Fine-grained MoE with ScatterMoE: **25–35 %**. Budget 30 %. |
| Dropless | OLMoE's shipped config is `moe_dropless: true`, `moe_mlp_impl: sparse` (verified). On one GPU there is no reason to drop tokens. |

**Optimizer-state arithmetic (the thing that actually decides model size).** Bytes per parameter on the GPU:

| Scheme | weights | master/compensation | grads | optimizer | **B/param** | 5.12 B model | 9.76 B model |
|---|---|---|---|---|---|---|---|
| A — fp32 master + fp32 AdamW | 2 | 4 | 2 | 8 | **16** | 82.0 GB ✗ | 156 GB ✗ |
| B — fp32 master + 8-bit AdamW (2110.02861) | 2 | 4 | 2 | 2 | **10** | **51.2 GB ✓** | 97.6 GB ✗ |
| C — Kahan-compensated bf16 + 8-bit AdamW | 2 | 2 | 2 | 2 | **8** | 41.0 GB ✓ | 78.1 GB ✗ |
| D — bf16 + stochastic rounding + 8-bit AdamW | 2 | 0 | 2 | 2 | **6** | 30.7 GB ✓ | 58.5 GB ⚠ |
| E — bf16 + **Muon** (single bf16 momentum, 2502.16982) | 2 | 0 | 2 | 2 | **6** | 30.7 GB ✓ | 58.5 GB ⚠ |
| F — D/E + CPU-offloaded optimizer state | 2 | 0 | 2 | 0 (host) | **4** | 20.5 GB ✓ | 39.0 GB ✓ |

**So: is 10 B total trainable on one A100 80 GB? Yes, but only under scheme D/E or F**, leaving ~11 GB for
activations — and scheme D/E means abandoning fp32 master weights, which is the single most common cause
of silent divergence. **5.1 B is trainable under scheme B with fp32 master weights intact and 18 GB of
headroom.** That asymmetry, plus §3.2, is why v1 is 5.1 B.

Two non-obvious memory traps on the way:
* **The logits tensor.** A 32 768-token micro-batch × 64 000 vocab in bf16 is **4.19 GB** (and 9.96 GB with
  Qwen's 151 936 vocab), plus an fp32 softmax copy. Mandatory fix: fused/chunked cross-entropy
  (Liger-Kernel 2410.10989 or Cut-Cross-Entropy 2411.09009) → ~0.1 GB.
* **CPU-offloaded optimizer for 10 B params** moves ~20 GB of gradients host-ward *per step*. At ~20 GB/s
  effective PCIe that is ~2 s of transfer plus a multi-second CPU Adam step, against a ~50 s step —
  10–20 % overhead. Survivable, not free. Colab's high-RAM A100 host has ~83 GB, so fp32 offloaded states
  for 10 B params (80 GB) would **not** fit; only 8-bit states (20 GB) would.

---

## 3. What actually transfers to our scale

### 3.1 The token budget, computed honestly

Assumptions: A100 80 GB, bf16 peak 312 TFLOP/s; full activation checkpointing; seq 4096.
Active non-embedding params 0.94 B ⇒ FLOPs/token = 6N (fwd+bwd) + 2N (recompute) + 12·L·s·d (attention)
= **9.94e9 FLOP/token**.

| Phase | MFU | tok/s | tokens/A100-hour |
|---|---|---|---|
| Dense | 45 % | 14 131 | **50.9 M** |
| MoE (ScatterMoE) | 30 % | 9 420 | **33.9 M** |

**300 A100-hours ⇒ ~10.5 B training tokens.** (100 A100-hours ⇒ ~3.5 B.) That is the whole budget. For
scale: OLMoE saw 5 T, Qwen3 36 T, Ling-mini-2.0 20 T. **We have 0.03–0.2 % of a competitor's data.**

### 3.2 Consequence 1 — total params must scale with data

Tokens per **total** parameter:

| Model | tokens/total-param |
|---|---|
| OLMoE-1B-7B | 725 |
| Qwen3-30B-A3B | 1180 |
| Ling-mini-2.0 | 1230 |
| DeepSeek-V3 | 22 |
| **Prophet @ 12 B total, 10.5 B tokens** | **0.9** |
| **Prophet @ 5.1 B total, 10.5 B tokens** | **2.1** |

Information-theoretically, 10.5 B tokens is ≈25 GB of text; 12 B bf16 parameters is 24 GB of *capacity*.
There is nothing to put in them. More parameters at fixed FLOPs never *raises* loss in theory, but in
practice each extra expert (a) dilutes the router's training signal, (b) adds a permanently
under-trained weight matrix that must still be quantised and shipped, and (c) costs VRAM and quantisation
error at inference for nothing. **The 8–12 B total target is not supported by a 10 B-token budget.** It
becomes justified at ≳100–200 B tokens, i.e. ~3000–6000 A100-hours, or immediately if the trunk is
inherited from an external checkpoint (§3.4).

Ablation **A10** measures exactly this and should be run before committing.

### 3.3 Consequence 2 — the "beat Qwen3-1.7B" goal is unreachable from scratch

Using the Chinchilla approach-3 fit `L(N,D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28` (absolute values are
not transferable across tokenizers/data; the *deltas* are what matter):

| Model | N (non-emb) | D | predicted L |
|---|---|---|---|
| Prophet dense-equivalent, from scratch | 0.94 B | 10.4 B | **2.695** |
| Prophet MoE, from scratch (assume 2.5× effective params) | 2.35 B eff. | 10.4 B | **2.599** |
| Qwen3-1.7B | 1.41 B | 36 T | **2.071** |

MoE buys **0.096 nats**. The data deficit costs **0.62 nats**. Inverting the fit: to reach Qwen3-1.7B's
loss with an *optimal* MoE at 0.94 B active you need **≈4.7 T tokens ≈ 140 000 A100-hours — 466× our
budget.** Distillation from a strong teacher buys perhaps 2×; better data another 1.5×. That leaves a
~150× gap.

**This is arithmetic, not pessimism, and R05's recommendation is that the project state it explicitly
rather than discover it at month four.** Three coherent responses:

* **(a) Reset the benchmark goal.** Target "best model trainable in 300 A100-hours", which is a real and
  defensible research claim, and where MoE is a genuine 2–3× win.
* **(b) Inherit a trunk.** Initialise from an Apache-2.0 open base (Qwen3-1.7B-Base, 36 T tokens) and
  upcycle it into Prophet's MoE. The *sparsity architecture* is still ours; the pretraining is not.
* **(c) Raise the compute.** ~3000 A100-hours changes the answer qualitatively (100 B tokens, 8–12 B
  total becomes correct, and the 8-12B target snaps into place).

R05 recommends **(a) as the research programme and (b) as the shipping programme**, run in that order —
they share 100 % of the architecture and ~90 % of the code.

### 3.4 Consequence 3 — dense-first, then upcycle

| Evidence | Says |
|---|---|
| Sparse upcycling 2212.05055 | Upcycling beats both from-scratch MoE and continued dense training **when the extra budget is small relative to the original**; from-scratch wins once the extra budget is large. |
| OLMoE 2409.02060 | Upcycled OLMo-1B (2 T) → 8 experts/top-2, +610 B tokens: **from-scratch matched or beat upcycling** at that (large) budget. Attributed to the dense checkpoint's parameters already sitting in a dense-optimal basin. |
| Nvidia 2410.07524 | Nemotron-4 15B upcycled, 1 T tokens: **67.6 % MMLU vs 65.3 %** for continued dense training. Virtual-group init enables fine-grained upcycling; softmax-then-topk > topk-then-softmax. |
| Llama-3 upcycling 2412.09952 | Efficient upcycling recipes for Llama-3-class models |
| **Drop-Upcycling 2502.19261 (ICLR'25)** | The key result. Naive upcycling (r=0) "struggles to improve beyond the pre-trained model"; full re-init (r=1) throws away the knowledge. **Partial re-initialisation with r=0.5 is best at both 8×152M and 8×1.5B** — precisely our scale — winning on training loss *and* task average, and fixing the convergence slowdown that made upcycling lose in the long run. Their 5.9 B-active MoE matched a 13 B dense model in the same family at **~1/4 the training FLOPs**. |

Our budget is unambiguously on the **upcycling side** of the 2212.05055 crossover. And dense-first has
four independent single-GPU advantages that have nothing to do with quality:

1. **Throughput**: 50.9 M vs 33.9 M tokens/A100-hour → the dense phase is **1.5× cheaper per token**.
2. **Memory**: a dense 1.1 B trunk trains at scheme A (16 B/param = 17.6 GB) — fp32 master weights, fp32
   Adam, huge micro-batches, zero exotic-optimizer risk.
3. **Router cold-start**: routers freeze early (OLMoE's saturation result). Upcycling hands the router a
   well-conditioned feature space at step 0 instead of asking it to co-evolve with random features
   during the only 10 B tokens we will ever have.
4. **Interruption safety**: Colab preempts. Phase 1 always ends with a shippable dense checkpoint
   (which *is* Prophet-mini's big sibling). From-scratch MoE has no such fallback.

### 3.5 What does *not* transfer

* **7× efficiency leverage.** 2507.17702 makes EL an explicit power law in the compute budget. Plan 2–3×.
* **Very high sparsity (1/32 like Ling, 1/27 like Qwen3-Next).** Those need trillions of tokens.
* **Expert-choice routing.** Non-causal; unusable for a decoder.
* **"Shared experts are useless" (OLMoE).** That result is at 5 T tokens where the router is fully
  converged and routing flexibility is the scarce resource. At 10 B tokens the scarce resource is
  *gradient signal per parameter*, and an always-on expert is a guaranteed-gradient path. Every 2025–26
  production MoE at high granularity (DeepSeek-V3, Ling 2.0, Qwen3-Next, Granite 4.0-h) reinstated the
  shared expert. **We keep one, and measure it (A2).**
* **Multi-GPU everything.** No expert parallelism, no device-level balancing loss, no all-to-all. This is
  actually a simplification: dropless single-device MoE has no capacity factor and no dropped tokens.

---

## 4. Recommendation for Prophet

### 4.1 The decision, stated flatly

**Do MoE. Fine-grained, token-choice, top-8, one shared expert, aux-loss-free bias balancing with a
global-batch auxiliary loss as a safety net and a router z-loss. Train dense first for ~40 % of the token
budget, then Drop-Upcycle (r≈0.5, validated by A7) and spend the rest as MoE. Size v1 at 5.1 B total /
1.07 B active — not 10 B — and grow to 9.8 B by expert cloning when a second compute tranche arrives.**

### 4.2 Prophet-M v1 — exact configuration

```
vocab                 64 000  (tied input/output embedding)
d_model               2048
n_layers              24        (layer 0 = dense SwiGLU FFN, d_ff 5632; layers 1..23 = MoE)
attention             GQA, 16 query heads / 4 kv heads, head_dim 128, RoPE, QK-RMSNorm
expert MLP            SwiGLU, d_expert = 512          (d_expert/d_model = 0.25, matching Ling 2.0 / Qwen3-Next)
routed experts        E = 64
top-k                 k = 8                            (routed activation ratio 12.5 %)
shared experts        1, width 512, always on
router                Linear(2048 -> 64), fp32, sigmoid scoring
selection             top-k over (sigmoid_score + bias_i); gate value = unbiased score, renormalised to sum 1
routed_scaling_factor 2.5
balancing             aux-loss-free bias, gamma = 1e-3 (-> 0 over the last 3 % of tokens)
                      + global-batch aux LBL, alpha = 1e-3
                      + router z-loss,        beta  = 1e-3
capacity factor       none (dropless)
```

**Parameter accounting**

| Component | Params |
|---|---|
| Embedding (tied) | 131.1 M |
| Attention, 24 layers | 251.7 M |
| Dense FFN (layer 0) | 34.6 M |
| Routed experts, 23 × 64 × 3 × 2048 × 512 | 4 630 M |
| Shared experts, 23 × 1 × 3 × 2048 × 512 | 72.4 M |
| Routers, 23 × 2048 × 64 | 3.0 M |
| **Total** | **5.123 B** |
| **Active / token** (incl. tied output head) | **1.072 B** (0.940 B non-embedding) |
| Activation ratio | 20.9 % of total; 12.5 % of routed experts |
| Total / active leverage | **4.8×** |

**Growth path to the 8–12 B target (v2):** clone each routed expert into two using Drop-Upcycling's
partial re-initialisation (r ≈ 0.5) → E=128, unchanged top-k, unchanged active params → **9.757 B total /
1.075 B active, 9.1× leverage.** This is a ~5 A100-hour surgery plus continued training, and it is
exactly the operation Phase 2 already implements. **Do it when the token budget exceeds ~40 B**, not before.

### 4.3 Training path

| Phase | Budget | What | Deliverable |
|---|---|---|---|
| **0 — Ablations** | 30 h | §7 suite at 90 M active / 3 B tokens | Locked architecture; A7 and A10 decide r and E |
| **1 — Dense trunk** | 90 h | Prophet-Dense-1.1B (d_model 2048, 24 L, d_ff 5632), scheme A optimizer, MFU ~45 % | **4.6 B tokens**; a shippable dense 1.1 B model |
| **2 — Upcycle** | 5 h | Split each FFN's 5632 columns into 11 groups of 512, replicate to 64 experts, apply Drop-Upcycling re-init at rate r; init router from expert-group centroids; 200 M-token router-only warmup at 10× router LR with the trunk frozen | Prophet-M v1 initial checkpoint |
| **3 — MoE main** | 150 h | Scheme B optimizer (8-bit AdamW + fp32 master), ScatterMoE kernels, global-batch LBL | **5.1 B tokens** |
| **4 — Anneal / distil** | 25 h | High-quality + instruction data, LR→0, `gamma`→0, optional logit distillation from a 4–8 B teacher, light 4-bit QAT on the experts | **0.85 B tokens**; final checkpoint |
| **Total** | **300 h** | | **~10.5 B tokens** |

If the goal "beat Qwen3-1.7B" is hard-binding, replace Phase 1 with **Phase 1′: adopt Qwen3-1.7B-Base as
the trunk** (Apache-2.0; 28 L, d_model 2048, d_ff 6144, vocab 151 936) and redirect its 90 hours into
Phase 3. Note the consequence: preserving Qwen's 6144-wide FFN at top-8 requires d_expert = 768, which
puts the active count at **~1.85 B, not 1.07 B** (still only 1.85 GB/token at FP8 → 970 tok/s ceiling on a
5090). Shrinking below that discards FFN capacity the base already paid 36 T tokens for, and the model
will be *worse* than its own base for a long time. **Do not upcycle a strong base into a narrower
active-FFN.**

### 4.4 Prophet-mini — dense, for the iPhone

```
d_model 1280, 28 layers, d_ff 3456 (SwiGLU), GQA 10 q / 2 kv heads, head_dim 128, vocab 64 000 tied
=> 564 M params.  Dense. No MoE. No routing.
```

| | int4 | int8 |
|---|---|---|
| Weights | 0.30 GB | 0.56 GB |
| KV @ 8 K ctx (fp16) | 0.24 GB | 0.24 GB |
| Total working set | **~0.6 GB** | ~0.9 GB |
| iPhone ceiling @60 GB/s | **201 tok/s** | 106 tok/s |

Rationale (§2.6): on an 8 GB phone the *total*-parameter footprint is the binding constraint, not
bandwidth — a 5.1 B MoE at int4 is 2.7 GB of weights, which is technically under the entitlement cap but
leaves no room for the app, and cannot use the ANE at all. A dense 564 M model uses the ANE, fits with
6× margin, and is 2× faster than the MoE would be even if the MoE could run. **The mini is where sparsity
loses.** Chunk-level MoE (BlockFFN-style) is the v2 research item if capacity becomes the limiter.

### 4.5 Memory budgets — proof it fits

**Training, one A100 80 GB, Phase 3, micro-batch 8 × 4096 = 32 768 tokens, full activation checkpointing:**

| Item | Bytes/param | GB |
|---|---|---|
| bf16 weights | 2 | 10.2 |
| fp32 master weights | 4 | 20.5 |
| bf16 gradients | 2 | 10.2 |
| 8-bit AdamW m, v (blockwise, 2110.02861) | 2 | 10.2 |
| **Optimizer subtotal** | **10** | **51.2** |
| Stored layer inputs (24 × 32 768 × 2048 × 2 B) | | 3.2 |
| Recompute peak + expert scatter buffers | | ~2.0 |
| Fused cross-entropy (Liger / CCE) instead of a 4.19 GB logits tensor | | ~0.1 |
| CUDA context, cuBLAS/Triton workspaces, allocator fragmentation | | ~5.0 |
| **Total** | | **≈ 61.5 / 80 GB** ✓ (18 GB headroom) |

*If Colab hands you a 40 GB A100:* switch to scheme F (offload the 8-bit states) → 20.5 GB optimizer +
10 GB other = ~31 GB ✓, at ~15 % step overhead. The run survives.
*Compare v2 (9.76 B):* scheme B would need 97.6 GB ✗. Only scheme D/E fits (58.5 GB, 11 GB headroom) —
which is why v2 waits.

**Inference, RTX 5090 32 GB (KV in FP8, 128 K context = 3.1 GB):**

| Precision | Weights | KV @128 K | Workspace | Total | Fits 32 GB |
|---|---|---|---|---|---|
| Prophet-M v1 bf16 | 10.2 GB | 3.1 | ~2 | **15.3 GB** | ✓ |
| Prophet-M v1 FP8 | 5.1 GB | 3.1 | ~2 | **10.2 GB** | ✓✓ |
| Prophet-M v1 MXFP4 (experts) + bf16 (attn/emb/router) | 2.9 GB | 3.1 | ~2 | **8.0 GB** | ✓✓✓ (fits a 12 GB card) |
| Prophet-M **v2** bf16 | 19.5 GB | 3.1 | ~2 | **24.6 GB** | ✓ |
| Prophet-M **v2** MXFP4 | 5.3 GB | 3.1 | ~2 | **10.4 GB** | ✓✓ |

**No expert offloading is ever required on any target.** That is a deliberate design constraint, and §2.5
is the evidence for why it is worth ~10× in latency.

**Mac Studio:** capacity irrelevant; v1 FP8 decodes at a 764 tok/s ceiling on an M3 Ultra (≈250–400 tok/s
realistic through MLX/llama.cpp).

### 4.6 PyTorch sketch — MoE block + router

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ProphetRouter(nn.Module):
    """Sigmoid-scored top-k router.
    Balancing = aux-loss-free additive bias (DeepSeek-V3, 2412.19437)
              + global-batch auxiliary LBL   (2501.11873)
              + router z-loss                (ST-MoE, 2202.08906).
    The bias affects SELECTION only, never the gate value -- that is the whole
    point: no gradient interference with the LM objective.
    """
    def __init__(self, d_model, n_routed, top_k, bias_lr=1e-3, accum_steps=16):
        super().__init__()
        self.n_routed, self.top_k, self.bias_lr = n_routed, top_k, bias_lr
        self.weight = nn.Parameter(torch.empty(n_routed, d_model))
        nn.init.normal_(self.weight, std=d_model ** -0.5)
        # NOT a Parameter: updated by a sign rule, never by the optimizer.
        self.register_buffer("bias", torch.zeros(n_routed))
        # ring buffer of per-micro-batch expert counts -> global-batch LBL under grad accumulation
        self.register_buffer("count_ring", torch.zeros(accum_steps, n_routed))
        self.register_buffer("ring_ptr", torch.zeros((), dtype=torch.long))

    def forward(self, x):                                   # x: [T, d_model]
        logits = F.linear(x.float(), self.weight.float())   # router always fp32
        scores = torch.sigmoid(logits)                      # [T, E]
        _, idx = torch.topk(scores + self.bias, self.top_k, dim=-1)   # selection uses the bias
        gates = scores.gather(-1, idx)                               # gate value does NOT
        gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-9)

        z_loss = logits.logsumexp(dim=-1).pow(2).mean()

        with torch.no_grad():
            counts = torch.bincount(idx.reshape(-1), minlength=self.n_routed).float()
            self.count_ring[self.ring_ptr] = counts
            self.ring_ptr.copy_((self.ring_ptr + 1) % self.count_ring.shape[0])
            f = self.count_ring.sum(0)
            f = f / f.sum().clamp_min(1.0)                   # GLOBAL-batch load fraction
        p = scores.mean(0)
        p = p / p.sum().clamp_min(1e-9)                      # this micro-batch's gate mass
        lb_loss = self.n_routed * (f * p).sum()              # gradient flows through p only

        return idx, gates.to(x.dtype), lb_loss, z_loss, counts

    @torch.no_grad()
    def step_bias(self, counts, gamma=None):
        """Call ONCE per optimizer step with the summed counts of the global batch.
        Decay gamma -> 0 over the final ~3% of training (DeepSeek-V3)."""
        g = self.bias_lr if gamma is None else gamma
        self.bias += g * torch.sign(counts.mean() - counts)


def grouped_moe(x, idx, gates, w13, w2):
    """Reference (correct, slow) grouped-expert forward.
    Replace the python loop with scattermoe.parallel_experts.ParallelExperts
    (2403.08245) or megablocks dMoE (2211.15841) -- both fuse the gather/scatter
    into the GEMM and avoid materialising the k copies of x below.
    NOTE: torch._grouped_mm needs sm90+; Colab's A100 is sm80 -> use Triton.
    """
    T, D = x.shape
    E, _, two_de = w13.shape                      # w13: [E, D, 2*d_expert] (gate|up fused)
    k = idx.shape[1]
    flat = idx.reshape(-1)                        # [T*k]
    order = torch.argsort(flat)                   # group (token, slot) pairs by expert
    tok = order // k                              # source token of each pair
    seg = torch.bincount(flat, minlength=E)
    xs = x[tok]                                   # [T*k, D]
    out = torch.empty_like(xs)
    start = 0
    for e, n in enumerate(seg.tolist()):
        if n == 0:
            continue
        h = xs[start:start + n] @ w13[e]
        g, u = h.chunk(2, dim=-1)
        out[start:start + n] = (F.silu(g) * u) @ w2[e]
        start += n
    out = out * gates.reshape(-1)[order].unsqueeze(-1)
    y = torch.zeros(T, D, device=x.device, dtype=out.dtype)
    return y.index_add_(0, tok, out)


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w13 = nn.Linear(d, 2 * h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
    def forward(self, x):
        g, u = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


class ProphetMoE(nn.Module):
    """Prophet-M v1 MoE block: 64 routed experts of width 512, top-8, 1 shared."""
    def __init__(self, d_model=2048, d_expert=512, n_routed=64, top_k=8,
                 n_shared=1, scale=2.5, lb_coef=1e-3, z_coef=1e-3):
        super().__init__()
        self.scale, self.lb_coef, self.z_coef = scale, lb_coef, z_coef
        self.w13 = nn.Parameter(torch.empty(n_routed, d_model, 2 * d_expert))
        self.w2 = nn.Parameter(torch.empty(n_routed, d_expert, d_model))
        for w, fan_in in ((self.w13, d_model), (self.w2, d_expert)):
            nn.init.normal_(w, std=fan_in ** -0.5)
        self.router = ProphetRouter(d_model, n_routed, top_k)
        self.shared = SwiGLU(d_model, n_shared * d_expert) if n_shared else None
        self.aux_loss = None          # picked up by the training loop each step
        self.last_counts = None       # fed to router.step_bias() once per optimizer step

    def forward(self, x):                                  # x: [B, S, d_model]
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        idx, gates, lb, z, counts = self.router(xf)
        y = grouped_moe(xf, idx, gates, self.w13, self.w2) * self.scale
        if self.shared is not None:
            y = y + self.shared(xf)
        self.aux_loss = self.lb_coef * lb + self.z_coef * z
        self.last_counts = counts
        return y.view(B, S, D)
```

Drop-Upcycling initialisation (Phase 2), in words: for each dense FFN of width 5632, partition the
intermediate dimension into groups of 512; for expert *e*, copy group *(e mod 11)*'s columns of
`w_gate/w_up` and the matching rows of `w_down`; then **re-initialise a random fraction r of the
intermediate indices** (same indices across gate/up/down so the expert stays coherent) from the empirical
distribution of the original weights. r=0 is function-preserving and stalls; r=1 discards knowledge;
r≈0.5 is the ICLR'25 optimum — confirm with ablation A7 at our token budget, where a smaller r (0.25) may
win because we have far less recovery data than they did.

---

## 5. Compute & memory budget

### 5.1 Compute

| Quantity | Value |
|---|---|
| A100 80 GB bf16 peak | 312 TFLOP/s |
| Assumed MFU, dense / MoE | 45 % / 30 % |
| FLOPs per token (0.94 B active non-emb, seq 4096, full recompute) | 9.94e9 |
| Throughput, dense / MoE | 14 131 / 9 420 tok/s |
| Tokens per A100-hour, dense / MoE | 50.9 M / 33.9 M |
| **Total tokens at 300 A100-hours (per §4.3 split)** | **10.5 B** |
| Total tokens at 100 A100-hours | 3.5 B |
| Total training FLOPs at 300 h | 1.0e20 |
| Qwen3-1.7B training FLOPs (6·1.7e9·36e12) | 3.7e23 (**3700×**) |
| A100-hours to match Qwen3-1.7B from scratch with an optimal MoE | **≈140 000 (466×)** |

### 5.2 Training memory (Phase 3, scheme B)

Reproduced from §4.5: **61.5 GB of 80 GB**, 18 GB headroom. Sensitivity:

| Change | Δ memory |
|---|---|
| E 64 → 128 (v2) | +46.4 GB ✗ under scheme B; needs scheme D/E |
| micro-batch 32 768 → 65 536 tokens | +3.5 GB |
| drop fp32 master (scheme D, stochastic rounding) | −20.5 GB (and −stability) |
| offload 8-bit states to host (scheme F) | −10.2 GB GPU, +10.2 GB host, +10–15 % step time |
| forget fused cross-entropy | **+4.2 GB** (bf16 logits) **+8.4 GB** more if an fp32 copy is made |

### 5.3 Inference memory & speed summary

| Target | Config | Resident | Ceiling tok/s | Realistic tok/s |
|---|---|---|---|---|
| RTX 5090 32 GB | v1 MXFP4 + FP8 KV 128 K | 8.0 GB | 3155 | 850 (llama.cpp-class) – 1420 (tuned) |
| RTX 5090 32 GB | v1 FP8 | 10.2 GB | 1672 | 450 – 750 |
| RTX 5090 32 GB | **v2** MXFP4 | 10.4 GB | 3126 | 840 – 1400 |
| Mac Studio M3 Ultra | v1 FP8 | 10.2 GB | 764 | 250 – 400 |
| Mac Studio M4 Max | v1 MXFP4 | 8.0 GB | 961 | 300 – 480 |
| iPhone 17 Pro | **mini, dense 564 M, int4** | 0.6 GB | 201 | 80 – 130 (ANE) |

### 5.4 Sensitivity: the active-parameter knob

Active params set both training tokens and decode speed. At a fixed 300 A100-hours:

| Active (non-emb) | tokens affordable | 5090 MXFP4 ceiling | comment |
|---|---|---|---|
| 0.55 B | 17.4 B | 5100 tok/s | more data, weaker per-token model |
| **0.94 B (recommended)** | **10.5 B** | **3155 tok/s** | ≈ compute-optimal N for 1e20 FLOPs (Chinchilla N* ≈ 0.92 B) |
| 1.85 B (Qwen3-1.7B upcycle) | 5.6 B | 1600 tok/s | only sane if the trunk is inherited |

Note that the recommended point is **already Chinchilla-compute-optimal for our budget**
(N* = √(C/120) = 0.92 B). This is the deepest argument for MoE at our scale: with a dense model,
"compute-optimal" and "enough capacity to be useful" are the same number, and 0.92 B is not enough
capacity. MoE decouples them — we stay compute-optimal at 0.94 B *active* while carrying 5.1 B of
parameters. That decoupling, not the raw efficiency leverage, is the real prize.

---

## 6. Risks & failure modes

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **The headline goal (beat Qwen3-1.7B) is unreachable from scratch** (§3.3) | Certain | Project-defining | Decide now between goal-reset, trunk inheritance, or more compute. Do not start Phase 1 before this is settled. |
| R2 | 8–12 B total is over-parameterised for 10 B tokens; experts undertrained, quantise badly, add nothing | High | Wasted VRAM + quality | Ship v1 at 5.1 B; run **A10** to measure the marginal value of total params at our exact token budget; expand later by cloning |
| R3 | Colab preemption; 20 GB checkpoints (bf16 weights + 8-bit states) are slow to write to Drive | Certain | Lost hours | Checkpoint to local NVMe every 20 min, async-upload every 2 h; keep weights-only fast checkpoints between full ones; make the router bias/count buffers part of the checkpoint |
| R4 | `torch._grouped_mm` is sm90+; the A100 is sm80 → the modern PyTorch MoE path silently doesn't apply | Certain | 3–10× slowdown | Pin ScatterMoE (2403.08245) or MegaBlocks (2211.15841); benchmark on day 1; verify the fallback isn't a python loop |
| R5 | Dead / collapsed experts, permanently (routers freeze early — OLMoE saturation) | Medium | Capacity loss | Aux-loss-free bias with γ=1e-3 from step 0; log per-expert load CV and max-violation every 100 steps; hard-fail the run if any expert's share < 0.2× uniform for 2 000 steps; router warmup at 10× LR in Phase 2 |
| R6 | bf16 instability if scheme D is used for v2 (no fp32 master) | Medium | Divergence | Stay on scheme B for v1. If v2 needs scheme D, use Kahan compensation or Muon (2502.16982) and keep the router in fp32 |
| R7 | Fused-CE forgotten → 4–10 GB logits tensor → OOM | Medium | Lost hours | Liger-Kernel (2410.10989) or CCE (2411.09009) wired in from the first commit |
| R8 | Experts quantise worse than dense weights (fewer tokens/expert, per-expert outliers) | Medium-High | −2–4 pts at 4 bits | Keep router + attention + embeddings ≥8 bit; MXFP4 only on expert weights; 4-bit QAT during Phase 4; evaluate at the deployment precision, never at bf16 only |
| R9 | Realised decode is 27 % of ceiling, not 45 % | High | 1.6× slower than promised | Treat the fused-MoE kernel + CUDA graphs as a first-class deliverable; benchmark against gpt-oss-20b and Qwen3-30B-A3B on the same box |
| R10 | KV traffic (48 KB/token) dominates weight traffic past ~24 K context, erasing the MoE advantage | High at long ctx | Speed claim collapses | GQA 4 kv-heads is the floor; coordinate with the attention track on MLA / sliding-window hybrid; report tok/s at 4 K **and** 32 K |
| R11 | Upcycling at r=0.5 stalls because we only have 5 B recovery tokens (Drop-Upcycling used ≫) | Medium | Phase 3 wasted | **A7** sweeps r ∈ {0, 0.25, 0.5} at *our* token ratio before committing; r=0 is function-preserving so the downside is bounded |
| R12 | Micro-batch LBL forces local uniformity, kills specialisation (2501.11873) | Medium | Silent quality loss | Global-batch LBL via the count ring buffer (implemented in §4.6); verify with a per-domain expert-usage heatmap |
| R13 | Benchmark trap: good perplexity, bad MMLU. Knowledge benchmarks need data, not parameters | High | Bad decisions | Track MMLU/ARC/TriviaQA separately from perplexity; never justify more experts with a perplexity delta alone |
| R14 | iPhone: no ANE path for token-routed MoE | Certain | mini would run 3–5× slower on GPU | mini ships dense (§4.4); chunk-MoE (BlockFFN 2507.08771) is v2 research only |

---

## 7. Ablation plan

Shared proxy (each run ≤ 6 A100-hours):

```
d_model 768, 12 layers, GQA 12q/4kv head_dim 64, vocab 32 000 tied, seq 2048
dense reference : d_ff 2048                        -> ~90 M active non-emb params
MoE reference   : d_expert 192, E 64, top-k 8, 1 shared  -> ~365 M total / ~90 M active
data            : 3 B tokens from the project's pretraining mix (fixed seed, fixed order)
metrics         : val loss (held-out, 4 domains) + HellaSwag + ARC-easy + PIQA + LAMBADA
                  + expert load CV, max-violation, routing entropy, per-domain expert heatmap
cost            : 6*90e6*3e9 + attention ~= 1.9e18 FLOP @35% MFU  ->  ~4.0 A100-hours
```

Run in this order; **A7 and A10 gate the whole programme and must finish first.**

| # | Question | Arms | Decides | Cost |
|---|---|---|---|---|
| **A7** | **Training path** — from-scratch MoE vs dense-then-upcycle at *matched total FLOPs* | (a) MoE 3 B tok from scratch; (b) dense 1.5 B tok → upcycle r=0 → MoE 1.5 B tok; (c) same with r=0.25; (d) r=0.5 | §4.3 Phase 1/2 split and r | 4 runs, 16 h |
| **A10** | **How much total capacity does 3 B tokens actually pay for?** | total/active ∈ {2, 4, 8, 16} at fixed active (E ∈ {16,32,64,128} with d_expert scaled inversely at fixed k·d_expert) | v1 expert count E; validates or kills the 8–12 B target | 4 runs, 16 h |
| **A1** | Granularity at iso-active | (E,d_e,k) = (16,768,2), (32,384,4), (64,192,8), (128,96,16) | d_expert / k. Expect OLMoE's pattern: big jump to 32, +1–2 % to 64, then flat or worse | 4 runs, 16 h |
| **A2** | Shared expert | n_shared ∈ {0, 1} at E=64,k=8 | Whether to keep the shared expert. OLMoE says no at 5 T tokens; hypothesis is it flips at 3 B | 2 runs, 8 h |
| **A3** | Balancing scheme | aux α=0.01 / aux α=0.001 / aux-free bias γ=1e-3 / bias + aux 1e-3 | §4.2 balancing block | 4 runs, 16 h |
| **A4** | LBL scope | micro-batch vs global-batch (16-step ring) at α=1e-3 | Validates 2501.11873 at our batch size | 2 runs, 8 h |
| **A5** | Router z-loss | β ∈ {0, 1e-3, 1e-2} | β; count bf16 loss spikes and max router-logit magnitude | 3 runs, 12 h |
| **A6** | Gate function | softmax→topk→renorm vs sigmoid→topk→renorm; scaling factor ∈ {1.0, 2.5} | Router head | 4 runs, 16 h |
| **A8** | Dense-first layers | first_k_dense_replace ∈ {0, 1, 2} | Whether layer 0 stays dense | 3 runs, 12 h |
| **A9** | Expert activation | SwiGLU vs ReLU² | Free activation sparsity for a future Deja-Vu-style loader | 2 runs, 8 h |
| **A11** | Memory layer add-on | +1 product-key memory layer (256 K keys, top-32) at mid-depth vs +equivalent expert params | Whether 2412.09764 beats more experts for factual recall at our budget | 2 runs, 8 h |
| **A12** | MoD for *training* throughput | MoD every-other-block, capacity 12.5 %, vs baseline, at iso-wallclock | Whether to buy ~1.3× more tokens per A100-hour | 2 runs, 8 h |

**Minimum viable suite (gates the design): A7 + A10 + A2 + A3 = 14 runs ≈ 56 A100-hours.** That is 19 % of
a 300-hour budget and it is the highest-return 19 % available — A10 alone determines whether the model is
5 B or 12 B, and A7 determines whether 90 hours go into a dense trunk or into MoE. **Reduce to A7 + A10
(8 runs, 32 h) if the budget is 100 hours.**

Pre-registered kill criteria:
* If A10 shows <0.02 nats improvement from total/active 4→8 at 3 B tokens, **freeze v1 at E=64 and do not
  build v2** until the token budget grows.
* If A7's best upcycled arm does not beat the from-scratch arm by ≥0.015 nats, **drop the dense-first
  phase** and train MoE from scratch for the whole budget (saves the Phase 2 engineering).
* If the MoE proxy does not beat an *iso-FLOP dense* proxy by ≥0.05 nats at 3 B tokens, **MoE is the wrong
  call at this budget** — ship a dense 1.1 B and put the compute into data quality instead. (Add this
  dense iso-FLOP control run to every table; it is the honest baseline and it is cheap.)

---

## 8. References

Network note: `arxiv.org` and most paper mirrors were unreachable from this environment. Configuration
numbers marked † in §2.1 were verified against live repository files (`raw.githubusercontent.com`,
`github.com`); paper results were extracted via web search over paper pages. arXiv IDs marked ‡ are from
memory and were not re-verified — check them before citing externally.

**MoE architecture**
- GShard — arXiv 2006.16668
- Switch Transformer — arXiv 2101.03961
- ST-MoE: Designing Stable and Transferable Sparse Expert Models (router z-loss) — arXiv 2202.08906
- Mixture-of-Experts with Expert Choice Routing — arXiv 2202.09368
- Mixtral of Experts — arXiv 2401.04088
- **DeepSeekMoE: Towards Ultimate Expert Specialization** — arXiv 2401.06066
- DeepSeek-V2 (MLA + DeepSeekMoE) — arXiv 2405.04434
- **DeepSeek-V3 Technical Report** (aux-loss-free balancing, γ=1e-3, α=1e-4) — arXiv 2412.19437
- **OLMoE: Open Mixture-of-Experts Language Models** — arXiv 2409.02060 (config verified: 64 experts, top-8, d_model 2048, 16 layers, dropless, `moe_loss_weight=0.01`, `moe_zloss_weight=0.001`, lr 4e-4, global batch 1024×4096, 5 T tokens)
- JetMoE: Reaching Llama2 Performance with 0.1M Dollars — arXiv 2404.07413
- GRIN: GRadient-INformed MoE — arXiv 2409.12136 ‡
- Qwen3 Technical Report — arXiv 2505.09388 (Qwen3MoeConfig defaults verified: 128 experts, top-8, `moe_intermediate_size=768`, `router_aux_loss_coef=0.001`, `norm_topk_prob=False`)
- gpt-oss model card — arXiv 2508.10925 (21 B/3.6 B and 117 B/5.1 B, MXFP4 MoE weights, 16 GB / 80 GB verified from `openai/gpt-oss`)
- Ling 2.0 / Ling-mini-2.0 — `github.com/inclusionAI/Ling-V2` (16.26 B total, 1.43 B active, 1/32 activation, 20 T tokens, aux-loss-free sigmoid routing, FP8 training, 300+ tok/s on H20 — verified)
- IBM Granite 4.0 language models — `github.com/ibm-granite/granite-4.0-language-models` (`GraniteMoeHybridConfig` defaults verified: `router_aux_loss_coef=0.001`, `shared_intermediate_size`)
- ReMoE: Fully Differentiable MoE with ReLU Routing — arXiv 2412.14711

**Scaling laws & MoE-vs-dense**
- Unified Scaling Laws for Routed Language Models — arXiv 2202.01169
- Scaling Laws for Fine-Grained Mixture of Experts — arXiv 2402.07871 ‡
- **Towards Greater Leverage: Scaling Laws for Efficient MoE LMs** — arXiv 2507.17702 (300+ models to 28 B; Efficiency Leverage is a power law in activation ratio and compute; granularity has a non-linear optimum; Ling-mini-beta 0.85 B active ≡ 6.1 B dense on 1 T tokens at 7× less compute)
- **MoE Can Surpass Dense LLMs Under Strictly Equal Resource** — arXiv 2506.12119 (ICLR'26; 250+ MoE runs at 2 B & 7 B; optimal activation rate ≈20 % at 7 B)
- The power of fine-grained experts: granularity boosts expressivity in MoE — arXiv 2505.06839

**Upcycling**
- **Sparse Upcycling: Training MoE from Dense Checkpoints** — arXiv 2212.05055
- **Upcycling LLMs into Mixture of Experts** (Nvidia; virtual groups, softmax-then-topk; Nemotron-4 15B 67.6 % vs 65.3 % MMLU at 1 T tokens) — arXiv 2410.07524
- Llama 3 Meets MoE: Efficient Upcycling — arXiv 2412.09952
- **Drop-Upcycling: Training Sparse MoE with Partial Re-initialization** — arXiv 2502.19261 (ICLR'25; r=0.5 optimal at 8×152M and 8×1.5B; 5.9 B-active MoE ≈ 13 B dense at ~1/4 the FLOPs; `github.com/Taishi-N324/Drop-Upcycling`)

**Load balancing**
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized MoE Models — arXiv 2501.11873 (global-batch LBL ≫ micro-batch LBL)

**Alternative sparsity**
- Large Memory Layers with Product Keys — arXiv 1907.05242
- **Mixture of A Million Experts (PEER)** — arXiv 2407.04153 (C4 ppl 20.63 vs MoE 21.41 vs PKM 21.92 at 6e18 FLOPs)
- **Memory Layers at Scale** — arXiv 2412.09764 (beats dense at >2× compute and MoE at matched compute+params on factual tasks; to 128 B memory params / 1 T tokens)
- Monet: Mixture of Monosemantic Experts — arXiv 2412.04139 ‡
- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time — arXiv 2310.17157
- ProSparse — arXiv 2402.13516
- **Q-Sparse: All LLMs can be Fully Sparsely-Activated** — arXiv 2407.10969 (>40 % sparsity on q/k/v/o/up/down, >60 % on gate; 300 M–7 B scaling laws)
- **Mixture-of-Depths** — arXiv 2404.02258 (12.5 % capacity every other block; up to 1.5 % better isoFLOP ppl; 220 M model 66 % faster per step)
- BlockFFN: Chunk-Level Activation Sparsity for End-Side Acceleration — arXiv 2507.08771
- CoSMoEs: Compact Sparse Mixture of Experts (on-device) — arXiv 2503.00245
- MobileMoE: Scaling On-Device Mixture of Experts (2026) ‡

**Inference systems**
- Fast Inference of MoE Language Models with Offloading (Mixtral-offloading) — arXiv 2312.17238
- **Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models** — arXiv 2402.07033, ICLR'25 (>3 tok/s unquantised Mixtral-8x7B on one 24 GB GPU; 8.2–10.1× vs Mixtral-offloading, 19.4–22.5× vs DeepSpeed-MII — verified from `github.com/efeslab/fiddler`)
- MoE-Infinity: activation-aware expert offloading — arXiv 2401.14361 ‡
- PowerInfer — arXiv 2312.12456 (OPT-30B, RTX 4090, 11.69 tok/s, 11.7× llama.cpp)
- llama.cpp `--n-cpu-moe` / `-ot` — `github.com/ggml-org/llama.cpp` discussion #21112 (measured: 120 B MoE Q3_K_XL on RTX 3090 ≈16 tok/s; 35B-A3B IQ4_XS on 8 GB laptop dGPU ≈27 tok/s; 122 B FP8 with `--cpu-moe` 7.1 tok/s at 5.6 GB VRAM)

**Training systems**
- MegaBlocks: Efficient Sparse Training with MoE — arXiv 2211.15841
- **ScatterMoE: Scattered Mixture-of-Experts Implementation** — arXiv 2403.08245 (`github.com/shawntan/scattermoe`, Triton, ~700 LoC, no padding/copies)
- 8-bit Optimizers via Block-wise Quantization — arXiv 2110.02861
- Adam-mini — arXiv 2406.16793
- **Muon is Scalable for LLM Training** (Moonlight 16B-A3B) — arXiv 2502.16982
- Liger-Kernel (fused linear cross-entropy) — arXiv 2410.10989
- Cut Your Losses in Large-Vocabulary Language Models — arXiv 2411.09009
- Training Compute-Optimal Large Language Models (Chinchilla) — arXiv 2203.15556
