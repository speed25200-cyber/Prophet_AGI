# R01 — Tokenization: killing the tokenizer bottleneck without killing our compute budget

**Track:** R01 (Tokenization / input representation)
**Status:** Decision-ready
**Date:** 2026-09-03
**Bottom line:** Do **not** build a from-scratch tokenizer-free Prophet. Build **Prophet-Tok v1** — a purpose-engineered 32,768-entry byte-fallback BPE that repairs every tokenizer pathology that shows up in our target benchmarks — and keep the byte/patch frontend as a *late-stage retrofit* (Bolmo-style byteification) behind a hard quantitative gate. Rationale, numbers and the full spec for both are below.

---

## 1. Problem statement

Tokenization is the last hand-designed, non-learned, non-differentiable stage in an otherwise end-to-end system. It is fixed before the first gradient step, cannot be changed afterwards without discarding the model, and it silently imposes an information bottleneck. Concretely, seven distinct failure modes, ordered by how much they cost *us*:

### 1.1 The embedding/head parameter tax is brutal at our scale (this is our #1 issue)

A vocabulary costs `V × d` parameters for the input embedding and another `V × d` for the output head (unless tied). At frontier scale this is rounding error. At 0.3–1B parameters it dominates:

| Model | d_model | Vocab | Embedding params | Share of total |
|---|---|---|---|---|
| Gemma-3-270M | 640 | 262,144 | 167.8M | **~62%** of 270M |
| Gemma-3-1B | 1152 | 262,144 | 302M | **30%** of 1.0B (reported: 302M emb / 698M non-emb) |
| Llama-3.2-1B | 2048 | 128,256 | 262.7M (tied) | ~21% |
| Qwen3-1.7B | 2048 | 151,936 | 311.2M (tied) | ~18% |
| Prophet-mini candidate (d=768, L=24) | 768 | 262,144 | 201.3M | **54%** |
| same, V=32,768 | 768 | 32,768 | 25.2M | **13%** |

Vocabulary-reduction experiments confirm this is real and recoverable: Gemma-3-270M shrinks from 270M → 141M parameters by vocabulary trimming alone (kaitchup, 2025). For an 8GB iPhone target, spending 50–60% of the weight budget on a lookup table that is mostly dead weight for English+code is indefensible.

### 1.2 The output softmax is a first-class FLOP and memory item at small scale

Computed for a Prophet-mini-class body (d=768, L=24, 170M body params → 340 MFLOP/token forward):

| Vocab | Head MFLOP/token | Share of forward FLOPs |
|---|---|---|
| 256 (byte) | 0.4 | 0.1% |
| 16,384 | 25.2 | 6.9% |
| 32,768 | 50.3 | 12.9% |
| 65,536 | 100.7 | 22.9% |
| 128,256 | 197.0 | **36.7%** |
| 262,144 | 402.7 | **54.2%** |

And the logit activation is frequently the single largest tensor in a small-model training step. For one micro-batch of 32,768 positions (e.g. B=8 × S=4096) in fp32:

| Vocab | Logits | + softmax grad |
|---|---|---|
| 256 | 0.03 GB | 0.07 GB |
| 32,768 | 4.29 GB | 8.59 GB |
| 128,256 | 16.81 GB | **33.6 GB** |
| 262,144 | 34.36 GB | **68.7 GB** |

On a single 80GB A100 a 128k-vocab head forces either chunked cross-entropy kernels or a 4× smaller micro-batch. A 32k vocab is comfortable; a byte vocab is free.

### 1.3 Digit segmentation destroys arithmetic

Llama-3 and modern GPT tokenizers emit 1-, 2- and 3-digit tokens, and chunk **left to right**, which means `1234567` becomes `123|456|7` — the place values are misaligned with the arithmetic algorithm. Single-digit tokenization (PaLM, early Llama, Qwen, Gemma) is consistently better for arithmetic in controlled comparisons. Reformatting integers right-to-left raised GPT-3.5 arithmetic accuracy from **75.6% → 97.8%** and GPT-4 from **84.4% → 98.9%** (beren.io, 2024). GSM8K and MATH are on our target list; this is free points.

### 1.4 Multilingual fertility ("token tax")

Subword tokenizers trained on English-dominant corpora produce up to **10–15×** more tokens per unit of meaning for some languages (Petrov et al., arXiv:2305.15425). Since attention is quadratic in sequence length, a 2× fertility increase is ~4× the cost of a long-context step. Fertility (tokens/word) is a reliable *negative* predictor of downstream accuracy across models and subjects (arXiv:2509.05486). Llama 4 cut its token premium ~25% versus Llama 3.1 by moving 128k → ~200k vocab. *For Prophet this is largely out of scope* — at 8–20B training tokens we cannot be a serious multilingual model — but it matters for a "next-generation" claim.

### 1.5 Code and whitespace waste

Indentation is the highest-frequency structure in Python and it is exactly what naive pre-tokenizers fragment. SuperBPE (arXiv:2503.13423) showed the opposite lever: allowing merges *across* whitespace cuts sequence length by up to **33%** at fixed 200k vocab, giving **+4.0%** average over 30 downstream tasks (**+8.2% MMLU**) at **27% less inference compute**. That is the largest single "free lunch" in the tokenizer literature and it points *away* from byte-level.

### 1.6 Glitch / under-trained tokens

Tokenizer construction is decoupled from training, so vocabularies contain entries the model essentially never saw (`SolidGoldMagikarp`). "Fishing for Magikarp" (arXiv:2405.05417, EMNLP 2024) gives automated detectors (unembedding-norm outliers + prompted-repetition probes) and shows these are prevalent across all major open models. **At our data budget this gets worse, not better:** at ~10–20B training tokens, a 128k vocab has an enormous under-trained tail. This is a direct argument for a smaller vocabulary at low data budgets.

### 1.7 Character blindness and train/test mismatch

Models cannot reliably count characters, reverse strings, or spell, because the character identity is not in the input. CUTE-style benchmarks expose this starkly: OLMo-3-7B scores **56.9** on CUTE while its byteified sibling Bolmo-7B scores **78.6**. Separately, the prompt-boundary problem (a prompt ending mid-token puts the model off-distribution) is real but milder than feared — "Broken Tokens?" (arXiv:2506.19004) finds LMs are surprisingly robust to non-canonical tokenizations.

**Which of these actually move our target benchmarks?** MMLU/MMLU-Pro/GPQA/HellaSwag/ARC are essentially tokenizer-neutral (they are data-bound). GSM8K/MATH are strongly affected by §1.3. HumanEval/MBPP by §1.5. IFEval partially by §1.7. §1.1/§1.2 are pure budget wins. §1.4 and most of §1.7 are *not* measured by our suite. **This asymmetry is the crux of the recommendation in §4.**

---

## 2. State of the art

> **Sourcing note.** `arxiv.org`, `huggingface.co`, `aclanthology.org` and `openreview.net` were blocked by this session's network egress proxy. Primary sources I *could* read directly: GitHub (source code and configs — `huggingface/transformers` BLT implementation, `goombalab/hnet`, `rosinality/halite`, `openai/parameter-golf`). Everything else comes from search-engine full-text summaries. Numbers taken from secondary sources are marked **†**. All architecture/config numbers for BLT and H-Net below are read directly out of source code and are exact.

### 2.1 Tokenizer-free and dynamic-patching architectures

| # | Approach | arXiv / venue | Year | Core mechanism | Reported result |
|---|---|---|---|---|---|
| 1 | **CANINE** | 2103.06874, TACL | 2021 | char hashing + strided downsample, encoder-only | Beats mBERT on TyDi QA with 28% fewer params † |
| 2 | **ByT5** | 2105.13626, TACL | 2021 | raw bytes into T5, deep encoder / shallow decoder | Competitive at small scale; strongly noise-robust; **~+33% pretraining compute** and slower inference † |
| 3 | **Charformer (GBST)** | 2106.12672 | 2021 | learned gradient-based subword blocks | ~28–100% faster than byte baselines † |
| 4 | **Hourglass Transformer** | 2110.13711 | 2021 | fixed-rate hierarchical up/down-sampling | First to make hierarchical byte LM competitive † |
| 5 | **Perceiver AR** | 2202.07765 | 2022 | cross-attend long input to small latent set | Decouples input length from depth cost † |
| 6 | **Dynamic Token Pooling** | 2211.09761, ACL'23 | 2022 | 2-layer MLP boundary predictor over Hourglass; ~1M extra params; 12L (2+8+2), ~41M | Variable-rate pooling beats fixed-rate at equal cost † |
| 7 | **MEGABYTE** | 2305.07185, NeurIPS'23 | 2023 | fixed patch size (8), global (758M) + local (262M), ctx 8192 | arXiv bpb **0.678** vs Transformer 0.816, Perceiver-AR 0.791 † |
| 8 | **ByteFormer** | 2306.00238, TMLR | 2023 | transformers on raw file bytes | Modality-agnostic classification † |
| 9 | **MambaByte** | 2401.13660, COLM'24 | 2024 | Mamba SSM on bytes; fixed-size state | Competitive with / beats subword Transformers; **2.6×** decode speedup via tokenized-draft speculative decoding † |
| 10 | **bGPT** | 2402.19155 | 2024 | byte models as digital-world simulators | Native binary/multimodal † |
| 11 | **SpaceByte** | 2404.14408, NeurIPS'24 | 2024 | big blocks inserted **only after whitespace-like bytes**; ~6.3 B/patch | PG-19 bpb **1.009** vs MEGABYTE 1.083, byte-Transformer 1.138; matches SentencePiece subword baseline † |
| 12 | **T-FREE** | 2406.19223 | 2024 | words → sparse hashed character-trigram activations; no reference corpus | **>85%** reduction in embedding+head parameters at competitive quality † |
| 13 | **MrT5** | 2410.20771 | 2024 | learned token deletion inside ByT5 encoder | Up to ~50% seq-length reduction at similar quality † |
| 14 | **BLT (Byte Latent Transformer)** | 2412.09871, ACL'25 | 2024 | **entropy-model-driven** dynamic patching; local enc → global → local dec + hash n-gram embeddings | First FLOP-controlled byte scaling study to 8B/4T bytes; **up to 50% inference FLOP savings**; 8B/1T FLOP-matched vs Llama-3 BPE: ARC-E **79.6/77.6**, ARC-C **52.1/53.3**, HellaSwag **80.6/79.1**, PIQA **80.6/80.7**, MMLU **57.4/58.1** † |
| 15 | **EvaByte** | HKU NLP + SambaNova (blog + code) | 2025 | 6.5B byte LM, EVA linear-ish attention, **multibyte prediction n=8** | Rivals tokenizer-based LMs with **5× less data** (1.5T bytes); **~2×** faster decode † |
| 16 | **H-Net** | 2507.07955 | 2025 | **fully end-to-end learned dynamic chunking**, no external entropy model, vocab **256**, U-Net style, Mamba2 outer stages | DC naturally compresses to **4.5–5 bytes/chunk**; 1-stage byte H-Net > BPE Transformer at matched compute+data; **2-stage** overtakes tokenized Transformer perplexity after **30B bytes** and matches downstream evals of a **2× larger** tokenized Transformer; ~**4×** data efficiency on DNA † |
| 17 | **AU-Net** | 2506.14761 | 2025 | autoregressive U-Net: bytes → words → 2-word → 4-word pooling; splitting must be stable to rightward insertion | Matches/outperforms BPE baselines at identical pretraining budget with comparable GPU throughput † |
| 18 | **H-Net++** | 2508.05628 | 2025 | + Transformer mixer, 2-level latent hyper-prior | Gains concentrated in morphologically rich languages † |
| 19 | **FLEXITOKENS** | 2507.12720, Findings-ACL'26 | 2025 | learnable boundary predictor with a *flexible* (non-fixed-rate) objective | Up to **+10%** downstream vs subword and other gradient tokenizers; less over-fragmentation † |
| 20 | **Multiscale Byte LM** | 2502.14553 | 2025 | hierarchical byte LM to ~1M positions | Causal million-length modeling † |
| 21 | **Bolmo** | 2512.15586 (Ai2) | 2025 | **byteification retrofit** of OLMo-3: freeze backbone, train local mLSTM enc(1)/dec(4) + non-causal boundary predictor + LM head (**9.8B tok ≈ 43B bytes**), then end-to-end (**39.3B tok ≈ 173B bytes**) | **<1%** of a normal pretraining budget; Bolmo-7B ≈ OLMo-3-7B on broad evals, **CUTE 78.6 vs 56.9**, and **+16.5%** on STEM vs BLT-7B † |
| 22 | **FastBLT** | 2605.08044 (Meta+Stanford) | 2026 | BLT-Diffusion / self-speculation / diffusion+verification for **parallel byte decoding** | Up to **92%** memory-bandwidth reduction in some configs † |
| 23 | **Scratchpad Patching** | 2605.09630 (Google DeepMind) | 2026 | transient in-patch "scratchpads" triggered by next-byte entropy, to fix **patch lag** (byte predictions inside a patch use a stale patch representation) | Decouples compute allocation from patch size; post-hoc inference-compute tuning † |
| 24 | **ATDC** | 2605.30080 (Fujitsu) | 2026 | curriculum on the **compression ratio** (low → high) for H-Net-style DC | More stable training and better final BPB than fixed compression on FineWeb-Edu-100B † |
| 25 | **Kronecker Embeddings** | 2605.29459 | 2026 | structured/factorized byte-level token representations | Parameter-efficient embeddings † |

### 2.2 Tokenizer-side (non-byte) approaches — the "boring" competition

| Approach | arXiv | Year | Result |
|---|---|---|---|
| **Scaling Laws with Vocabulary** | 2407.13623, NeurIPS'24 | 2024 | 33M→3B models, ≤500B chars. Optimal V grows with compute; most LMs use too-small vocabularies. Llama2-70B's optimum ≥**216k** (vs 32k). 32k→43k gave ARC-C **29.1 → 32.0** at fixed 2.3e21 FLOPs † |
| **Over-Tokenized Transformer** | 2501.16975, ICML'25 | 2025 | Decouple input/output vocab; scale *input* n-gram vocab via tiled matrix factorization. **Log-linear** loss vs input-vocab size; large input vocab ≈ **2× larger baseline at no extra cost** † |
| **SuperBPE** | 2503.13423 | 2025 | superword merges: **−33%** tokens at V=200k, **+4.0%** avg / **+8.2%** MMLU, **−27%** inference compute † |
| **Length-MAX** | 2511.20849 | 2025 | graph-partition vocabulary objective: **14–18%** fewer tokens than BPE (10k–50k V); GPT-2 124M/355M/1.3B need **17–19%** fewer steps to a fixed val loss; **−18%** embedding+KV memory; **−11.7%** LAMBADA ppl † |
| **Compute Optimal Tokenization** | 2605.01188 (FAIR) | 2026 | **988** BLT models, 50M→7B, compression rates set by construction. Compute-optimal scaling is in **bytes per parameter, not tokens**; optimal compression rate ≠ BPE's, and **decreases with compute** — i.e. *at low compute you want MORE compression, not less* † |
| **Multi-token prediction** | 2404.19737 | 2024 | n independent heads on a shared trunk: 13B solves **+12% HumanEval / +17% MBPP**; **up to 3×** faster inference via self-speculation † |
| **Fishing for Magikarp** | 2405.05417, EMNLP'24 | 2024 | automated under-trained-token detection † |

### 2.3 BLT-1B, exactly as released (read from `huggingface/transformers/models/blt/configuration_blt.py`)

This is the most useful concrete reference point, and the numbers are *not* what the paper's framing suggests:

```
local encoder    :  1 layer,  d=1024, 16 heads, ffn=2730→2816, vocab 260, cross_attn_k=2
local decoder    :  9 layers, d=1024, 16 heads, ffn=2816, cross-attention at EVERY layer
global transformer: 25 layers, d=2048, 16 heads, ffn=5632, max_pos 4096
entropy "patcher":  14 layers, d=768, 12 heads, ffn=2048, vocab 260, max_pos 8192  (frozen, eval-only)
patching_mode = "entropy", patching_threshold = 1.335442066192627, patch_size = 4
hash n-gram: group sizes [3,4,5,6,7,8], 1 hash fn, vocab 500,002 per group
```

Derived parameter counts (mine):

| Component | Params |
|---|---|
| global (25L, d=2048) | 1,284.5M |
| local encoder (1L, d=1024) | 12.8M |
| local decoder (9L, d=1024) | 115.6M |
| entropy patcher (14L, d=768) | 99.3M |
| **hash n-gram table (6 × 500,002 × 1024)** | **3.07B** |

**The hash n-gram table alone is 3.07B parameters — 2.4× the "1B" model it serves.** That is 6.1 GB in bf16 and ~43 GB with fp32 master weights + AdamW moments. On a single 80GB A100 that is most of the device before you have loaded anything else. This is the single most important under-reported fact about BLT for a compute-constrained team.

And where the FLOPs go (forward, ignoring attention quadratics):

| avg bytes/patch | Total MFLOP/byte | local enc | local dec | **entropy patcher** | global |
|---|---|---|---|---|---|
| 4.5 | 1026 | 2.5% | 22.5% | 19.3% | 55.6% |
| 6 | 884 | 2.9% | 26.2% | **22.5%** | 48.5% |
| 8 | 777 | 3.3% | 29.8% | **25.6%** | 41.3% |
| 16 | 616 | 4.2% | 37.5% | **32.2%** | 26.1% |
| *BPE 1.3B dense, d=2048/25L, V=128k, 4.4 B/tok* | **703** | — | — | — | — |

So BLT-1B-as-released is **1.26× the FLOPs/byte** of an equivalent BPE model at p=6, and **~half of its compute is byte-level machinery** (entropy model + 9-layer local decoder). The advertised "50% inference FLOP savings" comes from co-scaling patch size *with* model size at 8B+, not from the 1B configuration.

### 2.4 H-Net, exactly as released (`goombalab/hnet/configs/*.json`, `hnet/modules/dc.py`)

```
hnet_2stage_XL: arch_layout ["m4", ["T1m4", ["T27"], "m4T1"], "m4"]
                d_model [1024, 1536, 2048], d_intermediate [0, 4096, 5504]
                vocab_size 256, heads [16,16,16], SWA window [1023, 1023, -1]
hnet_1stage_L : ["m4", ["T22"], "m4"], d_model [1024, 1536], vocab 256
```
`m4` = 4 Mamba-2 blocks (d_state 128, d_conv 4, expand 2, chunk 256); `T27` = 27 Transformer blocks. Outer stages are **Mamba-2**, not attention — this matters for our deployment targets (see §6).

The routing module (verbatim mechanism from `dc.py`):
- `q = Wq h[:-1]`, `k = Wk h[1:]`, both initialised to **identity**;
- `boundary_prob = clamp((1 − cos_sim(q, k)) / 2, 0, 1)`; position 0 forced to 1.0;
- hard boundary iff `argmax > 0.5`; **no** straight-through estimator — gradients reach the router through the de-chunking EMA;
- **DeChunkLayer** expands chunks back to byte resolution with an EMA whose decay is `dt = log(1/(1−p))`, implemented by *reusing the Mamba-2 `mamba_chunk_scan_combined` kernel* (A = −1, B = p, C = 1).

The auxiliary **ratio loss** (verbatim from `rosinality/halite`, matching the paper) with target compression `N`:

```python
def ratio_loss(N, boundary_mask, boundary_prob):
    F = boundary_mask.float().mean()      # fraction of positions chosen as boundaries
    G = boundary_prob.float().mean()      # mean predicted boundary probability
    return (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
```

### 2.5 The single most important small-scale data point I found

`openai/parameter-golf`, record `records/track_non_record_16mb/2026-03-29_HNet_ByteVsSubword_Study` — a **matched** byte-vs-subword H-Net study at **17.5M params** on 8×H100 with a 10-minute wall-clock cap, on FineWeb. This is almost exactly our ablation regime, and it is the only rigorous matched comparison at this scale I could find. Read directly from the repo:

| Config | BPB | Size | Steps | Wall clock |
|---|---|---|---|---|
| **byte260** (V=260, 1 byte/token), best 10-min | **1.4116 ± 0.013** | 15.78 MB | 4,520 | 10 min |
| **sp1024** (SentencePiece V=1024), best 10-min | **1.3734** | 15.99 MB | 4,466 | 10 min |
| byte260, extended to **4 hours** | **1.3595** | 15.96 MB | 85,242 | 4 h |
| (earlier byte H-Net attempt, PR #1044) | 1.8989 | 22M params | — | 1×4090 |

Architecture: 9 layers total = 2 outer (encoder+decoder) + 5 main, d=512, MLP×2, 20 matched runs across 4 hyperparameters.

Findings that transfer directly to us:
- **At matched wall clock and matched artifact size, byte-level loses to subword by ~2.8% BPB (1.4116 vs 1.3734).** It needs **~24× the wall clock** (10 min → 4 h) to reach 1.3595 and only barely pass the subword 10-minute run.
- Learned boundaries *do* work: whitespace agreement reaches **97.4%**, average chunk length **5–6 bytes** (router starts at ~120 boundaries per 256-byte window, ends at 42–47). Qualitatively the trained router segments `[The ][quick ][brown ][fox ]...` — it discovers words from the LM objective alone.
- **Depth around the chunk/dechunk interface matters more than depth in the compressed stage**: OUTER_LAYERS 1→2 improved byte BPB 1.4526 → 1.4206.
- Best byte hyperparameters: `TARGET_AVG_CHUNK_LEN=9`, `RATIO_LOSS_WEIGHT=0.05`, `OUTER_LAYERS=2`, `HNET_LR_DIFF=0.75–0.85` (the router path wants a *lower* LR than the rest of the model).
- Byte chunking is *more regular* than subword chunking (chunk-size CV 0.45 vs 0.76).

### 2.6 Wall-clock evidence at larger scale

A tokenizer paper's training-cost table (arXiv:2511.20849 †, normalized to 8×H200) reports **BLT: 160 GPU-hours for 42B tokens** vs **byte-level BPE: 68 GPU-hours for 72B tokens** — i.e. BLT delivered **~0.26 B tokens/hour vs 1.06 B tokens/hour, roughly 4× slower per unit of text**. This is consistent with my FLOP accounting (1.26×) plus a large MFU penalty (~3×) from variable-length gather/scatter and the entropy model.

---

## 3. What actually transfers to our scale

Brutally honest assessment. Our envelope: **one A100 80GB, a few hundred A100-hours**.

### 3.1 First, the budget reality that frames everything

| Assumed MFU | Total FLOPs in 300 A100-h | Tokens at N_active = 1.3B | Tokens at N_active = 0.4B |
|---|---|---|---|
| 0.20 (byte/patch hierarchical) | 6.7e19 | 8.6B | 28.1B |
| 0.30 (MoE, grouped GEMM) | 1.0e20 | 13.0B | 42.1B |
| 0.42 (clean dense BPE + FA2 + compile) | 1.4e20 | 18.1B | 59.0B |

Two consequences that R01 must state plainly:

1. **Qwen3-1.7B saw ~36T tokens. Our best case is ~13–18B. That is ~2,800× less data.** We will not beat it on MMLU/GPQA by architecture. R01 cannot fix this; the tokenizer decision must therefore be judged on *cost* and on the *specific* benchmarks it does move (GSM8K/MATH/HumanEval/IFEval), not on a vague "better representation" claim.
2. At 200–300 A100-hours the **compute-optimal active-parameter count is ~400–700M, not 1–1.5B** (D/N ≈ 20 requires D ≈ 8–14B at N = 0.4–0.7B). Flag to the architecture/scaling track: the stated "1–1.5B active" target is ~2–3× over-parameterized for this budget.

### 3.2 Throughput: the decisive table

Design "byte-patch" below is the *lean* frontend I recommend in §4.3 (no entropy model, no hash n-grams, d_local = 512/768, 3 encoder + 4 decoder layers, avg 5.5 bytes/patch). All numbers are mine, forward-pass matmul FLOPs, ×3 for training, MFU as stated.

| Scale | Design | MFLOP/byte (fwd) | Assumed MFU | Training throughput | vs BPE |
|---|---|---|---|---|---|
| **main**, global 1.3B active MoE | BPE V=32k, 3.7 B/tok | 739 | 0.30 | **42.2 kB/s** | 1.00× |
| **main** | byte-patch, p = 5.5 | **622** | 0.22 | **36.8 kB/s** | **0.87×** |
| **main** | byte-patch, p = 7.0 | 513 | 0.22 | 44.6 kB/s | 1.06× |
| **mini**, global 299M dense (d=1024, L=24) | BPE V=32k, 3.7 B/tok | 180 | 0.42 | **243 kB/s** | 1.00× |
| **mini** | byte-patch, p = 5.5 | 215 | 0.22 | **106 kB/s** | **0.44×** |
| **mini** | BLT-as-published (entropy + 9L dec) | 484 | 0.22 | 47 kB/s | 0.19× |

**The key structural insight: a byte/patch frontend has a fixed per-byte cost, so its relative overhead shrinks as the global model grows.** At 1.3B active the frontend is only 18% of FLOPs and byte-patch is *FLOP-cheaper* than BPE (622 vs 739) — the MFU penalty is what makes it 13% slower. At 300M dense the frontend is 45% of FLOPs and byte-patch is **2.3× slower**. So:

- **Prophet-main (1.3B active) sits right at the crossover.** A byte frontend is affordable *if and only if* we can get MFU ≥ 0.26 on the patch kernels.
- **Prophet-mini (300–600M dense) is firmly below the crossover.** A from-scratch byte frontend there is a 2.3× tax we cannot pay.

### 3.3 What fails below ~3B params / ~100B tokens

Ranked, with the evidence:

1. **BLT as published — fails hard.** The 100M-parameter entropy patcher runs over *every byte* and costs 19–26% of total FLOPs; the 3.07B hash n-gram table costs ~43 GB of A100 memory with optimizer state; the 9-layer local decoder costs another 22–30%. The paper's own limitation section concedes that the scaling laws it used were fitted for BPE transformers and that architectural choices may change beyond 1B, and the search-surfaced summary is explicit: patch-size-8 models "start at a significantly worse point compared to BPE Llama 2 at 1B but end up better than BPE at 7B" †. **We live at the bad end of that sentence.**
2. **Byte-level from scratch at matched wall clock — fails.** The parameter-golf 17.5M matched study (§2.5): −2.8% BPB at equal wall clock, needing ~24× compute to catch up. ByT5's +33% pretraining cost † is the same story from 2021.
3. **H-Net's *quality* claims do transfer; its *engineering* does not.** DC genuinely learns word boundaries from bytes (97.4% whitespace agreement at 17.5M params). But the released H-Net puts **Mamba-2** in the outer stages and implements de-chunking by reusing the Mamba-2 SSD kernel. That means: a Triton dependency on Colab, no clean `torch.compile` path, no MLX/CoreML/ANE story for the iPhone target, and no vLLM path for the 5090. Also, the 2-stage headline ("overtakes tokenized Transformer after 30B bytes" †) is measured at 760M–1.3B with a 100B-token FineWeb-Edu budget — i.e. **the crossover happens at roughly our entire budget**, so we would spend everything just reaching parity.
4. **Fixed-patch MEGABYTE-style — fails.** Fixed patch size forces a bad trade at every position; Scratchpad Patching (2605.09630 †) names the mechanism precisely: **patch lag** — byte predictions inside a patch must use a stale patch representation to preserve causality, and the lag widens with patch size. Dynamic patching plus architecture changes were both necessary in BLT to match BPE scaling.
5. **Tiny vocabularies (≤4k) — fails, and the literature is unanimous.** Scaling Laws with Vocabulary (2407.13623), Over-Tokenized Transformer (2501.16975), SuperBPE (2503.13423) and Length-MAX (2511.20849) all say the same thing from four directions: at fixed compute, *more* compression per position is better. Compute Optimal Tokenization (2605.01188 †) is the sharpest: the optimal compression rate **decreases with compute**, so at *low* compute you want *higher* compression. That is a direct argument against 1-byte-per-position and for a healthy subword vocabulary.
6. **T-FREE-style trigram-hash embeddings — attractive but unproven at generative scale**, and the trie/decoding machinery is a substantial engineering lift for a benefit (>85% embedding-param reduction †) we can get 80% of by simply choosing V=32k instead of V=262k.

### 3.4 What *does* transfer

1. **Right-sizing the vocabulary.** Fitting a power law through the two anchors reported by Scaling Laws with Vocabulary (3B → V_opt ≈ 43k; 70B → V_opt ≈ 216k) gives exponent 0.513, hence `V_opt ≈ 43k · (N_nv / 3e9)^0.513`: **N=1.3B → V_opt ≈ 28k**, **N=0.4B → V_opt ≈ 15k**. This is a hard, quantitative, citable recommendation, and it is 5–9× smaller than what Gemma-3/Qwen3/Phi-4-mini use.
2. **Digit-level tokenization.** Cheap, well-evidenced (§1.3), and digits are only **1.5% of code bytes and 2.8% of English prose bytes** (measured on Python stdlib source and /usr/share/doc on this machine), so the sequence-length cost of splitting every digit is ~2–3%.
3. **Explicit whitespace/indent tokens for code.** Zero risk, measurable on HumanEval/MBPP.
4. **Multi-token prediction.** +12% HumanEval / +17% MBPP at 13B, up to 3× self-speculative decode (2404.19737 †). Orthogonal to tokenization, composes with either frontend, and it is the *only* item on this list that plausibly closes benchmark gaps at our budget.
5. **Byteification as a retrofit, not a pretrain.** Bolmo (2512.15586 †) is the key 2025 result for a compute-poor team: freeze the subword backbone, train a byte frontend to mimic it (9.8B tokens), then unfreeze (39.3B tokens) — and get **CUTE 78.6 vs 56.9** while staying level on broad evals. The order of operations matters: **BPE first, bytes last.**
6. **Deterministic, whitespace-anchored patching.** SpaceByte's rule (≈6.3 B/patch †) and AU-Net's rightward-insertion-stability requirement give a *zero-parameter, zero-instability, statically-shaped* boundary function — which is exactly what CoreML/ANE export needs.

---

## 4. Recommendation for Prophet

### 4.1 The decision

> **Ship Prophet-Tok v1: a purpose-engineered 32,768-entry byte-fallback BPE, plus 4-way multi-token prediction. Do not train a tokenizer-free Prophet from scratch. Reserve ~20 A100-hours at the end of the program for a Bolmo-style byteification retrofit (PPF-6, §4.3), applied to the finished checkpoint, and only if the §7 gates pass.**

Why, in one paragraph: every tokenizer pathology that our target benchmark suite actually measures (arithmetic, code indentation, glitch tokens, embedding/head waste, logit memory) is fixable *inside* a BPE tokenizer for approximately zero cost and zero risk. The pathologies that require going byte-level (character manipulation, noise robustness, multilingual fertility) are **not in our benchmark suite**. Against that, going byte-level from scratch costs 1.15× wall clock at main scale and 2.3× at mini scale, forfeits `torch.compile`/FlashAttention/vLLM/MLX/CoreML maturity, breaks `lm-eval-harness` compatibility, and — decisively — **breaks the shared-vocabulary property that lets Prophet-mini serve as the draft model for Prophet-main's speculative decoding**, which is worth more to our inference targets than CUTE points.

### 4.2 Prophet-Tok v1 — the concrete design to build now

**One tokenizer for the whole family** (main and mini). Shared vocabulary is a hard requirement: it enables (a) logit-level distillation main → mini, (b) mini-as-draft speculative decoding for main, (c) a single eval harness.

```yaml
name: prophet-tok-v1
algorithm: byte-level BPE (GPT-2 style bytes↔unicode surrogate map; nothing is ever UNK)
vocab_size: 32768                 # = 32,512 learned + 256 reserved special ids at the TOP
                                  # derived: V_opt ≈ 43k·(1.3e9/3e9)^0.513 ≈ 28k → round up to 2^15
normalization: NFC only           # no lowercasing, no accent stripping, no dummy-prefix hack
byte_fallback: true               # all 256 single bytes are in-vocabulary
tie_embeddings: mini=true, main=false     # mini: saves 25.2M @ d=768; main: 67.1M is 0.7% of 10B

pretokenizer_rules:               # applied before BPE; merges may never cross these
  - digits:        every digit is its own pre-token   -> [0-9] singletons, no multi-digit merges
  - newlines:      no merge may span '\n' or '\r'
  - indentation:   explicit tokens for runs of {2,4,6,8,12,16,20,24,28,32} spaces and {1,2,3,4} tabs
  - class changes: no merge across letter<->digit or letter<->punctuation, EXCEPT a curated
                   allowlist (contractions: 've 'll 're 'd n't 's ; markup: </ /> <!-- ; code: -> => :: ->
  - whitespace:    a leading single space MAY merge into the following word (standard);
                   SuperBPE-style multi-word merges are DISABLED in v1 (see note below)
  - unicode:       no merge may split a UTF-8 codepoint

training_corpus: the exact pretraining mixture, in the exact proportions
                 (per "Getting the most out of your tokenizer", arXiv:2402.01035)
                 target composition: 55% English web/edu, 25% code, 10% math/STEM, 10% multilingual

acceptance targets (measure before freezing):
  bytes_per_token(English)  >= 3.6
  bytes_per_token(code)     >= 3.4
  digit tokens              == 10 single-digit tokens, no others
  indentation fidelity      : 4/8/12/16-space runs are 1 token each
  glitch tokens after pretrain (Magikarp detector, arXiv:2405.05417) < 0.1% of vocab
```

**Why 32,768 and not 128k+:** (a) V_opt from the scaling law is ~28k at 1.3B active; (b) at V=32k the output head is 4.5% of main-model forward FLOPs and 12.9% of mini's, versus 15.6%/36.7% at 128k; (c) the fp32 logit tensor for a 32,768-position micro-batch is 4.3 GB rather than 16.8 GB — this is what lets us use a large micro-batch on one A100; (d) at 8–18B training tokens a 128k vocabulary has an enormous under-trained tail (§1.6). **Note the honest counterpoint:** at 1.3B active, V=128k is actually marginally *FLOP-cheaper per byte* (710 vs 739 MFLOP/byte) because the head amortizes over 4.4 rather than 3.7 bytes. The 32k choice is justified by memory, by the scaling law, and by tail-token statistics — not by FLOPs. Ablation A1 (§7) decides it empirically.

**Why SuperBPE is disabled in v1 despite being the biggest reported free lunch (+4.0% avg, +8.2% MMLU †):** its gains were demonstrated at V=200k with multi-trillion-token pretraining. At V=32k, superword merges compete directly with subword coverage, and at 8–18B tokens the resulting long tail is exactly the under-trained-token failure we are trying to avoid. **Ablation A1b** tests a 32k SuperBPE variant; if it wins on BPB *and* arithmetic, adopt it.

**Multi-token prediction (adopt unconditionally):** 4 independent output heads on the shared trunk, `n = 4`, following arXiv:2404.19737; heads 2–4 are discarded after pretraining or kept for self-speculative decoding. Cost: `3 × V × d` extra params = 201M at d=2048 (2% of a 10B MoE) and, with sequential per-head loss computation, ~3× logit memory on the last block only. This is the highest expected-value item in R01.

### 4.3 PPF-6 — the gated byte/patch frontend (retrofit stage)

If and only if gates A2/A6/A7 pass, attach this to the finished Prophet checkpoint. Full spec:

```yaml
name: PPF-6  (Prophet Patch Frontend, target 6 bytes/patch)
byte_vocab: 260                   # 256 bytes + <bos> <eos> <pad> <mask>

boundary_function: DETERMINISTIC (no entropy model, no learned router in v1 of PPF)
  new patch begins at byte i if ANY of:
    R1  byte[i-1] is whitespace and byte[i] is not          # word start (SpaceByte)
    R2  byte[i] is an ASCII digit                            # single-digit patches
    R3  byte[i-1] is an ASCII digit and byte[i] is not
    R4  byte[i] >= 0xC0 (UTF-8 lead) and this is the 2nd lead byte since the last boundary
    R5  current patch length == max_patch_len
  max_patch_len: 12
  properties: causal; stable to rightward insertion (AU-Net requirement); zero parameters;
              static-shape-friendly for CoreML/ANE; identical at train and inference time.

local_encoder:  3 transformer layers, d=512,  GQA 8q/2kv, head_dim 64, ffn 1408 (SwiGLU),
                sliding-window attention, window 128 bytes, RoPE theta 10000
pooling:        per patch, concat[mean(bytes), last(byte)] -> Linear(1024 -> d_global), no bias
global:         the existing Prophet backbone, UNCHANGED (main d=2048 L28 MoE; mini d=1024 L24)
local_decoder:  4 transformer layers, d=768,  GQA 12q/3kv, ffn 2048 (SwiGLU),
                sliding-window self-attn (window 128) + cross-attention to the patch state
                at EVERY layer (BLT's released decoder sets cross_attn_all_layers=True;
                the parameter-golf study independently found interface depth matters most)
byte_heads:     4 MTP heads, each Linear(768 -> 260)          # 4-byte parallel decode
NOT included:   entropy patcher (saves 99M params / 22% of FLOPs)
                hash n-gram embedding table (saves 3.07B params / 43 GB of optimizer state)
                Mamba-2 outer stages (keeps the MLX / CoreML / vLLM path alive)
```

**Measured patch statistics for this exact rule** (I implemented R1–R5 and ran it on real corpora on this machine: Python stdlib source for code, `/usr/share/doc` + license texts for prose, synthetic digit-dense math text, synthetic Chinese):

| Corpus | bytes/patch | CV of patch length |
|---|---|---|
| English prose/docs | **5.36** | 0.65 |
| Python code | **6.41** | 0.64 |
| JSON | 9.01 | 0.47 |
| digit-dense math text | 1.35 | 0.59 |
| Chinese (synthetic) | 6.00 | 0.00 |
| **weighted 55/25/10/10 mix (en/code/math-prose/multilingual)** | **≈5.5** | — |

5.5 bytes/patch is **1.25× fewer global-model positions than a 128k BPE tokenizer** (4.4 B/tok) and **1.49× fewer than our own 32k tokenizer** (3.7 B/tok). `max_patch_len` sweep: 8 → p=4.63, 12 → p=5.36, 16 → p=5.57, 24 → p=5.65 (English) — 12 is the knee, and it also bounds worst-case patch length, which bounds the patch-lag problem (2605.09630) and keeps CJK/JSON from degenerating.

**Parameter cost of PPF-6:** 8.59M (encoder) + 2.10M (projection, main) + 42.87M (decoder + cross-attn + 4 MTP byte heads) = **53.6M total** (mini: 46.2M). Compare: BPE-32k untied embed+head at d=2048 = 134.2M; BPE-151936 untied = 622.3M. **PPF-6 is a net parameter *saving* of 80M at main scale.**

**Core module, PyTorch sketch:**

```python
import torch, torch.nn as nn, torch.nn.functional as F

WS   = torch.tensor([0x20,0x09,0x0a,0x0d,0x0b,0x0c])
DIG0, DIG9 = 0x30, 0x39

@torch.no_grad()
def patch_starts(b: torch.Tensor, max_patch: int = 12) -> torch.Tensor:
    """b: (B, L) uint8 byte ids -> (B, L) bool mask, True where a patch begins.
    Causal and stable to rightward insertion: mask[i] depends only on b[:i+1]."""
    B, L = b.shape
    prev = F.pad(b[:, :-1], (1, 0), value=0x20)                  # virtual leading space
    is_ws   = (prev.unsqueeze(-1) == WS.to(b.device)).any(-1)
    cur_ws  = (b.unsqueeze(-1)    == WS.to(b.device)).any(-1)
    is_dig  = (b >= DIG0) & (b <= DIG9)
    pre_dig = (prev >= DIG0) & (prev <= DIG9)
    lead    = b >= 0xC0                                          # UTF-8 lead byte

    m = (is_ws & ~cur_ws) | is_dig | (pre_dig & ~is_dig)         # R1 R2 R3
    m[:, 0] = True
    # R4 (every 2nd UTF-8 lead byte) and R5 (hard cap) need a running counter.
    # Precompute once per shard offline and store alongside the byte stream;
    # at inference this is a 20-line incremental loop with two int counters.
    return _apply_cjk_and_cap(m, lead, max_patch)                # returns (B, L) bool


def pool_patches(h: torch.Tensor, start: torch.Tensor, n_patch: int):
    """h: (B, L, d_l) encoder states; start: (B, L) bool. -> (B, n_patch, 2*d_l)
    Segment mean + last byte of each patch. Uses index_add (no data-dependent shapes)."""
    B, L, d = h.shape
    pid = start.cumsum(-1) - 1                                   # (B, L) patch id per byte
    flat = pid + torch.arange(B, device=h.device).unsqueeze(1) * n_patch
    acc  = h.new_zeros(B * n_patch, d).index_add_(0, flat.reshape(-1), h.reshape(-1, d))
    cnt  = h.new_zeros(B * n_patch).index_add_(
                0, flat.reshape(-1), torch.ones_like(flat, dtype=h.dtype).reshape(-1))
    mean = (acc / cnt.clamp(min=1).unsqueeze(-1)).view(B, n_patch, d)
    last_idx = torch.zeros(B, n_patch, dtype=torch.long, device=h.device)
    last_idx.scatter_(1, pid.clamp(min=0), torch.arange(L, device=h.device).expand(B, L))
    last = h.gather(1, last_idx.unsqueeze(-1).expand(-1, -1, d))
    return torch.cat([mean, last], dim=-1)                        # (B, n_patch, 2*d_l)


class PPF6(nn.Module):
    def __init__(self, d_local=512, d_dec=768, d_global=2048,
                 n_enc=3, n_dec=4, n_mtp=4, max_patch=12, window=128):
        super().__init__()
        self.max_patch, self.n_mtp = max_patch, n_mtp
        self.byte_emb = nn.Embedding(260, d_local)
        self.enc  = nn.ModuleList(SWABlock(d_local, 8, 2, 1408, window) for _ in range(n_enc))
        self.up   = nn.Linear(2 * d_local, d_global, bias=False)
        self.down = nn.Linear(d_global, d_dec, bias=False)
        self.dec  = nn.ModuleList(
            XAttnBlock(d_dec, 12, 3, 2048, window, d_kv=d_dec) for _ in range(n_dec))
        self.dec_in   = nn.Linear(d_local, d_dec, bias=False)     # byte-level skip connection
        self.heads    = nn.ModuleList(nn.Linear(d_dec, 260, bias=False) for _ in range(n_mtp))
        self.enc_norm = nn.RMSNorm(d_local); self.dec_norm = nn.RMSNorm(d_dec)

    def encode(self, byte_ids):
        start   = patch_starts(byte_ids, self.max_patch)
        n_patch = int(start.sum(-1).max())
        h = self.byte_emb(byte_ids)
        for blk in self.enc:
            h = blk(h)
        h = self.enc_norm(h)
        patches = self.up(pool_patches(h, start, n_patch))         # (B, n_patch, d_global)
        return patches, h, start                                   # h kept for the skip

    def decode(self, patch_out, h_bytes, start):
        """patch_out: (B, n_patch, d_global) from the global backbone.
        CAUSALITY: byte t inside patch k may only see patch states < k (patch lag).
        We therefore shift the patch stream by one before broadcasting."""
        pid   = (start.cumsum(-1) - 1).clamp(min=0)
        ctx   = self.down(F.pad(patch_out, (0, 0, 1, 0))[:, :-1])   # shift right by one patch
        ctx_b = ctx.gather(1, pid.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))
        x = self.dec_in(h_bytes)
        for blk in self.dec:
            x = blk(x, kv=ctx_b)                                    # self-attn + cross-attn
        x = self.dec_norm(x)
        return [head(x) for head in self.heads]                     # n_mtp × (B, L, 260)
```

**Retrofit training recipe (Bolmo-style, two stages):**

| Stage | What trains | Objective | Text volume | A100-hours (main) | A100-hours (mini) |
|---|---|---|---|---|---|
| **A** | PPF-6 only; backbone frozen | KL to the parent BPE model's next-token distribution, aggregated over the bytes of each parent token, + byte CE | 1.5 GB | ~8.8 | ~3.1 |
| **B** | everything, backbone LR × 0.1 | byte CE + 4-byte MTP | 4.0 GB | ~30 | ~10.5 |

(Derived from 622 MFLOP/byte (main) / 215 (mini), MFU 0.22, ×2.33 for a frozen-backbone step, ×3 for a full step.) **Total retrofit: ~39 A100-h for main, ~14 A100-h for mini.** Bolmo used 43B + 173B bytes; we use 1.5 + 4.0 GB, which is 1/29th and 1/43rd — so expect a *partial* recovery of Bolmo's gains. Recommendation: **retrofit the mini model only** (14 A100-h ≈ 5% of budget), where character robustness and on-device weight budget matter most, and where a failed retrofit costs us nothing because the BPE mini still ships.

### 4.4 What we explicitly reject and why

| Rejected | Reason |
|---|---|
| BLT entropy patcher | 99.3M params over every byte = 19–26% of FLOPs; a second model to train, version and deploy |
| BLT hash n-gram embeddings | 3.07B params = 6.1 GB bf16 / ~43 GB with AdamW; kills our 80GB A100 and our 32GB/8GB inference targets |
| H-Net learned router (in v1) | boundary collapse risk, requires per-path LR scaling (`HNET_LR_DIFF≈0.75`) + ratio-loss tuning + an ATDC-style compression curriculum (2605.30080); a dynamic-shape router blocks CoreML/ANE export. Keep as ablation A2b. |
| Mamba-2 outer stages | no MLX/CoreML/vLLM path; Triton fragility on Colab |
| Vocab ≥ 128k | 30–54% of mini's parameters, 31–47% of its forward FLOPs, 17–34 GB logit tensors, huge under-trained tail at 8–18B tokens |
| Vocab ≤ 8k | contradicted by four independent 2024–2026 results (§3.3 item 5) |
| Pure character/ByT5-style (no patching) | +33% pretraining cost † for a 4.4× longer sequence; strictly dominated by patching |

---

## 5. Compute & memory budget

### 5.1 Parameters

| Component | Prophet-main (d_g=2048) | Prophet-mini (d_g=1024) |
|---|---|---|
| Prophet-Tok v1 embed (V=32,768) | 67.1M | 33.6M |
| Prophet-Tok v1 head | 67.1M (untied) | tied (0) |
| 3 extra MTP heads | 201.3M | tied variant: 100.7M |
| *(alternative)* V=151,936 embed+head | 622.3M | 311.2M |
| **PPF-6 frontend (if retrofitted)** | **53.6M** | **46.2M** |
| PPF-6: local encoder 3L d=512 | 8.59M | 8.59M |
| PPF-6: patch projection | 2.10M | 1.05M |
| PPF-6: local decoder 4L d=768 + xattn + 4 byte heads | 42.87M | 36.58M |

Prophet-mini candidates, total parameters and int4 weight footprint (relevant to the 8GB iPhone target):

| Candidate | Total | int4 weights |
|---|---|---|
| A. BPE-128k, d=768, L=24 (tied) | 268.4M | 134 MB |
| B. BPE-32k, d=768, L=26 (tied) | 209.2M | 105 MB |
| **C. BPE-32k, d=1024, L=24 (tied)** ← recommended | **332.4M** | **166 MB** |
| D. PPF-6 + d=1024, L=24 global | 322.5M | 161 MB |
| E. PPF-6 + d=1152, L=26 global | 437.8M | 219 MB |

Candidate A spends 197M of its 268M on a lookup table. Candidate C spends the same weight budget on 33% more depth×width. That single swap is the largest quality-per-byte win available in R01.

### 5.2 FLOPs per byte of text (forward, matmul only)

| Design | MFLOP/byte |
|---|---|
| main, BPE-32k @ 3.7 B/tok, 1.3B active | **739** |
| main, BPE-128k @ 4.4 B/tok, 1.3B active | 710 |
| main, PPF-6 @ p=5.5 | **622** |
| main, PPF-6 @ p=7.0 | 513 |
| mini, BPE-32k @ 3.7 B/tok, 299M body | **180** |
| mini, BPE-128k @ 4.4 B/tok | 196 |
| mini, PPF-6 @ p=5.5 | 215 |
| mini, BLT-as-published (d_l=768, 1 enc / 9 dec, + entropy) | 484 |
| *reference:* BLT-1B released config @ p=6 | 884 |

### 5.3 Expected single-A100 throughput and total budget

Assumed achievable MFU on one A100 80GB (bf16, peak 312 TFLOP/s): **0.42** dense BPE with FlashAttention-2 + `torch.compile`; **0.30** MoE with grouped GEMM; **0.22** byte/patch hierarchical (gather/scatter, variable-length cross-attention, no fused kernels). *A7 in §7 exists to measure the 0.22 figure, which is the single largest uncertainty in this report.*

| Run | Throughput | Text per A100-hour |
|---|---|---|
| mini, BPE-32k | 243 kB/s | 0.88 GB |
| mini, PPF-6 | 106 kB/s | 0.38 GB |
| main MoE, BPE-32k | 42.2 kB/s | 0.15 GB (≈ 41M tokens/h) |
| main MoE, PPF-6 @ p=5.5 | 36.8 kB/s | 0.13 GB |

**Program-level budget (300 A100-hours total):**

| Item | A100-hours | Share |
|---|---|---|
| §7 ablation rig (minimum viable set) | 24 | 8% |
| Prophet-main pretrain (BPE-32k, ≈8.6–13B tokens) | 190 | 63% |
| Prophet-mini pretrain (distilled from main, ≈25–40B tokens) | 55 | 18% |
| SFT / instruction tuning | 17 | 6% |
| PPF-6 retrofit of mini (gated) | 14 | 5% |

### 5.4 Inference-side budget (this is where PPF-6 earns its keep)

Weight traffic per output byte in the memory-bandwidth-bound single-stream decode regime (MoE weights NVFP4, local decoder fp16):

| Design | MB moved / output byte | 5090 @1.8 TB/s | M-Ultra @0.8 TB/s |
|---|---|---|---|
| BPE-32k, MoE 1.3B active, 1 step / 3.7 bytes | 175.7 | 10.2 kB/s | 4.6 kB/s |
| PPF-6 p=5.5 + 25M local decoder every byte | 158.3 (0.90×) | 11.4 kB/s | 5.1 kB/s |
| PPF-6 p=5.5 + **4-byte MTP** on the local decoder | **120.8 (0.69×)** | **14.9 kB/s** | **6.6 kB/s** |

KV cache for 100 KB of context (int8, GQA 32q/4kv, d=2048, L=24): BPE-32k **0.332 GB** (27,027 positions) vs PPF-6 **0.205 GB** (16,666 positions), plus a *bounded* 1 MB for the sliding-window local stages. **PPF-6 gives a 38% KV-cache reduction and a 31% decode-bandwidth reduction.** That is the real, quantified case for the retrofit — not benchmark scores.

---

## 6. Risks & failure modes

| # | Risk | Severity | Evidence / mechanism | Mitigation |
|---|---|---|---|---|
| R1 | **MFU on patch kernels is worse than 0.22** | **Critical** | BLT measured ~4× slower per unit of text than byte-BPE (160 h/42B vs 68 h/72B †) — far worse than its 1.26× FLOP ratio. Variable-length `index_add`/`gather`, dynamic `n_patch` recompiles, unfused cross-attention. | **A7 measures it before anything else is built.** Fix `n_patch` to a static bound (`ceil(L / p_min)`) with masking so `torch.compile` sees one shape. Kill PPF-6 if MFU < 0.18. |
| R2 | **The tokenizer cannot be changed after pretraining starts** | **Critical** | Structural. | Freeze the tokenizer only after A0/A1/A3/A4 pass. Reserve 256 special ids at the top of the vocabulary so chat/tool/FIM formats can be added later without re-embedding. |
| R3 | Under-trained tokens at 8–18B tokens | High | arXiv:2405.05417; our data budget is ~2,800× smaller than Qwen3's | V=32k (4× more updates/token than 128k). Run the Magikarp detector post-pretrain and re-initialise offenders to the mean of their byte-fallback decomposition. |
| R4 | **Digit-splitting inflates sequence length on math data** | Medium | Measured: digit-dense text drops to p=1.35 B/patch; but digits are only 1.5% (code) / 2.8% (prose) of bytes | Accept — the corpus-level cost is 2–3%. Cap it via R5 (`max_patch_len`) so no single number blows up a batch. |
| R5 | Patch lag degrades in-patch byte prediction | Medium | Named and measured by Scratchpad Patching (2605.09630 †): in-patch predictions must use a stale patch state | `max_patch_len = 12` bounds the lag. Cross-attention at every decoder layer. If A6 shows in-patch loss spikes, adopt SP's entropy-triggered scratchpads. |
| R6 | **Boundary collapse** if we ever enable a learned router | High | H-Net needs a ratio loss, `HNET_LR_DIFF ≈ 0.75`, and (per ATDC, 2605.30080 †) a compression curriculum to train stably | v1 is deterministic (zero collapse risk). If A2b enables learning, keep the deterministic rule as a floor: the router may only *add* boundaries, never remove them. |
| R7 | **CoreML / ANE cannot express dynamic patch shapes** | High (iPhone target) | ANE requires static shapes; data-dependent `n_patch` is not expressible | The deterministic rule can be evaluated on CPU ahead of the neural graph and padded to a static bound. A learned router cannot. This is the strongest single reason PPF-6 v1 is deterministic. |
| R8 | Eval-harness incompatibility | Medium | `lm-eval-harness` assumes a tokenizer and per-token log-likelihoods; byte models need custom scoring | Score everything in **bits-per-byte** from day one — the only metric that is fair across tokenizations. Budget ~1 engineer-week for a byte-scoring adapter *before* committing to PPF-6. |
| R9 | Losing the shared vocabulary breaks mini-as-draft speculative decoding | Medium | Structural | Either retrofit both models or neither. If only mini is byteified, main keeps a BPE draft model of its own. |
| R10 | CJK/JSON pathologies in the deterministic rule | Low | Measured: JSON reaches p=9.01 (fine), CJK is pinned at exactly 6.00 by R4 | `cjk_merge=2` and `max_patch_len=12` already bound both. Re-measure on the real mixture. |
| R11 | Chasing tokenization instead of the actual bottleneck | **Critical (strategic)** | We are ~2,800× short on data versus Qwen3-1.7B. No tokenizer recovers that. | Cap total R01 spend at **≤10% of program compute** (24 h ablations + 14 h retrofit = 38 h of 300). Everything above that belongs to data quality and distillation. |

---

## 7. Ablation plan

All runs at 50–150M non-embedding parameters on one A100 80GB, ≤6 A100-hours each. Metric is **held-out bits-per-byte** (tokenizer-invariant) plus targeted probes. Reference throughputs: a 150M-class BPE model does ~10.5 GB of text (2.8B tokens) in 6 A100-h at MFU 0.42; the byte-patch variant does ~3.2 GB (0.59B patches) in 6 A100-h at MFU 0.22 — **matched wall clock, not matched tokens, is the honest comparison** (this is precisely the parameter-golf protocol, §2.5).

### Minimum viable set — 24 A100-hours

| ID | Experiment | Cost | Pass gate | Kill criterion |
|---|---|---|---|---|
| **A7** | **Kernel MFU probe.** Implement `patch_starts` + `pool_patches` + variable-length cross-attention. Measure realized TFLOP/s at L=8192 bytes, static `n_patch` bound, with/without `torch.compile`. **Run this first — it is the cheapest way to kill PPF-6.** | 2 h | MFU ≥ 0.22 | MFU < 0.18 → drop PPF-6 entirely, ship Prophet-Tok v1, close R01 |
| **A0** | **Tokenizer audit, no training.** Build 6 candidates (V ∈ {8k,16k,32k,65k,128k} × {digit-split on/off}) on the real mixture. Measure bytes/token per domain, digit fidelity, indentation fidelity, round-trip exactness, merge-rule violations. | 0.5 h | bytes/token(en) ≥ 3.6 at V=32k | — |
| **A1** | **Vocabulary sweep.** 150M non-embedding params (d=768, L=20), fixed **3 GB of text**, V ∈ {8k, 32k, 128k}. Report BPB, MMLU-style cloze, a 6-digit arithmetic probe, a spelling probe. | 3 × 4 h = 12 h | V=32k within 0.5% BPB of the best | if V=128k wins BPB by >1.5% *and* fits in memory at our micro-batch, revisit to 49k |
| **A3** | **Digit-splitting.** V=32k with vs without single-digit pre-tokenization, matched wall clock. Probe: addition/subtraction/multiplication to 6 digits, chain-of-thought arithmetic. | 2 × 3 h = 6 h | digit-split ≥ +10 points absolute on the arithmetic probe at ≤1% BPB cost | if it costs >2% BPB and gains <5 points, drop it |
| **A4** | **Indentation tokens.** Same model, code-only eval (Python + JS BPB, HumanEval-style single-function completion). | 2 × 1.5 h = 3 h | ≥ 3% code BPB improvement | — |

### Extended set — a further 26 A100-hours, run only if A7 passes

| ID | Experiment | Cost | Pass gate |
|---|---|---|---|
| **A2** | **The decisive one: matched-wall-clock BPE vs PPF-6 from scratch.** 150M global, 6 A100-h each, identical data stream. Prior from parameter-golf: expect byte to lose ~2.8% BPB. | 2 × 6 h = 12 h | **PPF-6 within 1.5% BPB of BPE-32k at equal wall clock** → consider from-scratch byte. Loss > 3% → from-scratch byte is dead (as predicted); retrofit only. |
| **A2b** | **Learned router on top of the deterministic floor.** H-Net routing (identity-init Q/K, cosine dissimilarity, ratio loss with N=6, `lr_scale=0.75`), permitted only to *add* boundaries. | 6 h | ≥ 1% BPB improvement over deterministic at equal wall clock, and no boundary collapse over 4k steps |
| **A5** | **MTP heads** n ∈ {1, 2, 4} at V=32k. Report BPB, self-speculative acceptance rate, arithmetic + code probes. | 3 × 2 h = 6 h | n=4 ≥ n=1 on BPB and ≥ 2× accepted-token rate |
| **A6** | **Retrofit feasibility.** Take the A1 winner, freeze it, attach PPF-6, train Stage A on 0.3 GB with KL-to-parent. | 2 h | recover ≥ 97% of parent BPB within 0.3 GB **and** ≥ +15 points on the spelling/character probe |
| **A1b** | **SuperBPE at V=32k** vs standard BPE at V=32k, matched wall clock. | 2 × 3 h = 6 h | ≥ 1% BPB improvement and no arithmetic regression → adopt |

**Decision tree:**
- `A7 fails` → ship Prophet-Tok v1 + MTP. R01 closed. Total spend: 24 A100-h.
- `A7 passes, A2 loses by >3%, A6 passes` → ship Prophet-Tok v1 + MTP; retrofit **mini only** with PPF-6 at the end (14 A100-h). **This is the outcome I expect, at roughly 70% confidence.**
- `A7 passes, A2 within 1.5%` → reconsider a from-scratch PPF-6 for **main only** (where it is FLOP-favourable, §3.2), keeping mini on BPE. ~15% confidence.
- `A2 wins outright` → surprising; escalate and re-plan. ~5% confidence.

---

## 8. References

Sources I read directly (GitHub, primary):
- `huggingface/transformers` — `src/transformers/models/blt/configuration_blt.py`, `modeling_blt.py` (exact BLT-1B configuration, entropy patching, rolling polynomial hash n-grams, patch cross-attention)
- `goombalab/hnet` — `configs/hnet_{1,2}stage_{L,XL}.json`, `hnet/modules/dc.py` (RoutingModule, ChunkLayer, DeChunkLayer EMA via the Mamba-2 SSD kernel)
- `rosinality/halite` — `src/halite/transformers/models/hnet.py` (H-Net `ratio_loss` implementation)
- `openai/parameter-golf` — `records/track_non_record_16mb/2026-03-29_HNet_ByteVsSubword_Study/README.md` and `train_gpt_hnet_byte.py` (the 17.5M matched byte-vs-subword study)
- `facebookresearch/blt` (README), `zjysteven/Awesome-Byte-LLM` (survey index), `PiotrNawrot/dynamic-pooling`, `owos/flexitokens`, `OpenEvaByte/evabyte`, `lucidrains/MEGABYTE-pytorch`, `sail-sg/scaling-with-vocab`, `facebookresearch/compute-optimal-tokenization`

Papers (numbers marked **†** in the text come from search-engine full-text summaries, because `arxiv.org` / `huggingface.co` / `aclanthology.org` / `openreview.net` were blocked by this session's egress proxy; verify before quoting externally):

**Byte-level / tokenizer-free architectures**
1. Clark et al. *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* arXiv:2103.06874, TACL 2022.
2. Xue et al. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* arXiv:2105.13626, TACL 2022.
3. Tay et al. *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* arXiv:2106.12672, 2021.
4. Nawrot et al. *Hierarchical Transformers Are More Efficient Language Models.* arXiv:2110.13711, 2021.
5. Hawthorne et al. *General-purpose, long-context autoregressive modeling with Perceiver AR.* arXiv:2202.07765, 2022.
6. Nawrot et al. *Efficient Transformers with Dynamic Token Pooling.* arXiv:2211.09761, ACL 2023.
7. Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* arXiv:2305.07185, NeurIPS 2023.
8. Horton et al. *Bytes Are All You Need: Transformers Operating Directly On File Bytes.* arXiv:2306.00238, TMLR 2023.
9. Wang et al. *MambaByte: Token-free Selective State Space Model.* arXiv:2401.13660, COLM 2024.
10. Wu et al. *Beyond Language Models: Byte Models are Digital World Simulators (bGPT).* arXiv:2402.19155, 2024.
11. Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* arXiv:2404.14408, NeurIPS 2024.
12. Deiseroth et al. *T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings.* arXiv:2406.19223, 2024.
13. Kallini et al. *MrT5: Dynamic Token Merging for Efficient Byte-level Language Models.* arXiv:2410.20771, 2024.
14. Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* arXiv:2412.09871, ACL 2025.
15. EvaByte Team (HKU NLP + SambaNova). *EvaByte: Efficient Byte-level Language Models at Scale.* 2025 (blog + `OpenEvaByte/evabyte`).
16. Neitemeier et al. *Multiscale Byte Language Models — A Hierarchical Architecture for Causal Million-Length Sequence Modeling.* arXiv:2502.14553, 2025.
17. Pagnoni et al. *From Bytes to Ideas: Language Modeling with Autoregressive U-Nets (AU-Net).* arXiv:2506.14761, 2025.
18. Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net).* arXiv:2507.07955, 2025.
19. Owodunni et al. *FLEXITOKENS: Flexible Tokenization for Evolving Language Models.* arXiv:2507.12720, Findings of ACL 2026.
20. *H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages.* arXiv:2508.05628, 2025.
21. Minixhofer et al. (Ai2). *Bolmo: Byteifying the Next Generation of Language Models.* arXiv:2512.15586, 2025.
22. *Fast Byte Latent Transformer (BLT-D / BLT-S / BLT-DV).* arXiv:2605.08044, Meta + Stanford, 2026.
23. Zheng et al. (Google DeepMind). *Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models.* arXiv:2605.09630, 2026.
24. *Adaptive Targeted Dynamic Chunking for Tokenization-Free Hierarchical Models.* arXiv:2605.30080, Fujitsu Research of America, 2026.
25. *Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models.* arXiv:2605.29459, 2026.
26. *Fast and Expressive Multi-Byte Prediction with Probabilistic Circuits.* arXiv:2511.11346, 2025.

**Tokenizer design, vocabulary scaling and failure modes**
27. Petrov et al. *Language Model Tokenizers Introduce Unfairness Between Languages.* arXiv:2305.15425, NeurIPS 2023.
28. Dagan et al. *Getting the most out of your tokenizer for pre-training and domain adaptation.* arXiv:2402.01035, 2024.
29. Land & Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* arXiv:2405.05417, EMNLP 2024.
30. Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* arXiv:2407.13623, NeurIPS 2024.
31. Huang et al. *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling.* arXiv:2501.16975, ICML 2025.
32. Liu et al. *SuperBPE: Space Travel for Language Models.* arXiv:2503.13423, 2025.
33. *Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations.* arXiv:2506.19004, 2025.
34. *Bit-level BPE: Below the byte boundary.* arXiv:2506.07541, 2025.
35. Lundin et al. *The Token Tax: Systematic Bias in Multilingual Tokenization.* arXiv:2509.05486, 2025.
36. Dong & Su. *Length-MAX Tokenizer for Language Models.* arXiv:2511.20849, 2025.
37. *Compute Optimal Tokenization.* arXiv:2605.01188, FAIR, 2026.
38. Beren Millidge. *Integer tokenization is now much less insane* (2024-05-11) and *Right to Left (R2L) Integer Tokenization* (2024-07-07), beren.io.
39. Benjamin Marie. *Shrink LLMs with Vocabulary Reduction: From Gemma 3 270M to 141M.* kaitchup.substack.com, 2025.

**Decoding / prediction objectives**
40. Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction.* arXiv:2404.19737, 2024.
41. Gemma Team. *Gemma 3 Technical Report.* arXiv:2503.19786, 2025 (embedding/non-embedding parameter split).

**Numbers computed in this report (not cited from any paper)**
All parameter counts, FLOPs-per-byte figures, logit-memory figures, throughput estimates, KV-cache figures, decode-bandwidth figures, the `V_opt ≈ 43k·(N/3e9)^0.513` fit, and the measured patch statistics for the PPF-6 boundary rule (English prose 5.36, Python 6.41, JSON 9.01, CJK 6.00 bytes/patch; digit density 1.5%/2.8%) were derived or measured in this session. Scripts are reproducible from the specifications above.
