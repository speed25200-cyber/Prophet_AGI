# R06 — Data efficiency: winning quality-per-token at 1–2% of competitor pretraining FLOPs

**Track:** R06 · **Status:** research complete, decision-oriented · **Date:** 2026-09-03
**Scope:** pretraining corpus selection, filtering, mixture design, synthesis, curriculum, and a concrete 300B-token budget allocation for Prophet-main (~8–12B total / ~1–1.5B active MoE) and Prophet-mini (~300–600M dense).

> **Sourcing note.** `arxiv.org`, `huggingface.co`, `openreview.net` and most paper mirrors are blocked by this environment's egress proxy. Numbers below were obtained from web-search abstract/summary retrieval and from `github.com` (reachable) — principally the HuggingFace blog repo. **Every number tagged `[verify]` should be re-checked against the source PDF before it drives an irreversible decision.** HF dataset IDs should be confirmed with `huggingface_hub.list_repo_files` before the download script is finalized; IDs are given as released and a few have been renamed by their owners over time.

---

## 1. Problem statement

### 1.1 The compute gap, quantified

Using the standard `C ≈ 6·N_active·D` estimate for decoder-only pretraining FLOPs:

| Model | Active params N | Pretrain tokens D | Pretrain FLOPs `6ND` | × Prophet |
|---|---:|---:|---:|---:|
| **Prophet-main (this plan)** | 1.3e9 | 3.0e11 | **2.34e21** | **1.0×** |
| Prophet-mini (dense) | 0.45e9 | 3.0e11 | 8.1e20 | 0.35× |
| Gemma-3-4B | 4.3e9 | 4.0e12 | 1.03e23 | 44× |
| Phi-4-mini (3.8B) | 3.8e9 | 5.0e12 | 1.14e23 | 49× |
| Llama-3.2-3B | 3.2e9 | 9.0e12 | 1.73e23 | 74× |
| SmolLM3-3B | 3.1e9 | 1.12e13 | 2.08e23 | 89× |
| Qwen3-1.7B | 1.7e9 | 3.6e13 | 3.67e23 | 157× |
| Qwen3-4B | 4.0e9 | 3.6e13 | 8.64e23 | **369×** |

**Headline: Prophet gets 0.3%–2.3% of the pretraining FLOPs of every model on the kill list.** Median gap ≈ **80×**.
Matching Qwen3-4B's compute on a single A100 at 140 TFLOP/s effective would take **≈196 years**.

### 1.2 What that buys on one Colab A100

A100 80GB, bf16, realistic MFU 35–45% for a ~1.3B-active fine-grained MoE with fused kernels → **≈120–140 TFLOP/s effective**. At 1.3B active that is **≈18,000 tokens/s = 6.5e7 tokens/A100-hour**.

| Training tokens | A100-hours | Wall-clock @ 12 h/day |
|---:|---:|---:|
| 30B | 464 | 39 days |
| 100B | 1,548 | 129 days |
| 200B | 3,096 | 258 days |
| **300B** | **4,644** | **387 days** |

> **⚠ Brief-internal inconsistency, flagged deliberately.** The track brief states both "a few hundred A100-hours" **and** "100B–500B tokens for a ~1B-active model." Those differ by **~10×**. A few hundred A100-hours buys **~25–30B tokens**, not 300B. At 30B tokens (≈23 tokens/active-param) Prophet-main will be roughly Chinchilla-optimal but drastically under-trained relative to Qwen3-1.7B (≈21,000 tokens/param) and **will not beat it on MMLU**, no matter how good the data is. See §5 for the three ways out. The rest of this document assumes the **300B target** as instructed and specifies exactly what it costs.

### 1.3 The one structural advantage we have

**Prophet is compute-constrained, not data-constrained.** Competitors at 9–36T tokens must scrape the bottom of the quality barrel; we need only **300B of the ~40T open tokens available (0.75%)**. Therefore:

1. **Every quality filter is free for us.** FineWeb-Edu throws away 92% of FineWeb to keep 1.3T tokens — from our seat that is still 4.3 epochs of headroom. We can filter *far* harder than anyone has published (top 1–3% of the web rather than top 10%) with zero data-exhaustion penalty.
2. **Repetition research is a lever, not a constraint.** Muennighoff et al. (2305.16264) show ≤4 epochs costs ~nothing. That makes "100B elite tokens × 3 epochs" a *legitimate competitor* to "300B good tokens × 1 epoch." This is our single most important ablation (§7, A3).
3. **Every token must earn its place against the eval suite.** With 0.75% of the corpus in play, mixture design is a *selection* problem, not a *coverage* problem. Multilingual, low-value web, and long-tail domains are pure opportunity cost.

**Thesis of this track: the 80× FLOP gap must be closed by (a) top-percentile filtering, (b) harvesting already-released synthetic/rephrased corpora, and (c) an unusually fat LR-decay phase. Nothing else in the data pipeline has comparable leverage.**

---

## 2. Dataset landscape

Ranked shortlist. "Value to Prophet" is my judgment of marginal benchmark points per token *at our scale and eval targets*, not general quality.

### 2.1 Tier S — build the plan around these

| # | Dataset | HF ID | Tokens | License | Good for | Measured benefit |
|---|---|---|---:|---|---|---|
| 1 | **Nemotron-CC-v2** (real + synthetic English web) | `nvidia/Nemotron-CC-v2` | ~6.6T total; 2.5T new English incl. translated + rephrased | CC-BY-4.0 + NVIDIA Data Access Agreement (explicitly permits training *any* model, incl. proprietary; permits publishing benchmarks) | Backbone web for the whole run | v1 HQ subset: **+5.6 MMLU vs DCLM** at 8B/1T tokens (2412.02595). Full 6.3T matches DCLM on MMLU with **4× more unique real tokens** |
| 2 | **Nemotron-CC v1** | `nvidia/nemotron-cc` (also mirrored at `data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/`) | 6.3T (4.4T real + ~1.9T synthetic) | as above | Same; v1 is better documented and has the published ablations | 8B/15T beat Llama-3.1-8B by **+5 MMLU, +3.1 ARC-C** |
| 3 | **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` (also `-score-2`, ~5.4T) | 1.3T (score ≥3) | ODC-By 1.0 | Highest-density knowledge web; MMLU/ARC driver | 1.82B/350B: **MMLU 33→37, ARC 46→57** vs FineWeb; **10× fewer tokens than C4/Dolma for equal MMLU** (2406.17557) |
| 4 | **DCLM-baseline** | `mlfoundations/dclm-baseline-1.0` | ~3.8–4.0T (2.6T used for DCLM-7B) | CC-BY-4.0 per card (CommonCrawl-derived) | Reasoning/commonsense-heavy web; complements FineWeb-Edu | 7B/2.6T → **64% MMLU 5-shot**; fastText OH2.5+ELI5 top-10% worth **+6.6 MMLU** over prior best open corpus; **~+3.5 CORE** vs reference-data-only (2406.11794) |
| 5 | **Nemotron-CC-Math** | `nvidia/Nemotron-CC-Math-v1` | 133B (`-3+`), 52B (`-4+`) | NVIDIA open data / CC-BY-4.0 | **Best open math pretraining corpus, full stop** | 8B @300B: **MATH 44.2 EM (+9.6 vs FineMath-3+, +12.6 vs MegaMath-Web)**; **MBPP+ +4.6 vs MegaMath-Web, +14.3 vs FineMath-3+** (2508.15096, ICLR 2026) |
| 6 | **Stack-Edu** | `HuggingFaceTB/stack-edu` | ~125B, 15 languages | ODC-By 1.0 | Code with an edu-classifier applied; SmolLM2/3's code backbone | SmolLM3 promoted it into stage 2 specifically because raw Stack v2 was underperforming |
| 7 | **Cosmopedia v2** (in SmolLM-Corpus) | `HuggingFaceTB/smollm-corpus`, config `cosmopedia-v2` | 28B (v1: `HuggingFaceTB/cosmopedia`, 25B / 30M docs) | Apache-2.0 | Synthetic textbooks/blogs/stories; cheap MMLU/ARC lift | SmolLM2 used at **4%** of mix; cosmo-1b beat TinyLlama-1.1B on ARC-e/ARC-c/OBQA/MMLU |
| 8 | **MegaMath** | `LLM360/MegaMath` | 371.6B (279B web + 28.1B code + 64.5B synthetic) | ODC-By | Largest open math corpus; `MegaMath-Web-Pro` and `-Synth` are the useful parts | Beaten by Nemotron-CC-Math on MATH/MBPP+, but 7× larger — use for *volume* in the stable phase |
| 9 | **Ultra-FineWeb** | `openbmb/Ultra-FineWeb` | ~1T en + ~120B zh | Apache-2.0 on card; inherits FineWeb ODC-By upstream | Alternative/complement to FineWeb-Edu using verification-based fastText | Vendor extrapolation to 8B/8T: **+14.76 MMLU, +33.09 MATH** vs baseline data — *treat as an upper bound, not a measurement* `[verify]` |
| 10 | **Dolma 3 / Dolmino / Longmino** | `allenai/dolma3` (collection; subset IDs `[verify]`) | 9.3T pool → 5.9T mix; **Dolmino 100B mid-train**; **Longmino 50B long-context** | ODC-By | Best-documented *decontaminated* mix; Dolmino is the reference mid-training recipe | OLMo 2 7B mid-training on 50B Dolmino: **MMLU 59.8→63.7, +10.6 avg across suite** |

### 2.2 Tier A — use selectively

| Dataset | HF ID | Tokens | License | Notes |
|---|---|---:|---|---|
| **FineMath** | `HuggingFaceTB/finemath` (`finemath-3plus` ~34B, `finemath-4plus` ~9.6B, `infiwebmath-3plus` ~20.5B, `infiwebmath-4plus` ~8.5B) `[verify sizes]` | ~50–70B total | ODC-By | Highest precision-per-token math web before Nemotron-CC-Math existed. Still worth blending for source diversity |
| **The Stack v2** | `bigcode/the-stack-v2-dedup`, `-train-full-ids`, `-train-smol-ids` | 900B+ tokens (67.5TB raw) | Per-file OSS licenses; permissive-only in `train-*` subsets; opt-out honored | Requires Software Heritage S3 fetch by ID — **high pipeline cost**. Prefer Stack-Edu + Nemotron-CC-Code |
| **Nemotron-CC-Code / Nemotron-Pretraining-Code v1,v2** | `nvidia/Nemotron-CC-Code-v1`, `nvidia/Nemotron-Pretraining-Code-v1`, `-v2` | large | CC-BY-4.0 + NVIDIA agreement | Common-Crawl code extraction + curated GitHub. Easier to consume than Stack v2 |
| **Nemotron-Pretraining-SFT-v1 / Specialized-v1** | `nvidia/Nemotron-Pretraining-SFT-v1`, `nvidia/Nemotron-Pretraining-Specialized-v1` | — | CC-BY-4.0 (Wiki-Rewrite subset CC-BY-SA-4.0, Scientific-Coding GFDL — **exclude these two if we want a clean copyleft-free build**) | STEM/reasoning instruction data formatted for pretraining. Ideal anneal fuel |
| **Essential-Web v1.0** | `EssentialAI/essential-web-v1.0` | 24T, 23.6B docs, 12-category taxonomy per doc | ODC-By 1.0 | **Best tool for bespoke domain slices.** SQL-filter to build custom STEM (+24.5% rel.), web-code (+14.3%), medical (+8.6%) subsets vs SOTA (2506.14111) |
| **FineWeb2 / FineWeb2-HQ** | `HuggingFaceFW/fineweb-2`, `epfml/FineWeb2-HQ` | ~8TB / 1893 languages; HQ = top decile of ~20 languages | ODC-By | Only if multilingual is a product requirement. See §4.5 — I recommend ≤3% |
| **proof-pile-2** | `EleutherAI/proof-pile-2` | 55B (arXiv + OpenWebMath + AlgebraicStack) | Mixed (arXiv per-paper) | Formal/theorem-adjacent math. Small but complementary |
| **OpenWebMath** | `open-web-math/open-web-math` | 14.7B | ODC-By | Superseded by Nemotron-CC-Math / FineMath but useful as a validation slice |
| **ClimbMix** | `nvidia/ClimbMix` | 400B | NVIDIA license | Output of CLIMB (2504.13161) clustering-based iterative mixture bootstrapping over Nemotron-CC + smollm-corpus. **A pre-optimized mixture — worth benchmarking as a whole-plan baseline** |
| **OpenThoughts3** | `open-thoughts/OpenThoughts3-1.2M` | ~2–3B tokens | Apache-2.0 (verify per-source) | Reasoning traces; SmolLM3 used it for reasoning mid-training |
| **Llama-Nemotron post-training** | `nvidia/Llama-Nemotron-Post-Training-Dataset-v1.1` | — | NVIDIA | Reasoning + instruction. Note Llama-derived → naming obligations (§6.1) |
| **SmolTalk2 / Tulu 3** | `HuggingFaceTB/smoltalk2`, `allenai/tulu-3-sft-mixture` | ~1–2B | Apache / ODC-By | Instruction data for the anneal phase (IFEval driver) |
| **opc-annealing-corpus** | `OpenCoder-LLM/opc-annealing-corpus` | ~ tens of B | MIT-ish (verify) | Purpose-built code anneal set; exactly our use case |

### 2.3 Tier B — know about them, probably skip

| Dataset | HF ID | Tokens | Why deprioritized |
|---|---|---:|---|
| Zyda-2 | `Zyphra/Zyda-2` | 5T | Cross-dedup + filter over FineWeb-Edu/DCLM/Zyda-1/Dolma-CC — a re-blend of things we already take directly (2411.06068). Useful only as a convenience bundle |
| TxT360 | `LLM360/TxT360` (also `IFM/TxT360`) | ~5T unique, upsampled >15T | Excellent dedup work, but its value (global dedup across 99 CC snapshots + curated) is redundant once we take Nemotron-CC + DCLM |
| RedPajama-V2 | `togethercomputer/RedPajama-Data-V2` | ~30T raw w/ quality signals | Unfiltered by design; it ships *signals*, not a curated set. We'd be redoing DCLM's work |
| Dolma v1.7 | `allenai/dolma` | 3T | Superseded by Dolma 3 |
| GneissWeb | IBM release (`ibm-granite/GneissWeb` `[verify]`) | ~10T | Strong (2502.14907) but overlapping; no unique capability for us |
| SmolLM-Corpus `fineweb-edu-dedup` | `HuggingFaceTB/smollm-corpus` | 220B | A dedup'd 220B slice of FineWeb-Edu. **Actually a great convenience default for a 300B run** — see §4 |
| C4 / RefinedWeb / SlimPajama / The Pile | various | — | Strictly dominated on every published ablation |

### 2.4 The 2026 additions worth tracking

- **Nemotron-CC-v2.1** (`nvidia/Nemotron-CC-v2.1`) — incremental refresh of v2.
- **Nemotron 3 Nano** (2512.20848) — MoE hybrid Mamba-Transformer trained on the Nemotron-Pretraining-Dataset family; its data ablations are the closest published analogue to Prophet's architecture class.
- **open-sci-ref-0.01** (2509.09009) — *the* controlled cross-dataset reference study. Its ranking: **Nemotron-CC-HQ > DCLM-baseline > FineWeb-Edu**. This is the single most useful third-party arbitration of §2.1 and should be read in full before locking the mix.
- **Data Mixing for LLM Pretraining: A Survey and Outlook** (2604.16380) — current survey of mixture methods.
- **Scaling Laws for Mixture Pretraining Under Data Constraints** (2605.12715, Apple, May 2026) — 2,000+ runs; *mixtures tolerate substantially higher repetition than single-source training, with generic data acting as an implicit regularizer.* Directly relevant to our epoch-count decision.

---

## 3. Filtering, dedup and synthesis techniques that pay off

### 3.1 Measured deltas (the table to argue from)

| Technique | Reference | Ablation setup | Measured delta |
|---|---|---|---|
| **Edu classifier (Llama-3-70B-labelled, ≥3/5), keeps 8%** | FineWeb-Edu, 2406.17557 | 1.82B / 350B tokens | **MMLU 33→37 (+4.0), ARC 46→57 (+11)**; equals C4/Dolma MMLU with **10× fewer tokens** |
| **fastText OH-2.5+ELI5 classifier, top-10%** | DCLM, 2406.11794 | 7B / 2.6T | **+6.6 MMLU** vs prior best open corpus; **+3.5 CORE** vs reference-data-only |
| **Classifier ensemble + reduced heuristics + synthesis** | Nemotron-CC, 2412.02595 | 8B / 1T | **+5.6 MMLU** (HQ subset) vs DCLM; full set matches DCLM with **4× more unique tokens** |
| **Verification-based fastText (train classifier against a near-converged model's response to candidate data)** | Ultra-FineWeb, 2505.05427 | 1B / 100B; extrapolated to 8B/8T | Gains on MMLU/ARC-C/ARC-E/CSQA/OBQA; vendor extrapolation **+14.76 MMLU** `[verify]`. Cost: fastText inference ≈ **1,000 CPU-hours on 80 cores** vs **~6,000 H100-hours** for an LLM classifier |
| **LLM-annotated 12-facet taxonomy + SQL filters** | Essential-Web, 2506.14111 | — | Custom slices reach **STEM +24.5%, web-code +14.3%, medical +8.6%** relative to SOTA curated sets; math **−8.0%** (i.e. weaker than dedicated math corpora) |
| **Math-specific extraction (LaTeX-preserving HTML → text)** | Nemotron-CC-Math, 2508.15096 | 8B / 300B | **MATH +9.6 / +12.6**, **MBPP+ +4.6 / +14.3** over FineMath-3+/MegaMath-Web |
| **MinHash dedup, per-dump (independent) vs global across all dumps** | FineWeb, 2406.17557 | 1.82B ablations | **Global cross-dump dedup was NET NEGATIVE.** Aggressive global dedup strips frequently-reproduced (often high-quality) content and leaves the unique long tail, which is worse. Dedup *within* a dump; do not dedup *across* dumps |
| **Exact-substring + global fuzzy dedup (Nemotron-CC recipe)** | 2412.02595 | 8B | Enables 4.4T unique real tokens at DCLM-level quality — the counter-example to the above; the difference is that Nemotron-CC pairs it with *classifier ensembling* rather than relying on dedup for quality |
| **Exact/near dedup in general** | 2107.06499, SemDeDup 2303.09540, D4 2308.12284 | — | Dedup reliably reduces memorization and improves tokens-to-loss; SemDeDup reports ~2× efficiency on web data at fixed quality |
| **Synthetic rephrasing of real web (WRAP)** | 2401.16380 | 350M–1.3B on C4 | **~3× pretraining speedup**, **>10% perplexity reduction**, **>2% avg zero-shot** across 13 tasks |
| **≤4 epochs of repetition** | 2305.16264 | ≤9B params, ≤900B tokens | **Negligible loss penalty to 4 epochs**; meaningful gains to ~16; worthless by ~40 |
| **Repetition inside a mixture** | 2605.12715 (2026) | 2,000+ runs | Mixture training tolerates **substantially higher repetition** than single-source; optimal repetition scales predictably with target-data availability; larger models extract more from limited data |
| **LR-decay anneal on upsampled high-quality data** | Llama 3, 2407.21783 | 8B, **40B tokens** (0.27% of run) | **GSM8K +24.0%, MATH +6.4%.** (Negligible at 405B — the effect is *small-model-specific*, i.e. it applies maximally to us) |
| **Mid-training / anneal** | OLMo 2, 2501.00656 | 7B, **50B Dolmino tokens** (1.25% of run) | **MMLU 59.8→63.7 (+3.9); +10.6 average across the eval suite.** 13B: 63.4→67.5 |
| **WSD decay phase as a data-injection window** | MiniCPM, 2404.06395 | 1.2B/2.4B | ~10% decay completes convergence; enables injecting domain/SFT data *only* in decay. Also measured compute-optimal **tokens:params ≈ 192×** (vs Chinchilla 20×) |

### 3.2 Techniques that do NOT pay off for us

- **Perplexity filtering with a small reference LM** (CCNet-style). Cheap, but repeatedly shown weaker than classifier filtering; it selects for *fluent and average*, which is anti-correlated with the knowledge density we need. **Skip** except as a cheap pre-pass to kill garbage.
- **DSIR** (2302.03169, importance resampling toward a target distribution with hashed n-gram features). Elegant and very cheap, but its wins are largest when you have a *narrow* target distribution. Our target is "all of MMLU/GSM8K/HumanEval/IFEval/GPQA," which is broad. **Optional**, as a final re-ranker inside the anneal phase only, targeting the union of eval-adjacent text.
- **Ask-LLM / perplexity-correlation selection** (2402.09668). Works, but the LLM-scoring pass over hundreds of billions of tokens costs thousands of GPU-hours. **We cannot afford to re-score the web.** This is why the correct move is to *consume other people's scored corpora*.
- **Global cross-dump MinHash.** See table — net negative in FineWeb's ablations.

**Rule for Prophet: we never run a model-based classifier over raw Common Crawl. We buy pre-filtered corpora and spend our own CPU only on (a) 13-gram decontamination, (b) a cheap fastText re-rank to take the top slice of an already-filtered corpus, and (c) exact-dedup across the *blend*.**

### 3.3 Synthesis: what to generate vs. what to harvest

Published evidence is unambiguous that synthetic/rephrased text is the highest-value-per-token category available:

- **Nemotron-CC** generates ~1.9T synthetic tokens with four prompt families over *real* seed documents: *Wikipedia-style rephrase* (applied to low-quality docs, to salvage them), *Diverse QA Pairs*, *Distill*, *Extract Knowledge*, *Knowledge List* (applied to high-quality docs, to amplify them). This is the mechanism behind the +5.6 MMLU.
- **Cosmopedia / phi-series**: free-standing "textbook" generation, seeded from curated syllabi (Stanford, Khan Academy, OpenStax, WikiHow — ~20% of prompts) plus 112 topic clusters mined from RefinedWeb (~80%). Phi-4 reportedly used **~400B synthetic tokens across ~50 synthetic data types** `[verify]`.
- **WRAP**: ~3× speedup from style-diverse rephrasing of C4.
- **Persona-Hub** (2406.20094): 1B personas as a diversity axis for synthetic data at scale.
- **Qwen3** (2505.09388): bootstrapped its own corpus — Qwen2.5-VL to OCR PDFs, Qwen2.5-Math/Coder to synthesize domain data. Self-distillation of a *prior generation* is the state of the art for closing domain gaps.

**Cost reality check for generating our own.** Generating 30B synthetic tokens with vLLM: a 4B model on an A100 sustains roughly 5–10k output tok/s at high batch → **~1,200 A100-hours**. That is a quarter of our entire training budget for 10% of our tokens. A 1.7B generator at ~20k tok/s → ~420 A100-hours. Still very expensive.

**→ Decision: harvest, don't generate.** There are already **>2T tokens of released, permissively-licensed synthetic pretraining data**: Nemotron-CC synthetic (~1.9T), MegaMath-Synth (64.5B), Cosmopedia v1+v2 (53B), Nemotron-Pretraining-SFT-v1, MegaMath-Web-Pro. We take them for the price of bandwidth.
**Generate only ~2–5B bespoke tokens** (≈50–150 A100-hours, or a few hundred dollars of batch API) aimed at gaps nothing covers: IFEval-shaped constraint-following documents, GPQA-shaped graduate science QA, and tool/format-following text. Do this **in the anneal phase only**, where per-token leverage is highest.

---

## 4. Recommendation for Prophet

### 4.1 The five decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | **Three-phase WSD schedule with an unusually fat decay: 70% stable / 20% mid-train / 10% anneal.** | Llama 3 got +24% GSM8K from 0.27% of tokens; OLMo 2 got +10.6 avg from 1.25%. The effect is *strongest for small models* (Llama 3 405B saw ~nothing). Prophet is small, so the decay phase is our highest-leverage compute. SmolLM3 used 10% decay; MiniCPM ~10%. We use 10% anneal **plus** a 20% mid-training ramp. |
| **D2** | **Nemotron-CC-v2 is the web backbone, not FineWeb-Edu.** | open-sci-ref-0.01 ranks Nemotron-CC-HQ > DCLM > FineWeb-Edu; Nemotron-CC-HQ is +5.6 MMLU over DCLM at 8B/1T. FineWeb-Edu and DCLM stay in as diversity/decorrelation sources at ~32% combined. |
| **D3** | **~13% of all tokens are synthetic, all of it harvested and all of it seeded on real documents.** | Captures the WRAP/Nemotron effect while staying far from the collapse regime (§6.3). No free-running self-generation. |
| **D4** | **Math and code are over-weighted relative to any published small-model recipe: 12.3% math + 14.4% code = 26.7%.** | GSM8K/MATH/HumanEval/MBPP are 4 of our 9 target benchmarks, and math/code data is the only category with published *double-digit* benchmark deltas per unit of token share (Nemotron-CC-Math: +9.6 to +12.6 MATH). SmolLM3 ended at 13% math / 24% code in its decay phase — we start higher and end higher. |
| **D5** | **Multilingual capped at 2.5%; long-context capped at 1.4%.** | Every one of our nine target benchmarks is English. SmolLM3 could afford 12% multilingual across 11.2T tokens (1.3T multilingual tokens — more than our entire run). At 300B, 12% multilingual would cost 36B tokens ≈ 12% of everything for zero benchmark movement. Long-context is bought late and cheaply via RoPE-theta extension over a small budget. |

### 4.2 Phase A — Stable (210B tokens, 70%)

LR: warmup 1B → constant peak. Purpose: build the world model. Broadest mixture, highest token volume.

| Source | HF dataset ID (+config) | % of phase | Tokens | Epochs over source |
|---|---|---:|---:|---:|
| Nemotron-CC-v2, HQ real-web subset | `nvidia/Nemotron-CC-v2` | 22% | 46.2B | <0.05 |
| FineWeb-Edu (score ≥3) | `HuggingFaceFW/fineweb-edu` — or `HuggingFaceTB/smollm-corpus:fineweb-edu-dedup` for a pre-dedup'd 220B slice | 18% | 37.8B | 0.03 / 0.17 |
| DCLM-baseline | `mlfoundations/dclm-baseline-1.0` | 14% | 29.4B | <0.01 |
| Nemotron-CC synthetic (Diverse-QA + Distill + Extract-Knowledge) | `nvidia/Nemotron-CC-v2` (synthetic partitions) | 8% | 16.8B | ~0.01 |
| Stack-Edu (top ~10 languages) | `HuggingFaceTB/stack-edu` | 9% | 18.9B | 0.15 |
| Code (CC-extracted + curated GitHub) | `nvidia/Nemotron-CC-Code-v1`, `nvidia/Nemotron-Pretraining-Code-v2` | 5% | 10.5B | low |
| Nemotron-CC-Math-4+ | `nvidia/Nemotron-CC-Math-v1` (`4plus`) | 6% | 12.6B | 0.24 |
| MegaMath-Web-Pro + FineMath-4+ | `LLM360/MegaMath`, `HuggingFaceTB/finemath:finemath-4plus` | 3% | 6.3B | ~0.3 |
| Cosmopedia v2 (synthetic textbooks) | `HuggingFaceTB/smollm-corpus:cosmopedia-v2` | 4% | 8.4B | 0.30 |
| Curated reference: Wikipedia, StackExchange, books, science PDFs | `allenai/dolma3` curated subsets `[verify IDs]` | 5% | 10.5B | low |
| arXiv + formal math | `EleutherAI/proof-pile-2` | 3% | 6.3B | 0.11 |
| Multilingual (top 6 languages) | `epfml/FineWeb2-HQ` | 3% | 6.3B | <0.01 |
| **Total** | | **100%** | **210.0B** | |

### 4.3 Phase B — Mid-training (60B tokens, 20%)

LR: hold peak for the first 20B, then begin a slow cosine/linear descent to ~30% of peak. Purpose: capability injection (math, code, reasoning format) while the model still has plasticity. Context extended 4k → 16k in the last 20B of this phase.

| Source | HF dataset ID | % of phase | Tokens |
|---|---|---:|---:|
| Nemotron-CC-Math-4+ (upsampled) | `nvidia/Nemotron-CC-Math-v1:4plus` | 14% | 8.4B |
| MegaMath-Web-Pro + MegaMath-Synth + FineMath-4+ | `LLM360/MegaMath`, `HuggingFaceTB/finemath` | 8% | 4.8B |
| Stack-Edu (edu-score ≥4) + code anneal corpus | `HuggingFaceTB/stack-edu`, `OpenCoder-LLM/opc-annealing-corpus` | 16% | 9.6B |
| Nemotron-CC-v2 HQ real web | `nvidia/Nemotron-CC-v2` | 16% | 9.6B |
| FineWeb-Edu (score ≥4 only) | `HuggingFaceFW/fineweb-edu` | 10% | 6.0B |
| Nemotron-CC synthetic Diverse-QA | `nvidia/Nemotron-CC-v2` | 10% | 6.0B |
| Cosmopedia v2 + bespoke synthetic textbooks | `HuggingFaceTB/smollm-corpus:cosmopedia-v2` + ours | 6% | 3.6B |
| Curated reference (Wikipedia, StackExchange) | `allenai/dolma3` | 5% | 3.0B |
| Instruction data rendered as pretraining documents | `nvidia/Nemotron-Pretraining-SFT-v1`, `HuggingFaceTB/smoltalk2` | 8% | 4.8B |
| Long documents (16k+) | Dolma 3 Longmino-style pool `[verify ID]` + arXiv/books | 5% | 3.0B |
| Multilingual | `epfml/FineWeb2-HQ` | 2% | 1.2B |
| **Total** | | **100%** | **60.0B** |

### 4.4 Phase C — Anneal / decay (30B tokens, 10%)

LR: linear decay to 0. Context extended 16k → 32k in the last 8B. **This is where benchmarks are made.** Zero low-quality web. Run this phase **3× with different data orderings and model-soup the results**, following OLMo 2 — it is nearly free relative to the phase cost and reliably gains ~0.5–1 point.

| Source | HF dataset ID | % of phase | Tokens |
|---|---|---:|---:|
| Reasoning traces (long CoT, math + science + code) | `open-thoughts/OpenThoughts3-1.2M`, `nvidia/Llama-Nemotron-Post-Training-Dataset-v1.1` | 18% | 5.4B |
| Math: Nemotron-CC-Math-4+ + MegaMath-Web-Pro | `nvidia/Nemotron-CC-Math-v1`, `LLM360/MegaMath` | 16% | 4.8B |
| Code: Stack-Edu-Python/top-5 + opc-annealing-corpus | `HuggingFaceTB/stack-edu`, `OpenCoder-LLM/opc-annealing-corpus` | 14% | 4.2B |
| Instruction / constraint-following (IFEval driver) | `HuggingFaceTB/smoltalk2`, `allenai/tulu-3-sft-mixture`, `nvidia/Nemotron-Pretraining-SFT-v1`, **+ our bespoke 2–5B IFEval/GPQA-shaped set** | 14% | 4.2B |
| FineWeb-Edu score ≥4 (top-percentile web only) | `HuggingFaceFW/fineweb-edu` | 12% | 3.6B |
| Nemotron-CC synthetic Diverse-QA | `nvidia/Nemotron-CC-v2` | 10% | 3.0B |
| Curated reference (Wikipedia, StackExchange, textbooks) | `allenai/dolma3` | 6% | 1.8B |
| Cosmopedia v2 / synthetic textbooks | `HuggingFaceTB/smollm-corpus:cosmopedia-v2` | 6% | 1.8B |
| Long-context documents (32k) | Longmino-style pool | 4% | 1.2B |
| **Total** | | **100%** | **30.0B** |

### 4.5 Aggregate over the full 300B run

| Domain | Phase A | Phase B | Phase C | **Total tokens** | **% of 300B** |
|---|---:|---:|---:|---:|---:|
| English web (Nemotron-CC real + FineWeb-Edu + DCLM) | 113.4B | 15.6B | 3.6B | **132.6B** | **44.2%** |
| Code | 29.4B | 9.6B | 4.2B | **43.2B** | **14.4%** |
| Synthetic (harvested rephrase/QA/textbook) | 25.2B | 9.6B | 4.8B | **39.6B** | **13.2%** |
| Math | 18.9B | 13.2B | 4.8B | **36.9B** | **12.3%** |
| Curated reference (wiki/books/arXiv/StackExchange) | 16.8B | 3.0B | 1.8B | **21.6B** | **7.2%** |
| Instruction + reasoning traces | 0 | 4.8B | 9.6B | **14.4B** | **4.8%** |
| Multilingual | 6.3B | 1.2B | 0 | **7.5B** | **2.5%** |
| Long-context | 0 | 3.0B | 1.2B | **4.2B** | **1.4%** |
| **Total** | **210.0B** | **60.0B** | **30.0B** | **300.0B** | **100.0%** |

**Sanity checks.**
- No source is repeated more than ~2 epochs (only OpenThoughts3 approaches 2×), comfortably inside the ≤4-epoch free zone (2305.16264) and far inside the higher tolerance that mixtures permit (2605.12715).
- Instruction/reasoning is 0% in phase A and 32% in phase C — this is the deliberate "recency" exploitation.
- Web share falls 54% → 26% → 12% across phases; math+code rises 23% → 38% → 30% (phase C trades some math/code share for reasoning traces and instruction, which are themselves math/code-dense).

### 4.6 Scaled-down variants (if the budget really is a few hundred A100-hours)

Keep the *percentages* identical; scale the token counts. What changes:

| Total budget | Phase A / B / C | Changes to the mix |
|---|---|---|
| **300B** (4,644 A100-h) | 210 / 60 / 30 | As specified above |
| **150B** (2,322 A100-h) | 100 / 32 / 18 | Drop multilingual to 0. Redistribute to web + math. Reduce long-context to 24k. |
| **100B** (1,548 A100-h) | 62 / 23 / 15 | Drop multilingual and long-context entirely (do context extension post-hoc). Raise anneal to 15%. Consider 2 epochs over a 50B *elite* pool instead of 100B unique (see ablation A3). |
| **30B** (464 A100-h) | 18 / 7 / 5 | **This is a research prototype, not a competitor.** Raise anneal to 17%. Drop Prophet-main; train Prophet-mini (450M dense) at 30B tokens instead — 8.1e19 FLOPs, 160 A100-h — and use the remainder for ablations. Publish honestly as "SmolLM2-135M/360M class." |

### 4.7 Storage and streaming plan

**Data volume.**

| Quantity | Value |
|---|---|
| Unique tokens to materialize (300B + 20% selection headroom) | **360B** |
| Raw text at ~4.3 bytes/token | 1.55 TB uncompressed |
| Transfer volume (parquet + zstd, ~2.4× compression) | **≈ 0.65 TB one-time download** |
| Tokenized store, uint16 (**requires vocab ≤ 65,535**) | **0.72 TB** |
| Tokenized store, uint32 (vocab > 65,535, e.g. 128k) | 1.44 TB |
| Checkpoints (10B total params, bf16 + optimizer, 30 retained) | ~0.6–4 TB depending on optimizer-state policy |

> **Cross-track flag for R01/R02 (tokenizer):** choosing a **≤64k vocabulary halves our data storage and halves dataloader I/O**. At 128k vocab the token store doubles to 1.44 TB. This is a real, quantified cost of a large vocab that should enter the tokenizer decision.

**Streaming, and why bandwidth is a non-issue.**
Training consumes 18,000 tokens/s = **36 KB/s** at uint16. Even at 8× that rate the requirement is under 3 Mbps. **The corpus does not need to be local, and download bandwidth will never bottleneck training.** What *is* a problem: a Colab A100 runtime has only ~166–235 GB of local disk — it cannot hold 720 GB.

**Recommended pipeline:**
1. **Tokenize once, off Colab.** Pre-tokenize into **MosaicML Streaming (MDS)** shards of 64–128 MB, one shard stream per data source, with per-source token counts recorded. Tokenization cost: ~300B tokens ÷ (8 cores × ~1e6 tok/s) ≈ **10–12 CPU-hours** with HF `tokenizers` fast BPE. Do this on a cheap CPU VM, not on the A100.
2. **Host on Cloudflare R2** — $0.015/GB-month and **zero egress fees**. 750 GB ≈ **$11/month, $0 in transfer**. (S3 would charge ~$0.09/GB egress; with a 300B-token run re-reading shards across restarts that could be hundreds of dollars. B2 + Cloudflare Bandwidth Alliance is an equivalent alternative.)
3. **Stream with `mosaicml-streaming`**, `StreamingDataset(streams=[...], local='/content/cache', predownload=8*batch, cache_limit='40gb')`. Set per-`Stream` `proportion=` to the phase mixture percentages — this gives mixture control *at the dataloader*, so phase transitions are a config change, not a re-shard.
4. **Preemption safety.** MDS is deterministically resumable: persist `StreamingDataset.state_dict()` with every checkpoint. Colab kills sessions with no warning; checkpoint every ≤20 minutes to Google Drive or R2, and make the resume path the *default* path so an interrupted run restarts unattended.
5. **Never re-download.** Keep a persistent `/content/drive/…/mds_cache` of the hottest 40 GB (the anneal-phase shards especially, which are re-read 3× for the model soup).

---

## 5. Compute & storage budget

### 5.1 Training compute

| Item | FLOPs | A100-hours @140 TFLOP/s |
|---|---:|---:|
| Phase A — 210B tokens @1.3B active | 1.64e21 | 3,251 |
| Phase B — 60B tokens | 4.68e20 | 929 |
| Phase C — 30B tokens (×3 for the model soup: 3 × 10B, not 3 × 30B) | 2.34e20 | 464 |
| **Prophet-main pretraining total** | **2.34e21** | **4,644** |
| Prophet-mini (450M dense, 300B tokens) | 8.1e20 | 1,607 |
| Data ablations (§7, recommended 12% of the main run) | 2.8e20 | ~560 |
| Bespoke synthetic generation (2–5B tokens, 1.7B generator, vLLM) | — | 50–150 |
| **Grand total (main + ablations + synthesis)** | | **≈ 5,300 A100-hours** |

### 5.2 What that costs, three ways

| Route | Time | Cash |
|---|---|---|
| One Colab A100, 12 h/day | **~440 days** | Colab Pro+ compute units, and a year of calendar. **Not viable.** |
| Rented 8×H100 node (≈396 TFLOP/s/GPU at 40% MFU → 3,168 TFLOP/s) | **~205 GPU-hours = 8.5 days** for the main run; ~11 days with ablations | 8×H100 spot at $18–25/node-hour → **$3,700–6,600** |
| Rented 8×A100-80G node, spot | ~24 days | $8–12/node-hour → **$4,600–6,900** |

**Recommendation:** budget **~$5,000** for a rented multi-GPU node for the 300B run, and keep Colab for ablations, debugging, and evaluation. Alternatively, cut the target to **100–150B tokens** and/or **0.7–0.9B active** and stay on Colab; at 0.8B active × 150B tokens the run is **1,429 A100-hours** — reachable in ~4 months at 12 h/day.

### 5.3 The memory constraint that feeds back into the data plan

A 10B-total-parameter MoE on one 80 GB A100 with AdamW: 2 (bf16 weights) + 4 (fp32 master) + 8 (fp32 m,v) = **14 bytes/param = 140 GB**. It does not fit. Mitigations: 8-bit Adam (→ ~80 GB, still no room for activations), CPU-offloaded optimizer states (Colab A100 high-RAM ≈ 83 GB system RAM — also tight), or reducing total params to **~5–6B**. If total params shrink, active params likely shrink with them, which *reduces* FLOPs/token and *increases* the affordable token count. **This makes the memory decision a data-budget decision** — R01/R02 should be told that every 0.1B reduction in active params buys ~360 A100-hours or ~23B extra tokens.

### 5.4 Storage

| Item | Size | Cost |
|---|---:|---|
| Tokenized MDS shards (uint16, 64k vocab) | 0.72 TB | R2: $11/mo, $0 egress |
| Raw parquet staging (deletable after tokenization) | 0.65 TB | transient |
| Checkpoints (30 × bf16 weights only, 10B params) | 0.6 TB | R2: $9/mo |
| Eval + ablation artifacts | ~50 GB | negligible |
| **Total steady-state** | **≈ 1.4 TB** | **≈ $21/month** |

---

## 6. Risks

### 6.1 License contamination

| Risk | Assessment | Mitigation |
|---|---|---|
| **Copyleft leakage from Nemotron subsets** | `Nemotron-Pretraining-Wiki-Rewrite` is **CC-BY-SA-4.0** and `Nemotron-Pretraining-Scientific-Coding` is **GFDL**. Both are viral in the strict reading. | **Exclude these two subsets by name.** Everything else in the Nemotron pretraining family is CC-BY-4.0 + the NVIDIA Data Access Agreement, which explicitly permits training any model (open or proprietary) and permits publishing benchmarks. |
| **The Stack v2 opt-outs and per-file licenses** | Requires honoring `am-i-in-the-stack` opt-outs and using only permissive-license subsets. Files must be fetched from Software Heritage by ID, not downloaded directly. | Use **Stack-Edu** (ODC-By, pre-resolved) and **Nemotron-CC-Code** instead. Only touch raw Stack v2 if a specific language is missing. |
| **Distillation from restricted teachers** | Llama community licenses impose naming ("Llama" in the derived model name) and attribution obligations that propagate to models trained on Llama-generated data. Gemma Terms of Use propagate a prohibited-use policy. OpenAI/Anthropic/Google API terms forbid training competing models. | **Distill only from Apache-2.0 / MIT teachers: Qwen3 (Apache-2.0), DeepSeek-R1 (MIT), Mistral-Apache models.** Note `nvidia/Llama-Nemotron-Post-Training-Dataset-v1.1` is Llama-derived — if a fully unencumbered release is required, substitute `nvidia/Nemotron-Pretraining-SFT-v1` and Qwen-derived reasoning sets. |
| **ODC-By attribution chain** | FineWeb, FineWeb-Edu, Stack-Edu, Dolma, Zyda-2, Essential-Web are ODC-By 1.0 — attribution required, but no share-alike. | Maintain `docs/DATA_PROVENANCE.md` listing every source, version hash, license, and token count. Ship it with the model card. Cheap insurance. |
| **CommonCrawl-derived copyright exposure generally** | Unresolved industry-wide. Every competitor carries it. | Accept; document; do not train on anything with an explicit non-commercial or research-only clause. |

### 6.2 Benchmark contamination

**This is the risk most likely to silently invalidate the whole project.** FineWeb-Edu is *selected for educational content*, which is exactly the distribution MMLU is drawn from. DCLM's classifier was trained partly on OpenHermes-2.5, which contains benchmark-adjacent instruction data. Neither was decontaminated against our full eval suite by default.

**Mandatory pipeline step (cheap, ~4 CPU-hours):**
1. Build a Bloom filter of all **13-grams** (lowercased, punctuation-stripped, whitespace-normalized) from the *test/validation* splits of: MMLU, MMLU-Pro, GSM8K, MATH, HumanEval, HumanEval+, MBPP, MBPP+, IFEval, GPQA, HellaSwag, ARC-e/c, WinoGrande, OpenBookQA, PIQA, TriviaQA, DROP.
2. Drop any training document containing ≥1 hit. Log the **removal rate per source** — that number is itself a headline finding and belongs in the model card.
3. Hold out **private canary slices** never used in training, so we can detect leakage that the n-gram filter missed.
4. Re-run decontamination **after** the anneal mixture is finalized — anneal data (instruction sets, reasoning traces) is the highest-contamination-risk category by far.

**Also:** report both contaminated and decontaminated eval numbers if the removal rate on any benchmark exceeds ~0.5%.

### 6.3 Model collapse

At **13.2% synthetic**, and with **100% of that synthetic anchored to real seed documents** (rephrasing/QA-extraction, not free generation), we are far from the collapse regime described by Shumailov et al. (2305.17493). Gerstgrasser et al. (2404.01413) show that *accumulating* real + synthetic data — rather than *replacing* real with synthetic — avoids collapse entirely. Nemotron-CC ran ~30% synthetic at 8B/15T without collapse and beat Llama-3.1-8B.

**Guardrails:** (a) never exceed ~25% synthetic in any phase; (b) never generate synthetic text from Prophet's own outputs during pretraining; (c) monitor n-gram entropy and type-token ratio of each synthetic shard against its real seed corpus; (d) keep a synthetic-free validation set to detect distributional narrowing.

### 6.4 Colab bandwidth and preemption

**Bandwidth is not the risk** — training needs 36 KB/s (§4.7). The real risks are:
- **Preemption**: A100 sessions terminate without warning. Mitigate with ≤20-minute checkpoint cadence and deterministic MDS resume, with automatic restart as the default code path.
- **Local disk (166–235 GB)**: cannot hold the corpus. Streaming with a bounded `cache_limit` is mandatory, not optional.
- **Drive rate-limiting**: Google Drive throttles many small reads. **Never stream training data from Drive** — Drive is for checkpoints only.
- **The one-time 0.65 TB download**: at a realistic 50–200 MB/s this is 1–4 hours. Do it on a CPU VM, not inside a precious A100 session.

### 6.5 Mixture risk

Our mixture is chosen from published ablations at *other* people's scales and token budgets. Mixture optima are known to shift with N and D (2507.09404, 2605.12715). The 26.7% math+code share is aggressive and could cost general-knowledge performance. **This is what §7 exists to check** — and specifically ablation A2.

---

## 7. Ablation plan

### 7.1 Methodology (FineWeb protocol, scaled to our budget)

FineWeb's ablations used a **1.82B model on 28B tokens** (~607 A100-hours each) with 350B-token confirmations. SmolLM3 spent **161,280 of 437,760 GPU-hours (37%) on ablations**. We cannot do either. The FineWeb *method* is what transfers:

- **Change exactly one variable.** Identical architecture, tokenizer, optimizer, LR schedule, batch size, seed set, total token count. Only the data mixture varies.
- **Same total tokens per arm**, so results are per-FLOP comparable.
- **Multiple seeds.** 2 seeds minimum; report mean ± range. Small-model benchmark noise is large enough to invent 1-point "effects."
- **Cloze formulation (CF), not multiple-choice (MCF).** SmolLM3's team found models cannot do MCF early in training — MCF stays at chance and destroys the signal. Use CF for everything at this scale.
- **Select benchmarks on four criteria** (SmolLM3 playbook): *monotonic* in training tokens, *low noise*, *above random*, and *rank-consistent* with larger-scale outcomes.

### 7.2 Two-tier design

**Tier A — screening. 150M params (non-embedding), 1.6B tokens, seq 2048, ~4 A100-hours per run.**
At 100 TFLOP/s effective for a small model: `6 × 1.5e8 × 1.6e9 = 1.44e18 FLOPs → 4.0 h`. This is ~0.5× Chinchilla-optimal for 150M — enough to rank mixtures, not enough to move MMLU off chance.

**Tier B — confirmation. 400M params, 8B tokens, ~44 A100-hours per run.** Reserve for the **top-2 mixtures only** plus one baseline. Total ~130 A100-hours.

**Total ablation budget: 24 Tier-A runs (96 h) + 3 Tier-B runs (132 h) + eval overhead ≈ 250–560 A100-hours (5–12% of the main run).**

### 7.3 The exact early-signal eval suite

Run with `lighteval` or `lm-evaluation-harness`, all in **cloze/log-likelihood** form, averaged over the last 3 checkpoints.

**Primary — accuracy metrics with usable signal at 150M:**

| Benchmark | Format | Why |
|---|---|---|
| HellaSwag | CF, 0-shot | Highest-SNR, most monotonic metric at small scale. The workhorse. |
| ARC-Easy | CF, 0-shot | Above random early; sensitive to edu-filtered web quality |
| ARC-Challenge | CF, 0-shot | Noisier but rank-consistent; the FineWeb-Edu classifier moved this +11 |
| PIQA | CF, 0-shot | Physical commonsense; sensitive to *breadth* of web data |
| OpenBookQA | CF, 0-shot | Directly responsive to edu/science content share |
| CommonsenseQA | CF, 0-shot | Complements HellaSwag |
| WinoGrande | CF, 0-shot | Low ceiling at this scale but rank-consistent |
| SciQ | CF, 0-shot | Cheap proxy for science-knowledge share |
| LAMBADA (OpenAI) | accuracy | Long-range coherence; sensitive to sequence packing and doc-length distribution |
| MMLU (**CF, cloze**) | CF, 0-shot | The only way to get above-chance MMLU signal at 150M. **Never use MCF here.** |

**Primary — bits-per-byte (BPB) on held-out domain slices. This is the highest-SNR metric at small scale and should carry the most decision weight.** Paloma-style (2312.10523) per-domain evaluation on 8 slices held out from training:

`MMLU auxiliary-train` · `GSM8K solutions` · `held-out Stack-Edu Python` · `arXiv math (LaTeX)` · `Wikipedia` · `StackExchange` · `held-out FineWeb-Edu score-5` · `held-out multilingual (if the arm includes it)`

**Explicitly excluded at Tier A (all at floor, all pure noise):** MMLU-MCF, MMLU-Pro, GSM8K accuracy, MATH, HumanEval, MBPP, IFEval, GPQA. Introduce GSM8K/HumanEval only at Tier B and only in CF/pass@10 form.

### 7.4 The 8 ablations that actually matter

Ordered by expected information gain per A100-hour.

| ID | Question | Arms | Runs | Decision it settles |
|---|---|---|---:|---|
| **A1** | **Which web backbone?** | (a) 100% Nemotron-CC-v2-HQ (b) 100% FineWeb-Edu (c) 100% DCLM-baseline (d) 45/35/20 blend (e) ClimbMix as-is | 5 Tier-A | Whether to trust open-sci-ref's Nemotron>DCLM>FineWeb-Edu ranking at *our* scale, and whether blending beats the best single source (it usually does — decorrelated errors) |
| **A2** | **How much math+code can we afford?** | web:math:code at (a) 74:13:13 (b) 60:20:20 (c) 47:26:27 [our plan] (d) 35:32:33 | 4 Tier-A | The single riskiest number in §4. Watch for HellaSwag/PIQA regression as math+code rises |
| **A3** | **Quality vs. uniqueness at fixed compute.** *The highest-leverage ablation in this document.* | Fixed 1.6B training tokens drawn from: (a) 1.6B unique from the top-10% pool (b) 0.8B unique from the top-3% pool × 2 epochs (c) 0.4B unique from the top-1% pool × 4 epochs | 3 Tier-A + **1 Tier-B on the winner** | Whether Prophet should train on 300B "good" tokens or 100B "elite" tokens × 3 epochs. Grounded in 2305.16264 and 2605.12715. If (b) or (c) wins, the entire §4 plan is rewritten around a smaller, harder-filtered pool — **run this first** |
| **A4** | **Synthetic share.** | 0% / 8% / 15% / 25% / 40% synthetic (Nemotron-CC-synth + Cosmopedia-v2), real web backfilled | 5 Tier-A | Validates the 13.2% choice and locates the collapse/diminishing-return knee |
| **A5** | **Anneal fraction and anneal mixture.** | Fixed 1.6B total; decay phase = 5% / 10% / 20% of tokens; anneal mix = (i) math/code-heavy (ii) instruction/reasoning-heavy (iii) balanced [§4.4] | 5 Tier-A | The 10% anneal choice and the §4.4 mixture. Expect the largest absolute deltas of any ablation here |
| **A6** | **Multilingual opportunity cost.** | 0% / 3% / 12% FineWeb2-HQ, English backfilled | 2 Tier-A (0% is A2's arm c) | Confirms D5. Measure the English-benchmark cost per point of multilingual share |
| **A7** | **Decontamination impact.** | Identical mixture, with and without 13-gram decontamination | 2 Tier-A | Quantifies how much of our headline number is contamination. **Non-negotiable — run it.** |
| **A8** | **Confirmation at 400M.** | Best mixture from A1–A6 vs. a strong published baseline (SmolLM2's 60/40 FineWeb-Edu/DCLM + 4% Cosmopedia) vs. ClimbMix | 3 Tier-B | Final go/no-go on the §4 plan before committing 4,600 A100-hours |

### 7.5 What we deliberately do NOT ablate

- **Whether dedup helps** — settled (2107.06499). Just do per-dump MinHash + exact-substring, never global cross-dump MinHash.
- **Whether classifier filtering beats heuristics** — settled by FineWeb-Edu, DCLM, Nemotron-CC. Don't re-derive it.
- **DoReMi/RegMix/MixMin/DoGE machinery.** RegMix needs ~512 small runs; DoReMi and DoGE add >10% to base training cost; MixMin needs ~1% but still needs a per-target setup. **With only 8 domains and a strong literature prior, a hand-designed mixture validated by 24 targeted ablations dominates on cost-effectiveness.** If we later want an automated pass, **MixMin (2502.10510)** is the right choice — convex, ~1% of final training compute, and reported to beat RegMix — followed by **Chameleon (2505.24844)** at <2% overhead. Revisit only if A2 shows the mixture surface is unexpectedly sharp.
- **Per-example difficulty curricula.** See §7.6.

### 7.6 On curriculum learning (the honest answer)

Three things get called "curriculum," and they have very different evidence:

1. **Domain scheduling / multi-phase mixtures — STRONG evidence. Adopt.** This is what SmolLM3 (3 stages), OLMo 2/3 (Dolmino, Longmino), MiniCPM (WSD decay), Llama 3 (annealing) and Qwen3 (3 stages) all do, and the measured deltas are the largest in §3.1. §4's three-phase plan *is* our curriculum.
2. **Sequence-length curriculum — adopt, but for throughput not quality.** Everyone extends 4k → 32k → 64k/128k late (SmolLM3: two 50B-token stages with RoPE theta 1.5M then 5M, then YARN to 128k at inference). The benefit is that you don't pay quadratic attention on 99% of tokens. Treat it as a compute optimization; do not expect quality gains from ordering per se.
3. **Per-example difficulty ordering (easy→hard) — WEAK and inconsistent evidence at pretraining scale. Do not implement.** Random order is a strong baseline, and difficulty curricula add pipeline complexity, break dataloader determinism (which we need for preemption resume), and interact badly with data-parallel shuffling. If revisited, the current best method is the re-evaluation-curve approach of 2509.25380. **Not a priority for Prophet.**

**The recency effect is the real prize.** Because gradient updates late in a decayed LR schedule barely move weights but *do* sharpen the output distribution, the last ~10% of tokens is worth several times its weight in benchmark movement: Llama 3 got **+24% GSM8K from the last 0.27%** of its tokens; OLMo 2 got **+10.6 average points from the last 1.25%**. Prophet's 10% anneal phase is a deliberate ~8–40× over-allocation relative to those recipes, on the theory that the effect is strongest for small models (confirmed by Llama 3's 405B seeing ~nothing) — which is precisely our regime.

### 7.7 What the "existence proofs" actually did differently

| Model | The one thing that mattered |
|---|---|
| **SmolLM2 / SmolLM3** | Spent ~37% of total compute on **data ablations**, and built *new* datasets (FineMath, Stack-Edu, SmolTalk) whenever an existing one was too small or too noisy. Manual stage-by-stage mixture refinement keyed off measured performance at the previous stage. 60/40 FineWeb-Edu/DCLM + 4% Cosmopedia-v2 was the base. |
| **MiniCPM** | WSD scheduler + the discovery that compute-optimal **tokens:params ≈ 192×**, not 20×. Treated the decay window as a *data injection port* for domain and SFT data. |
| **Phi-3 / Phi-4** | Synthetic-first: ~400B synthetic tokens across ~50 generation types `[verify]`, with filtered web as *seasoning* rather than substrate. Proved that a 14B model on ~10T mostly-synthetic tokens beats much larger models on reasoning. |
| **Qwen3** | Bootstrapped its corpus with its own previous generation (Qwen2.5-VL for PDF OCR, Qwen2.5-Math/Coder for domain synthesis), then trained 3 stages: ~30T general, ~5T STEM/code-heavy, then long-context. Scale + self-synthesis. |
| **Nemotron-CC / Nemotron 3** | Industrialized rephrasing: 4 prompt families over real seeds, classifier ensembling instead of aggressive heuristics, 4× more usable unique tokens at DCLM quality. |
| **MobileLLM** | The outlier — its wins came from *architecture* (deep-and-thin, embedding sharing, block-wise weight sharing), not data. Relevant to R01/R02, not R06. |

**What Prophet copies:** SmolLM's ablation discipline and multi-stage refinement; MiniCPM's decay-as-injection-port; Nemotron's rephrased-web corpora (harvested, not generated); Phi's conviction that synthetic beats marginal web. **What Prophet cannot copy:** Qwen3's scale and Phi's synthesis budget.

---

## 8. References

Papers (arXiv IDs). Items marked ✓ were confirmed by retrieval during this session; unmarked items are cited from prior knowledge and **should be verified**. arXiv itself was network-blocked, so no PDF was read directly.

**Corpora**
- ✓ 2406.17557 — *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (FineWeb, FineWeb-Edu)
- ✓ 2406.11794 — *DataComp-LM: In search of the next generation of training sets for language models* (DCLM)
- ✓ 2412.02595 — *Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset*
- ✓ 2508.15096 — *Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset* (ICLR 2026)
- ✓ 2506.14111 — *Essential-Web v1.0: 24T tokens of organized web data*
- ✓ 2505.05427 — *Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data*
- ✓ 2411.06068 — *Zyda-2: a 5 Trillion Token High-Quality Dataset*
- ✓ 2504.02807 — *MegaMath: Pushing the Limits of Open Math Corpora*
- ✓ 2502.14907 — *GneissWeb: Preparing High Quality Data for LLMs at Scale*
- ✓ 2409.12568 — *InfiMM-WebMath-40B*
- 2310.06786 — *OpenWebMath*
- 2310.10631 — *Llemma* (proof-pile-2)
- 2402.19173 — *StarCoder 2 and The Stack v2*
- 2506.20920 — *FineWeb2* `[verify ID]`
- 2402.00159 — *Dolma* (v1.x); Dolma 3 / OLMo 3 via `github.com/allenai/dolma3` and the Ai2 OLMo 3 blog

**Filtering, dedup, selection**
- 2107.06499 — *Deduplicating Training Data Makes Language Models Better*
- 2303.09540 — *SemDeDup*
- 2308.12284 — *D4: Improving LLM Pretraining via Document De-Duplication and Diversification*
- 2302.03169 — *DSIR: Data Selection for Language Models via Importance Resampling*
- 2402.09668 — *How to Train Data-Efficient LLMs* (Ask-LLM, Density sampling)
- ✓ 2509.09009 — *open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison* — **the cross-dataset arbitration study; read this one in full**

**Mixtures**
- 2305.10429 — *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*
- 2310.15393 — *DoGE: Domain Reweighting with Generalization Estimation*
- ✓ 2312.02406 — *Efficient Online Data Mixing for Language Model Pre-Training* (ODM)
- 2407.01492 — *RegMix: Data Mixture as Regression for Language Model Pre-training*
- ✓ 2502.10510 — *MixMin: Finding Data Mixtures via Convex Minimization*
- ✓ 2505.24844 — *Chameleon: A Flexible Data-mixing Framework*
- ✓ 2403.16952 — *Data Mixing Laws*
- ✓ 2405.14908 — *BiMix: Bivariate Data Mixing Law for Language Model Pretraining*
- ✓ 2507.09404 — *Scaling Laws for Optimal Data Mixtures* (Apple)
- 2504.13161 — *CLIMB: CLustering-based Iterative Data Mixture Bootstrapping* (ClimbMix) `[verify ID]`
- ✓ 2604.16380 — *Data Mixing for Large Language Models Pretraining: A Survey and Outlook* (2026)
- ✓ 2606.08167 — *Explaining Data Mixing Scaling Laws* (2026)
- ✓ 2606.14971 — *FastMix: Fast Data Mixture Optimization via Gradient Descent* (2026)
- ✓ 2607.01104 — *CausalMix: Data Mixture as Causal Inference for Language Model Training* (2026)

**Repetition and data-constrained scaling**
- ✓ 2305.16264 — *Scaling Data-Constrained Language Models* (Muennighoff et al.) — ≤4 epochs ≈ free; code at `github.com/huggingface/datablations`
- ✓ 2605.12715 — *Scaling Laws for Mixture Pretraining Under Data Constraints* (Sedova, Seto, Schluter, Ablin; Apple, May 2026) — mixtures tolerate far higher repetition
- ✓ 2605.02364 — *InfoLaw: Information Scaling Laws for LLMs with Quality-Weighted Mixture Data and Repetition* (2026)
- 2203.15556 — *Training Compute-Optimal Large Language Models* (Chinchilla)

**Synthesis**
- 2401.16380 — *Rephrasing the Web (WRAP)*
- 2306.11644 / 2309.05463 — *Textbooks Are All You Need* I & II (phi-1, phi-1.5)
- 2404.14219 — *Phi-3 Technical Report*
- 2412.08905 — *Phi-4 Technical Report*
- 2406.20094 — *Scaling Synthetic Data Creation with 1,000,000,000 Personas* (Persona-Hub)
- ✓ 2604.13977 — *How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data* (2026)
- 2508.10975 — *BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining* `[verify ID]`
- 2305.17493 — *The Curse of Recursion / AI models collapse when trained on recursively generated data* (Shumailov et al., Nature 2024)
- 2404.01413 — *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*

**Recipes and reference models**
- ✓ 2502.02737 — *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*
- ✓ 2501.00656 — *2 OLMo 2 Furious*
- 2409.02060 — *OLMoE: Open Mixture-of-Experts Language Models*
- ✓ 2407.21783 — *The Llama 3 Herd of Models*
- ✓ 2404.06395 — *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD)
- 2505.09388 — *Qwen3 Technical Report*
- 2503.19786 — *Gemma 3 Technical Report*
- 2402.14905 — *MobileLLM*
- ✓ 2508.14444 — *NVIDIA Nemotron Nano 2*
- ✓ 2512.20848 — *Nemotron 3 Nano: Open, Efficient MoE Hybrid Mamba-Transformer Model for Agentic Reasoning*
- 2312.10523 — *Paloma: A Benchmark for Evaluating Language Model Fit* (per-domain BPB)
- ✓ 2509.25380 — *Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs*
- 2402.07871 — *Scaling Laws for Fine-Grained Mixture of Experts*

**Non-arXiv (fetched this session)**
- HuggingFace, *SmolLM3: smol, multilingual, long-context reasoner* — `github.com/huggingface/blog/blob/main/smollm3.md` (three-stage mixture percentages in §4 comparisons)
- HuggingFace, *The Smol Training Playbook* — ablation methodology, CF-vs-MCF finding, 1B/45B ablation setup, 161,280 vs 276,480 GPU-hour split
- HuggingFace, *Cosmopedia* — `github.com/huggingface/blog/blob/main/cosmopedia.md`
- HuggingFace, *SmolLM* — `github.com/huggingface/blog/blob/main/smollm.md` (SmolLM-Corpus: 28B + 4B + 220B)
- Ai2, *Olmo 3* blog and `github.com/allenai/dolma3` (9.3T pool → 5.9T mix, 100B Dolmino, 50B Longmino)
- NVIDIA, *Announcing Nemotron-CC* developer blog; `data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/`

---

## Appendix A — Immediate next actions

1. **Read open-sci-ref-0.01 (2509.09009) in full.** It is the cheapest way to validate or overturn D2 before we spend anything.
2. **Run ablation A3 first** (quality-vs-uniqueness, 3 runs, 12 A100-hours). If elite-and-repeated wins, §4 is rewritten around a ~100B-token pool and the whole compute picture improves.
3. **Build the decontamination Bloom filter now**, before any download — it changes what we keep.
4. **Resolve the tokenizer vocab size with R01/R02** (≤64k halves data storage; §4.7).
5. **Escalate the §1.2 budget inconsistency to project leadership.** 300B tokens at 1.3B active is 4,644 A100-hours ≈ $5,000 rented, not "a few hundred A100-hours." A decision is needed: raise the budget, shrink active params, or lower the token target and re-scope the competitive claim.
