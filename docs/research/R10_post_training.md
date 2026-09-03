# R10 — Post-training: SFT, distillation, RLVR and agentic capability on a shoestring

> Track R10 of the Prophet project (`docs/00_PROBLEM_LANDSCAPE.md` §10 — Gravity 5 · Leverage 5 · Risk 3).
> Decision-oriented report. Every number is either sourced or derived; derivations are shown.
>
> **Verification status.** This session had no network access to `huggingface.co`, `arxiv.org`,
> `openreview.net` or `allenai.org` (organisation egress policy). Everything marked **[V]** was
> re-verified in-session from a reachable primary or mirrored source (GitHub raw files, the Qwen3
> technical-report PDF shipped in `QwenLM/Qwen3`, the Gemma 3 report PDF on
> `storage.googleapis.com`, Meta's model card and licence in `meta-llama/llama-models`, the
> canonical Gemma Terms of Use mirrored in `aboutcode-org/scancode-toolkit`, TRL/alignment-handbook
> configs, and a scraped Hugging Face dataset-metadata registry). Items marked **[U]** are from prior
> knowledge and **must be re-verified before being acted on** — per `CLAUDE.md` rule 6
> ("un score non reproduit est marqué comme tel"). No claim in the recipe of §4 depends on a **[U]**
> item alone.

---

## 1. Problem statement

The base model is not the product. For sub-4B models the base→instruct gap is routinely larger than
the gap between two architectures — Qwen3-1.7B goes from **13.4 → 48.3** on AIME'24 purely by
switching from non-thinking to thinking mode inside the *same* post-trained weights **[V]**, and
DeepSeek-R1-Distill-Qwen-1.5B goes from a base Qwen2.5-Math-1.5B that scores near zero on AIME to
**28.9** with nothing but 800k SFT samples **[V]**. Post-training is where our benchmark table is won
or lost.

The constraint is brutal and specific:

| Resource | Prophet | Qwen3-4B (for scale) |
|---|---|---|
| Total training compute | ~300 A100-hours (single Colab A100 80GB, interruptible) | ~8.6e23 FLOPs |
| Post-training share | 20–40 % → **60–120 A100-hours** | Qwen3-8B on-policy distillation alone: **1,800 GPU-h**; its RL alternative: **17,920 GPU-h** **[V]** |
| Post-training token budget (derived, §3.1) | **~2.5–5.0 B tokens** on Prophet-M v1 | SmolLM3-3B: 140 B mid-training + 8 B SFT tokens **[V]** |
| GPU count | 1 | 384 H100 (SmolLM3) **[V]** |

So SmolLM3's published recipe — the single best-documented open recipe at our scale — costs roughly
**5,300 A100-hour-equivalents** in mid-training alone (140e9 tokens × 6 × 3e9 params ÷ 1.1e14
FLOP/s). We have **~2 %** of that. R10 therefore cannot be "run SmolLM3's recipe smaller"; it has to
be an *allocation* argument, exactly like the rest of the project.

Five sub-problems, in decreasing order of leverage:

1. **Which open data**, at what licence risk, and in what mixture (§2.1).
2. **Distillation**: off-policy (cheap, = SFT) vs on-policy logit KD (10× better per GPU-hour at
   Qwen scale, but needs a resident teacher and a **shared vocabulary** — R01's custom 32 k/64 k BPE
   vs Qwen3's 151,936) (§2.2, §3.3, §6.2).
3. **Preference optimisation**: worth ~5 A100-hours, or better spent on more SFT? (§2.3, §3.4).
4. **RLVR on one GPU**: is GRPO physically feasible for a 1.07B-active MoE? (§3.2 — the answer is
   *yes, under a narrow configuration*).
5. **Capability shaping that is nearly free**: hybrid think/no-think, instruction following, tool
   calling, contamination hygiene (§2.5, §4, §7).

**Configuration this report is costed against** (taken from the sibling tracks, 2026-09):

| Artifact | Spec | Source |
|---|---|---|
| Prophet-Dense-1.1B (trunk) | d_model 2048, 24 L, d_ff 5632, ~1.1 B dense | R05 §4.3 Phase 1 |
| **Prophet-M v1** | **5.123 B total / 1.072 B active**, 64 routed experts, top-8, 1 shared | R05 §4.2 |
| Prophet-M v2 (later) | 9.757 B total / 1.075 B active (expert cloning) | R05 §4.2 |
| Prophet-mini | **564 M dense**, d_model 1280, 28 L | R05 §4.4 |
| Tokenizer | custom byte-level BPE, **32,768** (R01) or **64,000** tied (R05) — *the two tracks disagree; R10 only depends on it being ≪ 128 k* | R01 §4.2 / R05 §4.2 |
| Extra heads | 4-way **multi-token prediction** | R01 §4.1 |
| Runtime depth dial | **Prophet-Loop** recurrent core, loop count `r` selectable at inference | R04 §4.1 |

Three Prophet-specific facts change the calculus versus everyone else's recipe. Two are
**advantages** we should exploit deliberately:

- **R02's bounded-state recurrent core makes RL rollouts cheap.** GRPO's dominant cost is
  autoregressive generation, and on a single GPU the binding constraint on rollout concurrency is
  KV-cache memory. A bounded-state core has *O(1)* state per sequence instead of *O(N)*, so we can
  hold far more concurrent rollouts in the same 80 GB. No one else training a transformer at this
  scale gets this.
- **R01's small vocabulary shrinks the logit tensor.** GRPO, DPO and GKD all compute per-token
  log-probabilities over the full vocabulary. For an 8,192-position packed micro-batch the bf16
  logit tensor is **0.54 GB at V=32,768** and **1.05 GB at V=64,000**, against **2.49 GB at Qwen3's
  V=151,936** (fp32: 1.07 / 2.10 / 4.98 GB). That is 2–5 GB of headroom per micro-batch handed
  straight to rollout batch size.

And one fact that is a **serious risk**: R05's MoE. Memory scales with *total* parameters (5.123 B),
not active ones (1.072 B), and at high rollout concurrency essentially every expert is touched per
batch, so generation throughput degrades toward that of a **dense 5 B** model. The measured
consequence is a ~3–4× rollout tax versus the dense trunk (§3.2). A fourth interaction is a
constraint: **4-way MTP heads** mean every post-training loss must be defined over the main head
only (or over all heads with a documented weighting) — decide once, in Stage 0, and keep it fixed.

---

## 1b. TARGET SCORES TABLE

The scores Prophet must beat. **All numbers below are pass@1 / strict-prompt accuracy as reported by
the cited source.** Different sources use different harnesses; the spread is real and is quantified
in the "protocol variance" note below.

### 1b.1 Reasoning ("thinking") mode

| Benchmark | DS-R1-Distill-Qwen-1.5B | Qwen3-1.7B (think) | Qwen3-4B (think) | SmolLM3-3B (think) | Source |
|---|---|---|---|---|---|
| MMLU-Redux | 45.4 | **73.9** | 83.7 | — | Qwen3 TR T19/T17 **[V]** |
| GPQA-Diamond | 33.8 | 40.1 | 55.9 | 41.7 | Qwen3 TR T19/T17 **[V]**; SmolLM3 card **[V]** |
| MATH-500 | 83.9 | 93.4 | 97.0 | — | Qwen3 TR T19/T17 **[V]** |
| AIME'24 | 28.9 | 48.3 | 73.8 | — | Qwen3 TR T19/T17 **[V]** |
| AIME'25 | 22.8 | 36.8 | 65.6 | 36.7 | Qwen3 TR **[V]**; SmolLM3 card **[V]** |
| LiveCodeBench v5 | 13.2 | 33.2 | 54.2 | 30.0 (v4) | Qwen3 TR **[V]**; SmolLM3 card **[V]** |
| IFEval (strict prompt) | 39.9 | 72.5 | 81.9 | 71.2 | Qwen3 TR **[V]**; SmolLM3 card **[V]** |
| Arena-Hard | 4.5 | 43.1 | 76.6 | — | Qwen3 TR T19/T17 **[V]** |
| BFCL v3 | 14.0 | 56.6 | 65.9 | 88.8 (BFCL, diff. harness) | Qwen3 TR **[V]**; SmolLM3 card **[V]** |
| ZebraLogic | 4.9 | 63.2 | 81.0 | — | Qwen3 TR **[V]** |
| Multi-IF | 13.3 | 51.2 | 66.3 | — | Qwen3 TR **[V]** |
| CodeForces (Elo) | 954 | — | 1671 | — | DeepSeek-R1 README / Qwen3 TR **[V]** |

### 1b.2 Non-reasoning ("instruct" / no-think) mode

| Benchmark | Llama-3.2-3B-Inst | Gemma-3-4B-IT | Phi-4-mini (3.8B) | Qwen3-1.7B (no-think) | Qwen3-4B (no-think) | SmolLM3-3B (no-think) |
|---|---|---|---|---|---|---|
| MMLU | 63.4 **[V]** | 58.1 **[V]** | 67.3 **[U]** | — | — | — |
| MMLU-Redux | — | — | 67.9 **[V]** | 64.4 **[V]** | 77.3 **[V]** | — |
| MMLU-Pro | — | **43.6** **[V]** | 52.8 **[U]** | — | — | — |
| GPQA-Diamond | 32.8 **[V]** | 30.8 **[V]** | 25.2 **[V]** | 28.6 **[V]** | 41.7 **[V]** | 35.7 **[V]** |
| GSM8K | 77.7 **[V]** | 89.2 **[V]** | 88.6 **[U]** | — | — | 72.8 (GSM-Plus) **[V]** |
| MATH / MATH-500 | 48.0 (MATH) **[V]** | 75.6 (MATH) **[V]** | 67.6 (MATH-500) **[V]** | 73.0 (MATH-500) **[V]** | 84.8 (MATH-500) **[V]** | — |
| AIME'24 / '25 | — | — | 8.1 / 5.3 **[V]** | 13.4 / 9.8 **[V]** | 25.0 / 19.1 **[V]** | — / 9.3 **[V]** |
| HumanEval | — | 71.3 **[V]** | ~70 **[U]** | — | — | — |
| MBPP | — | 63.2 **[V]** | 65.3 **[U]** | — | — | — |
| LiveCodeBench | — | 12.6 (T6) / 23.0 (T18) **[V]** | 10.4 (v5) **[V]** | 11.6 (v5) **[V]** | 21.3 (v5) **[V]** | 15.2 (v4) **[V]** |
| IFEval | 77.4 **[V]** | **90.2** **[V]** | 68.6 **[V]** | 68.2 **[V]** | 81.2 **[V]** | 76.7 **[V]** |
| Arena-Hard | — | — | 32.8 **[V]** | 36.9 **[V]** | 66.2 **[V]** | — |
| BFCL | 67.0 (v2) **[V]** | — | 31.3 (v3) **[V]** | 52.2 (v3) **[V]** | 57.6 (v3) **[V]** | 92.3 (diff. harness) **[V]** |
| BBH | — | 72.2 **[V]** | — | — | — | — |

**Protocol variance is large and must be respected.** The same Qwen3-1.7B thinking checkpoint is
scored differently by Qwen's harness and Hugging Face's `lighteval`:

| Metric | Qwen3 TR **[V]** | SmolLM3 card (lighteval) **[V]** | Δ |
|---|---|---|---|
| AIME'25 | 36.8 | 30.7 | −6.1 |
| GPQA-Diamond | 40.1 | 39.9 | −0.2 |
| IFEval | 72.5 | 74.2 | +1.7 |
| LiveCodeBench | 33.2 (v5) | 34.4 (v4) | +1.2 |

Consequence for R11: **we must publish our own re-measurement of every competitor under our single
harness**; quoting vendor tables against our own numbers is not a valid comparison. Gemma 3's own
report shows the same thing internally — LiveCodeBench 12.6 in Table 6 vs 23.0 in Table 18 for the
identical 4B IT checkpoint **[V]**.

### 1b.3 The bar has already moved (as of 2026-09)

Existence of the following was confirmed in-session from `unslothai/notebooks` (fine-tuning notebooks
exist for them) **[V]**; their *scores* were **not** verifiable here **[U]**:

- **Qwen3.5 (2B, 4B)** — multimodal, Apache-2.0 expected
- **Qwen3.8 (27B)**
- **Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B)**
- **Nemotron Nano 3 30B-A3B** (NVIDIA Open Model License)

Third-party notes seen in-session put Qwen3.5-4B at MMLU-Pro ≈ 79, LiveCodeBench v6 ≈ 56, GPQA-D ≈ 76
**[U, low confidence — untrusted source]**. If even approximately right, the sub-4B bar is now roughly
**+10 MMLU-Pro / +20 GPQA** above the Qwen3-4B column above.

**Recommendation.** Set Prophet's *committed* target against the §1b.1/§1b.2 table (the models named
in the project brief), and treat Qwen3.5-4B / Gemma-4-E4B as the *stretch* comparison to be measured
in R11 once their numbers are re-verified. Prophet's honest headline claim will be a
**capability-per-active-parameter and per-training-FLOP** claim, not an absolute SOTA claim.

**Prophet's concrete post-training targets** (thinking mode, own harness, Prophet-M v1 = 5.123 B total / 1.072 B active):

| Metric | Floor (must beat) | Target | Rationale |
|---|---|---|---|
| MATH-500 | 83.9 (R1-Distill-1.5B) | ≥ 90 | Reachable by distillation alone (§3.3) |
| AIME'25 | 22.8 | ≥ 33 | Tina hit 43.3 AIME'24 on a 1.5B with $9 of LoRA-RL **[V]** |
| GPQA-Diamond | 33.8 | ≥ 40 | Qwen3-1.7B level; knowledge-bound, hardest for us (§6.3) |
| LiveCodeBench | 13.2 | ≥ 30 | Needs verifiable-code RLVR |
| IFEval | 39.9 | ≥ 75 | Cheapest points on the board (§4, Stage 3) |
| BFCL v3 | 14.0 | ≥ 55 | Pure SFT-data problem (§2.1 tool block) |
| Arena-Hard | 4.5 | ≥ 40 | Preference stage + no-think polish |

---

## 2. State of the art

### 2.1 SFT / distillation data — ranked table

Licences below come from scraped Hugging Face dataset metadata (`Shekswess/open-corpus-registry`,
`data/datasets_all.jsonl`) cross-checked against dataset references in real training repos **[V]**,
except where marked. **A licence string on a card is not a licence audit** — re-check every card at
download time, and note that CC-BY/ODC-BY carry *attribution* obligations we must satisfy in the
model card.

**Tier A — take these (permissive, high quality, directly relevant).**

| # | HF dataset ID | Size | Licence | What it is / why |
|---|---|---|---|---|
| 1 | `open-thoughts/OpenThoughts3-1.2M` | 1.2 M (850k math / 250k code / 100k science) | **apache-2.0** **[V]** | Best open reasoning-trace corpus. Traces from **QwQ-32B** (Apache-2.0) → licence-clean. Trained OpenThinker3-7B to AIME'24 **69.0** / AIME'25 **53.3** / MATH500 **90.0** / LCB **51.7** **[V]** |
| 2 | `nvidia/Llama-Nemotron-Post-Training-Dataset` | 2.2 M math + 500 k code + reasoning | **cc-by-4.0** **[V]** | Use **only the DeepSeek-R1-generated splits** (`reasoning_r1`). SmolLM3 used exactly this + #1 for its 35 B-token mid-training set **[V]**. See §6.1 for the Llama-naming caveat |
| 3 | `HuggingFaceTB/smoltalk2` | ~24 SFT splits + Mid + Preference configs | card says apache-2.0; registry field empty **[V, partial]** | **The single most valuable artifact for us.** It is the *complete, per-split, weighted* dual-mode (think/no-think) mixture of a 3B model that beats Llama-3.2-3B. Full weights reproduced in §2.6 |
| 4 | `allenai/Dolci-Think-SFT` / `-7B` | reasoning traces (OpenThoughts3 + SYNTHETIC-2 + Nemotron + new) | **odc-by** **[V]** | OLMo 3's fully-open think-SFT mixture (Nov 2025). Fully documented provenance — the best "audited" option |
| 5 | `allenai/Dolci-Instruct-SFT` | **2,152,112** samples | **odc-by** **[V]** | OLMo 3 Instruct SFT: code, math, reasoning, IF, tools. `-No-Tools` variant also exists |
| 6 | `open-r1/OpenR1-Math-220k` | 220 k problems / R1 traces | **apache-2.0** **[V]** | R1 (MIT) traces over NuminaMath-1.5 prompts. Decontaminate against AIME'24 (§7) |
| 7 | `open-r1/Mixture-of-Thoughts` | 350 k (math/code/science) | registry empty; card states apache-2.0 **[V, partial]** | Curated blend used to train OpenR1-Distill-7B (AIME'24 **52.7**, MATH500 **89.0**) **[V]** |
| 8 | `allenai/tulu-3-sft-mixture` | **939,343** | **odc-by** **[V]** | Still the best *general* instruction mixture; sub-licences vary per subset |
| 9 | `allenai/tulu-3-sft-personas-instruction-following` | ~30 k | **odc-by** **[V]** | Synthetic verifiable-constraint IF data. In smoltalk2 as `tulu_3_sft_personas_instruction_following_no_think` **[V]**. **This is the IFEval lever** |
| 10 | `nvidia/OpenMathReasoning` | 306 k problems / 3.2 M CoT + 1.7 M TIR + 566 k GenSelect | **cc-by-4.0** **[V]** | AIMO-2 winner. Includes tool-integrated-reasoning traces (calculator/Python) |
| 11 | `nvidia/OpenCodeReasoning` | 735 k Python | **cc-by-4.0** **[V]** | Best open code-reasoning SFT set |
| 12 | `nvidia/Nemotron-Post-Training-Dataset-v1` / `-v2` | math, code, STEM, tool calling (+5 languages in v2) | **cc-by-4.0** **[V]** | Generated with DeepSeek-R1-0528 / Qwen3-235B → licence-clean chain |
| 13 | `Team-ACE/ToolACE` | 11,300 dialogues, 26 k APIs | **apache-2.0** **[V]** | Highest quality-per-row tool data; multi-turn with dependency chains |
| 14 | `NousResearch/hermes-function-calling-v1` | ~11.6 k, 5 configs | **apache-2.0** **[V]** | In smoltalk2 as `hermes_function_calling_v1_no_think` **[V]** |
| 15 | `nvidia/Nemotron-Agentic-v1` | multi-turn agentic tool use | **cc-by-4.0** **[V]** | Dec 2025; decompose-goal / decide-to-call / reason-over-output |
| 16 | `simplescaling/s1K-1.1` | 1 k | apache-2.0 (s1K) **[V]** | Tiny, extremely high-value long-CoT seed. In smoltalk2 as `s1k_1.1_think` **[V]** |
| 17 | `GAIR/LIMO` | 817 | **apache-2.0** **[V]** | "Less is more" reasoning seed |
| 18 | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 50 k | **odc-by** **[V]** | Execution-filtered code instructions |
| 19 | `AI-MO/NuminaMath-1.5` | ~900 k | **apache-2.0** **[V]** | Prompt source for RLVR (answers verifiable) |
| 20 | `glaiveai/glaive-function-calling-v2` | ~113 k | **apache-2.0** **[V]** | Bulk tool data; noisy 2-string format — prefer a cleaned ShareGPT recast |

**Tier B — usable with care.**

| HF dataset ID | Licence | Caveat |
|---|---|---|
| `Salesforce/xlam-function-calling-60k` | registry: **cc-by-4.0** **[V]**; widely also cited as CC-BY-NC-4.0 **[U]** | **Verify the card before commercial release.** In smoltalk2 as `xlam_traces_no_think` **[V]** |
| `open-r1/codeforces-cots` | **cc-by-4.0** **[V]** | 10 k problems / 100 k traces; heavy contamination risk vs LiveCodeBench |
| `teknium/OpenHermes-2.5` | **no licence declared** **[V]** | ~1 M rows, contains GPT-4 outputs → OpenAI-ToS grey zone. SmolLM3 still used it at weight 0.5 **[V]**. **Prophet: exclude from the commercially-released mix**, or use only for ablations |
| `allenai/WildChat-1M` / `-4.8M` | **odc-by** **[V]** | Real user turns with GPT-3.5/4 → same ToS grey zone; excellent prompt source if you regenerate responses yourself |
| `SynthLabsAI/Big-Math-RL-Verified` (+ `open-r1/Big-Math-RL-Verified-Processed`) | **apache-2.0** **[V]** | RLVR prompt pool with solve-rate difficulty bins |
| `zwhe99/DeepMath-103K` | **mit** **[V]** | Hard math RLVR prompts |
| `BytedTsinghua-SIA/DAPO-Math-17k` | **apache-2.0** **[V]** | The DAPO RLVR prompt set |
| `agentica-org/deepscaler-preview-dataset` | **mit** **[V]** | 40 k math RLVR prompts (DeepScaleR) |
| `Skywork/Skywork-OR1-RL-Data` | registry empty **[V]** | Large RLVR pool, licence unclear |
| `argilla/ifeval-like-data` | **other** **[V]** | IFEval-shaped; check overlap with `google/IFEval` |
| `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` | **llama3.3** **[V]** | Triggers the Llama naming clause (§6.1) |

**Tier C — DO NOT USE for a commercially released Prophet (non-commercial licences, verified).**

| HF dataset ID | Licence **[V]** |
|---|---|
| `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | **cc-by-nc-4.0** |
| `ServiceNow-AI/R1-Distill-SFT` | **cc-by-nc-sa-4.0** |
| `facebook/natural_reasoning` | **cc-by-nc-4.0** |
| `HuggingFaceH4/no_robots` | **cc-by-nc-4.0** |
| `Salesforce/APIGen-MT-5k` | **cc-by-nc-4.0** |
| `KodCode/KodCode-V1` | **cc-by-nc-4.0** |
| `MegaScience/MegaScience`, `EricLu/SCP-116K` | **cc-by-nc-sa-4.0** |
| `nickrosh/Evol-Instruct-Code-80k-v1` | **cc-by-nc-sa-4.0** |

This is a *material* finding: **AM-DeepSeek-R1-Distilled-1.4M, named in the brief as a candidate, is
CC-BY-NC-4.0** and is therefore out for anything we intend to release under a permissive licence.

**Preference data.**

| ID | Size | Licence | Note |
|---|---|---|---|
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | ~271 k pairs | **odc-by** **[V]** | Used verbatim by SmolLM3's APO stage (weight 0.5) **[V]** |
| `allenai/Dolci-Think-DPO-7B` | **150,000** pairs | **odc-by** **[V]** | Built with the *Delta-Learning* heuristic |
| `allenai/Dolci-Instruct-DPO` | **260,000** pairs | **odc-by** **[V]** | Includes multi-turn pairs |
| `allenai/ultrafeedback_binarized_cleaned` | ~60 k | **mit** **[V]** | Classic baseline |
| *(self-built)* `chosen = strong open model, rejected = weak open model` | free | — | **The cheapest preference data that exists.** SmolLM3 used Qwen3-32B (chosen) vs Qwen3-0.6B (rejected) **[V]**; AI2 calls the same trick *Delta Learning* **[V]** |

**RLVR prompt sets.**

| ID | Size | Licence | Domain |
|---|---|---|---|
| `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` | ~30 k | **odc-by** **[V]** | Tulu 3's RLVR set: math + verifiable IF constraints |
| `allenai/RLVR-IFeval` | — | **odc-by** **[V]** | IF-only verifiable |
| `allenai/Dolci-Think-RL-7B` | **102,014** prompts | registry empty **[V]** | Math + code + precise IF + general chat |
| `allenai/Dolci-Instruct-RL` | **169,964** prompts | registry empty **[V]** | Same domains, instruct mode |
| `allenai/Dolci-RL-Zero-{Math,Code,IF}-7B` | **13.3 k each** | **odc-by** **[V]** | Clean per-domain ablation sets — ideal for our §7 ablations |
| `nvidia/Nemotron-RL-instruction_following{,-structured_outputs}` | — | registry empty **[V]** | JSON-schema-constrained RLVR — directly maps to a product feature |
| `nvidia/Nemotron-RL-coding-competitive_coding` | — | **cc-by-sa-4.0** **[V]** | Unit-test-verified code |
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | — | **odc-by** **[V]** | Dec 2025 multi-domain blend |

**Ranking, for our budget.** If we can only afford ~4 sources: **OpenThoughts3-1.2M** (reasoning
backbone) → **smoltalk2 SFT config** (dual-mode chat/tools/IF, already weighted) →
**tulu-3-sft-personas-instruction-following + ToolACE + hermes-function-calling** (the IFEval and
BFCL levers, tiny and high-yield) → **Dolci-RL-Zero-{Math,Code,IF}** (RLVR prompts, 40 k total,
clean).

### 2.2 Distillation — algorithms

| Method | Paper | What it costs us | Verdict |
|---|---|---|---|
| **Sequence-level KD** (train on teacher-generated text) | Kim & Rush, `1606.07947` **[U]** | Exactly SFT. No teacher at train time. Offline generation can be *someone else's* (OpenThoughts3, Nemotron) → **free** | **Backbone of our recipe.** |
| **Word/logit-level KD (forward KL)** | Hinton et al. **[U]** | Teacher resident, or precomputed top-K logits on disk | Needs shared tokenizer |
| **MiniLLM (reverse KL)** | `2306.08543` **[U]** | Reverse KL avoids the student covering teacher modes it cannot represent — the right objective at a large capacity gap | Worth an ablation |
| **GKD / on-policy distillation** | `2306.13649` **[U]**; implemented in TRL as `trl.experimental.gkd.GKDTrainer` with `lmbda` (student-data fraction), `seq_kd`, `beta` (JSD interpolation: 0 → forward KL, 1 → reverse KL) **[V]** | Student generates, teacher scores. Structurally identical to GRPO with the verifier replaced by a teacher forward pass **[V]** | **The highest-value lever, if the tokenizer allows it (§6.2)** |
| **Distillation scaling laws** | Apple, `2502.08606` **[U]** | Key claims: student loss follows a broken power law in teacher loss and student size/tokens; there is a **capacity gap** (an *over-strong* teacher makes the student worse); distillation beats supervised training only when the teacher already exists or when the student's token budget is below a crossover | Directly implies **do not pick the biggest teacher** — ablate teacher size (§7 A3) |
| **DistiLLM / DistiLLM-2** | `2402.03898`, `2503.07067` **[U]** | Skew-KL + adaptive off-policy replay; cheaper than pure on-policy | Fallback if GKD is too slow |
| **Cross-tokenizer KD** (ULD `2402.12030`; approximated likelihood matching `2503.20083`) **[U]** | — | The only way to do logit-level KD across different vocabularies | **Load-bearing for Prophet if R01 ships a byte/patch frontend** |

**The decisive published number (verified).** Qwen3 TR Table 21, all starting from the same
off-policy-distilled Qwen3-8B checkpoint **[V]**:

| Method | AIME'24 | AIME'25 | MATH500 | LCB v5 | MMLU-Redux | GPQA-D | **GPU hours** |
|---|---|---|---|---|---|---|---|
| Off-policy distillation (baseline) | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | — |
| + Reinforcement learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | **17,920** |
| + **On-policy distillation** | **74.4 (93.3)** | **65.5 (86.7)** | **97.0** | **60.3** | **88.3** | **63.3** | **1,800** |

*(parentheses = pass@64)*. On-policy distillation is **better on every metric at ~1/10 the GPU
hours**, and it is the only one of the two that raises **pass@64** — i.e. it expands the student's
exploration space, whereas RL only sharpens what is already there **[V]**. Qwen's own text: "distillation
achieves significantly better performance than reinforcement learning while requiring approximately
only 1/10 of the GPU hours" **[V]**.

Corroborating (weaker) evidence: Thinking Machines' *On-Policy Distillation* report (Oct 2025) claims
~9–30× cost reduction vs RL for the same score on a Qwen3-8B student **[U — could not verify,
`thinkingmachines.ai` unreachable]**.

**Conclusion: distillation, not RL, is where our GPU-hours go.** RL is a finishing polish.

### 2.3 Preference optimisation — variants

All the following are available as a one-line `loss_type` in TRL's `DPOTrainer` **[V]**:

| `loss_type` | Method | Paper | Note |
|---|---|---|---|
| `sigmoid` (default) | DPO | `2305.18290` | Bradley-Terry logistic |
| `ipo` | IPO | `2310.12036` | Identity transform; less overfitting to the BT model |
| `hinge` | RSO/SLiC | `2309.06657` | β = 1/margin |
| `robust` | Robust DPO | `2403.00409` | Label-flip noise model via `label_smoothing ∈ [0,0.5)` |
| `exo_pair` | EXO | `2402.00856` | Reverse-KL preference optimisation |
| `nca_pair` | NCA | `2402.05369` | Absolute, not relative, likelihood |
| `bco_pair` | BCO | `2404.04656` | Binary classifier reward |
| `sppo_hard` | SPPO | `2405.00675` | Iterative Nash |
| `aot` / `aot_unpaired` | AOT | `2406.05882` | Optimal-transport stochastic dominance |
| **`apo_zero` / `apo_down`** | **APO** | **`2408.06266`** | **Anchored.** `apo_zero` boosts winners + downweights losers (use when the model is *worse* than the chosen responses — i.e. our case) |
| `discopop` | DiscoPOP | `2406.08414` | LLM-discovered loss |
| `sigmoid_norm` | **SimPO's length normalisation** | `2405.14734` | Divides by non-mask token count — the direct fix for DPO length bias |
| combine, e.g. `["sigmoid","bco_pair","sft"]` with `loss_weights` | MPO | — | Multi-objective |

Not in that table but relevant: **KTO** (`2402.01306`, unpaired, prospect-theoretic), **ORPO**
(`2403.07691`, reference-free, folds SFT and preference into one stage — attractive when GPU-hours
are scarce), **CPO** (`2401.08417`) **[all U]**.

**What the strong small-model recipes actually chose (verified):**

- **Tulu 3** (`2411.15124`): SFT → **length-normalised DPO** (`beta: 5`, lr 5e-7, 1 epoch, bsz 128)
  → RLVR (lr 3e-7, reward-model multiplier 0.0, i.e. purely verifiable) **[V, from
  `allenai/open-instruct/docs/tulu3.md`]**.
- **SmolLM3-3B**: SFT (lr 2e-5, 5 epochs, `max_length 65536`, FFD packing, `assistant_only_loss`,
  `max_grad_norm 0.2`, cosine-with-min-lr 0.1, warmup 0.03, Liger) → **APO-zero** (`loss_type:
  apo_zero`, `beta: 0.05`, lr 1e-6, 1 epoch, `max_length 24576`, `padding_free`, warmup 0.1) → **no
  RL at all** → MergeKit model soup **[V, verbatim from `huggingface/alignment-handbook/recipes/smollm3/`]**.
- **OLMo 3**: SFT → DPO (Delta-Learning pairs) → RLVR via `open_instruct/grpo_fast.py` **[V]**.

**Length bias** is the classic DPO pathology (`2310.03716`, `2403.19159` **[U]**): DPO reliably
inflates response length, which inflates LLM-judge scores (Arena-Hard / AlpacaEval) without real
quality gain. Fixes: `sigmoid_norm` (SimPO), length-normalised DPO (Tulu 3's `beta: 5` is on the
length-normalised objective), or length-controlled AlpacaEval (`2404.04475` **[U]**) for evaluation.
For Prophet, length inflation is not merely a scoring artifact — it is a **direct inference-cost
regression** on an iPhone. Length must be an explicit, monitored metric at every stage.

### 2.4 RLVR — algorithms

| Algorithm | Paper | Core idea | Cost signature |
|---|---|---|---|
| **GRPO** | DeepSeekMath `2402.03300` | Critic-free; advantage = group-normalised reward over G samples | No value net (frees ~50 % of training memory **[V]**); needs G = 8–32 |
| **Dr. GRPO** | `2503.20783` (Sea AI Lab) | Removes GRPO's 1/\|o_i\| length normalisation *and* the reward-std division → removes **length bias** (which otherwise makes wrong answers longer) and **difficulty bias**. Recipe cost: **27 h × 8 A100 = 216 A100-hours** for Qwen2.5-Math-7B **[V]** | Same as GRPO, strictly better-behaved |
| **DAPO** | `2503.14476` (ByteDance/Tsinghua) | 4 tricks: **Clip-Higher** (ε_low 0.2 / ε_high 0.28 **[V]**), **Dynamic Sampling** (drop groups with accuracy 0 or 1 — they carry zero advantage), **token-level** policy-gradient loss, **overlong reward shaping** (soft length penalty + overlong filtering). Qwen2.5-32B → **50 on AIME'24** with half the steps of DeepSeek-R1-Zero-Qwen-32B **[V]** | Dynamic sampling raises generation cost per *effective* sample |
| **GSPO** | `2507.18071` (Qwen) **[U]** | Sequence-level (length-normalised) importance ratio instead of token-level → stabilises **MoE** RL without routing-replay hacks | **Directly relevant to R05.** Now mainstream: Unsloth ships a "Qwen3.5 GSPO" notebook **[V]** |
| **VAPO** | `2504.05118` **[U]** | Value-model-augmented PPO; 60.4 AIME'24 on Qwen2.5-32B | Needs a critic → 2× memory. **Out for us** |
| **RLOO** | `2402.14740` **[U]** | REINFORCE leave-one-out baseline | Cheaper than PPO, comparable to GRPO |
| **REINFORCE++** | `2501.03262` **[U]** | Global-batch advantage normalisation | Simple, stable |
| **CISPO** | MiniMax-M1 `2506.13585` **[U]** | Clipped IS-weight; keeps low-probability but high-signal tokens | Interesting for long-CoT |

**Implementations.** `verl` supports PPO / GRPO / **GSPO** / ReMax / REINFORCE++ / RLOO / PRIME /
**DAPO** / **Dr.GRPO** / KL_Cov & Clip_Cov **[V]**, but is Ray-based and documented for "hundreds of
GPUs" **[V]**. **TRL** `GRPOTrainer` covers `loss_type ∈ {grpo, dapo, dr_grpo, sapo}` with
`beta` (KL weight) defaulting to **0.0** — i.e. *no reference model by default* — and vLLM in
`colocate` mode with `vllm_enable_sleep_mode`, `vllm_importance_sampling_correction=True` **[V]**.
`open-instruct` has `grpo_fast.py` **[V]**. **Unsloth** advertises 2× speed / 70–80 % less memory for
GRPO and "7× longer context RL vs. all other setups" **[V]**.

**LoRA-based RL is the key affordability result.** *Tina: Tiny Reasoning Models via LoRA*
(`2504.15777`): LoRA RL on **DeepSeek-R1-Distill-Qwen-1.5B** reaches **43.33 % pass@1 on AIME'24**
(base: 28.9 **[V]**) and **>20 % improvement** on general reasoning; **the best checkpoint is
reproducible for $9, and all experiments from scratch cost $526** **[V]**. Compare DeepScaleR-1.5B
(same base, full-parameter GRPO with an 8K→16K→24K context schedule) reaching 43.1 % AIME'24 **[V]**
at a published cost of thousands of A100-hours **[U]**. **Same score, ~3 orders of magnitude apart in
cost.** That single comparison determines our RL configuration.

### 2.5 Hybrid reasoning, IF, tools, long context

**Hybrid think/no-think.** Both Qwen3 and SmolLM3 converged on the identical design **[V]**:

- `/think` and `/no_think` flags in the **system message** (SmolLM3) or user turn (Qwen3).
- In no-think mode the assistant response is **pre-filled with an empty `<think>\n\n</think>` block**
  — this keeps the internal format identical in both modes and lets the *deployer* force no-think by
  concatenating the empty block into the template.
- Qwen3 additionally shows that **thinking-budget control emerges for free**: once the model handles
  both modes it can be interrupted mid-thought with a fixed stop-thinking string and will produce a
  final answer from partial reasoning. "This ability is not explicitly trained but emerges naturally
  as a result of applying Thinking Mode Fusion" **[V]**.
- **Cost of fusion, measured** (Qwen3 TR Table 22, Qwen3-32B) **[V]**: Stage-3 fusion moves AIME'24
  **−1.9**, LCB v5 **−1.2**, MMLU-Redux **−0.4** in thinking mode, while gaining CounterFactQA
  **+10.9**, LengthCtrl **+8.0**, IFEval **+5.4**, ToolUse **+7.1**, and creating mode-switch
  reliability (ThinkFollow 88.7). Stage-4 general RL then adds IFEval **+6.6**, Multi-IF **+8.4**,
  ToolUse **+15.1**, ThinkFollow **→ 98.9**, at a further AIME'24 **−0.5**. Qwen explicitly chose to
  "accept this performance trade-off to enhance the model's overall versatility" **[V]**.

That table is our single best guide to what a general-RL stage actually buys: **not maths — instruction
following, tool use, mode discipline and hallucination refusal.**

**Instruction following.** Two cheap, verified levers: (a) SFT on
`allenai/tulu-3-sft-personas-instruction-following` (~30 k, ODC-BY) **[V]**; (b) RLVR on
`allenai/RLVR-IFeval` / the IF half of `RLVR-GSM-MATH-IF-Mixed-Constraints` **[V]** — IF constraints
("write exactly 3 bullet points", "no commas") are *programmatically verifiable*, so they are the
cheapest possible reward function. Qwen3 credits rule-based rewards for IF explicitly, citing Tulu 3
**[V]**.

**Tool calling.** SmolLM3's chat template carries **two separate tool sections — XML Tools and Python
Tools** — and the authors state this categorisation "proved beneficial in our experiments for the
model's accurate interpretation of tool definitions in each format" **[V]**. Its SFT mixture includes
`xlam_traces_no_think`, `hermes_function_calling_v1_no_think`, and `smolagents_toolcalling_traces_think`
**[V]**. Result: BFCL 92.3 (no-think) / 88.8 (think) on their harness **[V]**.

**Long context.** SmolLM3 reports that **reasoning mid-training degrades RULER**, and that a
**linear merge of 0.9 × (APO model soup) + 0.1 × (mid-training checkpoint)** recovers the base
model's RULER score up to 128k **[V]**. Their APO data was capped at 24 k tokens because "the vast
majority of our reasoning dataset fell below this length" **[V]**. Long-context SFT sources in
smoltalk2: `LongAlign_64k_context_lang_annotated_lang_6_no_think` and
`LongAlign_64k_Qwen3_32B_yarn_131k_think` **[V]**.

### 2.6 The reference recipe, verbatim (SmolLM3-3B) **[V]**

Reproduced because it is the closest published analogue to Prophet's target and because it is the
only fully-specified one we can copy hyperparameters from.

- **Mid-training**: `smoltalk2` config `Mid`, exactly two splits at weight 1.0 each —
  `Llama_Nemotron_Post_Training_Dataset_reasoning_r1` and `OpenThoughts3_1.2M`. 35 B unique tokens,
  **4 epochs ≈ 140 B tokens**, ChatML, wrapped packing, `max_length 32768`, lr 2e-5,
  cosine-with-min-lr 0.1, warmup 0.03, `max_grad_norm 0.2`, Liger, 8 nodes GBS 128.
- **SFT**: `smoltalk2` config `SFT`, **24 weighted splits**, `max_length 65536`, **5 epochs**,
  `packing_strategy: ffd`, `assistant_only_loss: true`, same optimiser settings. Blog states the
  resulting mixture is **1.8 B tokens (1.0 B no-think + 0.8 B think), 12 non-reasoning + 10
  reasoning datasets, 4 epochs ≈ 8 B tokens**, loss masked on user turns *and tool results*.
  Notable weights: everyday-conversations 1.0, systemchats 1.0,
  `tulu_3_sft_personas_instruction_following` 1.0, `hermes_function_calling_v1` 1.0,
  `xlam_traces` 1.0, `smolagents_toolcalling_traces_think` 1.0, `s1k_1.1_think` 1.0,
  `Mixture_of_Thoughts_science` 1.0, `LongAlign_64k*` 1.0, `smol_magpie_ultra` **0.5**,
  `OpenHermes_2.5` **0.5**, `OpenThoughts3_1.2M_no_think` **0.4**,
  `smoltalk_multilingual8_Qwen3_32B_think` **0.3**, `OpenThoughts3_1.2M_think` **0.02**.
- **Alignment**: `apo_zero`, `beta 0.05`, lr 1e-6, 1 epoch, `max_length 24576`, `padding_free`,
  warmup 0.1, on `llama_3.1_tulu_3_8b_preference_mixture_no_think` (0.5) +
  `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think` (0.25).
- **Merge**: MergeKit soup of APO checkpoints, then linear 0.9/0.1 with the mid-training checkpoint.

The `OpenThoughts3_1.2M_think` weight of **0.02** against `..._no_think` at **0.4** is worth
internalising: at 3B scale, *raw long-CoT volume is not the binding constraint* — mixture balance is.

---

## 3. What actually transfers to our budget

### 3.1 The token budget is the real constraint

Achieved training throughput on one A100 80GB (SXM, 2,039 GB/s, 312 TFLOPS bf16 dense, **no FP8, no
FA3** per `CLAUDE.md`), with sequence packing, FlashAttention-2, Liger kernels and gradient
checkpointing:

| Model | Assumed MFU | Effective TFLOPS | tok/s (6·N_active FLOPs/tok) | **M tok / A100-h** |
|---|---|---|---|---|
| Prophet-Dense-1.1B trunk | 0.35 | 109 | 16,500 | **60** |
| **Prophet-M v1** (5.123 B total / 1.072 B active) | 0.24 | 75 | 11,600 | **42** |
| Prophet-mini, 564 M dense | 0.30 | 94 | 27,700 | **100** |

Therefore:

| Post-training budget | Dense-1.1B trunk | **Prophet-M v1** | mini 564 M |
|---|---|---|---|
| 60 A100-h | 3.6 B tok | **2.5 B tok** | 6.0 B tok |
| 95 A100-h | 5.7 B tok | **4.0 B tok** | 9.5 B tok |
| 120 A100-h | 7.2 B tok | **5.0 B tok** | 12.0 B tok |

**This is the single most important number in the report.** SmolLM3 spent 148 B post-training tokens.
We can spend **~4 B tokens on Prophet-M v1** — about **2.7 %**. Every mixture decision below follows from that ratio. In
particular: a 140 B-token reasoning mid-training is not merely expensive, it is **35× our entire
post-training budget**. We must get reasoning capability from *fewer, better* tokens
(OpenThoughts3-quality traces, s1K/LIMO-style curation) and from **on-policy distillation**, which
Qwen measured at 10× the sample-efficiency of RL **[V]**.

### 3.2 Is GRPO feasible on ONE A100 80GB for a 1.07B-active MoE? — VERDICT

**Memory.** Derived (bytes = params × dtype width):

| Configuration | Weights | Grads | Optimiser | Sub-total |
|---|---|---|---|---|
| Dense-1.1B trunk, bf16 + fp32 AdamW | 2.2 | 4.4 | 8.8 | **15.4 GB** |
| Dense-1.1B trunk, pure-bf16 + 8-bit AdamW | 2.2 | 2.2 | 2.2 | **6.6 GB** |
| mini 564 M, bf16 + fp32 AdamW | 1.1 | 2.3 | 4.5 | **7.9 GB** |
| **Prophet-M v1 (5.123 B), bf16 + fp32 AdamW** | 10.2 | 20.5 | 41.0 | **71.7 GB — leaves ~8 GB: NO** |
| **Prophet-M v1, pure-bf16 + 8-bit AdamW** | 10.2 | 10.2 | 10.2 | **30.7 GB — FITS** |
| **Prophet-M v1, LoRA r=32 on frozen bf16 base** | 10.2 | 0.08 | 0.5 | **10.8 GB — comfortable** |
| Prophet-M v2 (9.757 B), pure-bf16 + 8-bit AdamW | 19.5 | 19.5 | 19.5 | **58.5 GB — fragile** |
| Prophet-M v2, LoRA r=32 | 19.5 | 0.16 | 1.0 | **20.7 GB** |

*(This is a material improvement over a hypothetical 10 B-total v1: at 5.123 B, **full-parameter**
post-training of the MoE is feasible with an 8-bit optimiser — R05's decision to size v1 at 5.1 B
rather than 10 B is what makes R10's recipe possible on one GPU. Do not grow to v2 before
post-training is done.)*

On top of that, GRPO needs:

- **Reference policy**: **0 GB** — TRL's `beta` defaults to **0.0** **[V]**, and DAPO/Dr.GRPO drop the
  KL term entirely. With LoRA, the reference *is* the base with adapters disabled → free either way.
- **Colocated vLLM engine**: a second weight copy (2.6 GB dense / 20 GB MoE bf16 / 5.6 GB NF4) plus
  KV cache. Budget via `vllm_gpu_memory_utilization`; TRL recommends 0.3–0.4 for large N **[V]**, and
  `vllm_enable_sleep_mode=True` offloads vLLM params during the optimiser step **[V]**.
- **Logit tensor**: `packed_len × vocab × 2 bytes` per micro-batch, ×2–3 for backward. At 8,192
  positions: **0.54 GB (V=32,768)** / **1.05 GB (V=64,000)** / 2.49 GB (V=151,936). **R01's small
  vocabulary is a direct RL enabler** — and with 4-way MTP, budget the main head only unless the
  auxiliary heads are deliberately included in the RL loss.

**Verdict on memory: FEASIBLE.**
- Dense-1.1B trunk, full-parameter GRPO: ~15.4 (train) + ~20 (vLLM at util 0.25) + ~8 (activations
  + logits at V<=64k) = **~44 GB of 80 GB**. Comfortable.
- **Prophet-M v1 (5.123 B), full-parameter GRPO with 8-bit AdamW**: 30.7 (train) + 10.2 (vLLM weight
  copy) + ~14 (KV + activations) = **~55 GB**. Feasible, but with no margin for a bad batch - prefer
  **LoRA r=32 (10.8 GB train)** for the RL stages and keep full-parameter for the SFT stages.
- Prophet-M **v2** (9.757 B): **LoRA only.**

**Time.** Generation dominates. Anchor: vLLM on one H100 80GB bf16 gives ~6,300 output tok/s
aggregate for a 7B and ~1,200 tok/s for a 32B **[V]**. Bandwidth-scaling to A100 (×0.61) and
parameter-scaling gives **~11,000 tok/s** for the 1.1 B dense trunk (planning figure; realistic range
6,000–16,000). For **Prophet-M v1**, at high rollout concurrency essentially all 64 routed experts
are read per batch, so throughput tracks a **dense 5.1 B** model → ~5,300 tok/s bandwidth-bound,
derated for Ampere MoE kernels (no Hopper grouped-GEMM path) to **3,000–4,000 tok/s**: **a ~3× rollout
tax versus the trunk**. R02's bounded-state core removes the KV-cache ceiling on concurrency, so this
tax is expert-bandwidth-bound rather than memory-bound — which is the better of the two failure modes,
because it improves with better kernels (R07) rather than requiring more VRAM.

Derived GRPO step time at `P = 32` prompts × `G = 8` samples × `L` completion tokens (prompt 512 tok;
vLLM logprobs reused, so no separate old-policy forward pass):

**Dense-1.1B trunk** (generation ≈ 11,000 tok/s):

| L | Tokens/step | **Step** | **300 steps** |
|---|---|---|---|
| 1,024 | 0.26 M | **0.8 min** | **4.0 h** |
| 2,048 | 0.52 M | **1.5 min** | **7.3 h** |
| **4,096** | **1.05 M** | **2.8 min** | **13.9 h** |

**Prophet-M v1** (5.123 B resident → generation ≈ 3,000–4,000 tok/s):

| L | gen @3k | gen @4k | **Step (3k / 4k)** | **300 steps (3k / 4k)** |
|---|---|---|---|---|
| 1,024 | 87 s | 66 s | **2.0 / 1.7 min** | **10.1 / 8.3 h** |
| **2,048** | 175 s | 131 s | **3.9 / 3.1 min** | **19.3 / 15.6 h** |
| 4,096 | 350 s | 262 s | **7.5 / 6.1 min** | **37.6 / 30.3 h** |

Scaling the last row: `P=64, G=16, L=8192` on Prophet-M v1 is **~8× that** → **240–300 A100-hours per
300 steps**, i.e. the entire project budget for one RL run.

**VERDICT: GRPO on one A100 80GB is FEASIBLE — but only inside a narrow box.**

> **Feasible:** Prophet-M v1 (1.072 B active) · `max_completion_length ≤ 2048` · `G = 8` ·
> `≤ 32 prompts/step` · `beta = 0` (no reference model) · `loss_type = dr_grpo` or `dapo` ·
> LoRA r=32 · vLLM colocate + sleep mode · **≈ 16–19 A100-hours per 300 steps**.
> On the Dense-1.1B trunk the same configuration costs **7 h**, and `L = 4096` costs **14 h**.
>
> **Infeasible:** long-CoT RL (8 k–32 k completions) at **240–300 A100-hours per 300 steps** — the
> entire project budget for one run. Prophet-M **v2** (9.757 B) with a full-parameter optimiser.
> Any critic-based method (VAPO/PPO) — the value net doubles training memory **[V]**.
>
> **Direct consequence for the ordering of the program:** the dense trunk is **2.4× cheaper per RL
> step** than Prophet-M v1. Every RL *ablation* (§7 A5–A7, A9) should be run on the trunk or the
> mini, and only the final, chosen configuration re-run on the MoE.

**Cross-checks against published runs:**

| Reference | What | Cost | Scaled to ~1.1 B |
|---|---|---|---|
| Dr.GRPO / Oat-Zero **[V]** | Qwen2.5-Math-7B R1-Zero recipe | **27 h × 8 A100 = 216 A100-h** | ÷6.5 params → **~33 A100-h** |
| Tina **[V]** | LoRA RL on R1-Distill-Qwen-1.5B → **43.33 AIME'24** | **$9** best run; **$526** for *all* experiments | already at our scale |
| DeepScaleR-1.5B **[V]** | full GRPO, 8K→16K→24K schedule → 43.1 AIME'24 | thousands of A100-h **[U]** | **10× our whole budget** |
| Qwen3 Reasoning RL **[V]** | only **3,995 query-verifier pairs**; **170 steps** took Qwen3-235B AIME'24 **70.1 → 85.1** | — | **RLVR needs very few prompts, not many** |

That last row is the most actionable RLVR fact in the whole report: **Qwen used 3,995 prompts and 170
steps.** Our 300-step / 16-hour box is not a crippled version of the real thing — it is roughly the
*same* number of steps the frontier labs use. What we lack is rollout length, not step count.

### 3.3 Distillation: what fits

| Variant | Derived cost (Prophet-M v1 student) | Fits? |
|---|---|---|
| **Off-policy / sequence-KD** (SFT on someone else's teacher traces) | = SFT: **50 M tok / A100-h** | **Yes — this is the backbone.** Teacher inference is free because OpenThoughts/Nemotron already paid for it |
| **On-policy GKD**, teacher = Qwen3-4B (NF4), 150 M student tokens | generation 11.9 h (@3.5 k tok/s) + student bwd 3.6 h + teacher fwd ~3 h = **~18 h** | **Yes**, for ~150 M tokens |
| On-policy GKD, teacher = Qwen3-8B (NF4), 150 M tokens | 11.9 + 3.6 + ~6 = **~22 h** | Yes, but the teacher forward costs more than the student backward |
| On-policy GKD, teacher = 32 B | 32 B bf16 = 64 GB alone | **No.** Only via NF4 (≈18 GB) at heavy throughput cost, or offline top-K logits |
| Offline top-K teacher logits on disk | top-8 over 500 M tokens ≈ **24 GB**; top-64 over 1 B tokens ≈ **384 GB** | top-8/top-16 caching is viable; full-vocab caching is not. Note this only helps *after* the cross-vocabulary problem is solved (§6.2) |

**Teacher-size choice must be ablated, not assumed** — the distillation scaling law predicts a
**capacity gap** where a stronger teacher yields a *worse* student **[U, 2502.08606]**. Prophet's
student is ~1.07 B active; the safe teacher band is likely **4–8 B**, not 32 B.

### 3.4 Preference optimisation: is it worth the compute?

Derived cost of APO/DPO on 60 k pairs × ~1,500 tokens (chosen + rejected = 180 M tokens):

- policy fwd+bwd on both branches: 6·N·T ⇒ **3.6 h**
- reference log-probs, **precomputed once** (`precompute_ref_log_probs=True`): 2·N·T ⇒ **1.2 h**
- **Total ≈ 5 A100-hours.**

Five hours out of ninety, for a stage that SmolLM3 credits with "improvements across mathematics,
science, instruction following, coding, chat, and multilingual tasks" **[V]** and that Tulu 3 keeps
between SFT and RLVR **[V]**. **Yes — it is worth it**, and it is the best hours-to-points ratio in
the pipeline *provided the pairs are free*. They can be: the Delta-Learning / SmolLM3 heuristic
(chosen = strong open model, rejected = weak open model) costs only inference **[V]**, and
`allenai/Dolci-Think-DPO-7B` (150 k pairs, ODC-BY) already exists **[V]**.

Note the alternative allocation: 5 A100-h of extra SFT = 250 M more tokens ≈ +6 % on the SFT budget.
That is unlikely to move Arena-Hard by anything like what a preference stage does. **Keep the
preference stage; keep it to one epoch; keep max_length ≤ 8 k.**

### 3.5 What does *not* transfer

| Technique | Why not |
|---|---|
| 140 B-token reasoning mid-training (SmolLM3) | 35× our budget |
| Long-CoT RL at 16–32 k completions (DeepScaleR, DAPO) | 123+ A100-h per 300 steps (§3.2) |
| PPO / VAPO with a value network | doubles training memory **[V]**; no room |
| Disaggregated async RL (verl/SkyRL/PRIME-RL topologies) | requires ≥ 2 GPU pools by construction **[V]**; we are forced into colocated, hence synchronous |
| Full-parameter RL on the MoE | 160 GB of optimiser state |
| Reward-model-based RLHF | needs a trained RM (another model, another training run) — Tulu 3 sets the RM multiplier to **0.0** in its released RLVR configs anyway **[V]** |
| Process reward models (PRM) | a PRM forward pass over every trace is a second model plus a new synchronisation barrier **[V]** |

---

## 4. Recommendation for Prophet

### 4.0 Governing principles

1. **Distillation buys capability; RL buys behaviour.** Qwen3 Table 21 (10× cheaper *and* better)
   and Table 22 (general RL moves IFEval/ToolUse, not AIME) are the two load-bearing measurements.
   Spend ~60 % of the budget on distillation-flavoured SFT, ~20 % on RLVR, ~6 % on preferences.
2. **Copy SmolLM3's structure, not its scale.** Its recipe is fully public, its hyperparameters are
   in `alignment-handbook/recipes/smollm3/` **[V]**, and its model is the closest published analogue
   to Prophet.
3. **Every stage ends in a merge-able checkpoint.** SmolLM3 recovered its long-context regression for
   free with a 0.9/0.1 linear merge **[V]**; merging is CPU-only and is the cheapest capability
   insurance we have.
4. **Licence-clean by construction.** Teachers: Apache-2.0 or MIT only (§6.1). No Gemma-derived data
   in any released artifact, ever.
5. **Prototype on the dense trunk, ship on the MoE.** R05 upcycles to MoE *during* pre-training
   (Phase 2 of 4), so post-training operates on **Prophet-M v1**, not on a dense model. That is
   affordable at 5.123 B (§3.2) — but every RL *ablation* costs 2.4× more there than on the
   Dense-1.1B trunk that Phase 1 already produces. **Run ablations on the trunk (or the 564 M mini),
   run the final recipe on Prophet-M v1, and do not grow to v2 until post-training is finished.**

### 4.1 The recipe

All token counts are for **Prophet-M v1** (5.123 B total / 1.072 B active, **42 M tok / A100-h**).
On the Dense-1.1B trunk the same token counts cost **0.70×**; on the 564 M mini, **0.42×** (§3.1).

---

#### **Stage 0 — Reasoning mid-training (off-policy sequence-KD).** 0.85 B tokens · **20 A100-h**

*Goal: install long-CoT structure before any chat formatting exists. This is SmolLM3's mid-training,
compressed 140× — so curation quality replaces volume.*

| Source | Config / split | Weight | Licence |
|---|---|---|---|
| `open-thoughts/OpenThoughts3-1.2M` | math + code + science | **0.60** | apache-2.0 **[V]** |
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | **`reasoning_r1` splits only** | **0.25** | cc-by-4.0 **[V]** |
| `open-r1/OpenR1-Math-220k` | default (94 k verified) | **0.10** | apache-2.0 **[V]** |
| `simplescaling/s1K-1.1` + `GAIR/LIMO` | all | **0.05** | apache-2.0 **[V]** |

- Format: raw ChatML, **no** system prompt, **wrapped packing** (SmolLM3 deliberately avoids
  imposing structure at this stage) **[V]**.
- `max_length` 16384 (not 32768 — halves activation memory, and R02's bounded-state core changes the
  long-context calculus anyway; revisit after R02 lands).
- lr **2e-5**, cosine-with-min-lr 0.1, warmup 0.03, `max_grad_norm 0.2`, Liger, bf16, 8-bit AdamW.
- 2 epochs over ~425 M unique tokens (0.85 B total).
- **Gate**: MATH-500 ≥ 60 and a stable `<think>…</think>` structure, else stop and re-curate.

---

#### **Stage 1 — Dual-mode SFT (think / no_think).** 0.85 B tokens · **20 A100-h**

*Goal: chat, tools, instruction following, and the mode switch. Copy smoltalk2's structure exactly.*

Chat template: **adopt SmolLM3's verbatim** **[V]** — `/think` `/no_think` in the system message,
metadata block (knowledge cutoff / date / reasoning mode), **separate `### XML Tools` and
`### Python Tools` sections**, empty `<think>\n\n</think>` pre-fill in no-think mode, and a
`/system_override` escape. It is battle-tested, tokenizer-agnostic, and vLLM/SGLang already parse it
(`--tool-call-parser=hermes`) **[V]**.

**Mixture (target ≈ 55 % no-think / 45 % think by tokens):**

| Block | Source | Weight | Licence |
|---|---|---|---|
| Reasoning (think) | `smoltalk2:SFT/OpenThoughts3_1.2M_think` (or upstream) | **0.15** | **[V]** |
| Reasoning (no-think) | `smoltalk2:SFT/OpenThoughts3_1.2M_no_think` | **0.15** | **[V]** |
| General chat | `allenai/tulu-3-sft-mixture` (chat/creative/safety subsets) | **0.15** | odc-by **[V]** |
| **Instruction following** | `allenai/tulu-3-sft-personas-instruction-following` | **0.10** | odc-by **[V]** |
| **Tool calling** | `Team-ACE/ToolACE` (0.05) + `NousResearch/hermes-function-calling-v1` (0.04) + `nvidia/Nemotron-Agentic-v1` (0.03) | **0.12** | apache/cc-by **[V]** |
| Code | `nvidia/OpenCodeReasoning` + `bigcode/self-oss-instruct-sc2-exec-filter-50k` | **0.12** | cc-by-4.0 / odc-by **[V]** |
| Math (incl. tool-integrated) | `nvidia/OpenMathReasoning` (CoT + TIR) | **0.10** | cc-by-4.0 **[V]** |
| Science | `open-r1/Mixture-of-Thoughts` (science slice) | **0.05** | **[V, partial]** |
| Long context | `smoltalk2:SFT/LongAlign_64k*` (both modes) | **0.04** | **[V]** |
| **Abstention / when-not-to-call** | `nvidia/When2Call` + Tulu-3 CoCoNot subset | **0.02** | cc-by-4.0 **[V-ish]** |

- **Excluded on purpose**: `teknium/OpenHermes-2.5` (no licence + GPT-4 outputs),
  `allenai/WildChat-*` (ToS grey), everything in Tier C.
- Trainer: `assistant_only_loss: true`, **mask tool-result turns too** (SmolLM3 does) **[V]**,
  `packing_strategy: ffd`, `max_length 16384`, lr **1e-5** (half of SmolLM3's, because we do fewer
  epochs on a smaller model), 3 epochs, `max_grad_norm 0.2`, warmup 0.03, Liger.
- **Save this checkpoint. It is the merge anchor.**
- **Gate**: IFEval ≥ 60, BFCL v3 ≥ 45, MATH-500 ≥ 75, ThinkFollow-equivalent ≥ 90.

---

#### **Stage 2 — On-policy distillation (GKD).** ~150 M student tokens · **18 A100-h**

*The highest-leverage stage in the pipeline (Qwen3 Table 21). Only run it if §6.2's tokenizer
condition is met.*

- Trainer: `trl.experimental.gkd.GKDTrainer` **[V]**. `lmbda = 0.75` (mostly on-policy),
  `beta = 0.5` (JSD midway between forward and reverse KL — ablate 0.1/0.5/0.9), `seq_kd = False`.
- **Teacher: `Qwen/Qwen3-4B-Thinking-2507` (Apache-2.0)**, bf16, resident (8 GB). Backup:
  `Qwen/Qwen3-8B` in NF4. **Explicitly NOT** Gemma (§6.1) and **NOT** Llama (naming clause).
  Consider Qwen3.5-4B if its licence is confirmed Apache-2.0.
- Prompts (no responses needed — the student generates them): 60 k drawn from
  `allenai/Dolci-Think-RL-7B` + `AI-MO/NuminaMath-1.5` + `nvidia/Nemotron-RL-*`, in a 50/30/20
  think/no-think/tool mix.
- `max_new_tokens = 2048`, temperature 1.0 for on-policy sampling.
- **Fallback if the tokenizer blocks logit KD**: replace this stage with 200 M tokens of
  *self-distillation-by-rejection-sampling* — student generates k=8 per prompt, keep only
  verifier-correct traces, SFT on them (this is RFT/STaR, needs no teacher and no shared vocab).
  Budget the same 18 h; expect roughly half the gain.
- **Gate**: AIME'25 and MATH-500 up vs Stage 1; **pass@64 must not decrease** (Qwen's diagnostic for
  whether distillation expanded or collapsed the exploration space **[V]**).

---

#### **Stage 3 — Preference alignment (APO-zero).** 60 k pairs · **5 A100-h**

- `loss_type: apo_zero`, `beta: 0.05`, lr **1e-6**, **1 epoch**, `max_length 8192`,
  `padding_free: true`, warmup 0.1, `max_grad_norm 0.2`, `precompute_ref_log_probs: true`
  (all values from SmolLM3's `apo.yaml` except `max_length`, reduced from 24576 for memory) **[V]**.
- Data:
  - `allenai/llama-3.1-tulu-3-8b-preference-mixture` — no-think half, weight **0.5** **[V]**
  - `allenai/Dolci-Think-DPO-7B` (150 k pairs, subsample 30 k) — think half, weight **0.5** **[V]**
  - Optional top-up, self-built with the **Delta-Learning heuristic**: chosen = Qwen3-4B-Thinking,
    rejected = Qwen3-0.6B, over our own Stage-1 prompts (inference-only cost) **[V]**.
- **Mandatory metric at this stage: mean response length.** If it grows > 15 %, switch
  `loss_type` to `sigmoid_norm` (SimPO length normalisation) **[V]** and re-run. Length inflation is
  an inference-cost regression on our targets, not just a judge artifact.
- **Gate**: Arena-Hard up, IFEval not down, mean length +≤15 %, **ECE not worse** (§6.5).

---

#### **Stage 4 — RLVR (Dr.GRPO / DAPO).** ~350 steps · **20 A100-h**  *(LoRA r=32)*

Two sub-runs from the same Stage-3 checkpoint, merged afterwards (this parallelises the *risk*, not
the compute):

**4a — Reasoning RLVR (~180 steps, 11.5 A100-h).**
- Config: `P = 32`, `G = 8`, **`max_completion_length = 2048`** (4096 would cost 37 h — see §3.2),
  `max_prompt_length = 512`, **LoRA r=32** on all linears incl. router (10.8 GB, §3.2).
- `loss_type: dr_grpo` (removes length and difficulty bias **[V]**), `beta: 0.0` (no reference
  model), `epsilon_low: 0.2`, `epsilon_high: 0.28` (DAPO clip-higher **[V]**),
  `scale_rewards: false`, **dynamic sampling** (drop groups with accuracy 0 or 1 **[V]**),
  soft overlong penalty in the last 512 tokens.
- lr **1e-6** constant, `num_iterations: 1`, vLLM `colocate` + `enable_sleep_mode`,
  `vllm_gpu_memory_utilization: 0.35`, `vllm_importance_sampling_correction: true` **[V]**.
- Prompts: **~4,000 curated pairs**, mirroring Qwen3's 3,995 **[V]** — filtered so the Stage-3 model
  solves them at pass@8 ∈ [1/8, 7/8] (learnable but not solved). Draw from
  `allenai/Dolci-RL-Zero-Math-7B` (13.3 k) + `-Code-7B` (13.3 k) + `zwhe99/DeepMath-103K` +
  `BytedTsinghua-SIA/DAPO-Math-17k` **[V]**.
- Rewards: exact-match on boxed answers (math); sandboxed unit tests (code), **executed
  out-of-process with a hard timeout**.

**4b — General RLVR: IF + tools + format (~250 steps, 8.5 A100-h).**
- Config: `P = 32`, `G = 8`, `max_completion_length = 1024` → **~2.0 min/step** on Prophet-M v1
  (0.8 min on the trunk, §3.2). This is the
  cheapest stage in the entire pipeline and, per Qwen3 Table 22, the one that moves IFEval (+6.6),
  Multi-IF (+8.4), ToolUse (+15.1) and ThinkFollow (→98.9) **[V]**.
- Prompts: `allenai/RLVR-IFeval` + IF half of `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` +
  `nvidia/Nemotron-RL-instruction_following-structured_outputs` **[V]**.
- Rewards, all rule-based (no reward model, no judge):
  1. IF constraint satisfaction (the Tulu-3 verifier family);
  2. JSON-schema validity for structured output;
  3. **Mode-following**: response respects the `/think` vs `/no_think` flag and emits well-formed
     `<think>` blocks (Qwen3's "ThinkFollow" **[V]**);
  4. Tool-call schema validity + parameter correctness against the declared signature;
  5. **Prophet-specific — compute penalty**: `reward ← reward − λ·(tokens_emitted/L_max) − μ·(r/r_max)`
     where `r` is **Prophet-Loop's** recurrent loop count (R04 §4.1), sampled per rollout. **This
     directly trains the model to solve problems at the lowest inference cost that still works** —
     the project thesis expressed as a reward function — and it is the natural RL analogue of Qwen3's
     emergent "thinking budget" **[V]**. It also supplies the *training signal* R04's ponder head
     needs in order for a low `r` to be a safe runtime choice on an iPhone. Start λ = μ = 0.05 and
     ablate (§7 A9).

---

#### **Stage 5 — Merge + calibration repair.** CPU only · **~2 A100-h of eval**

- MergeKit **linear soup** of {Stage-4a, Stage-4b} checkpoints, then **linear 0.9 / 0.1 with the
  Stage-1 SFT checkpoint** (SmolLM3's recipe, which recovered RULER at 128 k **[V]**). Sweep the
  0.85/0.90/0.95 mixing weight on the dev set.
- Re-measure ECE and abstention behaviour (R09 hand-off). If RLHF-style calibration damage appears,
  increase the weight on the SFT anchor.

---

#### **Stage 6 — v2 expert-cloning port (conditional, only after v1 ships).** **~15 A100-h**

R05's growth path clones every routed expert (E: 64 → 128, r ≈ 0.5 Drop-Upcycling re-init) giving
9.757 B total at unchanged active count. Post-training does **not** transfer for free across that
surgery: the router distribution changes.

- **LoRA-only** re-tuning (§3.2): r = 32 on all linears incl. experts and router, frozen bf16 base
  (20.7 GB).
- 150 M tokens of the Stage-1 mixture (router re-balancing) + 100 RLVR steps at `L = 1024`.
- **Use GSPO, not GRPO.** Sequence-level importance ratios were designed to stabilise MoE RL
  **[U, `2507.18071`]**, and the failure mode they address — router disagreement between the
  inference and training stacks — is documented and unmitigated in every open library (§6.7).
  It is already mainstream tooling (Unsloth ships a GSPO notebook **[V]**).

---

### 4.2 Ordered summary

| # | Stage | Algorithm | Tokens / steps | **A100-h (Prophet-M v1)** |
|---|---|---|---|---|
| 0 | Reasoning mid-training | seq-KD (SFT), full-param 8-bit AdamW | 0.85 B tok | **20** |
| 1 | Dual-mode SFT | SFT, assistant-only loss | 0.85 B tok | **20** |
| 2 | On-policy distillation | GKD (JSD, λ=0.75), teacher Qwen3-4B | 150 M student tok | **18** |
| 3 | Preference alignment | **APO-zero**, β=0.05 | 60 k pairs | **5** |
| 4a | Reasoning RLVR | **Dr.GRPO + clip-higher + dyn. sampling**, LoRA r=32 | 180 steps × (32×8×2048) | **11.5** |
| 4b | IF / tool / format RLVR | same, L=1024, LoRA r=32 | 250 steps × (32×8×1024) | **8.5** |
| 5 | Merge + eval | MergeKit linear soup + 0.9/0.1 anchor | — | **2** |
| — | Reserve (Colab preemption, restarts) | — | — | **10** |
| | **TOTAL (Prophet-M v1)** | | **~1.85 B tok + 430 RL steps** | **95** |
| 6 | v2 expert-cloning port (conditional) | LoRA SFT + **GSPO** | 150 M tok + 100 steps | **+15** |
| M | Prophet-mini 564 M full pass | same recipe at 0.42× | ~1.85 B tok | **+40** |

---

## 5. Compute budget table

Assumes a 300 A100-hour project total (`docs/00_PROBLEM_LANDSCAPE.md` uses "~300 A100-heures").

| Stage | Hours | % of post-training | % of project | Derivation (Prophet-M v1 @ 42 M tok/h) |
|---|---:|---:|---:|---|
| S0 Reasoning mid-training | 20 | 21 % | 6.7 % | 0.85 B tok ÷ 42 M tok/h |
| S1 Dual-mode SFT | 20 | 21 % | 6.7 % | 0.85 B tok ÷ 42 M tok/h |
| S2 On-policy distillation (GKD) | 18 | 19 % | 6.0 % | gen 150 M tok @3.5 k tok/s = 11.9 h + student bwd 3.6 h + teacher-4B fwd ~3 h (NF4) |
| S3 APO preference | 5 | 5 % | 1.7 % | 3.6 h policy + 1.2 h ref precompute (180 M tok) |
| S4a Reasoning RLVR | 11.5 | 12 % | 3.8 % | 180 × 3.85 min (L=2048, gen 3 k tok/s) |
| S4b IF/tool RLVR | 8.5 | 9 % | 2.8 % | 250 × 2.02 min (L=1024) |
| S5 Merge + full eval sweep | 2 | 2 % | 0.7 % | inference only (merge itself is CPU) |
| Reserve | 10 | 11 % | 3.3 % | Colab preemption ≈ 10 % |
| **Post-training subtotal** | **95** | **100 %** | **32 %** | within the 20–40 % mandate |
| S6 v2 expert-cloning port (cond.) | +15 | — | +5 % | LoRA, GSPO |
| Prophet-mini 564 M pass (cond.) | +40 | — | +13 % | 0.42× cost at the same token counts |

**BUDGET COLLISION — needs a program-level decision.** R05 §4.3 allocates **all 300 hours** to
pre-training (Phase 0 ablations 30 h + Phase 1 dense trunk 90 h + Phase 2 upcycle 5 h + Phase 3 MoE
main 150 h + Phase 4 anneal/distil 25 h = 300 h), leaving **zero** for R10. R10 needs **95 h**.
Three ways out, in order of preference:

1. **Absorb R10's Stage 0 into R05's Phase 4** (which already reserves 25 h for "instruction data,
   LR→0, optional logit distillation from a 4–8 B teacher" — that *is* reasoning mid-training under
   another name). Net saving ≈ 20 h; R10 then needs **75 h**.
2. **Cut R05 Phase 3 from 150 h to 95 h** (5.1 B → ~3.2 B MoE tokens). Justified by the asymmetry
   this report documents: the base→instruct delta at this scale is larger than the delta from 60 %
   more pre-training tokens (§1). This is the recommendation.
3. **Raise the program total to ~375 h.** Only if a second compute tranche actually appears.

Whichever is chosen, the split must be agreed **before** Phase 1 starts, because Phase 1's
deliverable (Prophet-Dense-1.1B) is also R10's ablation platform (§7).

**Compressed 60-hour variant** (if pre-training overruns): drop S2 entirely (−18 h), halve S0
(−10 h), keep S1/S3/S4b in full, cut S4a to 90 steps (−5.7 h) → **~61 h**. Expected cost: most of
the AIME/MATH headroom, little of the IFEval/BFCL/Arena-Hard headroom. **Protect S1, S3 and S4b —
they are the cheap benchmark points.**

**Stretch 140-hour variant**: add a second GKD round after S4 (teacher re-scores post-RL rollouts,
+18 h), 200 more RLVR steps at `L=2048` (+13 h), a full Prophet-mini pass (+40 h).

**Wall-clock reality check.** 95 A100-hours on interruptible Colab sessions (~12 h max) ≈ **10–14
sessions**, assuming ~85 % effective utilisation. Every stage must therefore be resumable
mid-epoch *including dataloader and RNG state* (`CLAUDE.md`, "Reprise"), and GRPO must additionally
checkpoint the rollout buffer and the vLLM engine state.

---

## 6. Risks & failure modes

### 6.1 Licence risk (the highest-severity, lowest-probability-of-detection risk)

**Verified licence positions of candidate teachers:**

| Teacher | Licence | Distillation allowed? | Contamination of *our* licence? |
|---|---|---|---|
| **Qwen3 / Qwen3.x (all sizes)** | **Apache-2.0** — "All our open-weight models are licensed under Apache 2.0" **[V]** | Yes | **None.** Attribution only |
| **DeepSeek-R1** | **MIT**, explicitly permitting "any modifications and derivative works, including, but not limited to, distillation for training other LLMs" **[V]** | **Yes, explicitly** | **None** |
| DeepSeek-R1-Distill-Qwen-{1.5,7,14,32}B | Apache-2.0 (Qwen2.5 base) **[V]** | Yes | None |
| DeepSeek-R1-Distill-Llama-{8,70}B | Llama 3.1 / 3.3 licence **[V]** | Yes, with conditions | **Naming clause** |
| **Llama 3.x** | Llama Community Licence. Verbatim: *"If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include 'Llama' at the beginning of any such AI model name."* Plus a prominent **"Built with Llama"**, a retained attribution notice, and the **700 M MAU** threshold **[V, verbatim from `meta-llama/llama-models/models/llama3_2/LICENSE`]** | Yes | **Would force the model to be named `Llama-Prophet`.** Unacceptable |
| **Gemma / Gemma 3 / Gemma 4** | Gemma Terms of Use. Verbatim: *"'Model Derivatives' means all … (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or **Output** of Gemma, to that model in order to cause that model to perform similarly to Gemma, **including distillation methods** that use intermediate data representations **or methods based on the generation of synthetic data Outputs by Gemma for training that model**."* §3.1 then requires that any Distribution of a Model Derivative (**including via a hosted API**) pass along the use restrictions as an enforceable provision, ship a copy of the Agreement, and carry the notice *"Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms"* **[V, verbatim from the canonical `gemma-tou-2025-03-24` text]** | Technically yes | **CATASTROPHIC.** Prophet would become a Gemma Model Derivative and could never be Apache-2.0. §3.3 says Google claims no rights in Outputs *per se*, but §1.1(e) is explicit that a model *trained on* those Outputs **is** a Derivative |
| Phi-4 / Phi-4-mini | MIT **[U — not re-verified in-session]** | Yes | None, if confirmed |
| gpt-oss | Apache-2.0 **[U]** | Yes | None, if confirmed |
| Nemotron 3 | NVIDIA Open Model License **[U]** | Check | Check |

**Hard rules for Prophet (adopt verbatim into `CLAUDE.md`):**

1. **No Gemma, ever** — not as a teacher, not as a judge, not as a synthetic-data generator, not in a
   preference pair, not in an ablation whose artifacts get released.
2. **No Llama-derived generation** in released artifacts. Consequences: exclude
   `Magpie-*-Llama-*` sets, and from `nvidia/Llama-Nemotron-Post-Training-Dataset` use **only the
   `reasoning_r1` (DeepSeek-R1-generated) splits**. Note SmolLM3 used that dataset and did not rename
   itself **[V]**; we should nevertheless take the conservative reading.
3. **Teacher whitelist**: Qwen3/Qwen3.x (Apache-2.0) and DeepSeek-R1 family with Qwen bases (MIT +
   Apache-2.0). Both permit distillation with attribution only.
4. **No non-commercial data** — the Tier C list in §2.1 is verified and includes
   `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`.
5. **No undeclared-licence data in releases** — `teknium/OpenHermes-2.5` has *no* licence field
   **[V]** and contains GPT-4 outputs.
6. Maintain `docs/DATA_PROVENANCE.md`: for every training row, source ID, licence, generating model,
   and the split-level filter applied. CC-BY and ODC-BY both carry attribution obligations that must
   appear in the released model card.

### 6.2 Vocabulary mismatch blocks logit-level distillation — **cross-track issue with R01**

Top-K logit / GKD distillation requires the teacher and student to share a vocabulary.

**Good news from R01's decision.** R01 §4.1 chose **Prophet-Tok v1, a purpose-built byte-level BPE**
— *not* a tokenizer-free byte model. So the tooling problem disappears: `lm-eval-harness`, vLLM,
MLX and CoreML all work with a normal BPE, and R01 explicitly lists "logit-level distillation
main → mini" and "mini-as-draft speculative decoding" as reasons for keeping **one shared vocabulary
across the Prophet family**. **Intra-family distillation (Prophet-M → Prophet-mini) is therefore
native, exact and free of any alignment approximation.** That is a genuinely valuable asset: it makes
the 564 M mini a *distillation* target rather than a from-scratch training target.

**Bad news.** Prophet-Tok v1 is a **custom 32,768-entry** (R01) / **64,000-entry** (R05) vocabulary,
and Qwen3's is **151,936**. Cross-*family* logit distillation — the Stage-2 lever worth 10× RL per
GPU-hour **[V]** — still requires vocabulary alignment.

*(Note also the unresolved 32,768-vs-64,000 disagreement between R01 §4.2 and R05 §4.2. R10 does not
depend on which wins — both are ≪ 128 k and both make the logit tensor cheap — but it must be
resolved before any tokenizer-coupled artifact is built.)*

| Mitigation | Cost | Verdict |
|---|---|---|
| **Sequence-level KD** (train on teacher *text*) | zero — text is tokenizer-agnostic | **Always available. This is why S0/S1 are the backbone and why the recipe is robust to this risk.** |
| **Cross-tokenizer KD** (ULD `2402.12030`; approximated likelihood matching `2503.20083`) **[U]** | research risk, unproven at ~1 B | Ablate before committing S2 (§7 A2b) |
| **Rejection-sampling self-distillation (RFT/STaR)** | no teacher at all, no shared vocab needed | **The designated S2 fallback** |
| **Prophet-M → Prophet-mini logit KD** | native (shared vocab, R01) | **Adopt unconditionally** for the mini |
| Keep a Qwen3-vocabulary variant just for S2 | two tokenizers, forfeits R01's thesis | Rejected |

**Action for R01/R05/R10 coordination:** (a) settle 32 k vs 64 k; (b) run §7 A2b before Stage 2 is
budgeted. If cross-tokenizer KD fails its gate, delete Stage 2's GKD variant and reallocate its 18
hours to rejection-sampling self-distillation plus ~120 more RLVR steps at `L=1024`.

### 6.3 Teaching hallucination via SFT on unknown facts

`docs/00_PROBLEM_LANDSCAPE.md` §9 already states the corollary: *"ne jamais faire de SFT sur des faits
que le modèle de base ignore — cela lui enseigne à halluciner."* The supporting result is Gekhman et
al., *"Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"* (`2405.05904`) **[U]**: SFT
examples containing facts absent from pre-training are learned slowly and, once learned, **linearly
increase the model's tendency to hallucinate**.

At 1.072 B active parameters and ~2 bits/parameter of knowledge capacity, most world facts in any large
SFT mixture are "new knowledge" for Prophet. Mitigations:

1. **Prefer procedural over factual data.** Maths, code, logic and tool use teach *operations*, not
   facts. This is why the Stage-0/1 mixtures are ~70 % reasoning/code/math.
2. **Filter factual SFT rows** by base-model confidence: drop rows whose gold answer the *base* model
   assigns very low probability under few-shot prompting (the Gekhman "Unknown" bucket).
3. **Train abstention explicitly**: `nvidia/When2Call` **[V]** and Tulu 3's CoCoNot subset teach
   *not* calling a tool / *not* answering. Weight 0.02 in Stage 1.
4. **Add an abstention reward in Stage 4b**: on unanswerable prompts, reward the refusal. Qwen3's
   in-house `CounterFactQA` benchmark exists for exactly this and improved **+10.9** during mode
   fusion **[V]**.
5. Hand the calibration metric (ECE, AUROC of confidence vs correctness) to R09 as a **gate** at
   every stage, not a post-hoc measurement.

### 6.4 Reward hacking

| Hack | Symptom | Mitigation |
|---|---|---|
| **Length hacking** | mean response length climbs, accuracy flat | `dr_grpo` removes the 1/\|o\| normalisation that *rewards* length on wrong answers **[V]**; monitor length as a first-class metric |
| **Difficulty bias** | model over-optimises easy prompts | Dr.GRPO removes the reward-std division **[V]**; dynamic sampling drops all-correct groups **[V]** |
| **Format hacking** | emits `\boxed{}` with no reasoning, or a `<think>` block with filler | reward well-formedness *and* correctness jointly; hold out a format-blind verifier |
| **Verifier exploitation** | code that passes tests without solving (`assert True`, reading the test file) | run tests **out-of-process, sandboxed, no network, hard timeout**; hold out a second private test set |
| **Entropy collapse** | rollout diversity → 0, pass@k stops improving | DAPO clip-higher (ε_high 0.28) **[V]**; Qwen3 explicitly "controls the model's entropy to increase steadily or remain stable" **[V]**; log token entropy every step |
| **Judge hacking** (if an LLM judge is ever used) | Arena-Hard up, human quality flat | prefer rule-based rewards; Tulu 3 sets the RM multiplier to 0.0 **[V]** |

### 6.5 RLHF/RLVR destroys calibration

GPT-4's technical report documented post-RLHF calibration degradation (ECE ~0.007 → ~0.074) **[U]**.
This is in direct tension with R09, which makes calibrated abstention a *product* claim. Mitigations:
measure ECE at every stage gate; keep the Stage-1 SFT checkpoint as a merge anchor (Stage 5); if
calibration degrades, increase the anchor weight — merging is free and reversible.

### 6.6 Capability regression from mode fusion

Measured, not hypothetical: Qwen3-32B lost **AIME'24 −1.9** and **LCB v5 −1.2** at thinking-mode
fusion and a further **−0.5 / −1.5** at general RL **[V]**. Budget for this: our S1 dual-mode SFT and
S4b general RLVR will cost us a few points of peak maths in exchange for IFEval/BFCL/ThinkFollow.
**That trade is correct for our benchmark table** (IFEval and BFCL are worth more points than the top
of AIME for a ~1 B-active model) — but it must be *measured*, and the pre-fusion checkpoint kept.

### 6.7 MoE-specific RL correctness hazard

DeepSeek-V3.2 documented that inference frameworks (vLLM/SGLang) and training frameworks
(FSDP/Megatron) implement MoE routing independently, and **floating-point differences in the gating
function can select different experts for identical inputs**, which "induces abrupt shifts in the
active parameter subspace, destabilises optimisation and exacerbates off-policy issues". Their fix
("Keep Routing" — record routing at sampling time and enforce it during the training forward pass) is
**implemented by no current open-source async RL library** **[V]**. A parallel issue exists for
top-p/top-k sampling masks ("Keep Sampling Mask") **[V]**.

For Prophet this is a **correctness**, not performance, risk in Stage 6. Mitigations: use GSPO
(sequence-level IS ratios are far less sensitive to per-token routing divergence) **[U]**; or run RL
*before* upcycling (our default plan); or log and compare routing decisions between the trainer and
vLLM on a fixed probe batch as a CI check.

### 6.8 Operational

- **Colab preemption** mid-GRPO-step loses the rollout buffer → checkpoint the buffer, the vLLM
  engine state, the dataloader position and RNG. 10 h of the budget is reserve for exactly this.
- **TRL's `GKDTrainer` lives in `trl.experimental`** **[V]** — the API can break between releases.
  Pin the TRL version in `pyproject.toml` and vendor the trainer if necessary.
- **vLLM + a novel architecture**: Prophet's recurrent/hybrid core (R02) will not be supported by
  vLLM out of the box. **A vLLM (or SGLang) backend for Prophet is a hard prerequisite of Stages 2
  and 4** — without fast batched generation, GRPO's 105 s generation phase becomes ~20 minutes and
  the whole RL plan collapses. **This should be scheduled as engineering work in R02/R07, not
  discovered during Stage 4.**

---

## 7. Ablation plan

Per `CLAUDE.md` rule 2, nothing enters the main recipe without an ablation. **Run A1–A9 on
Prophet-Dense-1.1B (R05 Phase 1's deliverable) or the 564 M mini — 1.4× and 2.4× cheaper per token
respectively, and 2.4× / 4× cheaper per RL step (§3.2) — and confirm only the winners on Prophet-M
v1.** Total ablation cost below is ~110 h if run on Prophet-M v1 and **~46 h on the mini**; budget it
against R05's Phase 0 (30 h of ablations) rather than against R10's 95 h.

| # | Question | Arms | Metric / gate | Cost |
|---|---|---|---|---|
| **A1** | Is a separate reasoning mid-training stage worth 20 h, vs folding it into SFT? | (a) S0+S1 (b) S1 only, same total tokens | MATH-500, AIME'25, GPQA | 2 × 6 h |
| **A2a** | Off-policy seq-KD vs on-policy GKD at **matched compute** | (a) 18 h more SFT (b) 18 h GKD | AIME'25, MATH-500, **pass@64** | 2 × 8 h |
| **A2b** | Does cross-tokenizer KD (Prophet-Tok v1 ← Qwen3 BPE) work at all? | ULD / likelihood-matching vs (i) a shared-vocab control and (ii) seq-KD only | KL on a held-out set; downstream MATH-500 | 6 h — **gate for S2; run before Stage 2 is budgeted** |
| **A2c** | Does native Prophet-M → Prophet-mini logit KD beat training the mini from the same data? | logit KD vs plain SFT, matched tokens | mini MATH-500, IFEval | 2 × 4 h |
| **A3** | Teacher size / capacity gap (`2502.08606`) | Qwen3-1.7B / 4B / 8B teachers | student MATH-500, AIME'25 | 3 × 5 h |
| **A4** | Is the preference stage worth 5 h vs 5 h more SFT? | (a) APO-zero (b) DPO-sigmoid (c) `sigmoid_norm` (d) extra SFT | Arena-Hard, IFEval, **mean length**, ECE | 4 × 2 h |
| **A5** | RLVR variant | `grpo` / `dr_grpo` / `dapo` / GSPO (TRL `loss_type` **[V]**) | AIME'25 and **length drift** at fixed steps | 4 × 4 h |
| **A6** | LoRA vs full-parameter RL (the Tina question) | full / r=16 / r=32 / r=64 | AIME'25 per A100-hour | 4 × 3 h |
| **A7** | Rollout length | `L` ∈ {1024, 2048, 4096} at fixed wall-clock | AIME'25 per hour | 3 × 4 h |
| **A8** | Think / no-think token ratio in S1 | 25 / 45 / 65 % think | joint (AIME'25, IFEval, mean length) Pareto | 3 × 5 h |
| **A9** | **Compute-penalty reward** (λ, μ) — Prophet-specific | λ,μ ∈ {0, 0.05, 0.15} | accuracy vs **tokens-to-answer** and vs Prophet-Loop depth `r`; must not degrade AIME'25 by >2 pts | 4 × 3 h |
| **A12** | MTP heads in the post-training loss | main head only / all 4 heads weighted | loss stability, downstream parity | 2 × 3 h |
| **A10** | Merge weight | 0.85 / 0.90 / 0.95 SFT-anchor | RULER-equivalent long-context + ECE + Arena-Hard | free (CPU + eval) |
| **A11** | **Contamination audit** (not optional) | — | see §7.1 | 4 h CPU |

### 7.1 Contamination hygiene (`CLAUDE.md` rule 5)

Every source entering any mixture passes this gate, and the result is published in R11:

1. **n-gram decontamination**: 8-gram (Llama-style) and 13-gram exact-match against the *full text* of
   every eval item — MMLU-Pro, GPQA, GSM8K, MATH-500, AIME'24/25, HumanEval(+), MBPP(+),
   LiveCodeBench, IFEval, BFCL, Arena-Hard, AlpacaEval. Drop the training row on a hit.
2. **Embedding near-duplicate sweep** (cosine > 0.95) — n-grams miss paraphrase, which is the dominant
   leakage mode for synthetic reasoning data.
3. **Known-hazard list**, source-specific:
   - `open-r1/OpenR1-Math-220k` and everything built on **NuminaMath** overlaps competition-maths
     sources → **explicitly check AIME'24**. **AIME'25 postdates most corpora and is our safer
     headline metric.**
   - `open-r1/codeforces-cots` overlaps **LiveCodeBench** problem sources → use an LCB version window
     strictly *after* our data cutoff, and report the window.
   - **GSM8K is saturated and web-contaminated**; report **GSM-Plus** and **GSM1k** (`2405.00332`
     **[U]**) alongside it, and treat a GSM8K-vs-GSM1k gap as a contamination readout.
   - **`google/IFEval` prompts must never enter training.** `allenai/RLVR-IFeval` and
     `allenai/tulu-3-sft-personas-instruction-following` are IFEval-*shaped*, not IFEval — verify by
     exact-prompt diff anyway.
   - **BFCL must be held out entirely.** `xLAM` and `ToolACE` share API surfaces with BFCL; check at
     the API-signature level, not just the string level.
   - MATH-500 is a *subset of the MATH test split* — any dataset built from MATH must be verified to
     use the train split only.
4. **Upstream decontamination is not transitive.** OpenThoughts3 and the Nemotron sets were
   decontaminated against *their authors'* eval suites, which are not ours. Re-run everything.
5. **Held-out canary**: keep 200 eval items entirely out of every pipeline, including our own
   ablations, and report their scores separately. A canary/main gap is the cheapest contamination
   alarm that exists.
6. **Report the audit**, including its failures (`CLAUDE.md` rule 6): rows dropped per source, hit
   rate per benchmark, and the residual risk.

---

## 8. References

**Verified in-session [V]** — reachable primary or mirrored sources.

1. **Qwen3 Technical Report**, arXiv `2505.09388` — read in full from `QwenLM/Qwen3/Qwen3_Technical_Report.pdf`.
   §4.1–4.5 (four-stage pipeline, strong-to-weak distillation), Table 17/18 (Qwen3-4B/8B),
   Table 19/20 (Qwen3-1.7B/0.6B, R1-Distill-1.5B, Phi-4-mini, Gemma-3-1B),
   **Table 21** (RL 17,920 GPU-h vs on-policy distillation 1,800 GPU-h),
   **Table 22** (thinking-mode fusion and general-RL deltas), §4.2 (3,995 RL prompts, 170 steps).
2. **DeepSeek-R1**, `deepseek-ai/DeepSeek-R1/README.md` — MIT licence with explicit distillation
   permission; distilled-model eval table; 800 k SFT samples.
3. **Gemma 3 Technical Report** — `storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf`,
   Tables 6 and 18 (Gemma-3-4B-IT).
4. **Gemma Terms of Use, 2025-03-24** — canonical text mirrored in
   `aboutcode-org/scancode-toolkit/src/licensedcode/data/licenses/gemma-tou-2025-03-24.LICENSE`.
   §1.1(e) "Model Derivatives" (distillation + synthetic-data clause); §3.1 distribution conditions.
5. **Llama 3.2 Community License** — `meta-llama/llama-models/models/llama3_2/LICENSE` (naming
   clause, "Built with Llama", 700 M MAU) and `MODEL_CARD.md` (Llama-3.2-3B-Instruct scores).
6. **SmolLM3** — model card mirrored at
   `yiyihum/rag291/data/hf_cards_2025/models/HuggingFaceTB__SmolLM3-3B.md` (all eval tables);
   blog post at `huggingface/blog/smollm3.md` (35 B mid-training tokens ×4 epochs = 140 B; 1.8 B-token
   SFT mixture; APO; MergeKit 0.9/0.1 merge).
7. **SmolLM3 recipes, verbatim** — `huggingface/alignment-handbook/recipes/smollm3/{sft/mid.yaml,
   sft/sft.yaml, dpo/apo.yaml}`: full 24-split weighted mixture, chat template, and every
   hyperparameter quoted in §2.6.
8. **Tulu 3 recipe** — `allenai/open-instruct/docs/tulu3.md` (dataset IDs, lr/β/epochs for SFT, DPO,
   RM, RLVR) and `README.md` (SFT→DPO→RLVR, `grpo_fast.py`). Paper: arXiv `2411.15124`.
9. **TRL documentation** — `huggingface/trl/docs/source/{grpo_trainer.md, dpo_trainer.md,
   gkd_trainer.md}`: GRPO loss types (`grpo`/`dapo`/`dr_grpo`/`sapo`), `beta` default 0.0, vLLM
   colocate/sleep mode; the full DPO `loss_type` table with paper links; GKD `lmbda`/`seq_kd`/`beta`.
10. **Open-R1** — `huggingface/open-r1/README.md`: Mixture-of-Thoughts (350 k), OpenR1-Math-220k,
    CodeForces-CoTs, OpenR1-Distill-7B results, `vllm_mode=colocate`, 8×H100 baseline.
11. **OpenThoughts** — `open-thoughts/open-thoughts/README.md`: OpenThoughts3-1.2M composition
    (850 k math / 250 k code / 100 k science, QwQ-32B teacher) and OpenThinker3-7B results.
12. **Dr. GRPO / Oat-Zero** — `sail-sg/understand-r1-zero/README.md`: **27 h × 8 A100**. Paper: arXiv `2503.20783`.
13. **DAPO** — `BytedTsinghua-SIA/DAPO/README.md` (Qwen2.5-32B → 50 on AIME'24, DAPO-Math-17k);
    ε_low 0.2 / ε_high 0.28 and `overlong_buffer` confirmed from published verl configs. Paper: arXiv `2503.14476`.
14. **Tina: Tiny Reasoning Models via LoRA** — `shangshang-wang/Tina/README.md`: **$9** best
    checkpoint, **$526** all experiments, **43.33 % AIME'24** from R1-Distill-Qwen-1.5B. arXiv `2504.15777`.
15. **DeepScaleR / rLLM** — `agentica-project/rllm/README.md`: GRPO with an 8K→16K→24K context
    schedule, 43.1 % AIME'24 on a 1.5 B.
16. **verl** — `volcengine/verl`: PPO, GRPO, **GSPO**, ReMax, REINFORCE++, RLOO, PRIME, DAPO,
    Dr.GRPO, KL_Cov/Clip_Cov.
17. **"The Async RL Training Landscape"** — `huggingface/blog/async-rl-training-landscape.md`:
    single-GPU vLLM throughput anchors (7B ≈ 6,300 tok/s, 32B ≈ 1,200 tok/s on one H100), the
    GRPO generation-time table, "critic-free frees ~50 % of training memory", colocated-vs-
    disaggregated analysis, §5.4 DeepSeek-V3.2 MoE "Keep Routing"/"Keep Sampling Mask", §5.5
    on-policy distillation as the same async problem.
18. **Unsloth** — `unslothai/unsloth/README.md` (GRPO memory claims, 500 K-context RL on 80 GB) and
    `unslothai/notebooks/README.md` (existence of Qwen3.5 2B/4B, Qwen3.8 27B, Gemma 4 E2B–31B,
    Nemotron Nano 3 30B-A3B; a Qwen3.5 GSPO notebook).
19. **HF dataset metadata registry** — `Shekswess/open-corpus-registry/data/datasets_all.jsonl`
    (308 datasets: ID, licence, stage, size, dates). Source of every licence string in §2.1 unless
    stated otherwise.
20. **NVIDIA Nemotron dataset index** — `NVIDIA-NeMo/Nemotron/README.md` (dataset IDs, licences,
    sizes: OpenMathReasoning 5.68 M samples / 306 K problems; Llama-Nemotron 2.2 M math + 500 K code).

**Cited from prior knowledge, NOT re-verified in-session [U]** — verify before acting.

21. GRPO / DeepSeekMath — arXiv `2402.03300`.
22. GSPO (Group Sequence Policy Optimization, Qwen) — arXiv `2507.18071`.
23. VAPO — arXiv `2504.05118`. RLOO — `2402.14740`. REINFORCE++ — `2501.03262`. CISPO / MiniMax-M1 — `2506.13585`.
24. DPO `2305.18290`; IPO `2310.12036`; KTO `2402.01306`; SimPO `2405.14734`; ORPO `2403.07691`;
    CPO `2401.08417`; APO `2408.06266`; RSO `2309.06657`; and the rest of the TRL `loss_type` table (§2.3).
25. Delta Learning Hypothesis (weak-vs-strong preference pairs), AI2 — arXiv `2507.06187`.
26. **Distillation Scaling Laws**, Apple — arXiv `2502.08606` (capacity gap; when distillation beats
    supervised training).
27. GKD / On-Policy Distillation of Language Models — arXiv `2306.13649`. MiniLLM (reverse KL) —
    `2306.08543`. Sequence-level KD — `1606.07947`. DistiLLM `2402.03898`, DistiLLM-2 `2503.07067`.
28. Cross-tokenizer distillation: ULD `2402.12030`; approximated likelihood matching `2503.20083`.
29. **Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"** — arXiv
    `2405.05904`. (Direct support for `00_PROBLEM_LANDSCAPE.md` §9.)
30. Length bias in RLHF/DPO — `2310.03716`, `2403.19159`; length-controlled AlpacaEval `2404.04475`.
31. GSM1k / contamination measurement — arXiv `2405.00332`.
32. Benchmarks: MMLU-Pro `2406.01574`; GPQA `2311.12022`; IFEval `2311.07911`; LiveCodeBench
    `2403.07974`; Arena-Hard `2406.11939`.
33. DeepSeek-R1 paper `2501.12948`; Phi-4-mini technical report `2503.01743`; Gemma 3 `2503.19786`;
    Llama-Nemotron `2505.00949`; OpenThoughts `2506.04178`; s1 `2501.19393`; LIMO `2502.03387`.
34. Thinking Machines Lab, *On-Policy Distillation* (Oct 2025) and *LoRA Without Regret* (Sept 2025)
    — blog posts; `thinkingmachines.ai` was unreachable in-session.
35. GPT-4 Technical Report — post-RLHF calibration degradation (ECE 0.007 → 0.074).

---

### Appendix A — one-line verdicts

| Question from the R10 brief | Verdict |
|---|---|
| Best open SFT data? | OpenThoughts3-1.2M + smoltalk2 + Tulu-3 personas-IF + ToolACE/Hermes-FC + Dolci. §2.1 |
| Which teachers may we legally distill? | **Qwen3/Qwen3.x (Apache-2.0)** and **DeepSeek-R1 (MIT, distillation explicitly permitted)**. **Never Gemma.** Llama only if we accept being named `Llama-Prophet`. §6.1 |
| On-policy or off-policy distillation? | Off-policy is the backbone (it is free); on-policy GKD is the single highest-value stage **if** the tokenizer allows it — 10× better per GPU-hour than RL at Qwen scale. §2.2, §6.2 |
| DPO or its variants — worth the compute? | **Yes, ~5 A100-h.** Use **APO-zero** (β=0.05, lr 1e-6, 1 epoch), switch to `sigmoid_norm` if length inflates. §3.4 |
| **Is GRPO feasible on one A100 for a 1.07B-active model?** | **YES.** Prophet-M v1 (5.123 B total), LoRA r=32, `L ≤ 2048`, `G = 8`, `P ≤ 32`, `β = 0`, Dr.GRPO + clip-higher, vLLM colocate: **~19 h per 300 steps** (**7 h** on the Dense-1.1B trunk). **NO** for long-CoT `L ≥ 8192` (240–300 h/300 steps) and **NO** for full-parameter RL on Prophet-M **v2** (9.757 B). §3.2 |
| What actually moves benchmarks under 4B? | Reasoning-trace distillation (MATH/AIME/LCB), then rule-based RLVR for IFEval/BFCL/mode-following (Qwen3 Table 22: ToolUse +15.1, IFEval +6.6, ThinkFollow →98.9). §2.5 |
| Biggest single risk? | The Gemma "Model Derivatives" clause — a single Gemma-generated row could make Prophet un-releasable under Apache-2.0 (§6.1). Then, in order: the R05/R10 **budget collision** (§5), the cross-vocabulary GKD gate (§6.2), and vLLM support for Prophet's architecture, which is a hard prerequisite of Stages 2 and 4 (§6.8). |
