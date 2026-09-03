# R04 — Reasoning and Test-Time Compute: Buying Capability with Depth Instead of Parameters

**Track:** R04 · **Status:** research complete, decision-ready · **Date:** 2026-09-03
**Scope:** depth-recurrent / looped transformers, adaptive compute, latent reasoning, explicit test-time search, reasoning distillation — evaluated strictly against Prophet's constraint envelope (single A100 80GB, a few hundred GPU-hours; inference on 1× RTX 5090 / Mac Studio MLX / iPhone 17 Pro ANE).

**Citation confidence convention used throughout:**
- `[V]` = number verified against a retrieved primary/secondary source during this research session.
- `[M]` = number recalled from prior knowledge, **not** re-verified in this session (arXiv egress was blocked; only `github.com` was fetchable). Treat `[M]` numbers as approximately right and re-check before they gate a decision.

---

## 1. Problem statement

### 1.1 The compute wall we are actually behind

Prophet's realistic training budget is ~200–350 A100-80GB-hours. At 312 TFLOP/s bf16 peak and a charitable 40–45 % MFU for a 1–2 B-active model, that is:

```
300 h × 3600 s × 312e12 × 0.42  ≈  1.4e20 FLOPs   (call it 1.5e20)
```

Compare against the models we are asked to beat:

| Model | Pretraining tokens | Params | Approx. pretrain FLOPs (6ND) | Ratio vs Prophet |
|---|---|---|---|---|
| Qwen3-1.7B | ~36 T `[M]` | 1.7 B | ~3.7e23 | **~2 500×** |
| Llama-3.2-3B | ~9 T `[M]` | 3.2 B | ~1.7e23 | ~1 100× |
| Gemma-3-4B | ~4 T `[M]` | 4.3 B | ~1.0e23 | ~700× |
| SmolLM3-3B | ~11 T `[M]` | 3 B | ~2.0e23 | ~1 300× |
| Huginn-3.5B (2502.05171) | 0.8 T `[V]` | 3.5 B, mean recurrence 32 `[V]` | ~1.3e23 (see §5.1) | ~850× |
| **Prophet (planned)** | **~20 B** | 9.5 B total / ~1.0–1.5 B active | **1.5e20** | 1× |

This is the single most important fact in this report and it constrains every recommendation below. **We cannot buy knowledge.** MMLU-Pro and GPQA are dominated by pretraining-token count; no test-time trick recovers 3 orders of magnitude of pretraining compute on a knowledge benchmark. What we *can* buy is **depth of computation per token**, which is exactly what GSM8K / MATH / HumanEval / ARC-challenge reward, and which the literature says is the axis where parameters are most substitutable.

### 1.2 The three things we are trying to decide

1. **Does depth-recurrence (a small block looped k times) actually substitute for parameters at our scale?** If a 4-layer core looped 8× behaves like a 32-layer stack, we store 4 layers and get 32 layers of compute. On a 32 GB 5090 and an 8 GB iPhone that is the difference between shipping and not shipping.
2. **Can inference depth be made a runtime dial?** One set of weights must run at r≈2 on an iPhone and r≈16–32 on a 5090, without retraining or a quality cliff.
3. **Is latent (in-activation) reasoning a substitute for token-space chain-of-thought, or only a complement?** Token CoT is slow on-device (each thought token is a full forward pass plus a KV-cache row); latent depth is not. But the evidence here is *negative* in an important way (§2.3, §3.3).

### 1.3 Why token-space CoT is expensive on our targets

Per emitted thought token, on-device: 1 full forward pass + 1 KV row per layer + 1 sampling step + detokenization. A 2 000-token reasoning trace on an iPhone at ~25 tok/s is **80 seconds before the first answer character**. A latent loop of depth 8 on a 140 M model costs ~0.9 GFLOP total (§5.2) and is bandwidth-free if the looped block is cache-resident. The asymmetry is roughly **three orders of magnitude in wall-clock per unit of "extra computation"** — which is why this track exists.

---

## 2. State of the art

### 2.1 Master table

| # | Method | Mechanism | Reported gain (real numbers) | Compute / memory overhead | Paper (arXiv) |
|---|---|---|---|---|---|
| 1 | **Universal Transformer** | Single block applied T times + ACT halting | First demonstration that weight-tied depth beats same-param shallow on algorithmic/LAMBADA tasks `[M]` | ×T FLOPs, ×1 params | 1807.03819 `[M]` |
| 2 | **ALBERT** | Cross-layer weight sharing (all layers tied) | ALBERT-large 18 M params vs BERT-large 334 M (**~18× param reduction**) at modest quality loss `[M]` | ×1 FLOPs, huge param saving | 1909.11942 `[M]` |
| 3 | **Huginn-3.5B / recurrent depth** | Prelude(2L) → **shared core(4L) looped r** → coda(2L); state randomly initialised, prelude embedding **re-injected every step** via concat+adapter | 3.5 B params, 800 B tokens; "improves… up to a computation load **equivalent to 50 B parameters**" `[V]`. GSM8K 8-shot **CoT**: 24.9 strict / **38.1 flexible** at r=32 `[V]`. GSM8K **without** CoT saturates at **4.9 %** across r=4→256 `[V]`. ARC-E 69.9, MMLU 31.4 at r=32 `[V]` (vs OLMo-2-7B: ARC-E 82.8, MMLU 60.6 `[V]`) | Effective depth 2+4r+2 = **132 layers at r=32**; **KV cache stored per recurrence step** (see §6.2) | 2502.05171 (NeurIPS'25) |
| 4 | **Reasoning with Latent Thoughts** | Iso-param / iso-FLOP controlled study of k-layer block looped L times | "a k-layer transformer looped L times **nearly matches** the performance of a kL-layer non-looped model, and is **significantly better** than a k-layer model" `[V]`, on addition, p-hop induction, math. GSM8K-style: near-0 at 1 iteration → **34.8 strict / 42.1 flexible at 32 iterations** `[V]` | ×L FLOPs, ×1 params | 2502.17416 (ICLR'25) |
| 5 | **Mixture-of-Recursions (MoR)** | Shared block + **per-token router** choosing recursion depth + recursion-wise KV caching + optional KV sharing across recursions | 135 M–1.7 B scale, 3 recursions ⇒ **~1/3 unique params**; "for models larger than 360 M, MoR matches or exceeds vanilla Transformer" `[V]`; new Pareto frontier at equal training FLOPs `[V]`; **up to 2.06× decode throughput** with continuous depth-wise batching + early exit `[V]`; ~2× inference throughput headline `[V]` | ×1 params/3; router adds <0.1 %; **at 135 M MoR is worse than vanilla** `[V]` — a scale caveat that directly threatens our ablation size | 2507.10524 (NeurIPS'25) |
| 6 | **Relaxed Recursive Transformers** | Convert a pretrained LLM to a looped model (single unique block repeated) + **layer-wise LoRA** to relax exact tying | Recursive Gemma-1B beats TinyLlama-1.1B, Pythia-1B and KD baselines, "recovers most of the performance of the original full-size Gemma-2B" `[V]`; **Continuous Depth-wise Batching + early exit → 2–3× throughput** `[V]` | ×1 params/L + LoRA; needs an existing pretrained donor | 2410.20672 (ICLR'25) |
| 7 | **Retrofitted Recurrence** | Convert an *existing* pretrained non-recurrent LM into a depth-recurrent one using a **curriculum of increasing recurrence** | "a curriculum of recurrences… **preserves performance while reducing total computational cost**"; on mathematics, converting a pretrained model to recurrent **beats simply post-training the original non-recurrent model at equal compute** `[V]` | Cheaper than continued pretraining `[V]`; exact deltas not retrieved | 2511.07384 |
| 8 | **Think-at-Hard (TaH)** | 2 latent iterations, but a **learned per-token decider** fires the 2nd iteration only on "hard" tokens; **depth-aware LoRA** retargets the 2nd pass to refinement; duo-causal attention across the iteration axis | On Qwen3-1.7B-Base / 0.6B, 9 benchmarks: **+3.8–4.4 %** over always-iterate at identical params while **skipping iteration on 93 % of tokens**; **+3.0–3.8 %** over single-iteration Qwen3 `[V]`. With <3 % extra params (LoRA + decider): **+5.3–6.2 %** and **+6.1–6.8 %** `[V]`. README states +8–11 % over fixed-2-iteration and +4–5 % over single-iteration Qwen3, skipping 94 % of 2nd iterations `[V]` | Mean depth ≈ 1.06× ; <3 % params | 2511.08577 (ICML/ICLR'26) |
| 9 | **DeepLoop** | Depth-scaling *initialisation/normalisation* rule for looped nets: Post-LN DeepNorm with **α = (2N)^{1/2}, β = (8N)^{−1/2}** for **unrolled** depth N (exponent moves 1/4 → 1/2 as loop count grows at fixed physical depth) `[V]` | Neutral at R=1; **consistently improves final val loss at larger loop counts** at GPT-2 small/medium `[V]` | Zero extra params, no gates/aux losses `[V]` | 2607.13491 |
| 10 | **ACT** (Graves) | Halting unit accumulates halting probability; ponder-cost penalty | Foundational; "ACT-based transformer training is **not as stable** as vanilla and needs careful hyper-parameter tuning" `[V]` | Biased gradient estimator `[V]` | 1603.08983 |
| 11 | **PonderNet** | Halting node predicts P(halt \| not halted before); **KL to a geometric prior** | "fully differentiable… **low-variance** gradient estimates (unlike REINFORCE) and **unbiased** (unlike ACT)" `[V]`; beats ACT on parity with less pondering and less total training compute `[V]` | One linear head; KL term | 2107.05407 |
| 12 | **Mixture-of-Depths** | Top-k **token** router per block decides process-vs-skip, static compute graph | Best config: route every **other** layer at **12.5 % capacity** `[V]`; matches baseline at **~50 % of FLOPs per forward** `[V]`; **up to 1.5 % better final perplexity at equal training FLOPs** `[V]`; isoFLOP-optimal-matching model **steps 66 % faster** `[V]`; **>50 % faster post-training sampling** `[V]` | Static graph = TPU/ANE friendly | 2404.02258 |
| 13 | **LayerSkip / early exit** | Layer-dropout curriculum + shared early-exit head → self-speculative decoding | **2.16× on CNN/DM summarisation, 1.82× coding, 2.0× semantic parsing** `[V]` | No auxiliary modules `[V]` | 2404.16710 |
| 14 | **CALM** | Per-token confident early exit with calibrated guarantees | ~3× speedup class of results `[M]` | Needs consistency handling for skipped KV | 2207.07061 |
| 15 | **COCONUT** | Feed last hidden state back as next input embedding — "continuous thought", curriculum from explicit CoT | GSM8K **34.9 % (Coconut) vs 42.9 % (explicit CoT)** `[V]` — i.e. latent **loses** on arithmetic; ProsQA **97.0 % vs 77.5 %** `[V]` — latent **wins** on search/planning; far fewer decoded tokens `[V]` | ~L extra forward passes, no KV growth for thoughts | 2412.06769 |
| 16 | **Latent-CoT limits (theory + measurement)** | Formalises the **Exploration–Execution trade-off** governed by "decisional certainty"; introduces a Symbolic Index | Reproduces the split: ProsQA **97.0 %**, GSM8K **34.1 %** `[V]`; proves **curriculum learning is necessary — direct training provably fails** due to distributional mismatch `[V]` | — | 2602.01148 |
| 17 | **Pause tokens** | Learnable `<pause>` tokens inserted at pretrain **and** finetune; delay readout | 1 B model: **+18 EM on SQuAD, +8 % CommonsenseQA, +1 % GSM8K** `[V]`; gains require pause tokens **during pretraining** `[M]` | +1 forward per pause token, +KV row each | 2310.02226 |
| 18 | **Filler tokens ("dot by dot")** | `......` in place of CoT | Solves 2 hard algorithmic tasks unsolvable without intermediate tokens `[V]`; but "learning to use filler tokens is **difficult and requires specific, dense supervision** to converge" `[V]`; useful class characterised by quantifier depth of a first-order formula `[V]` | Same as above | 2404.15758 |
| 19 | **Thinking-token skepticism** | Empirical audit of why thinking/pause tokens underperform in practice | Negative result — needed as a counterweight to #17/#18 | — | 2411.11371 |
| 20 | **Quiet-STaR** | Learn to generate token-level rationales everywhere via REINFORCE + mixing head | Mistral-7B **zero-shot GSM8K 5.9 → 10.9 %**, **CommonsenseQA 36.3 → 47.2 %** `[V]`; gains **increase with number of thought tokens** `[V]` | Very expensive: parallel rationale sampling at every position | 2403.09629 (Fast variant: 2505.17746) |
| 21 | **Implicit CoT by stepwise internalisation** | Start from explicit-CoT model, **progressively delete** CoT tokens while finetuning | GPT-2 Small solves **9×9 multiplication at ~99 %** (standard training fails past 4×4) `[V]`; **Mistral-7B >50 % GSM8K with zero intermediate tokens** `[V]` | Inference emits **no** reasoning tokens → maximal on-device win | 2405.14838 |
| 22 | **s1 / budget forcing** | Append `"Wait"` to extend, or force-terminate, the thinking block; 1 000 curated traces (s1K) | s1-32B **AIME24 50 % → 57 %** by extrapolating test-time budget `[V]` | Pure decoding-time; zero training cost | 2501.19393 |
| 23 | **Compute-optimal test-time scaling** | Difficulty-aware allocation between sequential revision and parallel search against a PRM | "**4× efficiency gain**" over best-of-N `[V]`; test-time compute can substitute for a ~14× larger model on easy/medium problems, but **pretraining wins on hard problems** `[M]` | Needs a verifier/PRM | 2408.03314 |
| 24 | **"Can 1B surpass 405B?"** | Systematic PRM × policy × difficulty sweep | **Llama-3.2-1B beats Llama-3.1-405B on MATH-500**; a 0.5 B model beats GPT-4o; 3 B beats 405 B; 7 B beats o1 and R1 `[V]`. **Caveat: the search is guided by a 7–8 B PRM, so total inference params ≫ 1 B** `[M]` | ×64–256 samples + PRM forward passes | 2502.06703 |
| 25 | **Large Language Monkeys** | Repeated sampling, measure coverage (pass@k) | Coverage is **log-linear in samples across four orders of magnitude** `[V]`; SWE-bench-Lite **15.9 % (1 sample) → 56 % (250 samples)** with DeepSeek-Coder-V2, beating the 43 % single-attempt SOTA `[M]` | Only converts to accuracy when a **verifier** exists `[V]` | 2407.21787 |
| 26 | **Verification is required** | Shows unverified sequential test-time scaling is provably suboptimal | Sharpens #25's caveat | — | 2502.12118 `[M]` |
| 27 | **R1 distillation** | SFT on 800 k R1-generated traces | 800 k samples "significantly enhances the reasoning abilities of smaller models" `[V]`; R1-Distill-Qwen-1.5B ≈ MATH-500 83.9 / AIME24 28.9 `[M]` — but the base was **Qwen2.5-Math-1.5B**, already math-pretrained | SFT only, no RL | 2501.12948 |
| 28 | **OpenThoughts** | Open reasoning SFT data recipes | OpenThoughts-114k → OpenThoughts2-1M → **OpenThoughts3-1.2M (850 k math / 250 k code / 100 k science)** `[V]`; OpenThinker3-7B: **AIME24 69.0, MATH500 90.0, GPQA-D 53.7, LCB 51.7** `[V]` (vs OpenThinker-7B 30.7 / 82.8 / 38.6 / 26.1 `[V]`) | Data only — **the cheapest capability we can buy** | 2506.04178 `[M]` |
| 29 | **LIMO** | 817 curated reasoning samples beat 100 k-sample SFT `[M]` | Extreme data efficiency for reasoning | Trivial compute | 2502.03387 |
| 30 | **Latent-CoT critique** | Probes Huginn's hidden space across recurrences | "little evidence for latent CoT"; probing is **discontinuous** across recurrent blocks; "**we still NEED explicit CoT** to achieve optimal performance" `[V]` | — | 2507.02199 (COLM'25 wkshp) |
| 31 | **Mixture of LoRAs for recursion** | Per-recursion LoRA mixtures to relax tying | Direct successor to #6/#8 | — | 2512.12880 |
| 32 | **AdaPonderLM** | Gated pondering LM with token-wise adaptive depth | Modern PonderNet-for-LMs | — | 2603.01914 |
| 33 | **Token-level adaptive latent CoT pretraining** | Adaptive latent CoT injected during **pretraining** | Most aligned with a from-scratch build | — | 2602.08220 |
| 34 | **Bridging latent & explicit reasoning w/ looped transformers** | Ties loop count to CoT step count | Directly addresses the §2.3 gap | — | 2606.31779 |
| 35 | **Hyperloop Transformers / Training-Free Looped Transformers / Loop as a Bridge** | 2026 follow-ups on loop scaling and post-hoc looping | Emerging; monitor | — | 2604.21254 / 2605.23872 / 2601.10242 |

### 2.2 Huginn-3.5B — exact configuration (verified from source)

Retrieved from `seal-rg/recurrent-pretraining`, `recpre/model_registry.py`, config `nebel-raven-3.5b` `[V]`:

| Field | Value |
|---|---|
| `n_embd` | 5280 |
| `num_attention_heads` / `num_key_value_heads` | 55 / 55 (**no GQA**) |
| `n_layers_in_prelude` | **2** |
| `n_layers_in_recurrent_block` | **4** |
| `n_layers_in_coda` | **2** |
| `mean_recurrence` | **32** |
| `mean_backprop_depth` | **8** (truncated BPTT) |
| `sampling_scheme` | `poisson-lognormal-filling` |
| `block_size` / `vocab_size` / `intermediate_size` | 4096 / 65536 / 17920 |
| `block_class_name` | `SandwichBlock` (norm before **and** after each sublayer) |
| `tie_embeddings` | True |

From `raven_modeling_minimal.py` `[V]`:
- State init: `x = torch.randn_like(input_embeds) * init_values["std"]` — **random**, not zero.
- Injection every step: `adapter(torch.cat([x, input_embeds], dim=-1))` — concat + linear back to `d`.
- Embedding scale `sqrt(d)`; output-projection init `sqrt(2/(5d)) / sqrt(2·n_layers)`.
- **KV cache is stored per recurrence step**: `HuginnDynamicCache` is indexed `cache[layer_or_recurrent_step][position]`. The code ships compression strategies (`modulo`, `anchoring`, `relative`) precisely because the naive cache is unaffordable. **This is the single biggest deployment landmine (§6.2).**
- Training-time recurrence is sampled per micro-batch: `rate = torch.poisson(...)`.
- `_prefill_with_varied_exit_steps()` + `PerIterationExitEvaluator` exist → per-token adaptive exit is implemented at inference.

Derived parameter accounting (d=5280, ffn=17920, SwiGLU): ≈ **395 M params per layer**, so the **shared core (4 layers) is ≈1.58 B params** and prelude+coda ≈1.58 B, plus 346 M tied embeddings ⇒ ≈3.5 B total. At r=32 the unrolled network is **2 + 128 + 2 = 132 layers**; a *non-recurrent* 132-layer model at this width would be ≈**52 B params** — which is exactly the paper's "computation load equivalent to 50 B parameters" claim `[V]`. **Parameter compression factor ≈ 15×.**

### 2.3 The most important negative result in this literature

Two independent measurements say **latent depth does not replace token-space CoT for arithmetic/multi-step math**:

- Huginn, GSM8K, CoT suppressed: **3.11 → 4.47 → 4.78 → 4.93 → 4.70 → 4.93 → 4.62 %** for r = 4, 8, 16, 32, 64, 128, 256 `[V]`. Flat from r=16 onward. With explicit CoT at r=32: **24.9 strict / 38.1 flexible** `[V]`. Latent depth buys ~1.8 points; the CoT prompt buys ~33 points.
- COCONUT: GSM8K **34.9 (latent) vs 42.9 (explicit CoT)** `[V]`, but ProsQA **97.0 vs 77.5** `[V]`. 2602.01148 explains the split theoretically as an **Exploration–Execution trade-off**: high decisional certainty ⇒ precise execution but no exploration; low certainty ⇒ good search but error accumulation `[V]`.

**Interpretation for Prophet:** recurrent depth is a *compute-per-token amplifier and parameter-compressor*, useful for the "execution engine" (parsing, retrieval-from-context, planning, per-step arithmetic reliability, code token-level correctness). It is **not** a replacement for emitting reasoning tokens on GSM8K/MATH. The winning configuration is **latent depth × short explicit CoT**, not one or the other.

---

## 3. What actually transfers to our scale

### 3.1 Transfers cleanly (high confidence)

1. **Iso-FLOP looped ≈ full-depth, ≫ iso-param shallow.** 2502.17416's controlled result (k⊗L ≈ kL⊗1 ≫ k⊗1) `[V]` is scale-free-ish and was demonstrated at exactly our size band. This is the load-bearing claim for Prophet.
2. **~3× unique-parameter reduction at ≥360 M.** MoR `[V]`. Our mini (300–600 M dense) sits right at that boundary; the main model is well above it.
3. **Randomised recurrence during training ⇒ runtime-tunable depth.** Huginn trains with Poisson-log-normal r (mean 32) and evaluates at r ∈ {1…256} `[V]`. This is *precisely* the "one checkpoint, three devices" property Prophet needs.
4. **Truncated BPTT (k≈4–8) is sufficient.** Huginn used `mean_backprop_depth=8` at r̄=32 `[V]` — backprop through 25 % of the unrolled graph. Activation memory becomes O(k·n_core) instead of O(r·n_core), which is what makes this trainable on a single 80 GB card.
5. **Selective per-token iteration is nearly free accuracy.** TaH: **+3.8–4.4 %** while iterating on **7 %** of tokens `[V]`. Applied on top of a *base already at our target scale* (Qwen3-1.7B / 0.6B) `[V]`. This is the highest measured accuracy-per-FLOP in the whole table.
6. **Early exit gives 2–3× decode throughput** (LayerSkip 1.8–2.2× `[V]`, Relaxed-Recursive continuous depth-wise batching 2–3× `[V]`, MoR 2.06× `[V]`). A looped model gets self-speculative decoding *for free*: r=1 is the draft, r=k is the verifier, same weights, no draft model in memory.
7. **Reasoning SFT data is the cheapest capability we can buy.** OpenThoughts3-1.2M `[V]`, OpenR1, R1-distill 800 k `[V]`, LIMO's 817-sample result `[M]`. Cost to us: a few A100-hours of SFT, not hundreds.

### 3.2 Transfers with major caveats

1. **Huginn's absolute quality is bad** (MMLU 31.4 at r=32 vs OLMo-2-7B 60.6 `[V]`). But this is a *data/token* story (800 B tokens, no instruction or reasoning SFT), not an indictment of recurrence. Do not use Huginn's absolute numbers as an upper bound; use the *iso-FLOP* studies (#4, #5, #6).
2. **MoR is worse than vanilla at 135 M** `[V]`. Our planned ablation size is 125 M (§7). **This is a live false-negative risk** and the ablation plan explicitly hedges it with a 350 M confirmation run.
3. **PRM-guided search "1B > 405B"** `[V]` requires a 7–8 B process reward model `[M]`. On a 5090 that is affordable (share the trunk, add a value head — §4.6). On an iPhone it is not.
4. **Best-of-N without a verifier saturates.** Coverage scales log-linearly `[V]`, but majority-vote selection plateaus around 10²  samples `[M]`, and 2502.12118 argues unverified sequential scaling is suboptimal `[M]`.

### 3.3 Does *not* transfer / should be avoided

1. **Pure latent reasoning as a CoT replacement on math** — see §2.3. Budget for short explicit CoT.
2. **Quiet-STaR-style rationale sampling at every token position** — the training cost is enormous (parallel rationale generation per position) and the reported gain (GSM8K 5.9→10.9 on a 7 B `[V]`) is small relative to what 50 k distilled traces buy.
3. **ACT's halting as originally formulated** — documented instability `[V]`. Use PonderNet's geometric-prior KL `[V]` or TaH's supervised two-stage decider `[V]`.
4. **Dynamic per-token depth on the ANE.** CoreML / ANE wants static shapes and a static graph. Token-level depth routing is a GPU/Metal feature; the phone gets *sequence-level* or *bucketed* depth (§4.7).
5. **From-scratch pretraining that expects to beat Qwen3-1.7B on MMLU-Pro/GPQA.** At 1.5e20 FLOPs this is arithmetically out of reach (§1.1). Target GSM8K / MATH / HumanEval / ARC-C first, and treat 2511.07384-style **retrofit of an open base** as the strategic hedge (§4.8).

---

## 4. Recommendation for Prophet

### 4.1 The one design: **Prophet-Loop** — a shared recurrent core with token-selective ponder depth, cross-step KV sharing, and depth-conditioned LoRA

```
tokens ─► embed (×√d) ─► PRELUDE  (n_pre dense layers)  ──►  e   [computed once]
                                                              │
                              ┌───────────────────────────────┘
                              ▼
      s₀ ~ N(0, σ²I)  ──►  ┌──────────────────────────────────────────┐
                           │  for i = 1 … r  (r sampled in training,  │
                           │                  dialled at inference)   │
                           │   x  = Adapter([ s_{i-1} ; e ])          │  ← input re-injection
                           │   x  = CORE(x, step=i)   (n_core layers, │
                           │            weights SHARED across i,      │
                           │            + depth-LoRA_i, + step embed) │
                           │   s_i = x                                │
                           │   λ_i = σ(w·LN(s_i))    ← ponder head    │
                           │   [router may freeze a token at step i]  │
                           └──────────────────────────────────────────┘
                              │
                              ▼
                        CODA (n_coda dense layers) ─► RMSNorm ─► lm_head (tied)
```

Six design decisions, each traced to evidence:

| Decision | Choice | Why |
|---|---|---|
| Topology | prelude → shared core → coda (Huginn shape) | Verified config `[V]`; prelude/coda absorb tokenisation and readout so the core learns a *pure iterated update* |
| Input injection | `Adapter(concat[s, e])`, `2d → d` linear | Exactly Huginn's mechanism `[V]`; prevents the loop from forgetting the prompt; costs 2d² params (8.4 M at d=2048) |
| State init | `randn × σ`, σ = 0.4/√d | Huginn `[V]`; random init forces path-independence → the map converges to an attractor rather than memorising a fixed trajectory, which is what makes r generalise beyond training depth |
| Normalisation | **Sandwich norm** (RMSNorm pre *and* post each sublayer), out-proj init `√(2/5d)/√(2L)` | Huginn's `SandwichBlock` `[V]` survived 4096-GPU training at unrolled depth 132. **Ablate against DeepLoop's α=(2N)^½, β=(8N)^{−½} DeepNorm rule** `[V]` |
| **KV across steps** | **Compute K,V only at step i=1; all later steps re-use them** (queries recomputed every step) | Kills the Huginn KV blow-up `[V]`; MoR's "recursive KV sharing" precedent `[V]`. Reduces KV cache **3.4–11×** (§5.4). Ablate `share` vs `modulo-2` vs `full` |
| Breaking exact tying | **Depth-conditioned LoRA** (rank 16 on W_q, W_o, and the MoE router) + additive step embedding | Relaxed Recursive Transformers `[V]`, TaH's depth-aware LoRA `[V]`, Mixture-of-LoRAs 2512.12880. Costs <0.1 % params, converts "same function r times" into "r related functions" |

### 4.2 Concrete configurations

**Prophet-Loop-9B (main; 5090 + Mac Studio)**

| | |
|---|---|
| d_model / heads / kv-heads | 2048 / 16 (head_dim 128) / 4 (GQA) |
| Prelude / Core / Coda | 4 dense / **4 MoE (shared, looped)** / 4 dense |
| Dense FFN (prelude+coda) | SwiGLU 5632 |
| Core MoE | 256 experts, **top-4**, expert FFN 1408, + 1 shared expert FFN 2816 |
| Router | depth-conditioned: `router(x + step_emb[i])` ⇒ **different experts at different loop steps** |
| Vocab / embeddings | 65 536, tied |
| **Total params** | **≈ 9.48 B** |
| **Active params @ r** | 566 M (r=1) · 753 M (r=4) · **1.00 B (r=8)** · **1.50 B (r=16)** · 2.50 B (r=32) |
| Effective depth | 8 + 4r layers → **40 @ r=8**, **72 @ r=16** |
| Training r̄ / k_bptt | curriculum to mean 8, k=3 |

**Prophet-mini-143M-Loop (iPhone 17 Pro / ANE)** — dense, no MoE

| | |
|---|---|
| d_model / heads / kv-heads | 1024 / 16 (64) / 4 (GQA) |
| Prelude / Core / Coda | 2 / **4 (shared)** / 2 |
| FFN | SwiGLU 2816 |
| Vocab | 49 152, tied |
| **Total params** | **≈ 142.6 M** (core block alone = 45.1 M) |
| Shipping depth | r = 2 default, r = 4 on "think harder", r = 8 plugged-in |

> Note on the brief's "mini ≈300–600 M dense": a 143 M looped core at r=8 delivers the *compute* of a 456 M dense model (§5.3) at 31 % of the weight footprint. If we want 300–600 M of *stored* capacity we should spend it on **width or vocabulary**, not depth — depth is the thing recurrence gives us for free.

**Ablation model (125 M class, §7)**: d=768, 12/4 heads, FFN 2048, vocab 32 768. Baseline `D16` = 16 dense layers = 125.9 M. Looped `L(2,3,2)@r=4` = 70.4 M params, identical effective depth 16.

### 4.3 Loss

Let `y` be targets, `r` the sampled depth for this micro-batch, `D ⊂ {1..r}` a small set of **read-out depths** (the coda + head applied at intermediate states), `λ_i` the ponder head's conditional halting probability at step i, and `p_i = λ_i ∏_{j<i}(1−λ_j)` the induced halting distribution.

```
L = L_final  +  α · L_anydepth  +  β · L_ponder  +  γ · KL(p ‖ Geom(λ_p))  +  δ · L_moe-balance

L_final     = CE( head(coda(s_r)), y )                       # the depth we were asked for
L_anydepth  = Σ_{d ∈ D}  w_d · CE( head(coda(s_d)), y )      # makes ANY depth a valid exit
L_ponder    = Σ_{i=1..r}  p_i · CE( head(coda(s_i)), y )     # PonderNet reconstruction (2107.05407)
KL(p‖Geom)  = Σ_i p_i log( p_i / ((1−λ_p)^{i−1} λ_p) )       # geometric prior, λ_p = 1/r̄
```

- `L_anydepth` is the mechanism that makes **r a runtime dial**. `D` = {1, ⌈r/2⌉, r} with `w = (0.15, 0.25, 0.60)`. Cost: 2 extra coda+head passes (coda is 4 of 40 effective layers ⇒ ~12 % overhead).
- `α = 0.3`, `β = 0.0` for phases A–B, `β = 0.5` and `γ = 0.01` only in phase C (see schedule). **Turn ponder on last** — halting heads destabilise early training (ACT's documented failure `[V]`).
- `δ = 0.01` load-balancing over the MoE router, computed **per loop step** (a step-agnostic balance loss lets step 1 hog the experts).

**Halting supervision (the TaH trick, adopted).** After phase B, run a cheap offline pass: for each token, record the smallest depth `d*` at which the argmax prediction becomes and stays correct. Train the ponder head with **binary cross-entropy against `1[i ≥ d*]`** for one epoch (TaH stage 1: `IterLabelDecider` `[V]`), then switch to the PonderNet KL objective (TaH stage 2: `MLPIterDecider` `[V]`). This gave TaH +3.8–4.4 % over always-iterate while skipping 93 % of iterations `[V]` and is far more stable than learning halting from scratch.

### 4.4 Training schedule — sampling depth

Total budget assumption: **200 A100-h ≈ 1.0e20 FLOPs ≈ 20 B tokens** for the main model at r̄=6 (§5.1).

| Phase | Tokens | r sampling | k_bptt | Loss terms on | Rationale |
|---|---|---|---|---|---|
| **A — warm start** | 0 → 4 B | **r = 1** fixed | — | `L_final` | Curriculum: 2511.07384 shows a *curriculum of recurrences* "preserves performance while reducing total computational cost" `[V]`; 2602.01148 proves **direct training on the deep objective provably fails** from distributional mismatch `[V]` |
| **B — ramp** | 4 → 9 B | `r ~ Uniform{1,2,3,4}` linearly shifting to `{2,3,4,5,6}` | 2 | `+ α·L_anydepth` | Introduces depth generalisation; anydepth loss keeps r=1 usable as a draft model |
| **C — deep** | 9 → 17 B | `r = 1 + PoissonLogNormal(μ=log 6, σ=0.5)`, clipped to [1, 24] | 3 | `+ L_anydepth` | Huginn's `poisson-lognormal-filling` `[V]`. Heavy tail is what makes r=16–32 work at test time despite r̄=6 |
| **D — ponder head** | 17 → 19 B | same, r̄=6 | 3 | `+ β·L_ponder + γ·KL` (+ TaH-supervised BCE for the first 0.3 B) | Halting learned on a stable base |
| **E — reasoning SFT** | ~1 B tok equiv | `r ~ {4, 8, 12}` | 4 | `L_final + L_anydepth` | OpenThoughts3 / OpenR1 / R1-distill traces `[V]`. Train at the depths we ship |

**Depth-sampling implementation detail that matters:** sample **one r per micro-batch, not per sequence.** Per-sequence r forces ragged loop counts and destroys throughput; per-micro-batch r keeps the graph static. Use a **depth-stratified gradient accumulation** cycle: within one optimizer step accumulate micro-batches at r ∈ {1, 4, 6, 12} so every update sees the whole depth range (this is what prevents the "trained at r̄=6, collapses at r=16" failure).

### 4.5 PyTorch sketch

```python
# prophet/model/loop.py  (sketch — omits MoE, RoPE, flash-attn plumbing)
import math, contextlib, torch, torch.nn as nn, torch.nn.functional as F

class SandwichBlock(nn.Module):
    """RMSNorm before AND after each sublayer (Huginn's SandwichBlock).
    Optional depth-conditioned LoRA makes the shared block a *family* of r blocks."""
    def __init__(self, cfg, depth_lora: int = 0):
        super().__init__()
        d = cfg.d_model
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.n3, self.n4 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)          # GQA + SwiGLU (or MoE)
        self.step_emb = nn.Embedding(cfg.r_max, d) if depth_lora else None
        if depth_lora:                                           # rank-`depth_lora` per step, on W_q / W_o
            r_, R = depth_lora, cfg.r_max
            self.lq_a = nn.Parameter(torch.zeros(R, d, r_)); self.lq_b = nn.Parameter(torch.zeros(R, r_, d))
            self.lo_a = nn.Parameter(torch.zeros(R, d, r_)); self.lo_b = nn.Parameter(torch.zeros(R, r_, d))
            nn.init.normal_(self.lq_a, std=1/math.sqrt(d)); nn.init.normal_(self.lo_a, std=1/math.sqrt(d))

    def forward(self, x, step, kv_cache=None, write_kv=False):
        if self.step_emb is not None:
            x = x + self.step_emb.weight[step]                   # cheap depth conditioning
        lora = None if self.step_emb is None else (self.lq_a[step], self.lq_b[step],
                                                   self.lo_a[step], self.lo_b[step])
        # write_kv=True ONLY on step 0 -> K/V are shared across recurrence steps (MoR-style)
        h = self.n2(self.attn(self.n1(x), kv_cache=kv_cache, write_kv=write_kv, lora=lora))
        x = x + h
        x = x + self.n4(self.mlp(self.n3(x)))
        return x


class ProphetLoop(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed  = nn.Embedding(cfg.vocab, cfg.d_model)
        self.prelude = nn.ModuleList([SandwichBlock(cfg) for _ in range(cfg.n_pre)])
        self.core    = nn.ModuleList([SandwichBlock(cfg, depth_lora=cfg.lora_rank)
                                      for _ in range(cfg.n_core)])          # <-- SHARED across steps
        self.coda    = nn.ModuleList([SandwichBlock(cfg) for _ in range(cfg.n_coda)])
        self.adapter = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)  # concat-injection
        self.norm_f  = nn.RMSNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight                                # tied
        self.halt    = nn.Sequential(nn.RMSNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = math.sqrt(2 / (5 * self.cfg.d_model))
            if getattr(m, "_is_out_proj", False):                # residual-stream writers
                std /= math.sqrt(2 * (self.cfg.n_pre + self.cfg.n_core + self.cfg.n_coda))
            nn.init.normal_(m.weight, std=std)

    def read_out(self, s, kv):
        for b in self.coda:
            s = b(s, step=0, kv_cache=kv, write_kv=kv.needs_write("coda"))
        return self.head(self.norm_f(s))

    def forward(self, idx, r, k_bptt=3, readout_depths=(), kv=None, targets=None):
        cfg, kv = self.cfg, (kv or KVCache(self))
        e = self.embed(idx) * math.sqrt(cfg.d_model)
        for b in self.prelude:
            e = b(e, step=0, kv_cache=kv, write_kv=kv.needs_write("prelude"))

        s = torch.randn_like(e) * cfg.init_std                    # random state init
        n_nograd = max(0, r - k_bptt)                             # truncated BPTT window
        logits, lam = {}, {}
        for i in range(r):
            ctx = torch.no_grad() if i < n_nograd else contextlib.nullcontext()
            with ctx:
                x = self.adapter(torch.cat([s, e], dim=-1))       # re-inject the prompt every step
                for j, b in enumerate(self.core):
                    # K/V written ONLY on the first recurrence step -> KV cache is O(n_pre+n_core+n_coda)
                    x = b(x, step=min(i, cfg.r_max - 1), kv_cache=kv,
                          write_kv=(i == 0) and kv.needs_write(f"core{j}"))
                s = x                                             # detached automatically while i < n_nograd
            if (i + 1) in readout_depths or i == r - 1:
                logits[i + 1] = self.read_out(s, kv)
                lam[i + 1]    = torch.sigmoid(self.halt(s)).squeeze(-1)

        if targets is None:
            return logits, lam
        return self.loss(logits, lam, targets, r)

    def loss(self, logits, lam, y, r, alpha=0.3, beta=0.0, gamma=0.0, lam_prior=None):
        ce = lambda lg: F.cross_entropy(lg[:, :-1].flatten(0, 1), y[:, 1:].flatten())
        L = ce(logits[r])
        depths = sorted(logits)
        if alpha:                                                  # "any depth is a valid exit"
            w = {d: v for d, v in zip(depths, (0.15, 0.25, 0.60))}
            L = L + alpha * sum(w[d] * ce(logits[d]) for d in depths)
        if beta:                                                   # PonderNet
            p, rem = {}, torch.ones_like(lam[depths[0]])
            for d in depths:
                p[d] = rem * lam[d]; rem = rem * (1 - lam[d])
            p[depths[-1]] = p[depths[-1]] + rem                    # force halt at r
            L = L + beta * sum((p[d].mean()) * ce(logits[d]) for d in depths)
            lp = lam_prior or (1.0 / r)
            prior = torch.tensor([(1 - lp) ** (d - 1) * lp for d in depths], device=L.device)
            prior = prior / prior.sum()
            q = torch.stack([p[d].mean() for d in depths]); q = q / q.sum()
            L = L + gamma * (q * (q.clamp_min(1e-9).log() - prior.log())).sum()
        return L


# ---- training loop: depth-stratified accumulation (one r per micro-batch) -------------
DEPTH_CYCLE = [1, 4, 6, 12]                 # phase C; every optimizer step spans the range
for step in range(max_steps):
    opt.zero_grad(set_to_none=True)
    for micro, r in zip(loader.take(len(DEPTH_CYCLE)), DEPTH_CYCLE):
        loss = model(micro.x, r=r, k_bptt=3,
                     readout_depths=(1, max(1, r // 2), r), targets=micro.y)
        (loss / len(DEPTH_CYCLE)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

**Inference, depth dial (the whole point):**

```python
@torch.inference_mode()
def generate(model, idx, device_profile, max_new=512):
    P = {"iphone":  dict(r_min=1, r_max=4,  mode="fixed",     r=2),
         "iphone+": dict(r_min=1, r_max=8,  mode="ponder",    tau=0.9),
         "mac":     dict(r_min=2, r_max=12, mode="ponder",    tau=0.95),
         "5090":    dict(r_min=2, r_max=32, mode="ponder",    tau=0.98, spec_draft_r=1),
        }[device_profile]
    ...
    # mode="fixed"  -> static graph, CoreML/ANE-exportable, r bucketed to {1,2,4,8}
    # mode="ponder" -> accumulate p_i; stop when Σp ≥ tau or i == r_max; floor at r_min
    # spec_draft_r  -> self-speculative decoding: draft with r=1, verify with r=r_max (LayerSkip 2404.16710)
```

### 4.6 Explicit test-time compute — what we bolt on, per device

| Device | Depth r | Token CoT | Search | Verifier |
|---|---|---|---|---|
| **iPhone 17 Pro (mini-143M)** | fixed r=2 (r=4 on "think harder", r=8 plugged-in) | short CoT, ≤256 tokens, **budget-forced** (s1's `"Wait"` / forced `</think>`, zero training cost, 2501.19393 `[V]`) | none | none |
| **Mac Studio (9B, MLX)** | ponder, r ∈ [2,12] | full CoT | self-consistency (maj@8) | none |
| **RTX 5090 (9B, FP4)** | ponder, r ∈ [2,32] + self-speculative (draft r=1) | full CoT | **best-of-16 / beam-of-4 over steps** | **PRM as a value head on the coda** — shares the entire trunk, costs `d×1` params, one extra head pass per candidate step |

The PRM-as-shared-head design is the key affordability trick: 2502.06703's "1 B beats 405 B" result needs a 7–8 B PRM `[M]`, which we cannot afford as a separate model. Training a scalar value head on the coda against Math-Shepherd-style automatic step labels (2312.08935 `[M]`) or PRM800K `[M]` costs ~2 A100-h and adds ~2 K params. Expect a *fraction* of the published gain — but 2408.03314's "4× efficiency gain" from difficulty-aware allocation `[V]` is mostly about *allocation*, which a cheap verifier already enables.

### 4.7 How depth is dialled down for iPhone and up for a 5090

**Down (iPhone / ANE):**
1. **Static bucketing.** Export exactly four CoreML graphs, r ∈ {1, 2, 4, 8}, sharing one weight blob (the core is stored once — that is the whole point). ANE requires static shapes; per-token routing is forbidden. Selection is per *request*, not per token.
2. **`L_anydepth` guarantees r=2 is a first-class model**, not a degraded one. Without this term, low-r inference falls off a cliff — this is the number-one reason to keep α>0 for the whole run.
3. **KV sharing across steps** makes the cache identical to an 8-layer model (§5.4) — 4 KB/token at fp8, 33 MB at 8 K context.
4. **Core cache-residency.** The core is 45.1 M params = **25 MB at int4**, which plausibly fits the A19 Pro's system-level cache. Looping then costs compute but **not DRAM traffic** (§5.5). This is the crucial device-level asymmetry: on a bandwidth-bound phone, depth is nearly free and parameters are not.

**Up (5090):**
1. Ponder head with τ=0.98, r_max=32; expected mean r ≈ 3–5 given TaH's finding that 93 % of tokens need no extra iteration `[V]`.
2. **Self-speculative decoding**: r=1 draft, r=r_max verify, same weights. LayerSkip reports 1.8–2.2× `[V]`; Relaxed-Recursive reports 2–3× with continuous depth-wise batching `[V]`.
3. **Continuous depth-wise batching** (2410.20672 `[V]`): tokens at different loop depths are packed into the same core-block matmul. This is what turns variable depth from a throughput disaster into a throughput *win*.
4. Then, and only then, spend the remaining budget on best-of-N + PRM.

### 4.8 Strategic hedge (read this before committing 200 A100-hours)

Given §1.1, there is a materially higher-expected-value path that this track must put on the table: **retrofit an open base into Prophet-Loop rather than pretrain the core from scratch.** 2511.07384 reports that converting a pretrained non-recurrent LM to recurrent, with a curriculum of recurrences, **beats simply post-training the original model at equal compute, on mathematics** `[V]`; 2410.20672 shows a recursive Gemma-1B recovers most of Gemma-2B's quality `[V]`. Concretely: take Qwen3-1.7B-Base (or SmolLM3-3B), map its 28 layers to prelude(4) + core(6, initialised by *averaging* or by *stepwise* layer assignment) + coda(4), attach depth-LoRA, and run the phase-B→E curriculum for 5–10 B tokens (~60–100 A100-h). This is *exactly* the setup TaH used to get +3.0–3.8 % over single-iteration Qwen3-1.7B `[V]`.

**Recommendation: run the §7 ablations on our own 125 M models (they are architecture questions, not data questions), but keep the retrofit as the default path for the shipped model unless "from scratch" is a hard project constraint.** If it is hard, we should still expect to lose on MMLU-Pro/GPQA and should target GSM8K / MATH / HumanEval / ARC-C.

---

## 5. Compute & memory budget

### 5.1 Training FLOPs

Per token, with truncated BPTT window `k` and recurrence `r`:

```
FLOPs_fwd ≈ 2 · [ P_dense + r · P_core + P_head ]
FLOPs_bwd ≈ 4 · [ P_dense + k · P_core + P_head ]        # only the last k steps are differentiated
```

**Prophet-Loop-9B**, P_dense = 361 M, P_core = 62.4 M *active*, P_head = 134 M, r̄=6, k=3:

```
fwd = 2 · (361 + 6·62.4 + 134) M = 2 · 869 M = 1.74 GFLOP
bwd = 4 · (361 + 3·62.4 + 134) M = 4 · 682 M = 2.73 GFLOP
tot ≈ 4.5 GFLOP / token
20 B tokens ⇒ 9.0e19 FLOPs ⇒ ≈ 180 A100-h at 42 % MFU.      ✅ inside budget
```

For comparison, **Huginn's run** at r̄=32, k=8, P_core=1.58 B, P_dense=1.58 B:
`fwd = 2·(1.58 + 32·1.58) = 104 GFLOP`, `bwd = 4·(1.58 + 8·1.58) = 56.9 GFLOP` ⇒ **161 GFLOP/token × 800 B ≈ 1.3e23 FLOPs ≈ 250 000–350 000 A100-equivalent hours.** We have ~1/1000 of that. Every design choice below is downstream of that ratio.

**Truncated BPTT is a training-cost *saving*, not just a memory saving:** at iso-effective-depth, the looped model's backward pass touches only `k` core repetitions instead of `r`. At r=4/k=2 in the ablation config this makes the looped run **~20 % cheaper per token than the equal-depth dense baseline** (§7).

### 5.2 Inference FLOPs per decoded token

`FLOPs(r) ≈ 2 · [P_dense + r·P_core + P_head]` (matmuls only; attention scores add ~2·L·s·d which is <5 % at s≤4 K).

**Prophet-mini-143M** (P_dense = 45.1 M, P_core = 45.1 M, P_head = 50.3 M):

| r | effective layers | params used | FLOPs/token |
|---|---|---|---|
| 1 | 8 | 140.5 M | 0.28 GFLOP |
| 2 | 12 | 185.6 M | 0.37 GFLOP |
| 4 | 20 | 275.8 M | 0.55 GFLOP |
| 8 | 36 | 456.2 M | 0.91 GFLOP |
| 16 | 68 | 816.8 M | 1.63 GFLOP |
| 32 | 132 | 1 538 M | 3.08 GFLOP |

**Prophet-Loop-9B** (P_dense = 361 M, P_core = 62.4 M active, P_head = 134 M):

| r | effective layers | active params | FLOPs/token |
|---|---|---|---|
| 1 | 12 | 566 M | 1.13 GFLOP |
| 4 | 24 | 753 M | 1.51 GFLOP |
| 8 | 40 | **1.00 B** | 2.01 GFLOP |
| 16 | 72 | **1.50 B** | 3.00 GFLOP |
| 32 | 136 | 2.50 B | 5.00 GFLOP |

### 5.3 Parameters saved (the headline)

A non-recurrent model with the same effective depth would need `4r` core layers instead of 4.

**Prophet-mini-143M** (11.27 M/layer, 50.3 M embeddings):

| r | eff. depth | Prophet-mini stored | dense equivalent | **params saved** |
|---|---|---|---|---|
| 4 | 20 | 142.6 M | 275.7 M | 133 M (**1.9×**) |
| 8 | 36 | 142.6 M | 456.1 M | 314 M (**3.2×**) |
| 16 | 68 | 142.6 M | 816.7 M | 674 M (**5.7×**) |
| 32 | 132 | 142.6 M | 1 537.8 M | 1 395 M (**10.8×**) |

**Prophet-Loop-9B** (2 243 M per core MoE layer):

| r | eff. depth | Prophet stored | dense-MoE equivalent | **params saved** |
|---|---|---|---|---|
| 4 | 24 | 9.48 B | 36.4 B | **3.8×** |
| 8 | 40 | 9.48 B | 72.3 B | **7.6×** |
| 16 | 72 | 9.48 B | 144 B | **15.2×** |

Sanity check against the literature: Huginn's own compression at r=32 is ≈15× (3.5 B stored, ≈52 B dense-equivalent) `[V]`, and the paper claims a *quality* equivalent of ~50 B `[V]`. MoR reports ~3× at 3 recursions `[V]`; ALBERT reports ~18× with full tying `[M]`. Our 7.6× at r=8 is squarely inside the demonstrated range.

**Fitting the targets:**
- 5090 (32 GB): 9.48 B at FP4 (~0.55 B/param incl. scales) = **5.2 GB weights**. A 72 B dense-MoE equivalent would be 40 GB — does not fit. **Recurrence is what makes the 5090 target reachable.**
- iPhone (≈8 GB RAM, ~3–4 GB usable): 142.6 M at int4 = **80 MB**. A 1.54 B dense equivalent (r=32) would be 850 MB — technically loadable but 10× the memory and 10× the DRAM traffic per token.

### 5.4 KV cache — the make-or-break number

Huginn stores KV **per recurrence step** `[V]`, so the cache has `n_pre + r·n_core + n_coda` layers. With cross-step KV sharing it is `n_pre + n_core + n_coda`, independent of r.

**Prophet-Loop-9B**, kv_dim = 4 heads × 128 = 512, fp8 K and V:

| Config | KV layers | bytes/token | 32 K context |
|---|---|---|---|
| Shared (recommended) | 12 | 12.3 KB | **0.39 GB** |
| Per-step, r=8 | 40 | 41.0 KB | 1.31 GB |
| Per-step, r=16 | 72 | 73.7 KB | 2.36 GB |
| Per-step, r=32 | 136 | 139 KB | 4.46 GB |

⇒ **3.3× saving at r=8, 11.3× at r=32.**

**Prophet-mini-143M**, kv_dim = 256, fp8:

| Config | KV layers | bytes/token | 8 K context |
|---|---|---|---|
| Shared | 8 | 4.1 KB | **33 MB** |
| Per-step, r=8 | 36 | 18.4 KB | 151 MB |
| Per-step, r=32 | 132 | 67.6 KB | 553 MB |

On an 8 GB phone with ~3–4 GB usable, 553 MB of KV for an 80 MB model is absurd. **Cross-step KV sharing is non-negotiable for the mini.**

### 5.5 Decode latency per token at depth k

Roofline, batch = 1, greedy. `t = max(bytes_moved / BW_eff , FLOPs / TFLOPS_eff)`.

**iPhone 17 Pro (mini-143M, int4).** Assumptions (flagged): peak LPDDR5X BW ≈ 60–77 GB/s `[M]`, achieved ≈ 40 GB/s; ANE achieved ≈ 1.5 TFLOP/s fp16 on this decode shape `[M]`. Core block int4 = **25 MB**; total weights = **80 MB**.

| r | weight traffic (core SLC-resident) | traffic-bound | FLOPs | compute-bound | **predicted** |
|---|---|---|---|---|---|
| 1 | 80 MB | 2.0 ms | 0.28 G | 0.19 ms | 2.0 ms → **500 tok/s** |
| 2 | 80 MB | 2.0 ms | 0.37 G | 0.25 ms | 2.0 ms → **500 tok/s** |
| 4 | 80 MB | 2.0 ms | 0.55 G | 0.37 ms | 2.0 ms → **500 tok/s** |
| 8 | 80 MB | 2.0 ms | 0.91 G | 0.61 ms | 2.0 ms → **500 tok/s** |
| 32 | 80 MB | 2.0 ms | 3.08 G | 2.05 ms | 2.05 ms → **490 tok/s** |

*Worst case (core NOT cache-resident, traffic = 80 + 25·(r−1) MB):* r=4 → 3.9 ms (256 tok/s); r=8 → 6.4 ms (156 tok/s); r=32 → 21 ms (48 tok/s).

**The headline result of this section:** at batch 1 on a bandwidth-bound device, arithmetic intensity is catastrophically low (<1 % of peak FLOPs are used), so **extra depth via loops is nearly free in wall-clock while extra depth via parameters costs linearly in DRAM traffic**. Even in the pessimistic non-resident case, r=8 costs 3.2× while buying the compute of a 3.2× larger model that would *also* cost 3.2× traffic — so recurrence is never worse than parameters and is much better when the core fits in cache. Apply a 2–3× real-world derate to all figures above (framework overhead, tokenizer, sampling, thermal): expect **~60–200 tok/s at r=2–8 on device.**

**RTX 5090 (9B, FP4).** BW 1 792 GB/s, achieved ≈ 1 250 GB/s; ≈400 TFLOP/s FP8 achieved-ish `[M]`. Per-token weight traffic = dense (503 M) + distinct experts touched (≤ 4 layers × r steps × 4 experts × 8.65 M):

| r | distinct expert params | traffic | traffic-bound | FLOPs | **predicted (theoretical)** |
|---|---|---|---|---|---|
| 1 | 138 M | 353 MB | 0.28 ms | 1.13 G | 3 500 tok/s |
| 4 | 554 M | 582 MB | 0.47 ms | 1.51 G | 2 150 tok/s |
| 8 | 1 107 M | 886 MB | 0.71 ms | 2.01 G | **1 410 tok/s** |
| 16 | 2 215 M | 1 495 MB | 1.20 ms | 3.00 G | 835 tok/s |
| 32 | 4 429 M (capped by 256-expert pool) | ≤2 700 MB | ≤2.2 ms | 5.00 G | ≥455 tok/s |

Derate 3–5× for real single-stream serving ⇒ **~300–450 tok/s at r=8**, entirely acceptable, and expert re-selection across steps (cache hits in L2) makes the high-r rows optimistic in our favour.

**Mac Studio M-Ultra (MLX):** ~800 GB/s `[M]`; roughly half the 5090 numbers; r ∈ [2,12] is the comfortable band.

### 5.6 Activation memory during training (single A100 80 GB)

With truncated BPTT window `k`, activations are stored for `n_pre + k·n_core + n_coda + |D|·n_coda` layers, i.e. **8 + 3·4 + 4 + 2·4 = 32 layer-equivalents** at k=3 for the 9B config — the same as training a 32-layer dense model, while the *forward* effective depth is 40 and can be pushed to 136 at inference. With `torch.utils.checkpoint` on the core only, add ~30 % recompute for another 2× memory headroom. **Sequence 4 096, micro-batch 4, bf16, d=2048 ⇒ ≈ 22 GB activations + 19 GB optimizer state (AdamW, bf16 params + fp32 m/v on the 9.5 B ⇒ needs 8-bit optimizer or sharding).** Note: at 9.5 B total params, AdamW fp32 states alone are 76 GB — **use 8-bit Adam + bf16 master weights, or the 9B config does not fit on one A100 at all.** This is a hard constraint that R02/R03 must confirm.

---

## 6. Risks & failure modes

Ordered by expected damage × probability.

### 6.1 🔴 "Latent depth doesn't help beyond r≈4 at our scale"
**Evidence for:** Huginn's non-CoT GSM8K is flat from r=16 (4.78 → 4.93 → 4.70 → 4.93 → 4.62) `[V]`. MoR at 135 M **underperforms vanilla** `[V]`. Gains from r=16→32 are "under half a point on several benchmarks" `[V]`.
**Evidence against:** 2502.17416's iso-FLOP result is clean and at our scale `[V]`; 2502.06703 and DeepLoop `[V]` show loop count keeps paying with the right normalisation.
**Mitigation:** Ablation A2 is the go/no-go. Require monotone val-loss improvement out to at least 2× the training mean depth. **If r>4 gives nothing at 125 M, re-run at 350 M before killing the idea** (MoR's 135 M-vs-360 M crossover is direct evidence of a size threshold).
**Kill criterion:** looped model fails to reach within 0.02 nats of the iso-depth dense baseline at 350 M ⇒ abandon deep recurrence, keep only r≤2 TaH-style refinement.

### 6.2 🔴 KV cache explosion
Huginn caches KV per recurrence step `[V]`; naive implementation gives 4.5 GB of KV at r=32/32 K on the 9B, and 553 MB on a *80 MB* phone model. **Mitigation:** cross-step KV sharing is in the design from day one (§4.1), with `modulo-2` as the fallback if sharing costs too much quality. Ablation A5 measures the quality/memory trade directly.

### 6.3 🟠 Vanishing / exploding gradients through the unrolled stack
At r=32 the unrolled net is 136 layers of *tied* weights — one shared update receives gradients from 32 visits, so the effective gradient scale grows with r. DeepLoop formalises this: the depth-scaling exponent must move **1/4 → 1/2** as loop count grows at fixed physical depth `[V]`.
**Mitigation:** (a) sandwich norm (Huginn's proven choice `[V]`); (b) out-proj init `√(2/5d)/√(2L)` `[V]`; (c) DeepNorm α=(2N)^½, β=(8N)^{−½} as ablation A4 `[V]`; (d) grad-clip 1.0; (e) truncated BPTT k≤4 bounds the product of Jacobians. **Watchdog:** log per-step grad-norm ratio ‖∂L/∂s_i‖/‖∂L/∂s_r‖ — if it exceeds 10 or falls below 0.05 within the k-window, the norm scheme is wrong.

### 6.4 🟠 "Latent overthinking" — more depth makes correct tokens wrong
TaH explicitly documents this: "most token predictions are already correct after the first pass, but are sometimes revised into errors in later iterations" `[V]`. This is why *always*-iterate underperforms *selective* iterate by 3.8–4.4 % at identical params `[V]`.
**Mitigation:** the `L_anydepth` term plus a **monotonicity regulariser** — penalise `max(0, CE(s_r) − CE(s_{r/2}))` so deeper is never worse. Plus the ponder head (which is a learned "stop before you break it").

### 6.5 🟠 Halting-head training instability
ACT is "not as stable as vanilla and needs careful hyper-parameter tuning" `[V]`. A halting head introduced too early collapses to always-halt-at-1 (degenerate) or never-halt (r_max every token).
**Mitigation:** (i) train halting only in phase D, on a converged base; (ii) TaH's **supervised** stage-1 with oracle `d*` labels before any RL-free differentiable objective `[V]`; (iii) PonderNet's geometric-prior KL rather than ACT's ponder cost `[V]`; (iv) hard floor `r_min` and ceiling `r_max` at inference so a broken head degrades to fixed depth.

### 6.6 🟠 Depth generalisation failure (train at r̄=6, deploy at r=16)
A model trained at a *fixed* r will not extrapolate. Huginn's Poisson-log-normal sampling with a heavy tail is what makes r=32→256 evaluation meaningful `[V]`.
**Mitigation:** depth-stratified gradient accumulation (§4.4) so every optimizer step spans {1,4,6,12}; log val loss at r ∈ {1,2,4,8,16,32} every 1 000 steps and treat divergence at high r as a training bug, not an eval curiosity.

### 6.7 🟠 Deployment: dynamic depth is hostile to ANE / CoreML
Per-token depth routing needs data-dependent control flow. CoreML/ANE wants static graphs.
**Mitigation:** ship **bucketed static graphs** r ∈ {1,2,4,8} sharing one weight blob on the phone; reserve per-token ponder for Metal/CUDA. Verify the export path in week 1 — a design that cannot be exported is worthless regardless of its ablations.

### 6.8 🟡 Gradient checkpointing / recompute overhead
Checkpointing the core costs ~30 % extra forward FLOPs. At r̄=6/k=3 the checkpointed region is only 3 core repetitions, so the real overhead is ~10 % of total. Acceptable. **But**: checkpointing interacts badly with the `no_grad` prefix — make sure the `i < n_nograd` steps are *not* also checkpointed (double work).

### 6.9 🟡 MoE × recurrence interaction (main model only)
If the router is step-agnostic, the same experts are selected at every loop step and the model degenerates to `r×` the same computation. **Mitigation:** depth-conditioned router (`x + step_emb[i]`), per-step load-balancing loss, and an ablation that logs expert-selection Jaccard overlap between consecutive steps (target < 0.5). Secondary risk: expert-cache thrashing on the 5090 as r grows — measured in §5.5.

### 6.10 🟡 We measure the wrong thing at 125 M
GSM8K is ~0 % at 125 M. Ablations must use **synthetic reasoning tasks with depth structure** — multi-digit addition, p-hop induction, parity, and templated symbolic word problems — which is exactly what 2502.17416 used to establish the k⊗L ≈ kL⊗1 result `[V]`. Val loss alone will *understate* the benefit of depth, because depth helps a small, reasoning-heavy slice of tokens.

### 6.11 🟡 The strategic risk (§1.1, §4.8)
The largest risk in this track is not architectural: it is that we spend 200 A100-hours pretraining from scratch and land at Qwen3-1.7B minus 15 points on MMLU-Pro because we are 2 500× short on pretraining FLOPs. Recurrence does not fix that. **Retrofit (2511.07384) or distillation into an existing base is the risk-adjusted play.**

---

## 7. Ablation plan

**Shared setup.** 125 M-class models, d=768, 12 q-heads / 4 kv-heads, SwiGLU 2048, vocab 32 768 tied, seq 2 048, FineWeb-Edu-dedup + 10 % open-web-math + 5 % StarCoder-py. **~3 B tokens per run ≈ 5–6 A100-h** at 35 % MFU (D16 baseline: 6N = 755 MFLOP/token ⇒ 3.1 B tokens in 2.36e18 FLOPs; the looped variant at r=4/k=2 costs **604 MFLOP/token, i.e. ~20 % cheaper**, so iso-token runs finish in ~4.8 h).

**Metrics for every run:** (a) held-out val loss (nats); (b) **DepthBench** — a synthetic suite of 4 tasks with tunable depth: n-digit addition (n=3..12), p-hop induction (p=2..8), parity over k bits, and 2–5-step templated symbolic word problems (per 2502.17416 `[V]`); (c) ARC-e / PIQA / HellaSwag zero-shot for sanity; (d) tokens/s decode at batch 1 on the A100; (e) peak train memory.

**Configurations named:** `D16` = 16 dense layers (125.9 M). `D7` = 7 dense layers (69.2 M, iso-param with the loop). `L(2,3,2)@r` = prelude 2 / core 3 / coda 2, looped r (70.4 M stored; effective depth 2+3r+2).

| ID | Question | Arms | Budget | **Kill / pass criterion** |
|---|---|---|---|---|
| **A1** | Does looping substitute for depth at iso-FLOP? *(THE experiment)* | `D16` · `D7` · `L(2,3,2)@r=4` | 3 × 6 h = **18 h** | **PASS** if `L@4` val loss ≤ `D16` + 0.02 nats **and** ≥ `D7` − 0.08 nats, at 56 % of `D16`'s params. **KILL the whole track** if `L@4` is worse than `D7`. |
| **A2** | Does depth generalise past training depth? | train `L(2,3,2)` with r ~ {2,3,4,5,6}; eval r ∈ {1,2,4,8,16,32} | **6 h** (1 train + free evals) | **PASS** if val loss decreases monotonically to r=8 and DepthBench-addition improves ≥ 5 pts from r=4→8. **Kill deep recurrence** (keep r≤2) if flat at r>4. |
| **A3** | How short can the BPTT window be? | k ∈ {1,2,4,8} at fixed r ~ {2..6} | 4 × 5 h = **20 h** (can drop k=8) | Pick the smallest k within 0.01 nats of k=8. Expect k=2–3. Reports the memory/quality Pareto directly. |
| **A4** | Which normalisation survives deep unrolling? | sandwich-norm · pre-LN · **DeepLoop DeepNorm** α=(2N)^½ β=(8N)^{−½} `[V]` | 3 × 6 h = **18 h**, 2 seeds each on the 2 survivors | Score = final val loss **and** number of grad-norm spikes >5×EMA. DeepLoop claims neutrality at R=1 and gains at large R `[V]` — verify at r=8. |
| **A5** | Can we share KV across recurrence steps? | full per-step KV · **share-from-step-1** · modulo-2 · anchoring | 3 × 5 h = **15 h** | **PASS sharing** if Δval ≤ 0.015 nats vs full, for a 3.3–11× KV reduction (§5.4). If Δ > 0.03, fall back to modulo-2. |
| **A6** | Which halting mechanism? | fixed r · entropy threshold · PonderNet KL · **TaH two-stage supervised decider** | 4 × 4 h = **16 h** (all start from the A2 checkpoint) | Compare accuracy at **matched mean FLOPs**. Target: TaH-style ≥ +2 pts DepthBench over fixed-r at equal mean depth, with mean r ≤ 1.3× the fixed baseline (TaH reports 93 % skip `[V]`). |
| **A7** | Does input re-injection matter? | concat+adapter · additive `s+e` · no injection | 3 × 5 h = **15 h** | Expect concat ≫ none. If additive ties concat, drop the 2d² adapter (saves 8.4 M on the 9B, 2.1 M on the mini). |
| **A8** | Does breaking exact weight tying help? | plain shared · +step-embedding · +depth-LoRA r∈{8,16,32} · +Mixture-of-LoRAs | 4 × 5 h = **20 h** | Adopt the cheapest arm within 0.01 nats of the best. Budget ≤ 0.5 % extra params. |
| **A9** | **Latent depth vs token CoT at matched wall-clock** *(the thesis test)* | `L@8`, no CoT · `L@1` + 8× filler/CoT tokens · `L@4` + 2× CoT | 3 × 6 h = **18 h** (SFT on templated CoT) | Decides §2.3 for *our* model. Expect: latent wins on p-hop/parity/retrieval, token-CoT wins on multi-digit arithmetic. Output = the shipping policy for r vs CoT length per task type. |
| **A10** | Does recurrence amplify distillation? | SFT on 200 k OpenThoughts3 traces at r=1 vs r=4 vs r=8 | 3 × 3 h = **9 h** | If r=4 SFT ≥ r=1 SFT + 3 pts on templated math, recurrence and distillation compose — which is the whole product thesis. |
| **A11** | *(Conditional, only if A1/A2 marginal)* Scale threshold | repeat A1 at 350 M (d=1024, 24 layers dense vs L(3,4,3)@r=4) | 3 × 12 h = **36 h** | Guards against MoR's documented 135 M→360 M crossover `[V]`. |

**Staging (do not run all of this at once):**
- **Stage 0 — go/no-go (24 h):** A1 + A2. If either fails and A11 also fails, R04's recommendation collapses to "r ≤ 2 TaH-style refinement + explicit CoT + budget forcing", which is still a real (cheap) product.
- **Stage 1 — make it trainable (53 h):** A3, A4, A5, A7.
- **Stage 2 — make it adaptive (36 h):** A6, A8.
- **Stage 3 — prove the product thesis (27 h):** A9, A10.
- **Total ≈ 140 A100-h** if every stage runs; Stage 0 alone is 24 h and answers the expensive question.

---

## 8. References

**Depth recurrence / looped transformers**
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* — **arXiv:2502.05171** (NeurIPS 2025). Code: `github.com/seal-rg/recurrent-pretraining`; config `nebel-raven-3.5b`.
- Saunshi et al., *Reasoning with Latent Thoughts: On the Power of Looped Transformers* — **arXiv:2502.17416** (ICLR 2025).
- Bae et al., *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation* — **arXiv:2507.10524** (NeurIPS 2025). Code: `github.com/raymin0223/mixture_of_recursions`.
- Bae et al., *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA* — **arXiv:2410.20672** (ICLR 2025).
- McLeish et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence* — **arXiv:2511.07384**. Code: `github.com/mcleish7/retrofitting-recurrence`.
- Wang et al., *Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models* — **arXiv:2511.08577** (ICML 2026). Code: `github.com/thu-nics/TaH`.
- Li, Zhang, Guo, Gu, Wang, *DeepLoop: Depth Scaling for Looped Transformers* — **arXiv:2607.13491**.
- Dehghani et al., *Universal Transformers* — **arXiv:1807.03819** `[M]`.
- Lan et al., *ALBERT: A Lite BERT* — **arXiv:1909.11942** `[M]`.
- *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference* — **arXiv:2310.10845** `[M]`.
- *Improving Recursive Transformers with Mixture of LoRAs* — **arXiv:2512.12880**.
- *Hyperloop Transformers* — **arXiv:2604.21254**; *Training-Free Looped Transformers* — **arXiv:2605.23872**; *Loop as a Bridge* — **arXiv:2601.10242**; *Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers* — **arXiv:2606.31779**; *LoopCoder-v2* — **arXiv:2606.18023**.

**Adaptive computation / early exit**
- Graves, *Adaptive Computation Time for Recurrent Neural Networks* — **arXiv:1603.08983**.
- Banino et al., *PonderNet: Learning to Ponder* — **arXiv:2107.05407**.
- Raposo et al., *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models* — **arXiv:2404.02258**.
- Elhoushi et al., *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding* — **arXiv:2404.16710**.
- Schuster et al., *Confident Adaptive Language Modeling (CALM)* — **arXiv:2207.07061** `[M]`.
- *Adaptive Computation with Elastic Input Sequence (AdaTape)* — **arXiv:2301.13195**.
- *Attention Is All You Need For Mixture-of-Depths Routing* — **arXiv:2412.20875**.
- *Accelerating Large Language Model Inference via Early-Exiting Algorithms* — **arXiv:2509.05915**.
- *AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth* — **arXiv:2603.01914**.
- *When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time…* — **arXiv:2601.00894**.

**Latent-space reasoning**
- Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)* — **arXiv:2412.06769**. Code: `github.com/facebookresearch/coconut`.
- Zou, Xiong, Liu, *Capabilities and Fundamental Limits of Latent Chain-of-Thought* — **arXiv:2602.01148**.
- Lu et al., *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer* — **arXiv:2507.02199** (COLM 2025 workshop). Code: `github.com/wenquanlu/huginn-latent-cot`.
- Goyal et al., *Think before you speak: Training Language Models With Pause Tokens* — **arXiv:2310.02226** (ICLR 2024).
- Pfau et al., *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models* — **arXiv:2404.15758**.
- *Rethinking Thinking Tokens: Understanding Why They Underperform in Practice* — **arXiv:2411.11371**.
- Zelikman et al., *Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking* — **arXiv:2403.09629**; *Fast Quiet-STaR* — **arXiv:2505.17746**.
- Deng, Choi, Shieber, *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step* — **arXiv:2405.14838**.
- *A Survey on Latent Reasoning* — **arXiv:2507.06203**.
- *Soft Tokens, Hard Truths* — **arXiv:2509.19170**; *Towards Inference-time Scaling for Continuous Space Reasoning* — **arXiv:2510.12167**; *Pretraining with Token-Level Adaptive Latent Chain-of-Thought* — **arXiv:2602.08220**.

**Explicit test-time compute / search / verifiers**
- Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* — **arXiv:2408.03314**.
- Liu et al., *Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling* — **arXiv:2502.06703**. Code: `github.com/RyanLiu112/compute-optimal-tts`.
- Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* — **arXiv:2407.21787**.
- Muennighoff et al., *s1: Simple test-time scaling* — **arXiv:2501.19393**.
- Setlur et al., *Scaling Test-Time Compute Without Verification or RL is Suboptimal* — **arXiv:2502.12118**.
- Lightman et al., *Let's Verify Step by Step* — **arXiv:2305.20050** `[M]`.
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations* — **arXiv:2312.08935** `[M]`.

**Reasoning distillation / data**
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — **arXiv:2501.12948**.
- Guha et al., *OpenThoughts: Data Recipes for Reasoning Models* — **arXiv:2506.04178** `[M]`. Code: `github.com/open-thoughts/open-thoughts`.
- Ye et al., *LIMO: Less is More for Reasoning* — **arXiv:2502.03387**.

---

## Appendix A — decision summary (one screen)

| Question | Answer | Confidence |
|---|---|---|
| Does looping substitute for parameters? | **Yes, at iso-FLOP: k⊗L ≈ kL⊗1 ≫ k⊗1** `[V]`, giving 3–15× parameter compression | High for ≥360 M; **unproven below** |
| Is depth a runtime dial? | **Yes**, if trained with randomised r + an any-depth read-out loss (Huginn evaluates r=1…256 from one checkpoint `[V]`) | High |
| Does latent depth replace CoT on math? | **No.** Huginn: 4.9 % no-CoT vs 38.1 % with CoT on GSM8K `[V]` | High |
| Best accuracy-per-FLOP mechanism found? | **TaH selective iteration:** +3.8–4.4 % at identical params, iterating 7 % of tokens `[V]` | High |
| Biggest deployment blocker? | **KV cache scaling with r** (4.5 GB at r=32/32 K) — solved by cross-step KV sharing (3.3–11× reduction) | High |
| Biggest strategic risk? | We are **2 500× short of Qwen3-1.7B's pretraining FLOPs**; recurrence does not fix knowledge benchmarks | High |
| Cheapest big win? | **Reasoning-trace distillation** (OpenThoughts3-1.2M / R1-800k) + **budget forcing** (zero training cost) | High |
