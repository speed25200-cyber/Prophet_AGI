# R09 — Hallucination, Calibration, and Knowing What You Don't Know

**Track owner:** R09 · **Status:** decision-ready · **Date:** 2026-09-03
**Scope:** why Prophet will hallucinate, what we can measure, what we can fix, and the one design we should build.

> **Number provenance.** Egress policy in this session blocked arxiv.org, huggingface.co, aclanthology.org,
> openai.com and most paper hosts; the shared web-search budget was exhausted mid-track. Every number below
> carries a provenance tag:
> **[V]** verified in-session from a reachable primary source (github.com / raw.githubusercontent.com / search snippet
> of the paper text); **[P]** from the paper as recalled, high confidence, *re-verify before quoting externally*;
> **[E]** my own arithmetic or estimate, derivation shown. Anything tagged [P] that a decision hinges on is
> flagged **VERIFY** inline. No number in this document is decorative — each one is used somewhere.

---

## 1. Problem statement

### 1.1 The three-line version

1. Prophet-main (~10B total / ~1.3B active) physically cannot store enough world knowledge to answer long-tail
   factual questions. The knowledge-capacity arithmetic (§1.3) puts our realistic ceiling at **~3 Gbit ≈ 375 MB**
   of extractable factual knowledge at int8, falling to **~1 Gbit ≈ 130 MB** at the 3–4 bit quantization we need
   for an 8 GB iPhone. That is roughly **5–20% of a "Wikipedia-equivalent"**.
2. Next-token pretraining plus benchmark-driven post-training make the model *assert anyway*. This is not a bug in
   the data; it is the argmax of the objective we train against (Kalai et al., 2509.04664).
3. Therefore the only route to "pulverizing the competition" on factuality is **not more parametric knowledge**.
   It is: *know precisely which questions you can answer, answer those, and route the rest to retrieval or to an
   explicit refusal.* At our scale, calibrated abstention is worth more benchmark points than any plausible
   knowledge gain — quantified in §1.5 and §6.

### 1.2 Why models hallucinate — the statistical account

**Kalai, Nachum, Vempala, Zhang, "Why Language Models Hallucinate" (2509.04664)** [V] gives the cleanest
mechanistic account, in two parts.

**(a) Pretraining: hallucination is a binary-classification error.** Partition plausible strings into *valid* and
*erroneous*. Define the **Is-It-Valid (IIV)** classification problem: given a candidate string, decide whether it
is valid. The paper's central reduction bounds the generative error rate of a density-estimating model below by
(roughly) **twice** the optimal IIV misclassification rate, minus calibration slack. [V/P] Consequence: if
"Adam Kalai's birthday is 3 March" is statistically indistinguishable from the true string given the training
distribution, then *any* well-fit generative model must emit false birthdays at a rate governed by that
indistinguishability. Hallucination is not a training failure — for a calibrated base model it is *mandatory*.

**(b) The singleton bound.** For "arbitrary facts" — facts with no learnable pattern (birthdays, phone numbers,
one-off dates) — the base-model hallucination rate is lower-bounded by the **singleton rate** `sr`: the fraction of
such facts appearing **exactly once** in the training corpus. [V] The paper's worked example: if 20% of birthdays
appear once in pretraining, base models should be wrong on **≥ 20%** of birthday queries. [V] This is the single
most important theoretical result for Prophet, because **our corpus is 200–400B tokens, so our singleton rate is
much higher than a frontier model's**, and SimpleQA-style benchmarks are *deliberately* built from questions that
frontier models get wrong — i.e. selected for high singleton rate.

**(c) Post-training keeps it there: the scoring-rule argument.** Almost every benchmark in the standard suite
(MMLU, ARC, GPQA, HellaSwag, GSM8K, TriviaQA-EM, and the raw SimpleQA accuracy column) is **binary 0/1 with no
penalty for a wrong answer**. Under 0/1 scoring, the expected-score-maximizing policy is *always guess*, for any
confidence > 0. Abstention is strictly dominated. So RLHF/RLVR against these evals actively trains guessing.
The paper's fix is **explicit confidence targets in the rubric**: state in the prompt

> "Answer only if you are more than *t* confident, since mistakes are penalized *t/(1−t)* points, correct answers
> receive 1 point, and 'I don't know' receives 0 points." [V]

Under that rubric the optimal policy is **behavioral calibration**: answer iff `P(correct) > t`. With t = 0.75 a
wrong answer costs 3 points; with t = 0.9 it costs 9. **This threshold `t` is the central runtime knob of the
design in §4.** The paper's own demonstration table (SimpleQA) [P, VERIFY]:

| Model | abstention rate | accuracy | error (hallucination) rate |
|---|---|---|---|
| o4-mini | ~1% | ~24% | ~75% |
| gpt-5-thinking-mini | ~52% | ~22% | ~26% |

Read that table carefully: **2 points of accuracy bought a 3× reduction in hallucination.** That is exactly the
trade Prophet should make, and at 1.3B active the trade is far more favourable than it is at o4-mini scale,
because our accuracy is lower and our error rate is higher.

### 1.3 Knowledge-capacity arithmetic — what Prophet *cannot* know

**Allen-Zhu & Li, "Physics of Language Models Part 3.3: Knowledge Capacity Scaling Laws" (2404.05405)** [V]:

| Finding | Value | Tag |
|---|---|---|
| Peak knowledge capacity, ≥1000 exposures per fact | **2 bits / parameter** | [V] |
| Capacity at **100 exposures** per fact | **~1 bit / parameter** | [P] VERIFY |
| int8 quantization | no capacity loss (still 2 bits/param) | [V] |
| int4 quantization | **capacity drops to ~0.7 bits/param** | [P] **VERIFY — high-impact** |
| MoE sparsity | "nearly fully efficient": capacity scales with **total**, not active params; peak ratio down **1.5×** in the 100-exposure regime | [V] |
| Junk data, 1:7 useful:junk token ratio, 100 exposures | **20× capacity loss** on the useful knowledge | [V] |
| Same, with a domain token (e.g. `wikipedia.org`) prepended to useful data | loss improves from **20× → 2×** | [V] |
| Architecture | GPT-2 + rotary **matches or beats** LLaMA/Mistral for knowledge storage (GatedMLP is less training-stable) | [V] |
| Headline | a 7B model stores ~14 Gbit ≈ "English Wikipedia + textbooks combined" | [V] |

Applying this to Prophet (assume 9.6B non-embedding params for the 10B-total MoE; 380M for the 450M-dense mini):

| Regime | bits/param | Prophet-main (9.6B total) | Prophet-mini (380M) | Tag |
|---|---|---|---|---|
| Idealized ceiling (1000 exposures, clean data, int8) | 2.0 | **19.2 Gbit** (2.40 GB) | 0.76 Gbit (95 MB) | [E] |
| 100 exposures | 1.0 | 9.6 Gbit (1.20 GB) | 0.38 Gbit (48 MB) | [E] |
| 100 exposures + MoE 1.5× penalty | 0.67 | 6.4 Gbit (800 MB) | — | [E] |
| + realistic web junk ratio, **no** domain tokens (20×) | 0.033 | **0.32 Gbit** (40 MB) | 0.019 Gbit (2.4 MB) | [E] |
| + realistic web junk ratio, **with** domain tokens (2×) | 0.33 | **3.2 Gbit** (400 MB) | 0.19 Gbit (24 MB) | [E] |
| **On-device operating point** (above, then int4: ×0.35) | 0.12 | **1.1 Gbit** (140 MB) | 0.066 Gbit (8 MB) | [E] |

**Converting bits to facts.** A long-tail (entity, relation, value) triple costs roughly *index + value* bits:
identifying one of ~10⁷ entities is ~23 bits, but that index is amortized across all of that entity's attributes;
a value drawn from ~10⁵ cities is ~17 bits, a date ~15 bits, a person-name ~20 bits. **~30 bits/fact** is a
defensible working figure [E]. Then:

- Server-side operating point (3.2 Gbit): **~107M facts**.
- On-device int4 operating point (1.1 Gbit): **~37M facts**.
- For scale: Wikidata holds ~1.6B statements over ~115M entities; English Wikipedia has ~7M articles.

**Conclusion, stated bluntly: Prophet-main can hold on the order of 3–10% of Wikidata's statements at fp16/int8,
and 1–3% of it after on-device quantization.** Everything outside that is a guaranteed hallucination unless we
abstain or retrieve. And note the asymmetry no one designs for: **quantizing for the phone deletes factual
knowledge specifically** — the on-device model will hallucinate materially *more* than the server model even
though it is nominally "the same model". That single fact is the strongest argument in this document for shipping
the abstention/retrieval policy as part of the on-device runtime rather than as a server-side nicety.

### 1.4 The exposure-count problem, and the one cheap intervention

Capacity is not the binding constraint — **exposures** are. At a 300B-token budget with 1 epoch, a fact mentioned
in 10 web documents gets ~10 exposures, i.e. we sit in the *worst* part of the Allen-Zhu curve (well below the
100-exposure regime, and diluted 20× by junk).

Two levers, both cheap, both from 2404.05405 [V]:

1. **Domain-token / provenance-prefix conditioning.** Prepend a source token (`<src:wikipedia>`,
   `<src:textbook>`, `<src:web-cc>`, `<src:code>`) to every pretraining document. Cost: ~0 (a few tokens per doc,
   <0.2% of the budget). Benefit: turns the junk penalty from 20× to 2× — i.e. **~10× more retained knowledge for
   free**. This is the single highest-ROI item in this entire report and it belongs in the data track (R0x),
   not here. It also gives us a free steering token at inference ("answer as if from `<src:wikipedia>`").
2. **Upsample a curated knowledge core.** English Wikipedia ≈ 4–5B tokens; add verbalized Wikidata + open
   textbooks → a ~6–8B-token "knowledge core". Repeating it **4–8 epochs** costs 24–64B tokens = **8–21% of a
   300B budget** [E] and moves core facts from ~10 to ~40–80 exposures. Repetition beyond ~4 epochs has strongly
   diminishing returns (Muennighoff et al., *Scaling Data-Constrained Language Models*, 2305.16264; Xue et al.,
   2305.13230) [P], so **4 epochs is the safe recommendation, 8 the aggressive one.**

Even with both, we do not reach the 1000-exposure regime for the long tail, and we never will. Which is why §4 is
about abstention and retrieval rather than about memorization.

### 1.5 Why abstention is worth real benchmark points at our scale (the arithmetic)

SimpleQA (Wei et al., 2411.04368) grades every response as **correct / incorrect / not_attempted** and reports
`F = harmonic_mean(correct, correct_given_attempted)`. Take a 1–4B model with 3% closed-book accuracy [E, see §2.4]:

| Policy | correct | correct-given-attempted | **F** | Tag |
|---|---|---|---|---|
| Always answer, 3% accurate (today's small models) | 3.0% | 3.0% | **3.0** | [E] |
| Abstain on 70%, 10% accurate on the rest | 3.0% | 10.0% | **4.6** | [E] |
| Abstain on 90%, 25% accurate on the rest | 2.5% | 25.0% | **4.5** | [E] |
| Abstain on 60%, retrieval on the abstained 60% at 45% accuracy | 28.2% | 28.2% | **28.2** | [E] |

**Calibrated abstention alone buys ~1.5× SimpleQA F with zero knowledge gain.** Abstention *plus* retrieval buys
an order of magnitude. And the "hallucination rate" headline — the fraction of questions answered *wrongly* —
falls from ~97% to ~25%. That is the claim we should be able to make and defend.

The counterweight, quantified in §6: on 0/1-scored multiple-choice evals abstention is **pure loss** (abstaining
on 20% of MMLU where we'd have scored 40% costs 8 points). This is why the design in §4 makes abstention a
*runtime policy with a threshold*, never a baked-in behavioural reflex.

---

## 2. State of the art

### 2.1 Detection & calibration methods

| Method | arXiv | Mechanism | Inference cost | Measured gain | On-device? | Tag |
|---|---|---|---|---|---|---|
| Max-softmax / mean token logprob | — | length-normalized sequence logprob | **1×** (free) | AUC-PR 83.2 NonFact / 54.0 Factual on SelfCheckGPT-WikiBio; PCC 57.0 | ✅ | [V] |
| Predictive / naive entropy | — | entropy over the token distribution | 1× (free) | weak baseline; consistently below semantic entropy | ✅ | [V] |
| **Semantic entropy** | Farquhar et al., *Nature* 630 (2024); Kuhn et al. 2302.09664 | sample k answers, cluster by bidirectional NLI entailment, entropy over *meaning* clusters | **k× gen (k≈10) + O(k²) NLI** | AUROC/AURAC "substantially higher" than naive entropy and P(True) across 5 QA sets × 4 model families; ~0.75–0.80 AUROC typical | ❌ (10 gens/answer) | [V]/[P] |
| **Semantic Entropy Probes (SEP)** | 2406.15927 | **linear probe on hidden states that predicts semantic entropy** — labels cost k samples at *training* time, inference is 1 forward pass | **1×, ~0 params** | retains most of SE's AUROC at ~10–30× lower cost; beats accuracy-trained probes under distribution shift | ✅✅ | [P] VERIFY |
| SelfCheckGPT | 2303.08896 | sample k stochastic responses, score consistency (BERTScore / QA / n-gram / NLI / LLM-prompt) | k× gen + NLI or LLM judge | **NLI variant: 92.50 AUC-PR NonFact, 66.08 Factual, 74.14 PCC**; unigram 85.63/58.47; GPT-3.5 prompt 93.42/67.09 | ❌ | [V] |
| P(True) / self-evaluation | Kadavath et al. 2207.05221 | ask the model "is this answer true? (A/B)" and read the token prob | +1 short forward | well-calibrated at ≥52B; degrades sharply at small scale | ⚠️ | [P] |
| **P(IK) value head** | 2207.05221 | trained scalar head predicting "do I know the answer", from hidden states | **~0** | generalizes across tasks; the direct ancestor of §4's design | ✅✅ | [P] |
| Verbalized confidence ("I'm 70% sure") | Lin et al. 2205.14334; Tian et al. 2305.14975 | model emits a number | +few tokens | on RLHF'd models verbalized confidence is **better calibrated than the logits**; ECE reduced ~50% relative | ✅ | [P] |
| Internal-state probe (SAPLMA) | 2304.13734 | MLP on a mid-layer hidden state, "is this statement true" | ~0 | ~70–80% detection accuracy | ✅ | [P] |
| INSIDE / EigenScore | 2402.03744 | eigenvalue spectrum of the covariance of k sampled responses' *embeddings* — semantic diversity in latent space, no NLI model | k× gen, no NLI | beats LN-entropy and lexical-similarity baselines | ⚠️ | [P] |
| **Lookback Lens** | 2407.07071 | linear classifier on the **ratio of attention mass on context vs. generated tokens**, per head per span | **~0**, features already computed | detects *contextual* (RAG) hallucination; transfers across models without retraining; guided decoding cuts XSum hallucination ~9.6% | ✅✅ | [P] |
| SAT-Probe | 2309.15098 | attention to constraint tokens predicts factual errors | ~0 | comparable to the model's own confidence, but *before* generation | ✅ | [P] |
| Conformal prediction / factuality | Quach et al. 2306.10193; Mohri & Hashimoto 2402.10978 | split-conformal on a held-out set → distribution-free bound on assert-and-wrong rate | ~0 at inference (offline calibration) | **provable** P(error) ≤ α at chosen coverage; back-off/hedging to satisfy it | ✅✅ | [P] |

### 2.2 Architectural & training interventions

| Method | arXiv | Mechanism | Cost | Measured gain | Verdict for Prophet | Tag |
|---|---|---|---|---|---|---|
| **DoLa** | 2309.03883 | contrast final-layer logits against a dynamically-chosen early layer | ~1.05–1.10× decode | "improves LLaMA-family on TruthfulQA by **12–17 absolute points**" | ⚠️ gains measured on LLaMA-1 *base* + TruthfulQA-MC; multiple later replications find little/no gain on instruction-tuned or small models. **Ablate, don't assume.** | [V] |
| **ITI** | 2306.03341 | shift activations along a "truth" direction in ~48 selected attention heads (α≈15) | ~0 | Alpaca True*Informative **32.5% → 65.1%** | ❌ direction is fitted *on TruthfulQA*; does not generalize; TruthfulQA-specific overfit | [V] |
| **CAD (context-aware decoding)** | 2305.14739 | contrast p(y\|context) against p(y\|no context); amplify the context's contribution | **2× forward** | large gains on summarization faithfulness and knowledge-conflict QA (reported up to ~2.9× on conflict benchmarks) | ✅ **but only in grounded/RAG mode**, where 2× is affordable and the benefit is greatest | [P] |
| **R-Tuning** | 2311.09677 | split SFT data into the model's *known* vs *unknown* set; append "I am sure" / "I am unsure"; SFT | 0 extra | better AP and calibration than uncertainty-based testing; **refusal ability generalizes to unseen tasks** | ✅✅ core of §4 | [V] |
| **[IDK] token** | 2412.06676 | add a dedicated `[IDK]` token to the vocabulary; shift probability mass onto it when the model would be wrong | 0 extra | improved factual precision, minor recall loss | ✅ adopted as the *action* token in §4 | [P] |
| **Alignment for Honesty** | 2312.07000 | honesty-oriented SFT with explicit IDK responses and an honesty metric | 0 extra | improves honesty without tanking helpfulness | ✅ | [P] |
| **RLCR** (RL with Calibration Rewards) | 2507.16806 | RL where the model outputs *answer + confidence*, reward = correctness + a bounded **Brier** term (a proper scoring rule ⇒ honesty is optimal) | RL cost | drives calibration error to ~0 **without accuracy loss**, where plain RLVR *worsens* calibration; confidence-weighted test-time scaling improves accuracy | ✅✅ core of §4 | [P] VERIFY |
| **Linguistic Calibration** | 2404.00474 | RL objective on *long-form* text so that a downstream decision-maker reading it is calibrated | RL cost | improves downstream decision accuracy | ⚖️ elegant, expensive, phase-3 at best | [P] |
| **FLARE** (active retrieval) | 2305.06983 | if any token in the upcoming sentence has p < θ (θ≈0.4), retrieve using that sentence as the query and regenerate | retrieval on demand | gains on ASQA / StrategyQA / 2WikiMultihop long-form | ✅ the token-level trigger in §4 | [P] |
| **Adaptive retrieval / popularity** | Mallen et al. 2212.10511 (PopQA) | retrieve only for low-popularity entities | retrieval on demand | small-LM accuracy tracks entity popularity log-linearly; adaptive beats *always*-retrieve, because retrieval **hurts** on head entities | ✅✅ direct empirical mandate for a confidence-gated retriever | [P] |
| **Self-RAG** | 2310.11511 | reflection tokens control *when* to retrieve and *whether* the output is supported | retrieval on demand | 7B Self-RAG reported to beat ChatGPT on biography FActScore (~81 vs ~72) and to massively beat vanilla Llama-2-7B on PopQA | ✅✅ the control-token pattern we copy | [P] VERIFY |
| **Chain-of-Verification** | 2309.11495 | draft → generate verification questions → answer them independently → revise | **3–5× tokens** | solid long-form factuality gains | ❌ latency-prohibitive on-device; keep as an optional server "careful mode" | [P] |

### 2.3 Retrieval-native pretraining — the case and the cost

| Approach | arXiv | Headline result | Tag |
|---|---|---|---|
| **RETRO** | 2112.04426 | 7.5B RETRO + 2T-token database ≈ **GPT-3 175B / Jurassic-1 178B** on the Pile — a ~25× parameter reduction | [P] |
| RETRO++ / "Shall we pretrain autoregressive LMs with retrieval?" | 2304.06762 | retrieval-pretrained LMs are better on open-domain QA and less toxic than same-size GPT | [P] |
| **Atlas** | 2208.03299 | 11B Atlas beats **540B PaLM** on NaturalQuestions in the 64-shot setting (~42% vs ~40%) | [P] VERIFY |
| kNN-LM | 1911.00172 | ~25% relative perplexity improvement on WikiText-103 from a nearest-neighbour datastore; a model + datastore beats training on the full corpus | [P] |
| In-context RALM | 2302.00083 | simply **prepending** retrieved documents to a frozen LM recovers much of the gain, with no architecture change | [P] |
| RA-DIT | 2310.01352 | lightweight dual instruction-tuning of LM + retriever, no pretraining change | [P] |
| Position paper | Asai et al. 2403.03187 | argues retrieval-native LMs are the route to reliability/attributability | [P] |

**Why Prophet should be retrieval-*aware*, not RETRO-native — the cost calculation** [E]:
embedding the pretraining corpus to build a chunked datastore with even a small 100M-param encoder costs
`2 × 1e8 × 4e11 = 8e19 FLOPs`; at a realistic 120 TFLOP/s that is **≈ 185 A100-hours *just to index***, before
storage (tens of billions of chunk vectors) and before the extra cross-attention FLOPs in every training step. On
a single-A100 budget that is a non-starter. Worse, the *inference-time* datastore is precisely what an 8 GB phone
cannot hold, so RETRO's architecture buys us a capability we cannot ship. **In-context RALM (2302.00083) gets most
of the benefit at ~5% of the cost and keeps the architecture a plain decoder — which also matters because
llama.cpp / MLX / CoreML will never support RETRO-style chunked cross-attention.**

### 2.4 Where 1–4B models actually score today

Verified anchors from OpenAI's `simple-evals` results table [V]: `gpt-4.1-nano` **7.6**, `gpt-4o-mini` **9.5**,
`o1-mini` **7.6**, `o3-mini` **13.4**, `gpt-4o` **38.8–40.1**, `Claude 3.5 Sonnet` **28.9**, `gpt-4.5-preview`
**62.5**. Those are *accuracy* columns, and they set the scale: **every small model is in single digits.**

| Model | SimpleQA (acc %) | TruthfulQA MC2 | HaluEval (recognition acc) | Tag |
|---|---|---|---|---|
| Qwen3-1.7B | ~2–3 | ~50–55 | ~50–58 | [E] VERIFY |
| Qwen3-4B | ~3–5 | ~55–60 | ~55–62 | [E] VERIFY |
| Llama-3.2-3B-Instruct | ~2–3 | ~50–52 | ~50–55 | [E] VERIFY |
| Gemma-3-4B-IT | ~2–4 | ~55–58 | ~55–60 | [E] VERIFY |
| SmolLM3-3B | ~2–3 | ~45–50 | — | [E] VERIFY |
| Phi-4-mini (3.8B) | ~2–4 | ~58–62 | ~55–62 | [E] VERIFY |
| gpt-4o-mini (reference upper bound for "small") | **9.5** | — | — | **[V]** |
| GPT-4o (reference frontier) | **38.8** | — | — | **[V]** |

Reference points for the other benchmarks [P]:
- **HaluEval** (2305.11747, 35k samples: 10k QA / 10k dialogue / 10k summarization / 5k human-annotated ChatGPT
  responses [V]). ChatGPT hallucination-*recognition* accuracy ≈ **62.6% QA / 72.4% dialogue / 58.5%
  summarization / 79.4% general** [P VERIFY]. Note that ~50% is chance on the binary task — **frontier models are
  barely above chance on QA hallucination recognition, and 1–4B models are at or below chance.**
- **TruthfulQA** (2109.07958). Base models ~38–45 MC2; instruction/preference-tuned ~50–62. Critical caveat:
  **DPO/RLHF inflates MC2 without improving real factuality**, and the generative `%true` metric is trivially
  gamed by "I have no comment" (100% true, 0% informative). **Never report `%true` without `%informative`.**
- **FActScore** (2305.14251): ChatGPT biographies ≈ 58%, retrieval-augmented systems ≈ 71%, LLaMA-65B in the
  20–40% band [P]. Small models generate biographies that are majority-false.
- **FACTS Grounding** (2501.03200, Google DeepMind): long-form grounded generation, judged in **two stages** —
  first "does the response actually address the request", then "is every claim supported by the provided
  document". Frontier models score ~60–85%. **The eligibility gate means over-abstention is explicitly
  penalized**, which makes FACTS Grounding the best single benchmark for proving our design is not just refusing.
- **SelfAware** (2305.18153): benchmark of *unanswerable* questions; measures whether a model knows what it
  doesn't know. Cheap to run, directly on-track.

**What "pulverizing the competition" means numerically.** Targets for Prophet-main, stated so they are falsifiable:

| Metric | Peer 1–4B today | **Prophet target** |
|---|---|---|
| SimpleQA **error rate** (fraction answered *wrongly*) | 90–97% | **≤ 25%** |
| SimpleQA F (closed-book, t = 0.75) | 2–5 | **≥ 12** |
| SimpleQA correct (with on-device retrieval) | n/a | **≥ 25%** |
| TruthfulQA (%true / %info jointly) | ~50 / ~95 | **≥ 85 / ≥ 75** |
| HaluEval recognition (avg) | ~50–60 | **≥ 70** |
| AUROC of confidence signal vs. correctness | not reported | **≥ 0.80** |
| ECE after full post-training (MC) | 0.07–0.15 typical post-RLHF | **≤ 0.03** |
| Conformal guarantee at α = 0.1 | none | **empirical assert-and-wrong ≤ 10% at ≥ 40% coverage** |

---

## 3. What actually transfers to our scale

**Transfers well (build on these):**

1. **Probes on internal states.** SEP (2406.15927), P(IK) (2207.05221), SAPLMA (2304.13734) and Lookback Lens
   (2407.07071) all cost ~0 at inference and are the *only* family that is affordable per-answer on a phone. The
   expensive part (sampling k answers to make labels) happens **offline, once, during training**.
2. **Self-consistency sampling — as a label generator, never as an inference-time method.** Semantic entropy and
   SelfCheckGPT are the best detectors we have (SelfCheck-NLI: 92.5 AUC-PR [V]), and 5–10× generation cost is
   irrelevant when it is a one-off labelling pass over 1M questions (§4.4 costs it at ~10 A100-hours).
3. **Known/unknown-gated SFT (R-Tuning, Gekhman).** *More* valuable at our scale, not less: our "known" set is
   small, so the fraction of SFT data that would teach hallucination is *larger*, so the filter removes more harm.
4. **Confidence-gated retrieval** (Mallen 2212.10511, FLARE 2305.06983, Self-RAG 2310.11511). Mallen's result —
   that retrieval *hurts* on head entities — means the gate is not an optimization, it is required for accuracy.
5. **Proper scoring rules in the RL reward (RLCR, 2507.16806).** Cheap to add to any RLVR pipeline we are already
   running, and it is the only known way to do RL without wrecking calibration.
6. **Split-conformal thresholding.** A few thousand held-out labelled questions buy a distribution-free guarantee.
   Nothing else in this report gives a *guarantee*.
7. **Domain/provenance tokens in pretraining** (2404.05405) — 10× effective knowledge retention for ~free [V].

**Does not transfer / high risk:**

- **ITI (2306.03341).** The truth direction is fitted on TruthfulQA and evaluated on TruthfulQA. Do not ship.
- **DoLa (2309.03883).** Gains were measured on LLaMA-1 *base* models on TruthfulQA-MC; replications on
  instruction-tuned and small models are weak. Keep as an ablation (A1), not a plan.
- **Semantic entropy at inference.** 10 generations × answer length per query. Kills phone latency and battery.
  Use it to *make labels* for a probe, which is exactly what SEP does.
- **SelfCheckGPT at inference.** Needs k samples *and* an external NLI model. Server-only "careful mode" at best.
- **Chain-of-Verification (2309.11495).** 3–5× tokens.
- **kNN-LM (1911.00172).** Datastore is orders of magnitude larger than the model. Not shippable to a phone.
- **RETRO-style pretraining (2112.04426).** ~185 A100-hours just to index the corpus [E] — see §2.3.

**Genuinely uncertain (this is the scientific risk of the track):** every probe result in the literature is
measured at **≥7B**. Detection AUROC is known to degrade with scale downward, because the probe is reading the
model's *own* self-knowledge and a 1.3B-active model has less of it. **Whether an internal-state confidence probe
still reaches AUROC ≥ 0.75 at 1.3B active is unmeasured, and ablation A1 (§7) exists to answer it before we spend
a single hour of the main pretraining run on this track.**

---

## 4. Recommendation for Prophet

### 4.0 The one design, in one paragraph

Add a **Prophet Confidence Head (PCH)**: a ~0.6M-parameter MLP on the trunk's last hidden state (plus six free
decoding statistics) that emits two calibrated probabilities at every position — **`c_know`** (would my parametric
answer be correct?) and **`c_ground`** (is what I'm about to say supported by the context?). Train it in three
phases: a free self-supervised *corpus-frequency* objective during pretraining, a *self-labelled empirical
correctness* objective (soft-label BCE — a proper scoring rule) after pretraining, and an *RLCR* Brier-reward RL
phase in post-training. Give the LM two control tokens, `<IDK>` and `<RETRIEVE>`, and a 16-bin verbalized
confidence token emitted **before** the answer so it can route. At decode time, apply **behavioral calibration**:
answer iff `c_know > t`, retrieve if `t_lo ≤ c_know < t`, abstain otherwise — where `t` is a *runtime* threshold
set by the deployment's own scoring rule, and calibrated by split-conformal to give a provable assert-and-wrong
bound. Finally, enforce one hard post-training rule: **never train the model to assert a fact it does not already
know** (Gekhman 2405.05904) — closed-book targets are filtered against the model's own known-set, and unknown
targets are *rewritten* into abstentions or retrieval requests rather than dropped.

### 4.1 Architecture

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ProphetConfidenceHead(nn.Module):
    """~0.6M params. Reads the trunk's last hidden state plus six statistics that
    the decoder has already computed, and emits two logits:
        [0] c_know   -- P(my parametric answer is correct)      (closed-book)
        [1] c_ground -- P(this span is supported by the context) (open-book)
    Temperature is a fitted buffer, never trained by SGD -- it is re-fitted after
    every post-training stage and after quantization (see 4.6)."""

    N_EXTRA = 6

    def __init__(self, d_model: int, d_hidden: int = 256):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.inp  = nn.Linear(d_model + self.N_EXTRA, d_hidden, bias=False)
        self.mid  = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out  = nn.Linear(d_hidden, 2)
        self.register_buffer("log_T", torch.zeros(2))   # per-output temperature

    def forward(self, h, extra, detach_trunk: bool = False):
        # h: [B,T,d] final hidden state; extra: [B,T,6]
        if detach_trunk:
            h = h.detach()
        z = torch.cat([self.norm(h), extra], dim=-1)
        z = F.silu(self.mid(F.silu(self.inp(z))))
        return self.out(z) / self.log_T.exp()           # [B,T,2] logits


@torch.no_grad()
def decode_stats(logits, h_mid, h_last, attn_ctx_mass):
    """Six features, all already available in the forward pass -- zero extra FLOPs
    of consequence. (4) is the Lookback Lens feature (2407.07071); (5) is a
    DoLa-style layer-contrast signal (2309.03883) reduced to one scalar."""
    p    = logits.softmax(-1)
    top2 = p.topk(2, dim=-1).values
    return torch.stack([
        -(p * p.clamp_min(1e-9).log()).sum(-1),              # 1 predictive entropy
        top2[..., 0],                                        # 2 max prob
        top2[..., 0] - top2[..., 1],                         # 3 margin
        attn_ctx_mass,                                       # 4 attention mass on retrieved ctx
        F.cosine_similarity(h_mid, h_last, dim=-1),          # 5 layer contrast
        logits.logsumexp(-1) - logits.max(-1).values,        # 6 energy gap
    ], dim=-1)
```

**Vocabulary additions (32 tokens total, ~65k params at d=2048):**
`<IDK>`, `<RETRIEVE>`, `</RETRIEVE>`, `<GROUNDED>`, `<UNSUPPORTED>`, and `<conf:00>…<conf:15>` (16 bins).
The confidence token is emitted **at answer-start, before the answer** — so it can be used to route *before*
spending generation tokens — with an optional retrospective `<conf2:k>` after the answer.

**Why both a token and a head?** The *token* participates naturally in RL (it is an action, so a Brier reward on
it is a proper scoring rule and gradients flow through the ordinary policy-gradient path). The *head* is a cheap
scalar readout that the runtime can query without sampling, and it is trained by distillation from the token
distribution. Both are essentially free; each covers the other's weakness.

### 4.2 Losses

```python
# ---------- Phase 0: pretraining auxiliary (free labels, no extra data) --------
# y_freq[t] = clamp(log10(1 + corpus_count(fact_span_at_t)) / 4, 0, 1)
# counts come from a count-min sketch built once over the corpus (CPU-only).
l_freq = F.mse_loss(know_logit[fact_mask].sigmoid(), y_freq[fact_mask])
loss   = lm_loss + LAMBDA_FREQ * l_freq          # LAMBDA_FREQ = 0.02

# ---------- Phase 1: self-labelled correctness (proper scoring rule) ----------
# y_know in [0,1] = (# of k sampled answers judged correct) / k     -- SOFT label.
# Soft-label BCE is a strictly proper scoring rule => the minimizer is the true
# P(correct), i.e. calibration is the optimum, not a post-hoc fix.
l_know   = F.binary_cross_entropy_with_logits(know_logit[m_know],  y_know[m_know])
# y_ground in {0,1} from NLI support labels + synthetic entity-corruption negatives
l_ground = F.binary_cross_entropy_with_logits(ground_logit[m_gnd], y_ground[m_gnd])
loss     = lm_loss + 1.0 * l_know + 1.0 * l_ground

# ---------- Phase 2: RLCR (Damani et al. 2507.16806) --------------------------
def rlcr_reward(is_correct: float, q: float, beta: float = 1.0) -> float:
    """q = confidence decoded from the <conf:k> token, in [0,1].
    correctness + a bounded Brier term. Brier is a proper scoring rule, so for a
    FIXED answer distribution the reward-maximizing confidence is the honest one:
    the policy cannot gain by over- or under-claiming."""
    return is_correct + beta * (1.0 - (q - is_correct) ** 2)
```

Three notes that matter:

- **Phase 0 gradients flow into the trunk** (λ = 0.02, ablation A0 checks the LM-loss cost). The point is not the
  frequency prediction itself; it is to shape the residual stream so that "how often did I see this" is *linearly
  readable* — which is what makes Phase 1's tiny probe work.
- **Phase 1 uses soft labels, not hard 0/1.** This is the whole calibration argument. Training against
  `1{greedy answer was correct}` gives a discriminator; training against `k_correct/k` gives an estimator of
  P(correct), and BCE against a soft target is minimized exactly at the true probability.
- **Temperature is never trained jointly.** It is fitted by 1-D search on a held-out split after *each* stage and
  after quantization. Joint training lets the model trade LM loss for apparent calibration.

### 4.3 Decoding policy — behavioral calibration with a runtime threshold

```python
def prophet_policy(c_know, c_ground, t, retrieval_available, in_context_mode):
    """t is the DEPLOYMENT's stated penalty point, not a tuned hyperparameter.
    Under the rubric 'correct=+1, wrong=-t/(1-t), IDK=0' (Kalai 2509.04664) the
    expected-value-optimal policy is exactly: answer iff P(correct) > t."""
    if in_context_mode:                       # grounded / RAG answer
        return "answer" if c_ground > t else ("hedge" if c_ground > 0.5 * t else "abstain")
    if c_know > t:
        return "answer"                       # parametric answer
    if retrieval_available and c_know > T_LO: # worth a retrieval round-trip
        return "retrieve"
    if retrieval_available:
        return "retrieve"                     # long tail: retrieval is the only hope
    return "abstain"                          # emit <IDK>
```

Threshold table shipped as a runtime preset (`t` is a single float in the config):

| Deployment | `t` | Rationale |
|---|---|---|
| Multiple-choice / 0/1-scored eval (MMLU, ARC, GPQA) | **0.0** | wrong answers are free; abstention is strictly dominated (§6.1) |
| General chat | 0.50 | wrong answer costs about as much as a refusal |
| Factual assistant / SimpleQA-style | 0.75 | a mistake costs 3 correct answers |
| Medical / legal / high-stakes | 0.90 | a mistake costs 9 correct answers |

**Token-level guard (FLARE, 2305.06983).** During generation, at every token the tokenizer/NER flags as an entity,
number or date, if `c_know` at that position drops below `t`, either (a) emit `<RETRIEVE>` with the current
sentence as the query and regenerate, or (b) if retrieval is unavailable, hedge the span
("some sources say…", "I'm not certain of the exact date"). This is what converts a per-answer confidence into
per-*claim* factuality, which is what FActScore and FACTS Grounding actually measure.

**Grounded mode adds CAD (2305.14739)** at 2× forward cost: only when retrieved context is present, decode with
`logits = (1+α)·logits(y|ctx,x) − α·logits(y|x)` with α ≈ 0.5–1.0. Server-side default on, on-device default off.

**Conformal wrapper (2306.10193 / 2402.10978), for the guarantee:**

```python
# split-conformal, run offline on n>=2000 held-out graded questions:
#   s_i   = 1 - c_know_i   for every item
#   tau   = the ceil((n+1)*(1-alpha))-th smallest s_i among CORRECT items
#   assert iff (1 - c_know) <= tau     =>   P(assert AND wrong) <= alpha
# Re-fit per domain (wiki / science / code / medical) -- thresholds do NOT transfer.
```

### 4.4 The post-training rule that avoids teaching hallucination

This is the part that is most often skipped and most expensive to get wrong. **Gekhman et al. (2405.05904)** [V]:
SFT examples introducing knowledge the model does not have are learned *much more slowly*, and once they are
learned they **linearly increase the model's hallucination rate**. The mechanism is straightforward: you are
training the model to produce confident factual assertions in exactly the regime where it has no signal, which
generalizes to "produce confident factual assertions everywhere".

**The Prophet rule, stated as a build-gate:**

> **No closed-book SFT target may contain a fact the base checkpoint does not already know.**
> Every closed-book factual SFT example is graded against the base model's own known-set (SliCK-style: k=8 samples
> at T=0.7 plus greedy → HighlyKnown / MaybeKnown / WeaklyKnown / Unknown). Then:
> - **HighlyKnown / MaybeKnown** → keep, target = the factual answer, `<conf:k>` = the empirical accuracy bin.
> - **WeaklyKnown** → keep, target = the factual answer, but `<conf:k>` set to the (low) empirical bin. These are
>   the examples that teach *hedged* answering.
> - **Unknown** → **rewrite, never drop.** Target becomes `<RETRIEVE>query</RETRIEVE>` if a retrieval corpus can
>   answer it, else `<conf:00> <IDK>`. Dropping these throws away the abstention signal, which is the entire
>   point; rewriting them is what teaches abstention.
> - **Hard cap: 0% of closed-book SFT targets may assert an Unknown fact.** Ablation A3 measures the slope of the
>   hallucination-vs-unknown-fraction line at our scale so this cap can be defended rather than asserted.

**The escape hatch that keeps the data budget large.** The Gekhman constraint applies *only to closed-book
targets*. If the answer is present in the provided context, the example teaches *reading*, not new knowledge, and
is safe in unlimited quantity. **Therefore Prophet's factual post-training mix should be ~80% grounded / 20%
closed-book.** This is not a compromise — it is the same conclusion the retrieval-native argument reaches, from a
different direction, and it is why §2.3's "retrieval-aware, not RETRO" recommendation is coherent with the rest of
the design.

**Preserving the ability to guess (critical, and easy to get wrong).** R-Tuning-style training that unconditionally
teaches "I am unsure" will destroy MMLU (§6.1). Fix: 50% of the Unknown-bucket examples are re-emitted under a
`Always give your best guess, even if unsure.` system prompt, with the target set to **the base model's own
most-frequent sampled answer** (self-distillation). No new facts are injected — we are training on the model's own
output distribution — so the Gekhman constraint is respected, while the *capability* to answer under a forced-guess
regime is preserved and made explicitly controllable.

**RLHF/DPO and calibration.** Preference optimization is known to destroy calibration: GPT-4's MMLU ECE went from
**0.007 (pretrained) → 0.074 (post-RLHF)**, ~10× worse [P VERIFY]; RLCR (2507.16806) reports the same qualitative
effect for plain RLVR. Mitigations, all cheap, all mandatory:
1. **Freeze the PCH during preference optimization**, then re-fit temperature and thresholds afterwards.
2. **Use RLCR's Brier-augmented reward** instead of a pure correctness reward wherever we have a verifier.
3. **Keep a calibration replay set** (~2% of tokens) in every post-training stage.
4. **Gate the release on ECE ≤ 0.03 and AUROC ≥ 0.80.** A checkpoint that regresses calibration does not ship,
   the same way a checkpoint that regresses MMLU does not ship.

### 4.5 Data pipeline (exact)

| # | Artifact | How | Size | Cost |
|---|---|---|---|---|
| 1 | `provenance tokens` | prepend `<src:*>` to every pretraining doc | +0.15% tokens | ~0 (data track) |
| 2 | `fact_freq.cms` | count-min sketch over (entity, entity) and (entity, value) n-grams across the full corpus | ~2 GB sketch | ~1 CPU-day, 0 GPU |
| 3 | `factspan` tags | cheap regex+gazetteer tagger marking entity/number/date spans in pretraining text; used for the Phase-0 mask | inline | CPU, streaming |
| 4 | `probe_questions` | 400k existing open QA (TriviaQA 95k, NQ-open 87k, EntityQuestions 176k, PopQA 14k, WebQ, HotpotQA) **+** 600k templated Wikidata questions with alias sets | ~1M questions | CPU only — **no teacher model needed** |
| 5 | `known_set.jsonl` | run base checkpoint: greedy + k=8 @ T=0.7 per question, alias-normalized exact match → `y_know ∈ {0,1/8,…,1}` + SliCK bucket | 1M × 9 gens × ~20 tok = **180M generated tokens** | **≈ 7–10 A100-h** [E] |
| 6 | `grounded_set.jsonl` | (document, question, answer) with **sentence-level support labels** from an off-the-shelf NLI model, plus synthetic negatives made by entity-swap corruption of grounded answers (free, and gives perfectly clean `y_ground=0` labels) | ~500k | ≈ 3 A100-h |
| 7 | `sft_mix` | apply §4.4 rules to 4–6 to produce answer / hedge / `<IDK>` / `<RETRIEVE>` targets with `<conf:k>` tokens | ~1.5M examples | CPU |
| 8 | `calib_split` | 20–50k held-out graded questions, stratified by domain, **never** used for training | 50k | reserved from 5 |
| 9 | `rl_prompts` | ~50k verifiable short-answer prompts for the RLCR phase | 50k | reserved from 4 |

Recompute steps 5 and 8 **after every base-model change** — the known-set is a property of the checkpoint, not of
the data. This is the main operational burden of the design and should be a scripted, one-command job.

---

## 5. Compute & memory budget

**Context.** A 300B-token pretraining run at 1.3B active params is `6 × 1.3e9 × 3e11 = 2.34e21` FLOPs; at a
realistic 90–120 TFLOP/s for a small MoE on one A100 that is **5,400–7,200 A100-hours** [E]. *(Flagging a
cross-track inconsistency: the stated project budget of "a few hundred A100-hours" and "200–400B tokens" differ by
~15×. This report therefore quotes costs both absolutely and as a fraction of the pretraining run.)*

| Item | Cost | % of pretrain | Notes |
|---|---|---|---|
| PCH parameters | **0.59M** (d=2048→256→256→2) | +0.005% of 10B | 1.2 MB at fp16 |
| Extra vocabulary (32 tokens) | 65k params | negligible | |
| Phase-0 aux loss training overhead | **< 1%** of pretraining wall-clock | ~40–70 A100-h | one extra small matmul per token + the MSE |
| PCH forward FLOPs per token | 2 × 0.59M = **1.2 MFLOPs** vs 2.6 GFLOPs for the model | **0.05%** | genuinely free |
| Fact-frequency sketch + tagging | 0 GPU | 0% | ~1 CPU-day |
| Known-set self-labelling (step 5) | **7–10 A100-h** | 0.15% | repeat per checkpoint |
| Groundedness labelling (step 6) | **~3 A100-h** | 0.05% | |
| Confidence SFT (300M tokens @ 1.3B active) | `6×1.3e9×3e8 = 2.34e18` → **5–8 A100-h** | 0.1% | |
| RLCR phase (50k prompts × 8 rollouts × 200 tok ≈ 80M gen tokens + updates) | **20–30 A100-h** | 0.4% | the only expensive item; A4 decides whether it earns its keep |
| Conformal / temperature fitting | **< 0.5 A100-h** | ~0 | 1-D searches on 50k held-out items |
| **Total added** | **≈ 40–60 A100-hours** | **≈ 1% of pretraining** | |
| Ablation programme (§7) | **≈ 32 A100-hours** | — | mostly on off-the-shelf 0.36–0.6B substrates |

**Inference / on-device memory:**

| Item | Cost |
|---|---|
| PCH weights | **1.2 MB** fp16, 0.6 MB int8 |
| PCH latency | < 0.05 ms/token; **0.05%** of decode |
| Confidence-token overhead | 1–2 tokens per answer |
| CAD grounded mode | **2× forward** — server default on, phone default off |
| On-device retrieval index (recommended tier) | top-1M entities, one sentence each: **~100–150 MB** PQ-64 vectors + **~250–400 MB** zstd text = **~400–550 MB**, mmap'd from flash, not resident |
| On-device retrieval index (full Wikipedia passages, for reference) | 6.5M passages × 64B PQ = 416 MB + ~2.5 GB compressed text — **too large for the default install**, ship as an optional download |
| Retrieval latency budget | ~30–80 ms for an mmap'd IVF-PQ lookup + 200–400 ms to re-read context; only paid on the ~40% of queries that route to retrieval |

**Note the 8 GB constraint honestly:** Prophet-main at 10B total, 3.5-bit weights ≈ **4.4 GB**, plus KV cache,
plus the runtime — the retrieval index cannot be resident. mmap from flash is the only option, and the confidence
gate is what keeps the number of flash round-trips down to ~1 per 2.5 queries instead of 1 per query.

---

## 6. Risks & failure modes

### 6.1 Over-abstention destroys 0/1-scored benchmarks — quantified

This is the biggest product risk and it is arithmetic, not opinion.

| Benchmark family | Scoring | Cost of abstaining on fraction *f* | Break-even threshold |
|---|---|---|---|
| MMLU / ARC / GPQA / HellaSwag (4-way MC) | 1 point for correct, 0 for wrong, **0 for abstain** | `f × a` points, where `a` is the accuracy we'd have had. Abstain on 20% of MMLU at a=40% → **−8.0 points** | `t = 0` → **never abstain**; even guessing at chance (25%) beats abstaining |
| SimpleQA (F) | harmonic mean(correct, correct-given-attempted) | abstention **helps** up to ~90% abstention (§1.5) | `t ≈ 0.5–0.75` |
| TruthfulQA (%true) | true/false | abstention **trivially maximizes** it — "I have no comment" scores 100% true, 0% informative | must report **%true AND %info** |
| FACTS Grounding | two-stage: eligibility gate then groundedness | abstaining fails the **eligibility** gate → scores 0 | abstention is explicitly punished — good benchmark |
| Real deployment | application-defined | — | 0.5 (chat) … 0.9 (medical) |

**Design consequence, and it is non-negotiable:** abstention must be a **decode-time policy governed by a runtime
`t`**, never a behaviour baked unconditionally into the weights. The model must retain full ability to produce a
best guess when `t = 0`. The forced-guess self-distillation channel in §4.4 exists precisely to protect this, and
ablation A5 exists to plot the whole trade-off curve before we commit.

### 6.2 Other failure modes

| Risk | Severity | Quantification / mitigation |
|---|---|---|
| **Probes don't work at 1.3B active.** All published probe results are ≥7B. | **Critical — kills the track** | Ablation A1 measures AUROC at 0.36–0.6B *before* any main-run commitment. Kill criterion: if the best probe AUROC < 0.70 at 0.6B, drop the head and fall back to logprob thresholding + always-retrieve. |
| **RLHF/DPO destroys calibration.** | High | GPT-4 ECE 0.007→0.074, ~10× [P]. Freeze the head during preference optimization; re-fit temperature after; use RLCR reward; **gate release on ECE ≤ 0.03**. |
| **Confidence reward hacking.** The policy could suppress answer diversity to make its own confidence look good. | Medium | Brier is proper *given* the answer distribution, but the answer distribution is also an action. Monitor accuracy and answer-entropy alongside calibration in A4; abort the RL phase if accuracy drops > 1 point. |
| **Quantization decalibrates and de-knowledges.** int4 reportedly costs ~2.8× knowledge capacity [P VERIFY]. | High | Re-fit temperature and thresholds **after** quantization (A7); expect and budget for a *higher* abstention rate on-device; consider keeping the knowledge-critical expert layers at int8 while quantizing attention harder. |
| **Threshold shift across domains.** τ fitted on Wikipedia QA will not hold on code or medicine. | Medium | Per-domain conformal calibration; ship 4 threshold sets; A7 measures the transfer gap. |
| **Self-labelling noise.** Exact-match grading calls paraphrases wrong, inflating the "unknown" bucket and over-training abstention. | Medium | Alias-set normalization from Wikidata + a small NLI check on disagreements. Measure grader precision on a 1k human-checked sample. |
| **Users hate abstention.** "I don't know" is a bad chat experience even when correct. | Medium | Never emit a bare `<IDK>`. The abstention surface must be *"I don't know — want me to look it up?"* plus the retrieval action. Abstention without a retrieval path is a product failure. |
| **Retrieval hurts on head entities** (Mallen 2212.10511). | Medium | This is exactly what the gate is for; A6 measures accuracy on head vs. tail splits separately and will show a *regression* if the gate is mis-tuned. |
| **Hallucination is provably not eliminable** (2401.11817). | Framing | We are not claiming elimination. We are claiming a 3–4× reduction in error rate at a ~2-point accuracy cost, with a conformal bound. Set external expectations accordingly. |

---

## 7. Ablation plan

All experiments use a small substrate so that none exceeds **6 A100-hours**. Ablations A1–A8 run on
**off-the-shelf 0.36–0.6B checkpoints** (SmolLM2-360M, Qwen3-0.6B) so there is **no pretraining cost**; only A0
trains from scratch, on a synthetic corpus where fact frequency is known exactly.

| # | Question | Setup | Metric / decision | Budget |
|---|---|---|---|---|
| **A0** | Does the Phase-0 aux loss cost LM quality, and can a head read "how often did I see this" off the residual stream? | Allen-Zhu-style synthetic bioS corpus with **controlled exposures** (1/3/10/100/1000 per fact). Train 100M from scratch on 3B tokens (`6×1e8×3e9=1.8e18` → **~4.2 h**), with and without `l_freq` at λ ∈ {0, 0.02, 0.1}. | (i) Δ LM loss; (ii) head AUROC for predicting per-fact recall; (iii) Δ downstream fact accuracy. **Go if Δ LM loss < 0.005 nats and AUROC > 0.85.** | 5 h |
| **A1** | **Do confidence signals work at all below 1B?** (the track's kill-switch) | SmolLM2-360M + Qwen3-0.6B on TriviaQA / PopQA / NQ-open / EntityQuestions (100k questions × 10 gens ≈ 20M tokens). Compare: max-prob, mean-logprob, entropy, semantic entropy (k=10 + NLI), SelfCheck-NLI (k=5), P(True), linear probe (SEP-style), PCH-MLP. | AUROC + AURAC + ECE. **Kill criterion: best AUROC < 0.70.** | 3 h |
| **A2** | Which layer, which position, which features? | Sweep probe layer ∈ {L/4, L/2, 3L/4, L}, position ∈ {answer-start, post-answer, both}; ablate each of the 6 `decode_stats` features. | ΔAUROC per feature; pick the final feature set. | 3 h |
| **A3** | **Replicate Gekhman at 0.5B and measure the slope.** | SFT Qwen3-0.6B with 0 / 10 / 25 / 50 / 100% Unknown-bucket examples asserted; hold everything else fixed. | Hallucination rate on held-out QA vs. unknown-fraction. **Deliverable: the numerical slope**, which sets the §4.4 cap defensibly. | 4 h |
| **A4** | Is the RLCR phase worth 20–30 A100-h at full scale? | 0.6B, 20k verifiable prompts, 4 rollouts. Arms: SFT-only, RLVR (correctness), **RLCR** (correctness + Brier). | Accuracy, ECE, Brier, AUROC, answer-entropy. **Go if RLCR holds accuracy within 1 pt of RLVR while cutting ECE ≥ 2×.** | 5 h |
| **A5** | **Quantify the over-abstention trade-off** (§6.1). | Sweep `t` ∈ {0, 0.1, …, 0.9} on the A1 substrate + PCH. | Plot SimpleQA-F, TruthfulQA (%true, %info), MMLU, and utility-at-penalty `U(t)=correct − t/(1−t)·wrong`, all vs. `t`. **Deliverable: the shipped threshold table.** | 3 h |
| **A6** | Confidence-gated vs. always-on vs. never retrieval. | PopQA split by entity popularity (head / torso / tail) + a 1M-passage Wikipedia index. Arms: never / always / gated-at-τ. | Accuracy per popularity bucket + retrieval call rate. **Target: gated ≥ always-on accuracy with ≤ 50% of the calls.** | 4 h |
| **A7** | Does calibration survive quantization and domain shift? | Quantize the A1 substrate to int8 / int4 / 3-bit; fit τ on Wikipedia-QA, test on medical + code QA. | ΔECE, ΔAUROC, conformal coverage vs. nominal α. **Deliverable: whether per-domain and post-quant re-fitting is mandatory (expected: yes).** | 3 h |
| **A8** | `<IDK>` token vs. scalar head vs. verbalized `<conf:k>` bins. | Three SFT arms on the same data. | AUROC, ECE, Δ LM loss, decode-time ergonomics. | 3 h |
| | | | **Total** | **≈ 33 h** |

**Sequencing:** A1 first (it is the kill-switch, 3 hours, and needs no training). Then A3 and A5 (these change the
post-training data plan for every other track). A0 before any main-run commitment. A4/A6/A7 before ship.

---

## 8. References

**Why models hallucinate / theory**
- Kalai, Nachum, Vempala, Zhang — *Why Language Models Hallucinate* — **2509.04664** [V]
- Allen-Zhu & Li — *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* — **2404.05405** (ICLR'25) [V]
- Kandpal et al. — *Large Language Models Struggle to Learn Long-Tail Knowledge* — **2211.08411**
- Xu, Jain, Kankanhalli — *Hallucination is Inevitable: An Innate Limitation of LLMs* — **2401.11817**
- Muennighoff et al. — *Scaling Data-Constrained Language Models* — **2305.16264**; Xue et al. — **2305.13230**

**Fine-tuning and hallucination**
- Gekhman et al. — *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* — **2405.05904** (EMNLP'24) [V]
- Zhang et al. — *R-Tuning: Instructing Large Language Models to Say "I Don't Know"* — **2311.09677** (NAACL'24) [V]
- Yang et al. — *Alignment for Honesty* — **2312.07000**
- Cohen et al. — *I Don't Know: Explicit Modeling of Uncertainty with an [IDK] Token* — **2412.06676**
- Yin et al. — *Do Large Language Models Know What They Don't Know?* (SelfAware) — **2305.18153**

**Detection & calibration**
- Farquhar, Kossen, Kuhn, Gal — *Detecting hallucinations in LLMs using semantic entropy* — **Nature 630, 625–630 (2024)**
- Kuhn, Gal, Farquhar — *Semantic Uncertainty* — **2302.09664**
- Kossen et al. — *Semantic Entropy Probes* — **2406.15927**
- Manakul, Liusie, Gales — *SelfCheckGPT* — **2303.08896** [V, numbers]
- Kadavath et al. — *Language Models (Mostly) Know What They Know* — **2207.05221**
- Lin, Hilton, Evans — *Teaching Models to Express Their Uncertainty in Words* — **2205.14334**
- Tian et al. — *Just Ask for Calibration* — **2305.14975**
- Azaria & Mitchell — *The Internal State of an LLM Knows When It's Lying* — **2304.13734**
- Chen et al. — *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection* — **2402.03744**
- Chuang et al. — *Lookback Lens* — **2407.07071**
- Yuksekgonul et al. — *Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors* — **2309.15098**
- Quach et al. — *Conformal Language Modeling* — **2306.10193**
- Mohri & Hashimoto — *Language Models with Conformal Factuality Guarantees* — **2402.10978**
- Zhu et al. — *On the Calibration of LLMs and Alignment* — **2311.13240**

**Decoding & training interventions**
- Chuang et al. — *DoLa: Decoding by Contrasting Layers* — **2309.03883** [V, +12–17 pts claim]
- Li et al. — *Inference-Time Intervention (ITI)* — **2306.03341** [V, 32.5→65.1]
- Shi et al. — *Trusting Your Evidence: Context-Aware Decoding* — **2305.14739**
- Damani et al. — *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR)* — **2507.16806**
- Band et al. — *Linguistic Calibration of Long-Form Generations* — **2404.00474**
- Dhuliawala et al. — *Chain-of-Verification* — **2309.11495**

**Retrieval**
- Borgeaud et al. — *Improving LMs by Retrieving from Trillions of Tokens (RETRO)* — **2112.04426**
- Wang et al. — *Shall We Pretrain Autoregressive LMs with Retrieval?* — **2304.06762**
- Izacard et al. — *Atlas* — **2208.03299**
- Khandelwal et al. — *Generalization through Memorization: Nearest Neighbor LMs* — **1911.00172**
- Ram et al. — *In-Context Retrieval-Augmented Language Models* — **2302.00083**
- Lin et al. — *RA-DIT* — **2310.01352**
- Asai et al. — *Self-RAG* — **2310.11511**; *Reliable, Adaptable, Attributable LMs with Retrieval* — **2403.03187**
- Jiang et al. — *Active Retrieval Augmented Generation (FLARE)* — **2305.06983**
- Mallen et al. — *When Not to Trust Language Models* (PopQA) — **2212.10511**
- Jeong et al. — *Adaptive-RAG* — **2403.14403**

**Benchmarks**
- Wei et al. — *SimpleQA* — **2411.04368**; grading + reference scores via `github.com/openai/simple-evals` [V]
- Lin, Hilton, Evans — *TruthfulQA* — **2109.07958**
- Li et al. — *HaluEval* — **2305.11747** [V, dataset sizes]
- Min et al. — *FActScore* — **2305.14251**
- Jacovi et al. — *FACTS Grounding* — **2501.03200**
- Gao et al. — *ALCE: Enabling LLMs to Generate Text with Citations* — **2305.14627**
