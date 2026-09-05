# A4 — Self-verification: the wall behind the agentic wall

> Track A4. Question: an agent cannot tell when it is wrong, so it cannot recover, cannot
> know when to stop, and cannot safely learn from what it did. W4 measured that in
> amortised reasoning **verification is 93 % of the cost**; 07_WALLS §D.5 records that
> consolidating even *correct* solutions can make a model worse. Is verification easier
> than generation for a model the way it is for a human? Which signals are free? And what
> should Prophet do differently when a result can be checked cheaply versus when it cannot?

## 0. Provenance — read before trusting any number here

The outbound proxy blocked **arxiv.org, ar5iv, alphaxiv, openreview.net, aclanthology.org,
nature.com, huggingface.co, proceedings.iclr.cc, research.google, medium.com** and most lab
blogs during this session. Every published number below therefore comes from **search-result
snippets** of the primary paper, or from GitHub READMEs, or from this repository's own
reports. Tags:

| Tag | Meaning |
|---|---|
| `[S]` | Read in a search snippet quoting the primary source. Reliable for headline numbers; table cells may be mis-attributed. |
| `[P]` | From memory or from an earlier Prophet report that itself flagged it. **Verify before spending compute on it.** |
| `[C]` | Computed in this session against this repository (`prophet.budget`, `configs/prophet_500m_probe.json`). Reproduce with the snippet in §9. |
| `[E]` | My estimate, with the arithmetic shown. |
| `[?]` | A 2026 preprint whose ID I could not open. Existence confirmed by search; content from the snippet only. |

The per-token FLOP numbers here differ slightly from W4's (k=16/k=2 = **5.17×** now versus
4.67× then) because `prophet.budget` has been edited since. Where I reuse W4's break-even
arithmetic I keep W4's units and say so.

---

## 1. Why self-correction fails

### 1.1 The phenomenon, with numbers

The canonical negative result is Huang et al., *Large Language Models Cannot Self-Correct
Reasoning Yet* (arXiv **2310.01798**, ICLR 2024) `[S]`. With no oracle feedback:

| Model | Task | Initial | After round 1 | After round 2 |
|---|---|---:|---:|---:|
| GPT-4 | GSM8K | **95.5** | 91.5 | **89.0** |
| GPT-3.5 | GSM8K | 75.9 | — | 74.7 |

The mechanism is visible in the transition matrix. On GSM8K, GPT-3.5 keeps its answer
74.7 % of the time; of the rest it turns **7.6 % of wrong answers right and 8.8 % of right
answers wrong** `[S]`. Self-correction is a coin flip whose expected value is slightly
negative, and each extra round is another flip. The authors' diagnosis is the one this
track is about: *the model cannot judge whether its reasoning is correct.*

Three later results sharpen it:

- **Finding the error is the bottleneck, not fixing it.** Tyen et al. (arXiv
  **2311.08516**, BIG-Bench Mistake) `[S]`: the best mistake-location accuracy on five
  unambiguous, objectively-graded tasks was **52.87 %** (GPT-4, direct step-level prompting).
  When the *location* of the first mistake is supplied by an oracle, backtracking recovers
  most of the lost accuracy. Correction ability is intact; detection is not.
- **Discrimination is not better than generation.** SELF-[IN]CORRECT (arXiv **2404.04298**)
  `[S]`: across tasks, models are *not reliably better at discriminating among their own
  candidates than at generating a first answer*. Song et al., *Mind the Gap* (arXiv
  **2412.02674**, ICLR 2025) `[S]` make it quantitative: the generation–verification gap
  is **non-positive for nearly all verification methods below ~7B** (Qwen-1.5/2 0.5B,
  Llama-2 7B), becomes positive only for medium/large models under chain-of-thought
  verification, and **scales monotonically with pre-training FLOPs**.
- **The blind spot is specific to the model's own output.** Self-Correction Bench (arXiv
  **2507.02778**) `[S]`: 14 open non-reasoning models corrected injected errors in *user*
  text but ignored identical errors in their *own* text at an average **64.5 %** blind-spot
  rate; appending the single token "Wait" removed **89.3 %** of it. The detector exists;
  the model does not invoke it on itself.

### 1.2 The mechanism

Three effects, all measured, compose into the failure:

1. **Verifier errors are coupled to generator errors.** *Variation in Verification*
   (arXiv **2509.17995**, ICLR 2026; 14 models 2B–72B, 12 benchmarks) `[S]`: a generator
   error raises the odds of a verifier error by **57×**. A model that produced a wrong step
   did so because its representation of the problem is wrong; the same representation is
   what it verifies with. This is why self-verification is not an independent second draw.
   The same paper finds verifier gains peak at only **~0.1** in accuracy and fall below
   **0.05** on both very easy and very hard problems; stronger verifiers (GPT-4o) added
   nothing over Qwen2.5-7B in those regimes.
2. **Verification is learned, not inherited from generation, and it comes first only for
   facts.** *The Future of Facts* (arXiv **2605.27564**) `[?]` traces generation and
   verification through acquisition, continual learning and updating in four model families
   at two scales each, and finds verification of *factual* claims is learned **before**
   generation. That is the recognition-versus-recall asymmetry, and it holds for knowledge.
   For *reasoning*, Mind the Gap says the opposite at small scale: the gap is negative.
   The asymmetry humans and complexity theory enjoy (NP: checking a certificate is cheap)
   exists for models only where the check is a *lookup*, not a *recomputation*.
3. **Self-improvement is sharpening onto the model's own verifier.** Huang et al.,
   *The Sharpening Mechanism* (arXiv **2412.01951**) `[S]`: iterated self-training cannot
   add information the verifier does not already contain. Combined with (1), a self-verified
   loop amplifies exactly the errors it shares with the generator. W4 §8.2 lists the
   rise-then-collapse curves this produces.

**Answer to the track's first question.** For models at our scale, verification is **not**
easier than generation *when the verifier is the same model checking its own reasoning*.
It **is** easier in three cases, and the design in §7 is built only on those: (a) the check
is an execution or lookup rather than a recomputation; (b) many candidates are available
and the verifier can compare them rather than judge one in isolation — Kadavath et al.
(arXiv **2207.05221**) `[S]` found showing five samples materially improves P(True), and
*Sample, Scrutinize and Scale* (arXiv **2502.01839**) `[S]` shows self-verification
accuracy *rising* with the pool size ("implicit scaling": Gemini 1.5 Pro on AIME 2024 goes
from 1/15 at pass@1 and 4/15 at consistency@200 to **8/15 at verification@200**, above
o1-preview's 7/15); (c) the verifier is a *separate* model trained on labels the generator
did not produce (§2).

### 1.3 Final answer versus step versus tool result

The three are different problems with different evidence and different costs.

| What is verified | Best published detector | Cost | What the evidence says |
|---|---|---|---|
| **Final answer** | Execution / exact match when available; else a trained ORM or confidence head | 0 – 1 forward | Solvable *only* by an external check or a separately trained head. Self-judgement is at chance for ≤7B (§1.1). |
| **Reasoning step** | Process reward model (§2) or generative critic | 1 forward per step, or a CoT per step | Hard even for frontier models: **52.87 %** first-error location `[S]`; on Hard2Verify (arXiv **2510.13744**, 1,860 human-graded steps from 200 frontier responses, 29 verifiers) weak verifiers degenerate to "all steps correct" — **TNR → 0 while TPR → 1** `[S]`. 7–8B PRMs score **26.6–42.1 average F1** on ProcessBench `[S]`. |
| **Tool result** | Structural checks (schema, type, invariants, exit code) | ~0 | Almost no benchmark measures whether agents *verify* tool outputs; the 2026 provenance survey (arXiv **2606.04990**) `[?]` documents that agents routinely narrate tool results that were never returned. The cheap part — did the call happen, did it return the type it should — is the part nobody checks. |

The practical ordering is the reverse of the intuitive one. Steps are the hardest to
verify and the most expensive; final answers are checkable only when a checker exists;
tool results are the cheapest to check and the most dangerous to skip. Prophet's agent loop
should spend its verification budget in that order: structural checks on every tool
result, execution on every final answer that admits one, and step verification never,
except as ranking (§2.4).

---

## 2. Process reward models at our scale

### 2.1 What PRMs buy, and what they cost, at the scale where they were measured

| Work | arXiv | Labels | Cost of labels | Result | Tag |
|---|---|---|---|---|---|
| Let's Verify Step by Step (PRM800K) | 2305.20050 | 800K human step labels | ~human-months | PRM **78.2 %** vs ORM 72.4 % vs majority 69.6 % on a MATH subset, best-of-1860 | `[S]` |
| Math-Shepherd | 2312.08935 | MC rollouts: a step is good iff completions from it reach the right answer | ~4 rollouts × steps × problems | Verification: DeepSeek-67B **93.3 GSM8K / 48.1 MATH** at BoN-256; step-PPO on Mistral-7B 77.9→**84.1** GSM8K, 28.6→**33.0** MATH | `[S]` |
| OmegaPRM | 2406.06592 | MCTS with binary search for the first error | 1.5 M annotations, automated | Gemini Pro MATH500 51→**69.4**, GSM8K 86.4→**93.6** | `[S]` |
| Implicit PRM | 2412.01981 | **Response-level** labels only; PRM = log-likelihood ratio of policy to reference | **1/38.8** of the MC pipeline | Beats an MCTS-labelled baseline with <1/38 of the data (Llama-3.1-8B-Instruct) | `[S]` |
| ThinkPRM | 2504.16828 | 1 K synthetic verification CoTs filtered on **8 K** labels (1 % of PRM800K) | small | Beats discriminative PRMs trained on all of PRM800K by **+8 %** (GPQA subset) / **+4.5 %** (LiveCodeBench) OOD | `[S]` |
| Qwen PRM lessons | 2501.07301 | Compares MC vs LLM-judge vs human labels | — | **MC-estimated labels are inferior** in accuracy and generalisation to judge/human labels; consensus filtering needed | `[S]` |
| Compute-optimal TTS | 2408.03314 | PRM from MC soft labels | — | **4×** efficiency over best-of-N; a small model beats a 14× larger one on easy/medium | `[S]` |
| 1B vs 405B | 2502.06703 | Off-the-shelf PRMs | — | 0.5B > GPT-4o, 3B > 405B, 7B > o1/R1 on MATH-500/AIME24 — **guided by a 7B PRM** | `[S]` |

### 2.2 The evidence on small PRMs specifically

ProcessBench (arXiv **2412.06559**) is the only benchmark that scores PRMs on *finding the
first wrong step* rather than on the downstream best-of-N number, and it is unflattering
`[S]`:

| PRM | Params | Avg F1 (GSM8K / MATH / Olympiad / Omni) |
|---|---:|---|
| RLHFlow-PRM-Deepseek | 8B | **26.6** (38.8 / 33.8 / 16.9 / 16.9) |
| Math-Shepherd-PRM | 7B | **31.5** (47.9 / 29.5 / 24.8 / 23.8) |
| Skywork-PRM | **1.5B** | **31.5** `[S]` |
| Skywork-PRM | 7B | 42.1 (70.8 / 53.6 / 22.9 / 21.0) |
| Qwen2.5-Math-7B trained on PRM800K | 7B | 56.5 (68.2 / 62.6 / 50.7 / 44.3) |
| Qwen2.5-Math-PRM-7B (judge+human-filtered labels) | 7B | ~73 `[P]` |

Two readings. First, **label quality dominates model size**: the same 7B backbone goes from
31.5 to ~73 depending on how the step labels were made, and the MC-labelled 7B PRMs are no
better than a 1.5B one. Second, **a 1.5B PRM with MC labels sits at 31.5 F1**, which on the
Olympiad/Omni splits is close to the degenerate "everything is correct" verifier Hard2Verify
describes. There is no published result showing a **≤1B PRM improving a ≤1B policy** by a
margin that survives a same-cost self-consistency baseline; 2502.06703's headline results
all use the 7B PRMs, and the paper reports that Math-Shepherd-style PRMs are sometimes
*worse* than majority voting on some policy/PRM pairs `[S]`.

The one encouraging small-verifier number is Weaver (arXiv **2506.18203**) `[S]`: an
ensemble of weak verifiers closes the generation–verification gap by **12.8 points (8B
generator) / 16.0 points (70B)**, and a **400M cross-encoder distilled from the ensemble
keeps 98.2 % of that** while saving **99.97 %** of the verifier FLOPs. A 400M verifier
works — *when distilled from something much stronger*. That is a donor-conversion result
(D10), not a train-from-scratch result.

### 2.3 Cost to train one on a single A100 `[C]`/`[E]`

Against `configs/prophet_500m_probe.json` (d=1536, prelude 2 / core 4 / coda 2) `[C]`:

| k | Effective depth | GFLOP/token (forward) | A100 decode tok/s (single stream) |
|---:|---:|---:|---:|
| 1 | 8 | 0.664 | 12,686 |
| **2** | 12 | **0.946** | **8,655** |
| 4 | 20 | 1.510 | 5,292 |
| **8** | 36 | **2.637** | **2,978** |
| 16 | 68 | 4.892 | 1,589 |

Batched rollouts are compute-bound, not bandwidth-bound; at an assumed 30 k tok/s sustained
for this model at k=2 `[E]` (0.946 GFLOP × 30 k = 28 TFLOP/s, well under the A100's dense
peak, so conservative):

| Pipeline | Tokens to generate | A100-hours at k=2 | At k=8 (×2.8) |
|---|---:|---:|---:|
| **MC step labels, Math-Shepherd style** — 10 k problems × 4 solutions × 6 steps × 4 completions × 200 tokens + 12 M for the solutions | ~204 M | **1.9 h** (5.7 h at 10 k tok/s) | 5–16 h |
| **Implicit PRM / ORM labels** — 10 k problems × 8 samples × 300 tokens, graded by execution | 24 M | **0.2–0.7 h** | 0.6–2 h |
| Head training on 240 k sequences × 300 tokens, 6 × 250 M FLOP/token | — | **~0.3 h** | — |

So a PRM is **2–6 A100-h** via MC labels or **under 1 A100-h** via outcome labels; the
rollouts dominate, the training is noise. The budget is not the obstacle. **Label quality
is**: the Qwen lessons paper says MC labels are the worst of the three sources, and we
have neither human annotators nor a strong judge model we are licensed to use (R10 §
licence guard). Whatever PRM we train will be a 31-F1 PRM, and 31 F1 is what "cannot find
the first error" looks like.

### 2.4 Recommendation for Prophet

Do not train a standalone PRM. Do the two things the evidence supports:

1. **An outcome-labelled value head on the coda** (R04 §4.6's design), trained
   implicit-PRM style on execution-verified rollouts — cost < 1 A100-h, labels are free and
   correct because a program produced them. Use it for **ranking** (best-of-*n* on the
   5090) and as one input to §7's verifier. Never use it for **admission** to memory.
2. **Let the free verifier be the PRM where one exists.** For code and arithmetic, a
   rollout's step label *is* whether the executed continuation passes; there is nothing to
   learn. Math-Shepherd's MC estimator is exactly this with a program in the loop.

The ablation that decides whether the head is worth its slot is A4-5 in §8: it must beat
adaptive self-consistency at equal cost, or it is dropped.

---

## 3. Free verifiers, and what share of agentic work they cover

### 3.1 The taxonomy

| Verifier class | Examples | Cost (GPU) | Cost (wall-clock) | Reliability | Prophet tier (W4 §6.4) |
|---|---|---:|---:|---|---|
| **Execution** | unit tests, compiler, interpreter, SQL engine, `assert`, type checker | 0 | 0.1 s – minutes CPU | High, bounded by test coverage; generated tests are themselves unverified | **T0** if the tests pre-exist; T1 if the agent wrote them |
| **Exact / symbolic** | numeric answer match, SymPy equivalence, unit conversion, JSON-schema, regex constraints, IFEval-style constraints | 0 | ms | Very high | **T0** |
| **Source grounding** | the claim appears in the retrieved document (NLI / span match) | small model | ms | Medium — NLI errors | T1 |
| **Consistency** | self-consistency / majority; adaptive variants | m × generation | m × | Medium; 30 % false-positive rate reported among pass@256 selections `[W4 §6.4]` | T1 |
| **Learned** | confidence head, ORM/PRM, LLM-judge | 0 – 1 forward | ms | AUROC 0.65–0.85 (§4) | T2 |
| **Human** | ask the user | 0 | seconds–hours | Highest, and finite | (the fallback) |

The one cheap variant of consistency worth having: **stop sampling when agreement is already
decided.** Adaptive-Consistency (arXiv **2305.11860**) cuts samples **2.8×** for **−0.03 %**
accuracy `[S]`; Early-Stopping SC (arXiv **2401.10480**) cuts them **80.1 % on GSM8K, 33.8 %
on MATH** `[S]`; DeepConf (arXiv **2508.15260**) filters traces by the model's own token
confidence and cuts generated tokens **84.7 %** at 99.9 % on AIME25@512 `[S]`. Applied to
W4's T1 column (`m = 8` deep samples = 37.4 k=2-units), a 3–5× reduction brings T1 to
**7–12 units** and its break-even from M ≥ 33 to **M ≥ 9–13**. Still not free; no longer
absurd.

**A scale effect the literature does not mention.** At 8,655 tok/s a 300-token answer costs
**35 ms** on the A100 and ~0.7 s on the phone; a `pytest` run costs 1–30 s. For a 250M-class
model, *execution is more wall-clock than generation*, the opposite of the frontier-model
regime. "Free" means free in GPU FLOPs and in *labels*; the agent's latency budget must
count the checker, and on-device the checker is the long pole.

### 3.2 Where free verifiers exist — and the selection bias in the benchmarks

Every agent benchmark with a number has a programmatic checker, because that is how it
got a number. TheAgentCompany (arXiv **2412.14161**) `[S]` wrote checkpoint evaluators for
all **175** office tasks; the best agent (Claude 3.5 Sonnet) completed **24 %** (34.4 % with
partial credit). SWE-bench has fail-to-pass tests for 100 % of instances by construction.
RLVR pipelines (Tulu 3, Qwen3, OLMo 3 — R10 §2.3) run on exactly three domains: **math,
code, and constraint-checkable instruction following**. The 2026 agentic-PRM work (arXiv
**2605.10325**, **2510.24636**) `[?]` exists precisely because long-horizon agentic tasks
have *sparse* verifiable rewards and most intermediate decisions have none.

What that means for the share of agentic work covered `[E]`, using a coding-agent
trajectory as the unit because it is the only one with measured components:

| Step type in an agent trajectory | Free verifier? | Approx. share of steps | Approx. share of *errors* |
|---|---|---:|---:|
| Run a command / call a tool | Structural (exit code, schema) — yes | 30–40 % | low per step, catastrophic when ignored |
| Edit code | Tests — yes **if coverage exists**; CodeMonkeys' oracle-selection ceiling (69.8 %) vs achieved (57.4 %) `[S]` measures the gap tests leave | 20–30 % | high |
| Localise / navigate | Post hoc only (did the fix work?) | 15–25 % | medium |
| Explain, summarise, decide scope, ask | **No** | 10–20 % | medium, unmeasured |

Roughly **half of the steps and most of the *checkable* errors** in a coding loop have a
free or structural verifier; for office/web tasks the share is lower and the checkers were
written by the benchmark authors, not available to the agent at run time. Outside code,
math and structured extraction, the honest number is **"a minority, and not the steps that
fail"**. That is the boundary §6's policy is built around.

---

## 4. Uncertainty signals

Each row is a signal Prophet can compute, its marginal cost in units of one k=2 forward
pass over the answer span, and the best *published* evidence that it predicts error.
"—" means I found no published AUROC; those are the rows §8 measures first.

| Signal | How Prophet gets it | Marginal cost | Published AUROC (error prediction) | Source | Tag |
|---|---|---:|---|---|---|
| **Token entropy / max-prob / margin** | logits, already computed | 0 | Consistently below semantic entropy; AUC-PR 83.2 NonFact on SelfCheckGPT-WikiBio for mean logprob | R09 §2.1; 2303.08896 | `[P]` |
| **Confidence head** (`heads.confidence_head`, R09's PCH) | one linear on the coda state | ~0 | R09's release gate is **AUROC ≥ 0.80**; P(IK)-style heads generalise across tasks at 52B | 2207.05221 | `[P]` |
| **P(True) prompt** | an extra short forward | ~0.1 | Well calibrated at ≥52B, degrades sharply at small scale; improves when 5 samples are shown | 2207.05221 | `[S]` |
| **Semantic entropy** | 10 generations + NLI clustering | **≥ 10** | ~0.75–0.80 typical across 5 QA sets × 4 families | Farquhar et al., *Nature* 630 (2024) | `[P]` |
| **Semantic entropy probe (SEP)** | linear probe on hidden state, labels from SE at training time | 0 | Retains most of SE's AUROC at "almost zero" cost; SE itself is a 5–10× overhead | 2406.15927 | `[S]` |
| **Hidden-state dynamics probe (ICR)** | probe on layer-wise residual updates | ~0 | **AUROC ≥ 0.66** on *unseen* datasets; cross-domain drop 8.61 % vs 10.18 % (SAPLMA) and 11.67 % (SEP) | 2507.16488 | `[S]` |
| **Exact-answer-token probe** | probe on the tokens of the answer, not the last token | ~0 | Beats last-token probes and logit methods; does **not** transfer across task types | 2410.02707 | `[S]` |
| **EigenScore (INSIDE)** | covariance spectrum of k sampled embeddings | k | +5.6 (CoQA) / +8.9 (SQuAD) AUROC points over lexical similarity | 2402.03744 | `[S]` |
| **Consistency of m samples** | m generations | m (2.8–5× less with early stopping) | The strongest verifier-free signal; saturates ~10² samples; **30 % FP rate at pass@256** | 2305.11860, 2401.10480, W4 §6.4 | `[S]` |
| **Depth disagreement** — argmax/KL between the coda read-out at iteration 2 and at iteration K | **free when `halting == "ponder"`** — `hidden_per_step` already holds the coda output at every iteration; otherwise one extra 2-layer coda pass at step 2 (~17 % of a k=2 pass) | 0 – 0.17 | **—** No published AUROC. Adjacent evidence: Huginn exits per token when successive-state KL < 5·10⁻⁴ `[S]` (2502.05171); on a 135M looped model mean successive KL falls 3.9·10⁻¹ → 8.5·10⁻⁶ from loop 2 to 16, the **median token converges by loop 6, ~10 % are still moving at loop 8**, and convergence depth is ordered by token type `[S]` (2607.14427 `[?]`); CALM makes early-exit consistency a calibrated guarantee (95 %) at up to 3× speedup `[S]` (2207.07061) | | `[S]`/— |
| **MTP-head disagreement** — KL between head-1's prediction for t+2 and the main head's prediction at t+1 | free when `return_mtp=True` | 0 | **—** Nothing published as an error predictor; the only related quantity is speculative-decoding acceptance rate, which is known to be lower on "hard" tokens | 2404.19737 (MTP); Medusa | — |
| **Execution result** | run it | 0 GPU, seconds CPU | Effectively 1.0 where the test is right; the residual is test quality (CodeMonkeys: 12.4-point gap between selection and oracle) | 2501.14723 | `[S]` |

Three things to take from the table.

1. **The cheapest useful signal with published evidence is a probe on the hidden state**
   (SEP / ICR / exact-answer-token): AUROC ~0.66–0.80, zero marginal cost, transfers
   imperfectly. Prophet's confidence head *is* this probe, trained on the labels R09
   specifies. Nothing on the list beats it per FLOP except execution.
2. **The two signals that are free *and unique to Prophet* have no published AUROC.**
   Depth disagreement is the model's own convergence test — the looped core either settled
   by iteration 2 or it did not — and the per-token convergence literature says ~10 % of
   tokens have not settled at the mean training depth. Whether *those* are the wrong ones is
   exactly what nobody has measured, and it costs nothing to find out (§8, A4-0).
3. **Every signal above AUROC 0.85 costs ≥ 10× or needs a program.** There is no cheap
   signal in the 0.9s. The policy in §6 is designed for verifiers at 0.7–0.8, not 0.95.

---

## 5. Verification in agent loops

### 5.1 How the deployed loops verify

| System | Verification step | Measured effect | Tag |
|---|---|---|---|
| **Reflexion** (2303.11366) | Generates its own unit tests, runs them, reflects on failures | HumanEval **91 %** pass@1 (GPT-4, vs 80 % baseline). Ablation: **without internal test generation and execution, accuracy falls to 52 % against a 60 % baseline** — reflection *without* a test signal is net negative | `[S]` |
| **Self-Refine** (2303.17651) | Model critiques its own output, no tools | Gains on math are "modest"; the authors attribute it to the inability to tell whether there is an error, and report gains of **5 %+** when an *external* source identifies the error | `[S]` |
| **CRITIC** (2305.11738) | Critique via tools (calculator, search, interpreter) | Tool-interactive critique improves GSM8K/QA; the paper's ablation shows the gain comes from the tool, not the self-critique | `[P]` |
| **Agentless** (2407.01489) | Regression tests + generated reproduction tests select among sampled patches by majority | **27.3 %** SWE-bench Lite at **$0.34/issue** (original); 40.7 % Lite / 50.8 % Verified with Claude 3.5 Sonnet. In the selection ablation, **generated reproduction tests give the largest gain** beyond majority voting and regression filtering | `[S]` |
| **CodeMonkeys** (2501.14723) | Jointly writes a test script and an edit; samples many trajectories; a selection state machine ranks by tests | **57.4 %** SWE-bench Verified at **~$2,300** per full run; oracle selection would reach the coverage of **69.8 %**; selection closes "about half" of the random→oracle gap | `[S]` |
| **R2E-Gym** (2504.07164) | Execution-based (agent-written tests) *and* execution-free (learned) verifiers over K candidates | Execution-based alone and execution-free alone both saturate around **42–43 %**; the **hybrid reaches 51 %** on SWE-bench Verified (32B open policy, pass@1 34.4 %) | `[S]` |
| **SWE-HERO** (2604.01496) `[?]` | Execution-based SFT: extra turns to author and run tests | "Trades temporal efficiency for a significant gain in resolution accuracy" — extra turns, higher resolve rate | `[S]` |
| **Live-SWE-agent** (2511.13646) | Self-evolving scaffold, tools written on the fly | **75.4 %** Verified / **45.8 %** SWE-bench Pro | `[S]` |
| **Thinking vs Doing** (2506.07976) | Scales *interaction* (act, observe, backtrack) rather than reasoning tokens | Acting more to gain information beats thinking longer per step on web tasks, even by prompting alone | `[S]` |
| Claude Code-style loops | Run tests, re-read diffs, ask the user when ambiguous | No published ablation; the structure matches the three-tier pattern above | — |

### 5.2 What the measurements say together

- **Verification without an external signal is worth ≤ 0 (Reflexion −8, Huang −4 to −6).
  Verification with execution is worth +11 to +30 points.** The entire measured benefit of
  "reflect" comes from the test that runs inside it.
- **The verifier's quality, not its existence, sets the ceiling.** Execution-based and
  learned verifiers each saturate at ~42–43 % and the hybrid at 51 % (R2E-Gym); CodeMonkeys
  leaves 12.4 points on the table between its selector and an oracle. Agent-written tests
  are themselves unverified generations, which is why a learned second opinion adds 8 points
  on top of them.
- **Verification is where the money goes.** Agentless at $0.34/issue is a *selection*
  pipeline over cheap samples; CodeMonkeys at ~$2,300/run spends most of it on the
  test-and-select loop over many trajectories. This is W4's 93 % measured in dollars.
- **The "verify budget" sweet spot is a knee, and it is early.** Every curve reported —
  Agentless's patch samples, CodeMonkeys' serial/parallel budget, adaptive consistency —
  shows most of the gain from the first 2–4 verified attempts, then a slow log-linear tail
  that only an oracle-quality verifier can harvest (Large Language Monkeys: coverage
  log-linear over four orders of magnitude, but converts to accuracy only with a verifier —
  R04 §2.1). For a verifier at AUROC ~0.8 the tail is not harvestable at all (§6.2), so the
  budget should be **n ≤ 4 attempts** with a learned verifier and **n ≤ 8–16** with an
  execution verifier, then ask.

---

## 6. The verifiability-aware policy

### 6.1 The quantity that decides everything

Define, in units of one k=2 forward over the answer span:

- `c_g` — cost of one attempt (1 at k=2; **2.79** at k=8; 5.17 at k=16 `[C]`).
- `c_v` — cost of checking one attempt.
- `ρ = c_v / c_g` — the **verifiability ratio**.
- `p` — probability an attempt is correct; `t`, `f` — the verifier's true- and
  false-positive rates at its operating point.

Sequential retry-until-accepted, with i.i.d. attempts, gives closed forms:

```
P(accepted answer is correct)  =  p·t / (p·t + (1−p)·f)
E[attempts]                    =  1 / (p·t + (1−p)·f)          (capped at n)
P(a wrong answer is accepted)  =  (1−p)·f / (p·t + (1−p)·f)   =: f_w
```

Evaluated at `p = 0.5` (a hard query for this model) `[E]`:

| Verifier | `ρ` | `(t, f)` | Accuracy after retry | E[attempts] | `f_w` (wrong-write rate) |
|---|---:|---|---:|---:|---:|
| None | 0 | t = f | **0.50** | 1 | — |
| Confidence head, AUROC ≈ 0.65 | 0 | (0.70, 0.50) | 0.58 | 1.67 | **42 %** |
| Confidence head, AUROC ≈ 0.80 | 0 | (0.80, 0.35) | **0.70** | 1.74 | **30 %** |
| Self-consistency m=8 (adaptive) | 3–5 | ≈ (0.85, 0.30) `[P]` | 0.74 | — | 26 % |
| Execution, imperfect tests | ~0 GPU | (0.95, 0.02) | **0.98** | 2.06 | **2 %** |

At `p = 0.2`: the head at 0.80 gives 0.36 after 2.3 attempts; execution gives 0.92 after
4.9 attempts (0.83 with n capped at 8).

### 6.2 What the numbers force

1. **A learned verifier converts retries into accuracy (+8 to +20 points at 1.7× cost) but
   cannot be trusted to admit anything.** Its wrong-write rate of 30–42 % means a memory
   written on its say-so caps future class accuracy at 58–70 % (07_WALLS §D.5), *below what
   a deep pass already gets*. Extending W4 §5.3's break-even with the harm term — saving per
   hit `= h·[(N−1)(1−f_w) − f_w·L]` where `L` is the cost of a confidently wrong answer in
   k=2 units — at `f_w = 0.30, N = 4.67, L = N` the saving per hit falls from 3.67 to **1.17**,
   and at `L = 10` it is **negative**. This reproduces W4's tier table from the verifier's
   ROC alone: **T2 never admits; T0 admits immediately; T1 quarantines.**
2. **Where `ρ ≈ 0` and `t ≈ 1`, spend freely.** Execution turns a 50 % model into a 98 %
   model for 2 attempts. There is no other 2× in this document.
3. **Where `ρ ≥ 3`, the second attempt is not worth generating.** Self-consistency at 0.74
   vs the head at 0.70 buys 4 points for 3–5× the cost; the head buys 20 points for 0×.
   Beyond one retry, *ask*.
4. **Depth is a retry with a selector.** A blanket k=8 pass costs 2.79× for R04's
   Huginn-shaped ~1.8-point gain. If depth disagreement flags 20 % of queries and those
   carry the whole gain (Gate 0's ≥ 5-point threshold), the conditional gain on flagged
   queries is 25 points at 2.79× on 20 % of traffic — **1.36× average cost for +5 points**,
   which is the only way the depth dial pays for itself on the phone.

### 6.3 The decision rule

```
inputs: c   = P(correct | signals)  from §7's verifier (calibrated)
        ρ   = verifier cost ratio for this task class   (0: execution/symbolic; ~0: head;
                                                         3–5: consistency; ∞: none)
        t_d = deployment threshold (R09 §4.3: 0.0 eval / 0.5 chat / 0.75 factual / 0.9 high-stakes)
        d   = depth disagreement on the answer span  (fraction of tokens whose argmax
                                                      differs between iteration 2 and K)
        n   = attempts so far

ACT / RETRY / ASK
  if a free verifier exists (ρ ≈ 0, t ≈ 1):
        act; run it; on failure retry at the same depth while n < 8 (16 on the 5090);
        then ask.                                   # 0.50 → 0.98 for ~2 attempts
  elif c ≥ t_d:
        act.                                        # R09's rule, unchanged
  elif d ≥ d* and n == 0 and the Gate-0 depth gain is ≥ 5 points:
        retry once at k_max.                        # 2.79× on the flagged minority
  elif ρ ≤ 1 and n < 2:
        retry once (resample), re-score with the head, keep the higher c.   # +8–20 pts
  else:
        ask the user, showing c.                    # on-device default beyond 2 attempts

CONSOLIDATE (W4 §6.4 tiers, now derived from f_w)
  T0 (execution / exact)          -> write to the main ledger, permanent, provenance = T0
  T1 (≥ 3 independent agreements) -> quarantine ledger, promote after g ≥ 3 later T1 hits or one T0
  T2 (head only, any c)           -> never; f_w ≥ 30 % caps the class below the deep pass
  T3 (nothing)                    -> refuse

STOP / CONTINUE (inner loop, per token, per iteration)
  continue the recurrent loop while  halt_probs says "not yet"  and  KL(step i ‖ step i−1) > κ
        (Huginn's κ = 5·10⁻⁴; ours is a config field to be swept)  and  i < k_max
  stop the outer loop when  a T0/T1 check passes,  or  n hits the cap,  or  c ≥ t_d with ρ = ∞
```

Two lines, as the brief asked:

> **Think and retry in proportion to how cheaply the result can be checked: unbounded
> attempts behind an execution check, one extra attempt behind a learned check, none behind
> nothing — then ask. Consolidate only what a program or three independent runs agreed on;
> a confidence head may gate acting, never remembering.**

---

## 7. Design for Prophet

### 7.1 Specification

**Module.** `prophet/verify/verifier.py` — a `Verifier` that consumes one `ProphetOutput`
(plus, optionally, a second at a different depth and an execution result) and emits a
calibrated `P(correct)` per answer span, a `Tier`, and the three decisions of §6.3. It is
a *reader* of signals the trunk already produces; it adds **one small MLP (≈ 0.1 M params)**
and no forward passes in the common case.

**Signals combined** (all per answer span, pooled over positions):

| # | Signal | From | Free? |
|---|---|---|---|
| 1 | Confidence head logit (`output.confidence`) | `heads.confidence_head` | yes |
| 2 | Mean token entropy, mean margin, min max-prob over the span | `output.logits` | yes |
| 3 | **Depth disagreement**: fraction of span tokens whose argmax differs between the iteration-2 read-out and the final one; mean KL | `output.hidden_per_step[1]` vs `output.hidden` when halting is on; else a second forward at `loop_k=2` | yes / 1× |
| 4 | Expected depth from `output.halt_probs` (a converged token halts early) | halting head | yes |
| 5 | **MTP disagreement**: mean KL between head-1's prediction of token t+2 and the main head's prediction of the same token one step later | `output.mtp_logits[0]` | yes |
| 6 | Execution result ∈ {pass, fail, unavailable} | the agent's tool layer | 0 GPU |

**Training labels.** Only labels a program produced. Three sources, in priority order:

1. **Execution-verified rollouts** on R10's RLVR prompt pools (math with checkable answers,
   code with tests, IFEval constraints): 50 k prompts × 4 samples × ~300 tokens = 60 M
   tokens → **0.6–2 A100-h at k=2** `[E]`, giving ~200 k `(span, correct ∈ {0,1})` pairs.
2. **Exact-match QA** from R09's `known_set.jsonl` (already budgeted at 7–10 A100-h there;
   reused, not re-spent).
3. Each example is run at **k=2 and k=8** so signal 3 has both read-outs — ×3.79 on the
   forward cost of (1), i.e. the whole label pass is **≈ 3–8 A100-h**.

Loss: soft-label BCE against `k_correct / k` per prompt (a proper scoring rule, as R09 §4.2
argues), temperature fitted post hoc on a held-out split, **never trained jointly**. The
verifier MLP is trained with the trunk **frozen**; the confidence head keeps R09's schedule.

**Calibration and the guarantee.** Split-conformal per domain (R09 §4.3) on ≥ 2 000
held-out graded items, so that `P(act ∧ wrong) ≤ α` holds for the chosen `t_d`.

**Provenance.** Every ledger write carries `(tier, verifier_version, c, d, n_attempts)`.
This replaces `DepthEpisode.verified: bool`, which W4 §6.2 correctly says cannot be
audited, revoked or used for eviction.

**Reversibility (CLAUDE.md rule 3).** `verifier.enabled = False` restores R09's plain
`c_know > t` rule; `depth_signal = False` drops the second forward; `λ = 0` on the ledger
remains reachable at run time.

**Devices.** iPhone: signals 1, 2, 4, 5 only (no second forward; halting on); one retry
max; ask early. 5090: all six, execution in a sandbox, best-of-*n* ranked by the value head
of §2.4.

### 7.2 PyTorch sketch

Consistent with `prophet/modeling/model.py` as of this session: `ProphetModel.forward(ids,
loop_k=..., return_mtp=..., halt_threshold=...)` returns a `ProphetOutput` with `logits`,
`hidden`, `confidence`, `mtp_logits`, `halt_probs`, `hidden_per_step`. `_project` is
private; the sketch calls it and marks the one-line public alias the model should grow.

```python
# prophet/verify/verifier.py  (sketch)
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prophet.modeling.model import ProphetModel, ProphetOutput


class Tier(IntEnum):
    GROUND_TRUTH = 0   # a program said yes                -> main ledger
    CONSENSUS = 1      # >= 3 independent runs agreed       -> quarantine
    LEARNED = 2        # the head said yes                  -> act, never consolidate
    UNVERIFIED = 3     # nothing                            -> refuse


@dataclass
class Verdict:
    p_correct: Tensor          # (batch,) calibrated
    tier: Tier
    depth_disagreement: Tensor # (batch,) fraction of span tokens that changed with depth
    decision: str              # "act" | "retry_depth" | "retry_sample" | "ask"
    consolidate: bool


@dataclass
class VerifierConfig:
    enabled: bool = True
    depth_signal: bool = True
    mtp_signal: bool = True
    shallow_step: int = 2          # iteration whose read-out is the "shallow" opinion
    d_star: float = 0.15           # depth-disagreement trigger for a k_max retry
    max_attempts_free: int = 8
    max_attempts_learned: int = 2
    depth_gain_points: float = 0.0 # filled from Gate 0; retry_depth is dead below 5


def _span_stats(logits: Tensor, span: Tensor) -> Tensor:
    """Mean entropy, mean margin, min max-prob over the answer span. ``span`` is a
    (batch, seq) bool mask. Free: the logits are already computed."""
    p = logits.float().softmax(-1)
    ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
    top2 = p.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    m = span.float()
    n = m.sum(-1).clamp_min(1)
    return torch.stack(
        [(ent * m).sum(-1) / n, (margin * m).sum(-1) / n,
         top2[..., 0].masked_fill(~span, 2.0).amin(-1)], dim=-1)


class Verifier(nn.Module):
    """Reads what the trunk already computed and says how much to trust it.

    Adds ~0.1 M parameters and, in the common case, zero forward passes: with
    ``recurrent.halting == "ponder"`` the coda read-out at every iteration is already in
    ``output.hidden_per_step``, so the shallow/deep comparison is a subtraction.
    """

    N_FEATURES = 8

    def __init__(self, cfg: VerifierConfig, d_hidden: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(self.N_FEATURES, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 1))
        self.register_buffer("log_T", torch.zeros(()))  # fitted post hoc, never by SGD

    # -- signals -------------------------------------------------------------------

    def depth_disagreement(
        self, model: ProphetModel, out: ProphetOutput, ids: Tensor, span: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fraction of span tokens whose argmax differs between a shallow read-out and the
        final one, and the mean KL(final || shallow). The looped core either settled by
        ``shallow_step`` or it did not; this is the model's own convergence test."""
        s = self.cfg.shallow_step - 1
        if out.hidden_per_step is not None and len(out.hidden_per_step) > s:
            # Free path: the ponder probe already ran the coda at every iteration.
            shallow_logits = model._project(model.norm_out(out.hidden_per_step[s]))
        else:
            # Paid path: one extra k=2 forward (1x, or 0.17x if the model exposes a
            # coda-only call on the stored core state).
            shallow_logits = model(ids, loop_k=self.cfg.shallow_step, return_mtp=False).logits
        deep = out.logits.float()
        changed = (deep.argmax(-1) != shallow_logits.argmax(-1)).float()
        m = span.float()
        n = m.sum(-1).clamp_min(1)
        frac = (changed * m).sum(-1) / n
        kl = F.kl_div(shallow_logits.float().log_softmax(-1), deep.log_softmax(-1),
                      log_target=True, reduction="none").sum(-1)
        return frac, (kl * m).sum(-1) / n

    @staticmethod
    def mtp_disagreement(out: ProphetOutput, span: Tensor) -> Tensor:
        """KL between MTP head 1's forecast of token t+2 and the main head's prediction of
        the same token one position later. Free when ``return_mtp=True``; no published
        evidence that it predicts error — A4-1 measures it."""
        if not out.mtp_logits:
            return torch.zeros(out.logits.shape[0], device=out.logits.device)
        fore = out.mtp_logits[0][:, :-1].float().log_softmax(-1)   # predicts t+2 at t
        main = out.logits[:, 1:].float().log_softmax(-1)            # predicts t+2 at t+1
        kl = F.kl_div(fore, main, log_target=True, reduction="none").sum(-1)
        m = span[:, 1:].float()
        return (kl * m).sum(-1) / m.sum(-1).clamp_min(1)

    # -- verdict -----------------------------------------------------------------------

    def forward(
        self,
        model: ProphetModel,
        out: ProphetOutput,
        ids: Tensor,
        span: Tensor,
        *,
        executed: bool | None = None,   # True/False from a program, None if unavailable
        agreements: int = 0,            # independent runs that reached the same answer
        attempts: int = 0,
        t_d: float = 0.5,
    ) -> Verdict:
        b = ids.shape[0]
        feats = [ _span_stats(out.logits, span) ]                               # 3
        conf = out.confidence if out.confidence is not None else out.logits.new_zeros(b, ids.shape[1])
        feats.append(((conf.float() * span).sum(-1) / span.sum(-1).clamp_min(1)).unsqueeze(-1))  # 1
        if self.cfg.depth_signal:
            frac, kl = self.depth_disagreement(model, out, ids, span)
        else:
            frac = kl = out.logits.new_zeros(b)
        feats += [frac.unsqueeze(-1), kl.unsqueeze(-1)]                          # 2
        depth = out.expected_depth() if out.halt_probs is not None else float(out.loop_k)
        feats.append(out.logits.new_full((b, 1), float(depth)))                  # 1
        feats.append((self.mtp_disagreement(out, span) if self.cfg.mtp_signal
                      else out.logits.new_zeros(b)).unsqueeze(-1))               # 1
        x = torch.cat(feats, dim=-1)
        p = torch.sigmoid(self.mlp(x).squeeze(-1) / self.log_T.exp())

        # A program's verdict overrides everything the head thinks.
        if executed is not None:
            p = torch.full_like(p, 0.99 if executed else 0.01)
            tier = Tier.GROUND_TRUTH
        elif agreements >= 3:
            tier = Tier.CONSENSUS
        elif self.cfg.enabled:
            tier = Tier.LEARNED
        else:
            tier = Tier.UNVERIFIED

        decision = self._decide(p, frac, tier, attempts, t_d)
        return Verdict(p_correct=p, tier=tier, depth_disagreement=frac,
                       decision=decision,
                       consolidate=tier <= Tier.CONSENSUS)  # T2 acts, never remembers

    def _decide(self, p: Tensor, frac: Tensor, tier: Tier, attempts: int, t_d: float) -> str:
        c = float(p.mean())
        d = float(frac.mean())
        if tier == Tier.GROUND_TRUTH:
            return "act" if c > 0.5 else ("retry_sample" if attempts < self.cfg.max_attempts_free else "ask")
        if c >= t_d:
            return "act"
        if (self.cfg.depth_signal and d >= self.cfg.d_star and attempts == 0
                and self.cfg.depth_gain_points >= 5.0):
            return "retry_depth"                      # 2.79x on the flagged minority only
        if attempts < self.cfg.max_attempts_learned:
            return "retry_sample"                     # +8..20 points at ~1.7x
        return "ask"
```

Where it plugs in: `consolidate_depth(..., require_verified=True)` becomes
`consolidate_depth(..., min_tier=Tier.CONSENSUS)` with the verdict's tier recorded on every
slot, which is the change W4 §6.4 asked for. The agent loop calls `verifier(...)` after
every answer span and after every tool result (with `executed` set from the structural
check). Three things the model should expose to make the free path free: a public
`project(hidden)` alias for `_project`, `hidden_per_step` populated even when
`halt_threshold` is `None` (it already is when the halting head exists), and a coda-only
call on a stored core state so the paid path costs 0.17× rather than 1×.

---

## 8. Ablation plan

Ordered by cost; the first one is free and decides whether the rest exist.

### A4-0 — Depth disagreement as an error predictor (no training; minutes)

**Question.** When the iteration-2 read-out and the iteration-8 read-out disagree on an
answer token, is the answer more likely to be wrong?

**Protocol.** On the A1 ablation checkpoint at ≥ 350 M (R04: recursion under-performs
below ~360 M), with `halting == "ponder"` so `hidden_per_step` is populated, run **one
k=8 forward** per example on (a) 1 000 GSM8K-style problems and (b) 500 code problems
with tests. Per example compute: depth-disagreement fraction and KL on the answer span
(§7.2), token entropy, confidence-head logit, and the correctness of the k=8 answer and of
the iteration-2 answer.

**Report three AUROCs**, because the signal has three possible uses:

| Target | Decision it would gate | Keep if |
|---|---|---|
| `wrong(k=8)` | act vs ask | AUROC ≥ 0.70 **and** ≥ entropy + 0.05 |
| `wrong(k=2)` | is the shallow pass safe? | AUROC ≥ 0.70 |
| `wrong(k=2) ∧ right(k=8)` | retry with more depth | precision ≥ 0.4 at the recall that flags ≤ 25 % of queries |

**Kill criterion.** If none of the three clears its bar, `depth_signal` defaults to
`False`, the verifier keeps signals 1, 2, 4, 5, and §6.3's `retry_depth` branch is removed.
Cost: 1 500 examples × ~500 tokens × one k=8 forward ≈ 0.75 M token-forwards ≈ **< 5
minutes on the A100** `[C]` (2 978 tok/s single-stream; batched, seconds).

### A4-1 — MTP disagreement (free; same run)

Same protocol, signal 5, same three targets. There is no published evidence either way;
one run settles it. Same kill criterion.

### A4-2 — The combined verifier (≈ 3–8 A100-h)

Train §7's MLP on execution-labelled rollouts (§7.1). Report AUROC and ECE per domain on
the held-out split, the conformal `α` achieved at each `t_d`, and — the number that matters
— **accuracy versus attempts** under §6.3 with `ρ = 0` (head) against adaptive
self-consistency at equal cost. Release gate as in R09: **AUROC ≥ 0.80, ECE ≤ 0.03**.

### A4-3 — The policy end to end (≈ 5 A100-h of evaluation)

On the three task families (code with tests, math with a checker, closed-book QA), plot
accuracy against mean cost in k=2 units for: no verification; blanket k=8; §6.3 with the
head only; §6.3 with execution. The claim to confirm or kill: **execution-gated retry
reaches ≥ 0.9 of the pass@8 coverage at ≤ 2.5× mean cost; head-gated retry buys ≥ 8 points
at ≤ 1.8×; and blanket k=8 is dominated by both.**

### A4-4 — Wrong-write rate by tier, longitudinally (rides on W4's E-W4)

Consolidate 200 episodes per tier into separate ledgers; measure `f_w` directly (fraction
of admitted episodes whose deep answer was wrong) and the held-out class accuracy after
1, 5, 20 rounds, memory on and off (`λ = 0`). Prediction from §6.1: T2 ledgers cross below
the no-memory baseline within 5 rounds; T0 ledgers do not. If T0 also crosses, the
consolidation *operator* is the problem (07_WALLS §D.5), not the verifier, and A4 hands
the finding to W3/W4.

### A4-5 — Value head versus confidence head for ranking (≈ 3–6 A100-h)

Train §2.4's implicit-PRM value head on the same rollouts; compare best-of-8 accuracy
ranked by (i) the value head, (ii) the confidence head, (iii) adaptive majority, at equal
generation cost. Keep the value head only if it beats (iii) by ≥ 3 points on two of three
families.

---

## 9. References

### Self-correction and the generation–verification gap
- Huang, J. et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* arXiv **2310.01798** (ICLR 2024). `[S]`
- Tyen, G. et al. *LLMs cannot find reasoning errors, but can correct them given the error location.* arXiv **2311.08516** (Findings of ACL 2024). `[S]`
- Jiang, D. et al. *SELF-[IN]CORRECT: LLMs Struggle with Discriminating Self-Generated Responses.* arXiv **2404.04298**. `[S]`
- Song, Y. et al. *Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models.* arXiv **2412.02674** (ICLR 2025). `[S]`
- Huang, A. et al. *Self-Improvement in Language Models: The Sharpening Mechanism.* arXiv **2412.01951**. `[S]`
- Tsai, Y.-C. et al. *Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot in LLMs.* arXiv **2507.02778**. `[S]`
- *Variation in Verification: Understanding Verification Dynamics in Large Language Models.* arXiv **2509.17995** (ICLR 2026). `[S]`
- *The Future of Facts: Tracing the Factual Generation-Verification Gap.* arXiv **2605.27564**. `[?]`
- Zhao, E. et al. *Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification.* arXiv **2502.01839** (ICML 2025). `[S]`
- Saad-Falcon, J. et al. *Shrinking the Generation-Verification Gap with Weak Verifiers (Weaver).* arXiv **2506.18203**. `[S]`
- Kadavath, S. et al. *Language Models (Mostly) Know What They Know.* arXiv **2207.05221**. `[S]/[P]`
- Wei, J. *Asymmetry of verification and verifier's law.* Blog, 15 July 2025. `[S]`
- Setlur, A. et al. *Scaling Test-Time Compute Without Verification or RL is Suboptimal.* arXiv **2502.12118**. (via W4)

### Process reward models
- Lightman, H. et al. *Let's Verify Step by Step.* arXiv **2305.20050**. `[S]`
- Wang, P. et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* arXiv **2312.08935** (ACL 2024). `[S]`
- Luo, L. et al. *Improve Mathematical Reasoning in Language Models by Automated Process Supervision (OmegaPRM).* arXiv **2406.06592**. `[S]`
- Yuan, L. et al. *Free Process Rewards without Process Labels.* arXiv **2412.01981**. `[S]`
- Zheng, C. et al. *ProcessBench: Identifying Process Errors in Mathematical Reasoning.* arXiv **2412.06559**. `[S]`
- Zhang, Z. et al. *The Lessons of Developing Process Reward Models in Mathematical Reasoning.* arXiv **2501.07301** (Findings of ACL 2025). `[S]`
- Khalifa, M. et al. *Process Reward Models That Think (ThinkPRM).* arXiv **2504.16828**. `[S]`
- Snell, C. et al. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv **2408.03314**. `[S]`
- Liu, R. et al. *Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling.* arXiv **2502.06703**. `[S]`
- *Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math.* arXiv **2510.13744** (ACL 2026). `[S]`
- Brown, B. et al. *Large Language Monkeys.* arXiv **2407.21787**. (via R04)

### Cheap verifiers and consistency
- Aggarwal, P. et al. *Let's Sample Step by Step: Adaptive-Consistency.* arXiv **2305.11860**. `[S]`
- Li, Y. et al. *Escape Sky-high Cost: Early-stopping Self-Consistency.* arXiv **2401.10480** (ICLR 2024). `[S]`
- Fu, Y. et al. *Deep Think with Confidence (DeepConf).* arXiv **2508.15260**. `[S]`
- Gou, Z. et al. *CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing.* arXiv **2305.11738**. `[P]`
- Madaan, A. et al. *Self-Refine.* arXiv **2303.17651**. `[S]`
- *Verifiable Process Rewards for Agentic Reasoning.* arXiv **2605.10325**. `[?]`
- *OpenReward: Learning to Reward Long-form Agentic Tasks via RL.* arXiv **2510.24636**. `[S]`

### Uncertainty signals
- Farquhar, S. et al. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 625–630 (2024). `[P]`
- Kossen, J. et al. *Semantic Entropy Probes.* arXiv **2406.15927**. `[S]`
- Zhang, Z. et al. *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection.* arXiv **2507.16488** (ACL 2025). `[S]`
- Orgad, H. et al. *LLMs Know More Than They Show.* arXiv **2410.02707** (ICLR 2025). `[S]`
- Chen, C. et al. *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection.* arXiv **2402.03744**. `[S]`
- Manakul, P. et al. *SelfCheckGPT.* arXiv **2303.08896**. (via R09)
- Schuster, T. et al. *Confident Adaptive Language Modeling.* arXiv **2207.07061** (NeurIPS 2022). `[S]`
- Geiping, J. et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* arXiv **2502.05171**. `[S]`
- *Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers.* arXiv **2607.14427**. `[?]`
- Gloeckle, F. et al. *Better & Faster LLMs via Multi-token Prediction.* arXiv **2404.19737**. `[P]`
- *Streaming Hallucination Detection in Long Chain-of-Thought Reasoning.* arXiv **2601.02170** (Findings of ACL 2026). `[?]`
- Damani, M. et al. *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR).* arXiv **2507.16806**. (via R09)

### Agent loops
- Shinn, N. et al. *Reflexion.* arXiv **2303.11366** (NeurIPS 2023). `[S]`
- Xia, C. S. et al. *Agentless: Demystifying LLM-based Software Engineering Agents.* arXiv **2407.01489**. `[S]`
- Ehrlich, R. et al. *CodeMonkeys: Scaling Test-Time Compute for Software Engineering.* arXiv **2501.14723**. `[S]`
- Jain, N. et al. *R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents.* arXiv **2504.07164** (COLM 2025). `[S]`
- *From SWE-ZERO to SWE-HERO: Execution-free to Execution-based Fine-tuning.* arXiv **2604.01496**. `[?]`
- *Live-SWE-agent.* arXiv **2511.13646**. `[S]`
- Shen, J. et al. *Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction.* arXiv **2506.07976**. `[S]`
- Xu, F. et al. *TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks.* arXiv **2412.14161**. `[S]`
- Yang, J. et al. *SWE-agent.* arXiv **2405.15793**; Wang, X. et al. *OpenHands.* arXiv **2407.16741**. (no verification ablation published)
- *From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents.* arXiv **2606.04990**. `[?]`

### Internal
- `docs/07_WALLS.md` §D (the 93 % and the 54 %), `docs/research/W4_compute_does_not_compound.md` §5–6, §8; `docs/research/R09_hallucination_calibration.md` §2.1, §4; `docs/research/R04_reasoning_test_time_compute.md` §2.3, §4.6; `docs/research/R10_post_training.md` §2.4; `prophet/memory/consolidate.py`; `prophet/modeling/model.py`.

### Reproducing the `[C]` numbers

```bash
python - <<'PY'
from prophet.config import ProphetConfig
from prophet.budget import inference_profile
cfg = ProphetConfig.from_json("configs/prophet_500m_probe.json")
for k in (1, 2, 4, 8, 16):
    p = inference_profile(cfg, device="a100_80gb", loop_k=k, context_len=4096)
    print(k, cfg.effective_depth(loop_k=k), round(p.flops_per_token/1e9, 3), round(p.decode_tok_s))
PY
```

---

## 10. Recommendation

1. **Run A4-0 before anything else.** It is one k=8 forward per example, no training, and
   it is the only experiment in this document that can make Prophet's verifier better than
   the generic probe every other project has: the looped core's own convergence, read for
   free from `hidden_per_step`.
2. **Do not train a standalone PRM.** At our label quality it lands at ~31 F1, which is the
   "everything is correct" verifier. Train an outcome-labelled value head on the coda for
   ranking only (< 1 A100-h), and let programs be the PRM where programs exist.
3. **Adopt the two-line policy of §6.3 and make the tier a hard type.** The confidence
   head gates *acting*; only execution or three independent agreements gate *remembering*;
   `require_verified: bool` becomes `min_tier: Tier` and the tier goes on every slot.
4. **Budget verification as latency, not FLOPs.** For a 250M model the test run is longer
   than the generation. The agent loop's timeouts, not its token counts, are where the 93 %
   will show up on device.
