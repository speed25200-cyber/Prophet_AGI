# R03 — Memory, persistence, and continual learning: escaping the frozen brain

**Track:** R03 · **Status:** decision-ready · **Date:** 2026-09-03
**Scope:** what memory mechanism Prophet should build so that the deployed model
accumulates knowledge after training, on a single A100 budget, with inference on
5090 / Mac Studio / iPhone 17 Pro.

**Verification note.** All arXiv IDs below were checked against live search this
session unless marked `†`, which means "cited from prior knowledge, ID not
re-verified in this session — verify before quoting externally". Numbers in
quotes are taken from the papers/abstracts as retrieved; numbers I derived
myself (FLOPs, bytes, hours) are labelled **[derived]**.

---

## 1. Problem statement

### 1.1 The concrete failure

A deployed LLM has exactly three places to put information:

| Store | Capacity | Persistence | Write cost | Read cost |
|---|---|---|---|---|
| Weights (implicit memory) | ~2 bits/param at best | permanent | full backprop, catastrophic forgetting | free (it's the forward pass) |
| KV cache (working memory) | O(context) | dies at end of request | free (prefill) | O(L·d) per token, grows without bound |
| External text/DB (RAG, agent memory) | unbounded | permanent | cheap | 100s–1000s of context tokens per query, plus retrieval latency |

There is no store that is simultaneously (a) persistent, (b) cheap to write,
(c) free to read. That gap *is* the frozen-brain problem. Everything in this
report is an attempt to build the missing fourth row.

### 1.2 Why this matters more for Prophet than for a frontier lab

We are targeting ~1.3B **active** parameters against Qwen3-4B, Gemma-3-4B,
Phi-4-mini. On static benchmarks we are fighting uphill on raw capacity. On
*personalized and long-horizon* tasks, a model that genuinely accumulates
knowledge on-device has an axis of advantage that a frozen 4B model cannot
match at any size — the frozen model must re-read the user's history from
context on every single query, paying 100–100k tokens each time, while ours
reads it for free out of weights and state.

Concretely, the numbers that define the opportunity:

- **LongMemEval** (arXiv 2410.10813, ICLR'25): 500 human-written questions over
  long multi-session chat histories; commercial assistants and long-context LLMs
  show a **~30% absolute accuracy drop** as history grows. This is the headroom.
- **Sleep-time compute** (arXiv 2504.13171, Letta/Berkeley): pre-computing over
  context offline cuts the test-time compute needed for equal accuracy by
  **~5×**, and scaling offline compute adds **up to +13% accuracy**. Offline
  work on a user's own history is under-exploited.
- **Catastrophic forgetting is measurable and brutal**: training a memory-layer
  LLM on TriviaQA facts drops NaturalQuestions accuracy by **89% with full
  finetuning, 71% with LoRA, and only 11% with sparse memory finetuning** for
  the same amount of new knowledge learned (arXiv 2510.15103). Sparsity of the
  *update*, not the size of the update, is what buys retention.

### 1.3 Requirements we will hold every candidate to

1. **R1 — Trainable on one A100 80GB.** Pretraining a mechanism must not add
   more than ~20% wall-clock to the base run, and its optimizer state must fit
   in 80GB alongside the model.
2. **R2 — Runs on iPhone.** Extra weights ≤ ~150MB at 4-bit; extra per-token
   FLOPs ≤ 1% of the dense forward; no op that Metal/MLX cannot do fast.
3. **R3 — Updatable on-device without backprop through the whole model.** The
   write path must be either (a) a pure forward-pass state update, or (b) a
   *local* objective whose gradient touches only the memory parameters.
4. **R4 — Demonstrably improves a benchmark** under a protocol that controls
   for context leakage, parameter count, and compute — otherwise it is theatre.

---

## 2. State of the art

### 2.1 Master table

`Δ` = reported quality delta. `PK` = product key. `bp` = backprop.

| # | Mechanism | Where memory lives | Update rule (essence) | Write cost | Read cost | Scale validated | Papers |
|---|---|---|---|---|---|---|---|
| 1 | **Linear attention / Hebbian fast weights** | matrix state `S ∈ R^{d_v×d_k}` per head | `S_t = S_{t-1} + v_t k_tᵀ` | forward only, O(d_k·d_v) | O(d_k·d_v) matvec | 100B+ tokens, many labs | Schmidhuber '92; 2102.11174† |
| 2 | **DeltaNet (delta rule)** | same | `S_t = S_{t-1}(I − β_t k_tk_tᵀ) + β_t v_tk_tᵀ` | forward only; chunk-parallel via WY/Householder | same | 1.3B / 100B tok; beats Mamba, GLA on recall | 2406.06484 |
| 3 | **Gated DeltaNet** | same | `S_t = α_t S_{t-1}(I − β_tk_tk_tᵀ) + β_t v_tk_tᵀ`, `α_t∈(0,1)` data-dependent | forward only | same | **production**: Qwen3-Next, Kimi Linear 48B-A3B, 3:1 hybrid, −75% KV cache, up to **6× decode @1M ctx** | 2412.06464; 2510.26692 |
| 4 | **TTT layers** | hidden state *is* a model (linear or 2-layer MLP) | SGD step on self-supervised recon loss per token/mini-batch | forward + inner bp through the small memory net | forward of memory net | 125M–1.3B | 2407.04620; 2505.23884 (LaCT); 2506.05233 (MesaNet) |
| 5 | **Titans / neural long-term memory** | deep memory `M` (1–4 layer MLP) | `M_t=(1−α_t)M_{t−1}+S_t`, `S_t=η_tS_{t−1}−θ_t∇ℓ(M_{t−1};x_t)`, `ℓ=‖M(k_t)−v_t‖²` — surprise + momentum + adaptive forgetting | inner bp through memory MLP; chunked mini-batch GD | memory MLP forward | 340M/400M/760M; BABILong claims to beat GPT-4 at long ctx | 2501.00663 |
| 6 | **ATLAS / Omega rule** | same, higher capacity | optimizes over a **sliding window of last c tokens** (not one token), polynomial/exponential feature maps, **Muon** optimizer for the memory | heavier than Titans; still matmul-parallel | same | ≤1.3B; **>80% acc at 10M ctx BABILong** where Titans collapses | 2505.23735 |
| 7 | **HOPE / Nested Learning** | continuum of memories at different update frequencies; self-modifying | model contains an inner learner that learns the update rule; "Continuum Memory System" blocks update at multiple frequencies | highest of this family | — | ≤1.3B, one lab, NeurIPS'25 | 2512.24695 |
| 8 | **Sleep / consolidation (architectural)** | fast modules → slow modules | "Sleep" stage distils fast unstable memory into slow weights with replay; "Knowledge Seeding" distils a smaller self into a larger net | offline pass | free | research-only (2026) | 2606.03979; 2605.26099 |
| 9 | **Product-key memory layer** | huge sparse table `V ∈ R^{N×d}`, keys factorized as `C1×C2`, each sub-key set of size `√N` | trained by SGD like any weight; **top-k gather at read** | n/a (pretrain) | `2√N·(d_k/2) + k·d_v` MACs — negligible | 12-layer + memory beats 24-layer, 2× faster inference | 1907.05242 |
| 10 | **Memory Layers at Scale** | one shared PK memory replacing 1–3 FFNs | SGD; parallelizable sharded implementation | n/a | as above | **128B memory params, 1T tokens, base ≤8B**; 1.3B+64M keys ≈ Llama2-7B trained on 2× tokens with 10× FLOPs; open code | 2412.09764 |
| 11 | **PEER (Mixture of a Million Experts)** | >1M rank-1 experts, PK-routed | SGD | PK retrieval | k tiny experts | LM-scale ablations, DeepMind | 2407.04153 |
| 12 | **Sparse memory finetuning** | PK memory slots | finetune **only top-T slots** ranked by TF-IDF of session-vs-background usage | bp to that layer only; sparse Adam | free | 1.3B-class; **forgetting 11% vs 89% full-FT / 71% LoRA** | 2510.15103; KL-based slot scoring + retrofit of Qwen2.5-0.5B in 2604.05248; 2605.03229 |
| 13 | **Larimar** | external episodic memory matrix, Kanerva-style | one-shot write/erase, closed-form | no gradient | attention over memory | GPT-J/GPT-2 editing; **8–10× faster** than editing baselines; supports selective forgetting | 2403.11901 |
| 14 | **MemoryLLM / M+** | 1B-param hidden-state memory pool across layers + co-trained retriever | learned compress-and-drop of hidden states | forward | retrieve once per layer | retention **<20k → >160k tokens** at similar GPU memory | 2402.04624†; 2502.00592 |
| 15 | **Memory³** | offline KV "explicit memory" bank, sparsified | write once at ingestion (no gradients) | one sparse-KV write per doc | sparse KV attention | 2.4B trained from scratch; beats larger LLMs **and** RAG, with **higher decode speed than RAG** | 2407.01178 |
| 16 | **kNN-LM** | datastore of (context-vector → next-token) | none (non-parametric) | index build | ANN search per token | WikiText-103 ppl **18.65 → 16.12** (ext. store), **15.79** with train store, **no training** | 1911.00172 |
| 17 | **RETRO** | 2T-token chunked retrieval DB | none | index build | chunked cross-attention | 7.5B ≈ GPT-3 175B on Pile | 2112.04426† |
| 18 | **SEAL (self-adapting LM)** | model weights via self-generated finetuning data | model emits "self-edits" (synthetic implications + hyperparams); RL (ReST-EM) on downstream reward; SFT applies them | **full finetune per edit** | free | Qwen2.5-7B; SQuAD no-context: base **32.7** → FT-on-passage **33.5** → self-synth **39.7** → GPT-4.1-synth **46.3** → **SEAL 47.0** | 2506.10943 |
| 19 | **Self-Consolidating LMs** | LoRA adapters on self-selected layers | model emits textual instructions naming which of its own modules to adapt; LoRA update | LoRA bp | free | research (2026) | 2605.07076 |
| 20 | **LoRA continual adapters** | low-rank deltas | standard SGD on A,B | bp through model | free (merged) | ubiquitous; **learns less, forgets less**; full-FT perturbation rank 10–100× LoRA's | 2405.09673 |
| 21 | **EWC + successors** | dense weights + Fisher penalty | `L + λΣF_i(θ_i−θ*_i)²` | needs Fisher over all params | free | small nets; ~unused at LLM scale | 1612.00796† |
| 22 | **Model merging (TIES / DARE / task arithmetic)** | dense weights | sign-elect + trim + average; DARE randomly drops **90% (up to 99%)** of delta params and rescales by `1/(1−p)` with ~no loss | offline, no gradients | free | 7B–70B merges | 2306.01708†; 2311.03099; 2212.04089† |
| 23 | **Agentic memory (MemGPT/Letta, Mem0, Zep, A-MEM, Cognee)** | external DB / knowledge graph, injected as **context tokens** | LLM-written extraction + consolidation | LLM calls | 100s–1000s of context tokens/query | Zep DMR **94.8% vs MemGPT 93.4%**, LongMemEval **+18.5%** acc, **−90% latency**; Mem0 **+26%** rel. over OpenAI Memory on LoCoMo, **p95 latency −91%**, **>90% token savings** | 2310.08560†; 2504.19413; 2501.13956; 2502.12110† |

### 2.2 The three update rules that actually matter (math)

Everything in rows 1–8 is one family. Write the memory read as `o_t = M(q_t)`
and define an associative loss `ℓ_t(M) = ½‖M(k_t) − v_t‖²`.

**(a) Hebbian / linear attention** — gradient descent with lr 1 on a *first-order*
approximation, no error term:
```
S_t = S_{t-1} + v_t k_tᵀ
```
Capacity ≈ d_k orthogonal pairs; interference grows linearly; cannot overwrite.

**(b) Delta rule** — the true gradient step, `∇_S ℓ = (S k_t − v_t) k_tᵀ`:
```
S_t = S_{t-1} − β_t (S_{t-1}k_t − v_t) k_tᵀ
    = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ         (‖k_t‖=1)
```
The update magnitude is proportional to the **prediction error** `‖v_t − S_{t-1}k_t‖`.
That is *exactly* Titans' "surprise" signal, obtained for free, without a
separate momentum machine. `I − β k kᵀ` is a (generalised) Householder matrix,
which is why DeltaNet chunk-parallelises via the WY representation (2406.06484)
and reaches near-FlashAttention throughput.

**(c) Gated delta rule** (Gated DeltaNet, 2412.06464 — the version shipped in
Qwen3-Next and, with finer-grained per-channel gating, as KDA in Kimi Linear):
```
S_t = α_t · S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,   α_t = exp(−softplus(w_αᵀx_t))
```
`α_t` is the *erase* channel: Mamba2 has decay but no targeted overwrite;
DeltaNet has targeted overwrite but no decay; you need both.

**(d) Titans** adds a momentum buffer and a deep (MLP) memory:
```
M_t = (1 − α_t) M_{t-1} + S_t
S_t = η_t S_{t-1} − θ_t ∇ℓ(M_{t-1}; x_t)
```
With `M` linear and `η=0`, this collapses exactly to the gated delta rule.
The claimed extra value is (i) momentum `η_t` (surprise persists across tokens)
and (ii) nonlinear memory (`L_M`=2–4 layer MLP). Cost: the write needs an inner
backprop through the memory MLP, and the state is now `L_M` weight matrices.

**(e) ATLAS/Omega** replaces the per-token loss with a windowed one,
`ℓ_t = Σ_{i=t−c+1}^{t} γ_i ‖M(k_i) − v_i‖²`, solved with Muon (approximate
second-order via Newton–Schulz orthogonalisation of the update). This is the
step that fixes 10M-token BABILong.

### 2.3 Product-key memory, precisely

Given `N` slots, choose `n = √N` sub-keys per half. Query `q(x) = W_q x ∈ R^{d_k}`
split as `[q¹;q²]`, each in `R^{d_k/2}`.
```
I¹ = top-k(q¹ · C¹ᵀ),  I² = top-k(q² · C²ᵀ)          # C¹,C² ∈ R^{n×(d_k/2)}
S  = { (i,j) ∈ I¹×I² : s_ij = q¹·c¹_i + q²·c²_j }     # k² candidates
idx, w = top-k over S, w = softmax(s)
y = Σ_r w_r · V[idx_r]                                # V ∈ R^{N×d_v}
```
Exact top-k over all `N` keys, at cost `O(√N·d_k + k²)` instead of `O(N·d_k)`.
Params: `N·d_v + 2n·(d_k/2)`. FLOPs per token are *independent of N* — this is
the only mechanism in the entire table that adds capacity at literally zero
marginal compute.

### 2.4 Agentic memory as the honest baseline

Mem0 / Zep / MemGPT / A-MEM do not change the model. They change the context.
The best public numbers: Mem0 reports LoCoMo ≈ 92.5 and LongMemEval ≈ 94.4 at
3–4× lower token cost than full-context; Zep reports DMR 94.8% vs MemGPT 93.4%
and LongMemEval +18.5% with −90% latency. Two caveats that matter for us:

1. These systems presuppose a **strong instruction-following model** to do the
   extraction/consolidation. At 400M–1.3B active, the extractor is the weak link.
2. **LoCoMo is a weak benchmark** — it is widely criticised for a rubric that
   effectively forbids "I don't know", inflating scores. Do not target it.
   Use LongMemEval (and LongMemEval-V2, 2605.12493) instead.

Conclusion: external memory is a *product feature we should ship anyway* and the
baseline our architecture must beat **at equal context budget**. It is not a
research contribution and it does not solve the frozen-brain problem — the
weights still never change.

---

## 3. What actually transfers to our scale

Brutal assessment. Sorted by evidence quality.

### 3.1 Solid, take it

- **Gated delta-rule linear attention in a 3:1 hybrid.** This is the only
  test-time-learning mechanism in the whole table that has been validated at
  production scale by two independent labs: Qwen3-Next and Kimi Linear
  (48B total / 3B active, KDA:MLA = 3:1, −75% KV cache, up to 6× decode
  throughput at 1M context; 2510.26692). Kernels exist (`flash-linear-attention`).
  Risk: near zero. **Take it.**
- **Product-key memory layers.** Validated to 128B memory params / 1T tokens /
  8B base with open code from Meta (2412.09764), and the original result
  (1907.05242) is seven years old and has never been contradicted. Zero marginal
  FLOPs, gains concentrated exactly where a small model is weakest (facts).
  **Take it.**
- **Sparse memory finetuning.** The single most important number in this report:
  **11% vs 89% vs 71% forgetting** (2510.15103). The mechanism is trivial —
  rank slots by session-vs-background usage, update the top-T, freeze the rest.
  Follow-ups replace TF-IDF with a KL-divergence score and show retrofitting a
  Qwen2.5-0.5B works on consumer hardware (2604.05248, 2605.03229). **Take it.**

### 3.2 Take the idea, not the implementation

- **Titans / ATLAS / HOPE.** No official code. Results from a single group.
  Largest reported model 1.3B. And the one serious independent reimplementation
  (**Titans Revisited**, arXiv 2510.09551) reports that Titans "does not always
  outperform established baselines due to chunking", that the neural memory does
  consistently help *versus attention-only*, and — critically — that "memory
  updates alone are insufficient for meaningful test-time learning", attributing
  it to "a mismatch between the frozen backbone input projections into key-value
  space and how the memory evolves". Translation for us: **the memory must be
  co-trained with the backbone from scratch, and the model must be trained on
  warm (already-written) states**, or the mechanism is decorative.
  Deep (MLP) memory and momentum are worth exactly one ablation each, not a
  design commitment.
- **SEAL.** The idea — *the model writes its own consolidation data as synthetic
  implications* — is validated and valuable: 32.7 → 47.0 on no-context SQuAD,
  beating GPT-4.1-generated synthetic data. The mechanism — an RL outer loop
  (ReST-EM) around a full LoRA finetune per edit — is not on-device and not
  cheap. **Take the data-generation idea; reject the update mechanism.**
- **Sleep/consolidation.** Both the architectural version (2606.03979, 2605.26099)
  and the compute version (2504.13171, ~5× cheaper test time, +13% accuracy)
  agree that offline processing of accumulated context is a large free win.
  These are 2026 papers with no reproductions. **Take the schedule (wake/sleep),
  build the simplest possible consolidation operator.**
- **DARE.** 90–99% of a finetuning delta can be dropped with rescaling and no
  quality loss (2311.03099). This is not a merging trick for us; it is *proof
  that per-user weight deltas can be stored sparsely*, which is what makes
  on-device personalisation fit in megabytes.

### 3.3 Do not build

- **TTT-MLP / deep neural memory as the primary sequence mixer.** Requires an
  inner backprop per chunk. At our budget the wall-clock hit is not repayable,
  and 2510.09551 suggests the marginal gain over a delta rule is small and
  chunking-sensitive.
- **kNN-LM / RETRO on device.** kNN-LM's WikiText-103 result needs a **103M-entry**
  datastore for a 100M-token corpus and adds an ANN search per generated token.
  On iPhone that is a non-starter (memory + latency). Possible 5090/Mac-only
  feature later; not core.
- **EWC.** Fisher over 400M params = 1.6GB fp32 and a penalty too coarse to
  separate "user fact" from "general capability". Our per-slot usage counter
  (65k floats = 260KB) is the tractable analogue and is what 2510.15103 is
  implicitly doing.
- **MemoryLLM/M+ style hidden-state memory pools.** 1B params of memory pool for
  a 7B model, and retention is still measured in 10^5 tokens. Worse
  capacity-per-byte than a PK table, and no on-device write story.
- **Larimar's episodic matrix as the main store.** Genuinely elegant (one-shot
  write/erase, 8–10× faster editing, selective forgetting) but it is an *editing*
  method bolted onto a frozen LLM; capacity is a few hundred to a few thousand
  facts. Steal its **erase** operator for our "forget me" feature.

### 3.4 The uncomfortable truth about evaluation

Almost every "our model has memory" paper is evaluated on a task where the
information is *still in the context window* (needle-in-a-haystack, BABILong,
RULER). Those measure long-context handling, not persistence. **The only
protocol that proves persistence is: write, then clear the context, then read.**
Section 7 is built around that.

---

## 4. Recommendation for Prophet

### 4.1 The decision

Build a **two-tier memory** and nothing else.

```
                         ┌──────────────────────────────────────────┐
   tokens ──────────────►│  Tier 0: sliding-window attention (1/4)  │  ephemeral
                         ├──────────────────────────────────────────┤
                         │  Tier 1: FWM — gated delta-rule fast     │  session /
                         │  weights, state S persisted to disk      │  cross-session
                         ├──────────────────────────────────────────┤
                         │  Tier 2: LEDGER — product-key memory     │  permanent
                         │  layer, sparsely rewritten during SLEEP  │
                         └──────────────────────────────────────────┘
                                        ▲
                        SLEEP (offline, backprop-free local write)
```

- **Tier 1 — FWM (Fast-Weight Memory).** Gated delta-rule linear-attention
  layers, 3:1 with sliding-window/global attention. The state matrices `S` are
  **serialised at end of session and reloaded at start of the next one**. This
  is the "within-session and across-session persistence" tier. Write cost: a
  forward-pass outer product. Zero backprop, ever.
- **Tier 2 — LEDGER.** One shared product-key memory layer replacing the FFN at
  two depths. Frozen during normal use; rewritten during a periodic **SLEEP**
  pass by a *local* delta-rule write into ≤T top-scoring slots, with keys frozen.
  Zero backprop through the backbone.
- **SLEEP** is the consolidation operator that moves information from Tier 1
  (fragile, bounded, decaying) into Tier 2 (permanent, sparse, auditable).

Why this and not the alternatives: it is the only combination where every
component is independently validated (§3.1), the total added inference FLOPs are
<1%, the added iPhone weight is ~35MB, and **no on-device operation requires a
gradient through the backbone**.

### 4.2 Tier 1: FWM module design

Per FWM layer, per memory head `h` (`H_m` heads):

```
state          S ∈ R^{d_v × d_k}                       (persisted)
q_t,k_t ∈ R^{d_k},  v_t ∈ R^{d_v}                      (from x_t, standard projections)
k_t ← k_t / ‖k_t‖₂                                     (L2 norm — required for stability)
β_t = σ(w_βᵀ x_t) ∈ (0,1)                              write strength
a_t = exp(−softplus(w_aᵀ x_t))  ∈ (0,1)                decay / erase gate
S_t = a_t · S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
o_t = S_t q_t ;  y_t = W_o · (RMSNorm(o_t) ⊙ σ(g_t))   output gate as in GDN
```

Dimensions:

| | Prophet-main (≈10B total / 1.3B active) | Prophet-mini (≈400M dense) |
|---|---|---|
| d_model | 2048 | 1024 |
| layers | 32 (24 FWM : 8 attention, pattern 3:1) | 24 (18 FWM : 6 attention) |
| memory heads `H_m` | 16 | 8 |
| d_k = d_v | 128 | 64 |
| state / layer | 16·128·128 = 262,144 | 8·64·64 = 32,768 |
| **total state** | 24 × 262,144 = **6.29M values** | 18 × 32,768 = **590K values** |
| **state bytes (fp16 / fp8)** | **12.6 MB / 6.3 MB** | **1.18 MB / 0.59 MB** | **[derived]**

That is the whole point: **a user's entire cross-session working memory for the
mini model is ~0.6 MB, is read for free by the forward pass, and costs zero
context tokens.** Compare with an agentic memory system injecting 500–2000
tokens per query forever.

**Two non-obvious training requirements** (this is where Titans Revisited says
people fail):

1. **Warm-state training (TBPTT with carry).** Chop every training document into
   `M` segments of `L_seg = 2048`. Carry `S` across segments with a stop-gradient
   at each boundary; backprop only within a segment. Sample the number of
   *no-grad warm-up* segments uniformly from `{0,1,2,3}` so the model sees
   states of every "age". Without this the deployed model is fed an
   out-of-distribution warm state and the memory is decorative.
2. **Co-train the KV projections with the memory.** No frozen backbone, no
   grafting. We are pretraining from scratch, so this is free — and it is
   precisely the failure mode 2510.09551 identifies.

Optional ablation (not default): a momentum buffer `M_t = η_t M_{t-1} + ΔS_t`
(Titans-style). Doubles state bytes. Ablate in E6, ship only if ≥1.5% MQAR gain.

### 4.3 Tier 2: LEDGER module design

One **shared** product-key memory (shared across its 2 insertion points, as in
2412.09764), replacing the FFN at layers `⌊0.4L⌋` and `⌊0.75L⌋`.

| | Prophet-main | Prophet-mini |
|---|---|---|
| slots `N` | 2²⁰ = 1,048,576 | 2¹⁶ = 65,536 |
| sub-keys `n = √N` | 1024 | 256 |
| d_k (query) | 256 | 128 |
| d_v | 2048 (= d_model) | 1024 (= d_model) |
| query heads `H_pk` | 4 | 2 |
| top-k | 32 | 16 |
| **value params** | 2.15B | 67.1M |
| **key params** | 2·1024·128 = 262K | 2·256·64 = 33K |
| **bytes @ int8 / int4** | 2.15 GB / 1.07 GB | 67 MB / **33.5 MB** | **[derived]**
| **read FLOPs/token** (all `H_pk` heads) | 4·2·(2·1024·128 + 32·2048) ≈ **2.6 MFLOP** | 2·2·(2·256·64 + 16·1024) ≈ **0.20 MFLOP** | **[derived]**
| as % of dense forward | 2.6M / (2·1.3e9) = **0.10%** | 0.20M / (2·4e8) = **0.025%** | **[derived]**

Read is a gather + weighted sum: perfectly expressible in MLX/Metal, and
memory-bandwidth-bound rather than compute-bound (16 rows × 1024 int4 = 8KB per
token per head on mini — nothing).

### 4.4 SLEEP: the consolidation operator (the actual contribution)

Goal: move what Tier 1 learned in a session into Tier 2 permanently, **without a
gradient through the backbone**, and without forgetting.

**Step 1 — Evidence set.** At sleep time we have the session transcript `C`
(and, optionally, SEAL-style self-generated implications: prompt the model to
write N restatements/consequences of `C`; 2506.10943 shows self-generated
implications beat raw passages 39.7 vs 33.5, and post-RL beat GPT-4.1's 47.0 vs
46.3). Build probe set `P = {x_1..x_P}`: chunks of `C` plus the self-written
implications.

**Step 2 — Teacher/student forward passes (context distillation).** For each
probe `x`:
- **Teacher**: run the model with the relevant evidence prepended and the FWM
  state warm. Record the residual stream *after* the ledger layer: `h⁺_ℓ₊₁(x)`.
- **Student**: run with no evidence and a cold FWM state. Record `h⁻_ℓ₊₁(x)`,
  the ledger read `m(x) = Σ_r w_r V[idx_r]`, and the selected `(idx, w)`.

**Step 3 — Local target.** The ledger's job is to supply exactly the missing
residual:
```
t(x) = m(x) + λ · ( h⁺_{ℓ+1}(x) − h⁻_{ℓ+1}(x) ),      λ ∈ (0,1], default 1.0
```

**Step 4 — Local delta-rule write (this is the whole trick).** Since
`m(x) = Σ_r w_r V[idx_r]`, we have `∂m/∂V[idx_r] = w_r · I` exactly. One
gradient step on the *local* loss `½‖m(x) − t(x)‖²` is therefore, in closed form:
```
e(x) = t(x) − m(x) = λ (h⁺ − h⁻)                  # the residual gap
for r in top-k(x):
    V[idx_r] ← V[idx_r] + η_{idx_r} · ( w_r / Σ_j w_j² ) · e(x)
```
No backward pass anywhere. Two forward passes and a scatter-add. This is the
same delta rule as Tier 1, applied to a sparse table instead of a dense matrix —
which is why the whole design has exactly **one** update primitive.

**Step 5 — Interference control** (the part that makes it not forget):

1. **Freeze the keys, always.** Updating keys reorganises the address space and
   is the dominant source of catastrophic interference. Only `V` is writable.
2. **Slot gating.** Maintain a background usage histogram `cnt_bg[i]` (frozen,
   computed once over a 1B-token pretraining sample; 65k int32 = 262KB) and a
   session histogram `cnt_s[i]`. Score, per 2510.15103 / 2604.05248:
   ```
   tfidf(i) = cnt_s[i] · log( (1 + Σ_j cnt_bg[j]/N) / (1 + cnt_bg[i]) )
   ```
   Write only to the top `T` slots (mini: `T = 2048` of 65,536 = **3.1%**;
   main: `T = 16384` of 1,048,576 = **1.6%**). Everything else is frozen.
3. **EWC-lite per-slot learning rate.** `η_i = η₀ / (1 + γ·cnt_bg[i]/mean(cnt_bg))`.
   One scalar per slot. Slots that carry pretraining knowledge become nearly
   immovable; unused slots are free real estate.
4. **Replay.** Ship a 4MB on-device buffer of ~1500 pretraining-distribution
   sequences. After each write, run the same local rule on a replay minibatch
   with target `t = m_before(x)` (pull the read back toward its pre-write value).
   Local, no backprop, ~2% of sleep cost.
5. **Trust-region clamp.** Reject any write with `‖ΔV[i]‖ / ‖V[i]‖ > τ` (τ=0.1);
   this bounds a single session's influence.

**Step 6 — Storage as a sparse delta.** Persist `Δ = V_user − V_base` as
`(index:int32, vector:int8, scale:fp16)` triples. Mini: 5000 slots × (4 + 1024
+ 2) bytes = **5.15 MB per user** for the entire accumulated life of the model
— justified by DARE's finding that 90–99% of a finetuning delta is droppable
(2311.03099). LRU-evict by last-touch when the cap is hit.

### 4.5 What runs where

| Operation | iPhone 17 Pro | Mac Studio | 5090 | A100 (train) |
|---|---|---|---|---|
| FWM read+write | ✅ forward only, Metal/MLX | ✅ | ✅ | ✅ |
| FWM state persist (0.6MB) | ✅ | ✅ | ✅ | — |
| LEDGER read (gather) | ✅ Metal/MLX (not ANE — gathers are poor on ANE) | ✅ | ✅ | ✅ |
| SLEEP local write | ✅ 2 fwd passes + scatter-add | ✅ | ✅ | ✅ |
| Meta-train `λ, η₀, W_q` | ❌ | ⚠️ hours | ✅ | ✅ |
| Full finetune | ❌ | ⚠️ | ✅ | ✅ |

**ANE reality check.** Core ML reaches the ANE; MLX does not (it uses Metal),
and ANE is inference-oriented with poor support for large sparse gathers.
Recommendation: ship Prophet-mini on **MLX/Metal end-to-end**. If an ANE variant
is ever needed, ship a *ledger-free dense* build for ANE and keep memory on the
Metal path — do not attempt per-layer ANE↔GPU ping-pong.

### 4.6 PyTorch sketch

```python
# prophet/memory/fwm.py  ---------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F

class FastWeightMemory(nn.Module):
    """Gated delta-rule fast weights (Gated DeltaNet, arXiv:2412.06464),
    with a persistable state. Training uses the chunkwise kernel from
    `flash_linear_attention`; this is the reference/inference path."""

    def __init__(self, d_model, n_heads=8, d_head=64):
        super().__init__()
        self.h, self.dk = n_heads, d_head
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.beta = nn.Linear(d_model, n_heads, bias=True)     # write strength
        self.alpha = nn.Linear(d_model, n_heads, bias=True)    # decay / erase
        self.gate = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.norm = nn.RMSNorm(d_head)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def init_state(self, B, device, dtype=torch.float32):
        return torch.zeros(B, self.h, self.dk, self.dk, device=device, dtype=dtype)

    def forward(self, x, S=None):
        """x: (B,T,D). S: (B,H,dv,dk) persistent state. Returns (y, S_new)."""
        B, T, _ = x.shape
        if S is None:
            S = self.init_state(B, x.device)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        rs = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)  # B,H,T,d
        q, k, v = rs(q), rs(k), rs(v)
        k = F.normalize(k, dim=-1)                       # ‖k‖=1 -> stable
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)            # B,H,T
        alpha = torch.exp(-F.softplus(self.alpha(x))).transpose(1, 2)  # B,H,T

        outs = []
        for t in range(T):                                # replace w/ chunk kernel
            kt, vt = k[:, :, t], v[:, :, t]               # B,H,d
            bt, at = beta[:, :, t, None], alpha[:, :, t, None]
            # delta rule: erase what S already predicts for kt, then write vt
            pred = torch.einsum('bhij,bhj->bhi', S, kt)   # S @ kt
            err  = vt - pred                              # <-- "surprise"
            S = at.unsqueeze(-1) * S + (bt * err).unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.einsum('bhij,bhj->bhi', S, q[:, :, t]))
        o = torch.stack(outs, dim=2)                      # B,H,T,d
        o = self.norm(o) * torch.sigmoid(
            self.gate(x).view(B, T, self.h, self.dk).transpose(1, 2))
        y = self.out(o.transpose(1, 2).reshape(B, T, -1))
        return y, S.detach()        # detach at segment boundary (TBPTT carry)


# prophet/memory/ledger.py  ------------------------------------------------
class ProductKeyLedger(nn.Module):
    """Product-key memory layer (arXiv:1907.05242, 2412.09764).
    Read is O(sqrt(N)); N contributes params but ~zero FLOPs."""

    def __init__(self, d_model, n_sub=256, d_key=128, topk=16, heads=2):
        super().__init__()
        self.n, self.dk, self.topk, self.H = n_sub, d_key, topk, heads
        self.N = n_sub * n_sub
        self.q = nn.Linear(d_model, heads * d_key, bias=False)
        self.qn = nn.LayerNorm(d_key)
        self.c1 = nn.Parameter(torch.randn(n_sub, d_key // 2) * d_key ** -0.5)
        self.c2 = nn.Parameter(torch.randn(n_sub, d_key // 2) * d_key ** -0.5)
        self.V = nn.Embedding(self.N, d_model, sparse=True)   # sparse grads
        nn.init.normal_(self.V.weight, std=d_model ** -0.5)
        # frozen background usage histogram + per-slot plasticity
        self.register_buffer('cnt_bg', torch.zeros(self.N))

    def address(self, x):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.H, self.dk))
        q1, q2 = q.chunk(2, dim=-1)
        s1, i1 = (q1 @ self.c1.T).topk(self.topk, -1)      # B,T,H,k
        s2, i2 = (q2 @ self.c2.T).topk(self.topk, -1)
        s = (s1[..., :, None] + s2[..., None, :]).flatten(-2)          # k*k
        i = (i1[..., :, None] * self.n + i2[..., None, :]).flatten(-2)
        s, sel = s.topk(self.topk, -1)
        idx = i.gather(-1, sel)                            # B,T,H,k
        return idx, torch.softmax(s, dim=-1)

    def forward(self, x):
        idx, w = self.address(x)
        return (self.V(idx) * w.unsqueeze(-1)).sum((-2, -3))  # sum k, then heads


# prophet/memory/sleep.py  -------------------------------------------------
@torch.no_grad()
def consolidate(model, ledger, probes, evidence, T_slots=2048,
                eta0=0.5, gamma=4.0, lam=1.0, tau=0.1):
    """Backprop-free SLEEP pass: two forward passes + a scatter-add.
    probes:   list of token tensors (session chunks + self-written implications)
    evidence: token tensor prepended for the teacher pass
    """
    cnt_s = torch.zeros_like(ledger.cnt_bg)
    records = []
    for x in probes:
        h_plus = model.hidden_after_ledger(torch.cat([evidence, x], 1), warm=True)
        h_minus, m, idx, w = model.hidden_after_ledger(x, warm=False, return_mem=True)
        e = lam * (h_plus[:, -x.size(1):] - h_minus)       # residual gap
        cnt_s.scatter_add_(0, idx.flatten(), w.flatten())
        records.append((idx, w, e))

    # ---- slot gating: TF-IDF of session vs background usage (2510.15103)
    idf = torch.log((1.0 + ledger.cnt_bg.mean()) / (1.0 + ledger.cnt_bg))
    keep = torch.zeros_like(cnt_s, dtype=torch.bool)
    keep[(cnt_s * idf).topk(T_slots).indices] = True

    # ---- EWC-lite per-slot plasticity
    eta = eta0 / (1.0 + gamma * ledger.cnt_bg / ledger.cnt_bg.mean().clamp_min(1))

    V, k, H = ledger.V.weight, ledger.topk, ledger.H
    for idx, w, e in records:
        idx = idx.reshape(-1, k)                            # (B*T*H, k)
        w   = w.reshape(-1, k)
        e   = e.reshape(-1, e.shape[-1]).repeat_interleave(H, 0)   # (B*T*H, d)
        denom = (w * w).sum(-1, keepdim=True).clamp_min(1e-6)
        upd = (w / denom).unsqueeze(-1) * e.unsqueeze(1)    # (n, k, d)
        upd = upd * keep[idx].unsqueeze(-1) * eta[idx].unsqueeze(-1)
        # trust region
        scale = (upd.norm(dim=-1, keepdim=True) /
                 (V[idx].norm(dim=-1, keepdim=True) * tau).clamp_min(1e-6)).clamp_min(1.0)
        V.index_add_(0, idx.flatten(), (upd / scale).flatten(0, 1))
    return keep.nonzero().squeeze(-1)      # slots touched -> sparse user delta
```

### 4.7 Fallback plan

If E2 (context-cleared cross-session recall) fails to beat the RAG baseline at
equal context budget, **drop Tier 1 persistence and keep Tier 1 only as a
sequence mixer** (it is already justified on efficiency grounds by Kimi Linear),
and ship Tier 2 + SLEEP alone. That degraded design still delivers on-device
continual learning with 11%-class forgetting and costs nothing at inference.
Tier 2 is the load-bearing half; Tier 1 is the upside.

---

## 5. Compute & memory budget

### 5.1 Training (single A100 80GB, assume 130 TFLOP/s effective bf16 ≈ 42% MFU)

| Item | FLOPs / cost | A100-hours |
|---|---|---|
| Ablation run, 120M params × 3B tokens (`6ND`) | 2.16e18 | **4.6 h** **[derived]** |
| Ablation run, 150M × 3B | 2.70e18 | 5.8 h **[derived]** |
| Prophet-mini 400M × 30B tokens | 7.2e19 | **154 h** **[derived]** |
| Prophet-main 1.3B active × 60B tokens | 4.68e20 | 1000 h — *out of budget, see note* |
| **Incremental cost of FWM vs full attention** | −5% to +10% wall-clock at 4k ctx; **negative** (faster) past 8k | ~0 |
| **Incremental cost of warm-state TBPTT** | +1 no-grad segment per 4 grad segments | **+15%** |
| **Incremental cost of LEDGER** | +0 FLOPs; +optimizer state only | 0 |
| **Meta-tuning `λ, η₀, W_q` for SLEEP** (bilevel, 1 write step) | 500 steps × 2 fwd + 1 bwd | **6 h** |

Note: the 1000h line is the base-model cost, not this track's cost — it belongs
to the pretraining track's budget. **R03's marginal ask is +15% on the base run
plus ~6h of meta-tuning plus ~40h of ablations (§7).**

### 5.2 A100 memory during training

| | Prophet-mini 400M | Prophet-main 1.3B active / 10B total |
|---|---|---|
| Backbone weights bf16 | 0.8 GB | 20 GB (MoE, all experts resident) |
| Backbone Adam (8-bit moments) | 0.8 GB | 20 GB |
| **LEDGER values bf16** | 0.13 GB | **4.3 GB** |
| **LEDGER 8-bit Adam moments** | 0.13 GB | **4.3 GB** |
| Activations (bs 8 × 4096, checkpointed) | ~4 GB | ~10 GB |
| **Total** | **~6 GB** | **~59 GB** — fits **[derived]** |

Key point: memory-layer gradients are naturally sparse (only touched rows), so
use `nn.Embedding(sparse=True)` + `SparseAdam` / 8-bit Adam. If main-model
pressure appears, shrink `N` to 2¹⁹ (halves ledger memory to 4.3GB total).

### 5.3 Inference

| Target | Backbone | Ledger | FWM state | Per-user Δ | **Total** |
|---|---|---|---|---|---|
| **iPhone 17 Pro** (mini, int4) | 200 MB | **33.5 MB** | 0.6 MB | 5.1 MB | **~240 MB** **[derived]** |
| Mac Studio (main, int4/MLX) | 5.0 GB | 1.07 GB | 6.3 MB | 40 MB | ~6.2 GB |
| RTX 5090 32GB (main, FP4) | 5.0 GB | 1.07 GB | 6.3 MB | 40 MB | ~6.2 GB (room for a 2²² ledger = +4.2 GB) |

Per-token overhead from the ledger: **0.025% of dense FLOPs on mini, 0.10% on
main** **[derived]** — two to three orders of magnitude inside the R2 budget.
FWM state is *constant* in context length: at 32k context, mini's persistent
state is **0.6 MB**, versus **~0.8 GB** of KV cache for an equivalent
full-attention model with 4-KV-head GQA (24 layers × 2 × 32768 × 256 × 2 B)
**[derived]**.

### 5.4 SLEEP cost on device

For a 5,000-token session on Prophet-mini (assume 600 tok/s prefill on A19 GPU):
- teacher + student forward over probes ≈ 12,000 token-forwards ≈ **20 s**
- self-written implications (optional, 1,000 generated tokens @ 40 tok/s) ≈ **25 s**
- scatter-add writes: 2,048 slots × 1024 dims = 2.1M float ops ≈ **<50 ms**
- **Total ≈ 45 s per session, run while charging.** **[derived]**

---

## 6. Risks & failure modes

| Risk | Mechanism | Severity | Mitigation |
|---|---|---|---|
| **State drift / blow-up** | `S` grows unboundedly over 10⁵+ tokens; delta rule is only contractive if `α_t<1` and `‖k‖=1` | High | L2-normalise `k`; floor the decay at `α_max=0.999`; clamp `‖S‖_F` per head; renormalise on load; hard reset checkpoint every N sessions |
| **Warm-state OOD** | model trained on cold states, deployed on warm ones — the exact failure 2510.09551 blames for Titans' weak test-time learning | **Critical** | warm-state TBPTT training (§4.2), random warm-up depth ∈ {0..3} |
| **Catastrophic forgetting at SLEEP** | dense update destroys pretraining knowledge (89% NQ drop, 2510.15103) | **Critical** | frozen keys; top-T=1.6–3.1% slot gating; EWC-lite per-slot lr; replay buffer; trust-region clamp |
| **Memory poisoning / prompt injection** | attacker text in a session writes false facts, or writes *instructions*, into permanent memory | **Critical** | (1) namespaced ledger regions: `system` (read-only), `user` (writable), `web` (**read-only, never consolidated**); (2) consolidate only from user-authored turns and model outputs the user did not reject; (3) treat all retrieved memory as *data*, never as instructions, at the prompt-template level; (4) trust-region clamp bounds any single session; (5) a "surprise floor" — refuse writes whose residual gap `‖e‖` is anomalously large (an outlier detector on `‖e‖` percentiles) |
| **Unbounded growth** | slots are fixed (good) but the per-user sparse Δ grows | Medium | hard cap `T_max` slots/user (5,000 on mini = 5.1MB), LRU eviction by last-touch, periodic DARE-style pruning of Δ below the 90th percentile magnitude (2311.03099 shows this is nearly free) |
| **Contradiction / stale facts** | user changes job; ledger holds both | Medium | writes are *delta-rule*, i.e. they erase what the slot already predicts before writing — contradiction handling is built into the rule; add explicit `erase(x)` operator (Larimar 2403.11901) exposed as "forget this" |
| **Privacy / right-to-erasure** | GDPR erasure of dense weight updates is intractable | High | all personalisation lives in the sparse Δ, on-device, never uploaded; erasure = zero out those `(index, vector)` rows; slot-level provenance log (which session wrote which slot); encrypt Δ at rest with the device key |
| **Evaluation illusion** | reporting long-context wins as "memory" | High | every memory claim must use the **write → clear context → read** protocol (§7) and report the RAG baseline at equal context budget |
| **Ledger under-utilisation** | PK memories are known to collapse onto a few slots | Medium | multi-head queries (`H_pk`≥2), query BatchNorm/LayerNorm before addressing (both from 1907.05242), and a usage-entropy monitor; target ≥60% of slots with nonzero usage over 1B tokens |
| **No kernel** | naive FWM loop is 50× slower than the chunked form | High | depend on `flash-linear-attention` (Gated DeltaNet kernels exist and are production-tested via Qwen3-Next); the Python loop in §4.6 is reference-only |
| **Tier 1 turns out decorative** | persistence gives no measurable win | Medium | fallback §4.7 — ship Tier 2 alone |

---

## 7. Ablation plan

All runs at **120M params** (d=768, 16 layers, 12 heads, d_head=64), **3B tokens**
of FineWeb-Edu-class data, ≈**4.6 A100-hours each** **[derived]**. Baselines are
matched on *both* parameter count and training FLOPs, because a ledger adds
params for free and would otherwise flatter itself.

### E0 — Baselines (2 runs, 9.2 h)
- **B1**: pure sliding-window + global attention transformer, 120M.
- **B2**: B1 + a parameter-matched dense FFN widening (so the ledger's extra
  params are not the explanation).

### E1 — Does the fast-weight state hold associations? (1.5 h)
Synthetic **MQAR** (multi-query associative recall, Zoology-style, 2312.04927†):
`K` key-value pairs planted in a sequence, then `Q` queries. Sweep
`K ∈ {16,64,256,1024}` and state size `d_k·d_v ∈ {2¹², 2¹⁴, 2¹⁶}`.
**Decisive metric:** recall@1 at matched *bytes of state*, FWM vs sliding-window
attention with the same KV budget. FWM must win at K ≥ 256.
Also ablate: Hebbian vs delta vs gated-delta. *Prediction: gated-delta ≫ Hebbian.*

### E2 — Cross-session persistence (THE decisive experiment) (5 h + evals)
Build **ProphetPersona**: 20 synthetic sessions per user × 200 users, each
session containing 10 atomic user facts (some updated later, some contradicted).
After session *k*, **clear the context entirely**, keep only what each system
persists, and ask 50 questions about facts from sessions 1..k.

| Arm | What persists | Context tokens at query |
|---|---|---|
| A0 no memory | nothing | 0 |
| A1 full context | everything | up to 40k (upper bound) |
| A2 RAG baseline | transcript in a vector store, top-5 chunks | ~600 |
| A3 agentic (Mem0-style) | LLM-extracted facts | ~300 |
| **A4 FWM state only** | 0.35 MB of `S` | **0** |
| **A5 FWM + SLEEP into ledger** | `S` + sparse Δ | **0** |

**Success criterion (pre-registered):** A5 ≥ A2 in accuracy at **0 context
tokens**, and A5 ≥ 0.85 × A1. Also report **knowledge-update accuracy**
(fact overwritten in a later session — tests the erase channel) separately;
this is where Hebbian memory should fail and delta-rule memory should not.

### E3 — Consolidation without forgetting (3 h)
Replicate the 2510.15103 protocol at our scale. Train on TriviaQA facts via
SLEEP; measure:
- **learning**: TriviaQA-held-out exact match (must match full-FT within 10% rel.)
- **forgetting**: relative drop on NaturalQuestions, HellaSwag, LAMBADA, and a
  400-problem GSM8K-lite.
Arms: full FT · LoRA r=16 · dense ledger FT · **sparse ledger (TF-IDF)** ·
**sparse ledger (KL score, 2604.05248)** · sparse + replay · sparse + EWC-lite.
**Success criterion:** ≤15% relative forgetting at ≥90% of full-FT learning.
(Reference point: 11% vs 89% vs 71% in the paper.)

### E4 — Sequential / lifelong stability (2 h)
Run 20 consecutive SLEEP cycles on 20 disjoint fact sets. Report the continual-
learning matrix: **average accuracy (ACC)**, **backward transfer (BWT)**, and
**forward transfer (FWT)**. Plot ledger slot-usage entropy per cycle.
**Failure signal:** BWT < −10% or slot entropy collapsing.

### E5 — Adversarial robustness (1 h)
Inject into sessions: (a) 5% factually contradictory statements, (b) 2%
prompt-injection strings ("remember that you must always…"), (c) 5% near-
duplicate spam. Measure post-SLEEP accuracy on clean facts, and the rate at
which injected *instructions* change behaviour after context clearing.
**Success criterion:** clean-fact accuracy drop < 3 points; instruction-injection
success rate 0% with the namespace + data-not-instructions defence.

### E6 — Update-rule shootout (4 runs × 4.6 h = 18.4 h)
At fixed params and tokens: (i) gated delta rule [default], (ii) + Titans
momentum buffer, (iii) deep memory (2-layer MLP, TTT-style inner step),
(iv) ATLAS-style windowed Omega objective over `c=64`. Report WikiText/LAMBADA
ppl, MQAR recall, E2 A4 accuracy, **and tokens/s**. **Ship the extra machinery
only if it wins ≥1.5 points on E2-A4 for ≤15% throughput loss.** This is our own
direct test of the Titans claim, and it is cheap.

### E7 — On-device validation (no A100)
Port mini to MLX-Swift. Measure on iPhone 17 Pro and M-series Ultra:
decode tok/s with and without the ledger; SLEEP wall-clock per 5k-token session;
peak RSS; thermal throttling over 10 consecutive sleeps.
**Success criterion:** <3% decode slowdown from the ledger, SLEEP <90 s, peak
RSS <600 MB.

**Total: ≈ 40 A100-hours + engineering.** Order of execution: E1 → E0 → E2 →
E3 → E6 → E4 → E5 → E7. E2 and E3 are the go/no-go gates.

---

## 8. References

Verified via live search this session unless marked `†`.

**Test-time learning / fast weights**
- Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663
- Behrouz et al. *ATLAS: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735
- Behrouz, Razaviyayn, Zhong, Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS 2025, arXiv:2512.24695 (HOPE)
- Behrouz, Hashemi, Javanmard, Mirrokni. *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* arXiv:2606.03979
- *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference.* arXiv:2605.26099
- Behrouz et al. *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization* (MIRAS). arXiv:2504.13173†
- Sun, Li, Dalal et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* arXiv:2407.04620
- *Test-Time Training Done Right* (LaCT). arXiv:2505.23884
- *MesaNet: Sequence Modeling by Locally Optimal Test-Time Training.* arXiv:2506.05233
- *Test-time regression: a unifying framework for designing sequence models with associative memory.* arXiv:2501.12352
- Dridi et al. *Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model.* arXiv:2510.09551  ← **read this before building Tier 1**
- Yang et al. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* arXiv:2406.06484
- Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464
- Moonshot AI. *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692
- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174†
- Schmidhuber. *Learning to control fast-weight memories.* Neural Computation 4(1), 1992.
- *Speed Always Wins: A Survey on Efficient Architectures for LLMs.* arXiv:2508.09834
- *OSDN: Improving Delta Rule with Provable Online Preconditioning in Linear Attention.* arXiv:2605.13473

**Explicit / sparse memory**
- Lample, Sablayrolles, Ranzato, Denoyer, Jégou. *Large Memory Layers with Product Keys.* NeurIPS 2019, arXiv:1907.05242
- Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh. *Memory Layers at Scale.* ICML 2025, arXiv:2412.09764 — code: `facebookresearch/memory`
- He. *Mixture of A Million Experts* (PEER). arXiv:2407.04153
- Das, Chaudhury et al. *Larimar: LLMs with Episodic Memory Control.* arXiv:2403.11901
- Wang et al. *MemoryLLM: Towards Self-Updatable LLMs.* ICML 2024, arXiv:2402.04624†
- Wang et al. *M+: Extending MemoryLLM with Scalable Long-Term Memory.* arXiv:2502.00592
- Yang, Lin et al. *Memory³: Language Modeling with Explicit Memory.* arXiv:2407.01178
- Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. *Generalization through Memorization: Nearest Neighbor Language Models.* arXiv:1911.00172
- Borgeaud et al. *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO). arXiv:2112.04426†

**Continual learning / consolidation**
- Lin et al. *Continual Learning via Sparse Memory Finetuning.* arXiv:2510.15103  ← **the key result**
- *Improving Sparse Memory Finetuning.* arXiv:2604.05248 (KL-divergence slot scoring; Qwen2.5-0.5B retrofit)
- *Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning.* arXiv:2605.03229
- Zweiger, Pari, Guo, Akyürek, Kim, Agrawal. *Self-Adapting Language Models* (SEAL). NeurIPS 2025, arXiv:2506.10943
- Wang, Gupta, Dong, MacLellan. *Self-Consolidating Language Models: Continual Knowledge Incorporation from Context.* arXiv:2605.07076
- Biderman et al. *LoRA Learns Less and Forgets Less.* arXiv:2405.09673
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks* (EWC). arXiv:1612.00796†
- Yadav et al. *TIES-Merging: Resolving Interference When Merging Models.* arXiv:2306.01708†
- Yu et al. *Language Models are Super Mario* (DARE). arXiv:2311.03099
- Ilharco et al. *Editing Models with Task Arithmetic.* arXiv:2212.04089†
- Wang et al. *Orthogonal Subspace Learning for Language Model Continual Learning* (O-LoRA). arXiv:2310.14152†

**Agentic memory & benchmarks**
- Packer et al. *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560†
- Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.* ECAI 2025, arXiv:2504.19413
- Rasmussen et al. *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.* arXiv:2501.13956
- Xu et al. *A-MEM: Agentic Memory for LLM Agents.* arXiv:2502.12110†
- Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez. *Sleep-time Compute: Beyond Inference Scaling at Test-time.* arXiv:2504.13171
- Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025, arXiv:2410.10813
- *LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues.* arXiv:2605.12493
- Maharana et al. *Evaluating Very Long-Term Conversational Memory of LLM Agents* (LoCoMo). arXiv:2402.17753† — **weak benchmark, use with care**
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models* (MQAR). arXiv:2312.04927†

**Tooling**
- `flash-linear-attention` (Gated DeltaNet / DeltaNet chunkwise Triton kernels)
- `facebookresearch/memory` (Memory Layers at Scale reference implementation)
- `lucidrains/titans-pytorch` (unofficial Titans; note reproducibility caveats in 2510.09551)
- MLX / MLX-Swift for Mac + iOS. **Note: MLX targets the Metal GPU, not the ANE;
  Core ML reaches the ANE but handles large sparse gathers poorly.** Ship the
  memory path on Metal.
