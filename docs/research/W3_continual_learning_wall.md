# W3 — The continual learning wall: why gradients forget, and what the third tier actually is

**Track:** W3 · **Status:** design-ready, one gate unfunded · **Date:** 2026-09-03
**Builds on:** [`R03_memory_continual_learning.md`](R03_memory_continual_learning.md),
[`../06_MEMORY.md`](../06_MEMORY.md), and the implemented two-tier system in
`prophet/memory/` (`ledger.py`, `consolidate.py`, `session.py`).
**Scope:** the question posed to this track — *is the stability–plasticity dilemma
solvable by gradient-based systems at all, and what is the third tier we are missing?*

---

## 0. Provenance and verification status — read before using any number

**The egress proxy blocked every primary source.** Direct fetches to `arxiv.org`,
`export.arxiv.org`, `huggingface.co`, `openreview.net`, `semanticscholar.org`,
`ar5iv.labs.arxiv.org`, `ui.adsabs.harvard.edu`, `aclanthology.org`,
`proceedings.neurips.cc`, `nature.com`, `pmc.ncbi.nlm.nih.gov`, `alphaxiv.org` and
`paperswithcode.com` all returned `403 to CONNECT (policy denial)` from the gateway.
Probing confirms the denial is host-level, not transport-level. **No PDF or abstract page
was read directly during this session.**

The only working research channel was the `WebSearch` tool, which returns search-engine
summaries. Its per-session budget (200 calls, shared with sibling agents) was exhausted
before the last two verification queries could run.

Consequence — every figure below carries a marker:

| Marker | Meaning |
|---|---|
| **[S]** | Taken from a **search-result summary** this session. Title and arXiv ID confirmed in the result list; the number itself was **not** read from the paper. Treat as ±plausible, verify before spending compute on it. |
| **†** | Cited from prior knowledge. arXiv ID **not** confirmed this session. Verify before quoting externally. |
| **[derived]** | Arithmetic done here, from stated inputs. Reproducible. |
| **[ours]** | Measured in this repository (`tests/test_memory.py`), reported in `docs/06_MEMORY.md`. |

This is the same degradation R03 hit and flagged in `docs/research/README.md`. It is worse
this time: R03 got some live fetches through; this session got none.

---

## 1. Why gradient descent forgets

Not "catastrophic forgetting happens". The mechanism, in four separable parts that have
**different fixes** — which is why single-mechanism mitigations plateau.

### 1.1 Forgetting is negative gradient alignment. That is the whole first-order story.

Take a network with shared parameters θ, an old loss `L_old` and a new one `L_new`. One
SGD step `Δθ = −η ∇L_new` changes the old loss, to first order, by

```
ΔL_old ≈ ∇L_old · Δθ = −η ⟨∇L_old, ∇L_new⟩ = −η ‖∇L_old‖ ‖∇L_new‖ cos φ
```

Forgetting is *exactly* the case `cos φ < 0`. Nothing more mysterious is required. This
identity is the entire basis of the transfer–interference framing (Riemer et al., MER,
arXiv:1810.11910 [S]) and of gradient-surgery methods (GEM, arXiv:1706.08840 †).

The function-space version is sharper. In the NTK regime the change in the network's
output on old inputs after a step on new inputs is

```
Δf(X_old) = −η · K(X_old, X_new) · ∇_f L_new
```

where `K` is the neural tangent kernel. Forgetting is therefore governed by the **NTK
overlap matrix** between tasks: zero overlap ⇒ zero forgetting, high overlap ⇒ the new
update drags old outputs with it (Doan et al., arXiv:2010.04003 [S]). The counter-intuitive
corollary, which their paper states explicitly: **forgetting increases as the two tasks
become more aligned** — up to a point (see §1.3).

### 1.2 The dilemma is a rank budget, and it is finite.

Orthogonal Gradient Descent (Farajtabar et al., arXiv:1910.07104 [S]) projects the new
gradient onto the orthogonal complement of stored old-task gradients; Gradient Projection
Memory (Saha et al., arXiv:2103.09762 [S]) does the same with the old-task *feature*
subspace. Both **guarantee** first-order non-interference. Both make the price explicit:

- parameter space has dimension `d`;
- each protected task consumes `r` directions of that space;
- after `T = d/r` tasks the free subspace is empty and **plasticity is exactly zero**.

This is the cleanest statement of the stability–plasticity dilemma available: it is not a
pathology of SGD, it is **linear-algebraic bookkeeping**. In its strong form —
unbounded new knowledge, fixed parameters, zero interference — the dilemma is not
"hard", it is *arithmetically impossible*. Any system claiming to have solved it has
either grown parameters, accepted interference, or bounded the knowledge stream.

The information-theoretic version of the same bound: a transformer stores
**≈2 bits of extractable knowledge per parameter, even quantised to int8**
(Allen-Zhu & Li, "Physics of LM 3.3", arXiv:2404.05405 [S]). Prophet-mini at 229M
parameters therefore has a hard ceiling of ≈458 Mbit ≈ 57 MB of factual capacity, all of
which is already spoken for by pretraining. **This single number decides most of §4.**

### 1.3 Curvature: forgetting is a second-order quantity, and the geometry is fixable.

At the old task's optimum `∇L_old ≈ 0`, so the first-order term vanishes and

```
L_old(θ* + Δ) ≈ L_old(θ*) + ½ Δᵀ H Δ
```

Forgetting is governed by the **Hessian** — which is precisely what EWC penalises with a
diagonal Fisher approximation, `L_new + λ Σ_i F_i (θ_i − θ*_i)²` (Kirkpatrick et al.,
arXiv:1612.00796 †). Two consequences that matter more than EWC itself:

- **Flat minima forget less.** Dropout, learning-rate decay and batch size act as
  regularisers that widen task minima and measurably reduce forgetting (Mirzadeh et al.,
  arXiv:2006.06958 [S]). Note the ID: R03's tooling and several secondary sources
  mis-cite this as 2010.04495, which is the *Linear Mode Connectivity* paper by an
  overlapping author set. Both are real; they are different papers.
- **A joint solution usually exists and SGD just misses it.** Multitask and continual
  solutions are frequently connected by simple low-error curves in weight space
  (arXiv:2010.04495 [S]). This is the strongest available argument that the *practical*
  dilemma is a **search failure, not a representational one** — the low-forgetting weights
  are nearby; plain sequential SGD does not walk to them.

### 1.4 Where forgetting lives, and what it is not

- **Depth.** Deeper layers are disproportionately the source of forgetting, and mitigation
  methods work by stabilising them (Ramasesh et al., arXiv:2007.07400 [S]).
- **Similarity is non-monotone.** Maximal forgetting occurs at *intermediate*
  representational similarity between tasks (same paper [S]). Orthogonal tasks do not
  interfere; identical tasks help; the dangerous regime is "almost the same" — which is
  exactly the regime a personalisation update lives in.
- **Scale mitigates.** Large pretrained transformers and ResNets are substantially more
  resistant to forgetting than from-scratch models, improving systematically with model
  and pretraining-set size (Ramasesh, Lewkowycz & Dyer, ICLR 2022 [S]). Prophet, at
  229–500M, sits at the **bad end** of this curve. We get less protection from scale than
  anyone we are compared against.
- **Much of it is not knowledge loss at all.** "Spurious forgetting"
  (arXiv:2501.13453 [S], ICLR 2025) shows that large sequential-training performance drops
  frequently reflect a collapse of **task alignment**, not destruction of knowledge — the
  facts are still in there, the model has lost the interface to them. Their fix is
  embarrassingly cheap: **freeze the bottom layers**, which improves all four continual
  scenarios tested. Design consequence: *any forgetting number reported as task accuracy
  conflates two mechanisms with different cures*, and a large part of what we must protect
  is an interface, not content.
- **Forgetting is low-rank and therefore predictable.** The matrix of
  (new task) × (forgotten upstream example) is well approximated by low-rank multiplicative
  structure; matrix completion predicts which upstream examples a new task will damage, and
  up-weighting the predicted ones for replay measurably reduces forgetting
  (Jin & Ren, arXiv:2406.14026 [S], ICML 2025). **This is directly actionable for us:
  replay selection should not be uniform.**

### 1.5 Capacity or interference? Both — and which one binds depends on scale.

The honest answer is that the field's two headline results point in opposite directions
and are both right:

| Evidence | Says |
|---|---|
| Sparse memory finetuning: NQ drops **89%** (full FT) / **71%** (LoRA) / **11%** (sparse memory) *at equal new knowledge learned* (arXiv:2510.15103 [S]) | Interference, not capacity. The same knowledge fits; the update's **support** decides whether it destroys anything. |
| Knowledge capacity is 2 bits/param, int8 included (arXiv:2404.05405 [S]) | Capacity, eventually. At 229M there is no room for a life's worth of facts, however sparse the update. |

For Prophet the resolution is: **interference binds over a session, capacity binds over a
deployment.** Both must be handled, and — this is the load-bearing inference for §4 —
they must be handled by *different tiers*, because the tier that fixes interference
(sparse, disjoint, addressable writes: our ledger) is the wrong place to fix capacity, and
the tier that has capacity (the trunk) is the wrong place to write facts.

### 1.6 Why LoRA does not rescue this, and the Prophet-specific hazard

- Forgetting grows as a **shifted power law in the number of non-embedding parameters
  fine-tuned and in the number of update steps**, and is strongly predicted by a linear
  function of the fine-tuning loss. It cannot be escaped by early stopping or by tuning
  fewer parameters (arXiv:2401.05605 [S], Llama-2-7B-chat).
- LoRA's solutions are **structurally different** from full fine-tuning: they introduce
  high-ranking singular vectors absent from the base ("intruder dimensions"). LoRA forgets
  less, but its forgetting is *localised* in those dimensions, and models carrying them are
  worse models of the pretraining distribution and adapt less robustly to task sequences
  (arXiv:2410.21228 [S]).
- **Prophet-specific.** Our trunk has a weight-shared recurrent core applied `k` times
  (`RecurrentCoreConfig`, `default_loop_k = 4`, swept at inference). A parameter
  perturbation inside the core is therefore applied **k times per forward pass**. A LoRA
  of nominal norm ε behaves like a perturbation of effective norm ~kε at k=8, and its
  interference scales with the depth dial the user is free to move. **Do not put a
  continual-learning adapter inside the looped core.** This constraint is not in any paper;
  it falls out of our architecture and it eliminates the most obvious design.

### 1.7 The 2026 mechanistic picture (single-lab, unreplicated)

Two 2026 papers claim to localise the mechanism at LLM scale. Both are single-source; I
could not read either. Recorded because they are the only recent mechanistic work found:

- arXiv:2601.18699 [S]: decomposes forgetting into gradient interference in attention
  weights, representational drift in intermediate layers, and loss-landscape flattening
  around prior minima; reports CKA dropping **0.32–0.47** in intermediate layers and
  **15–23%** of lower-layer attention heads severely disrupted, with gradient alignment
  predicting forgetting severity.
- arXiv:2605.09608 [S] ("Geometry Conflict"): frames forgetting as a *state-relative
  update-integration failure* — each update lands on a model state that already encodes
  prior updates, and forgetting occurs when the covariance geometries conflict.

Both are restatements of §1.1–§1.3 with better instrumentation. Neither changes the design.

### 1.8 Answering the question as asked

> *Is the stability–plasticity dilemma solvable at all by gradient-based systems, or does
> it require a different substrate?*

Three claims, in decreasing confidence:

1. **In its strong form it is unsolvable, by any substrate.** §1.2 is a counting argument:
   finite parameters, finite rank, ~2 bits each. Anything that keeps learning forever must
   grow, forget, or externalise. This is not a fact about gradients.
2. **In its practical form it is already solved to within useful tolerance, by gradient
   systems, using three ingredients** — replay (§3), locality/sparsity of the update
   support (§3), and distillation-based anchoring to the pre-update model (§2). None of
   these requires a new substrate. The evidence is Ibrahim et al.: LR re-warm + re-decay +
   ~5% replay **matches full retraining from scratch** on final loss and benchmarks
   (arXiv:2403.08763 [S]).
3. **The part that genuinely wants a different substrate is not accuracy, it is cost.**
   What gradient descent cannot do is write one fact, on a phone, in 50 ms, without a
   training job and without touching parameters that other capabilities depend on. That is
   the property our closed-form ledger write already has, and it is why it forgets 11%-class
   rather than 89%-class. **The substrate change that matters is not "not gradients" — it
   is "an addressable store with disjoint write support."** We have that. The third tier
   is therefore a *supplement with a narrow job description*, not a replacement, and §4 is
   about keeping that job description narrow.

---

## 2. Complementary learning systems, and who has actually built one

### 2.1 The theory, and the correction the working thesis is missing

The original CLS argument (McClelland, McNaughton & O'Reilly, *Psychological Review*
102(3):419–457, 1995 †) is usually summarised as "fast hippocampus, slow neocortex,
replay in between". That summary is what the working thesis for this track assumes. The
**updated** version (Kumaran, Hassabis & McClelland, *Trends in Cognitive Sciences*
20(7):512–534, 2016 [S] — title and venue confirmed) adds two points that change the
engineering:

1. Discovery of structure in ensembles of experiences **depends on an interleaved
   learning process**. Not on transfer of individual episodes — on *interleaving across
   many*. A single episode has no business becoming a weight.
2. **Once structured knowledge has been acquired, new *consistent* information can be
   integrated rapidly** [S]. Consolidation rate is gated by consistency with existing
   schema (the animal-learning evidence is Tse et al., *Science* 316:76–82, 2007 †).

So the correct reading of CLS is not "drain the fast store into the slow one". It is:

> **Distil the regularities *across* episodes, at a rate set by their consistency with what
> the slow system already knows. Keep the exceptions in the fast store forever.**

That is a materially different specification from the working thesis, and it is the one
§4 implements.

### 2.2 Classical ML instantiations

| System | Mechanism | Status |
|---|---|---|
| **Deep Generative Replay** (Shin et al., arXiv:1705.08690 [S], NIPS'17) | "Scholar" = generator + solver; the generator replays *synthesised* past data so old data need not be stored. Explicitly motivated by the hippocampus as short-term store. | The canonical memory→weights loop. Everything below is a variation. |
| **CLS-ER** (Arani et al., arXiv:2201.12604 [S], ICLR'22) | Episodic replay buffer + **two** semantic memories built as EMAs of the student at different rates (short-term plastic, long-term stable). No task boundaries. | The most faithful CLS implementation in ML. Small-vision scale. |
| **Sleep Replay Consolidation** (Tadros et al., *Nature Communications* 13:7742, 2022 [S]) | Map the trained net to a spiking net, drive it with noise reflecting the training distribution, apply **local Hebbian plasticity offline**. Old tasks that were forgotten are *recovered*. | Strong existence proof that an offline, unsupervised, local-rule phase can undo forgetting. Not language-scale. |
| **Learning without Forgetting** (Li & Hoiem, arXiv:1606.09282 [S]) | Preserve old behaviour by distilling the *pre-update model's own softened outputs* on the new data. | The anchor loss in §4 is this, at LLM scale. |
| **Wake-Sleep Consolidated Learning** (arXiv:2401.08623 [S]) | Explicit wake/NREM/REM phases. | Vision scale. Recorded, not adopted. |

### 2.3 Who has actually built hippocampus→neocortex **for LLMs**

The honest finding of this track: **almost nobody**. The field is cleanly split into
(a) memory that never becomes weights — RAG, agentic memory, Cartridges, KV compression —
and (b) weights that never had a memory — continual pretraining, SFT, model editing. The
crossing points are few, recent, and mostly unreplicated:

| Work | What crosses the gap | Numbers |
|---|---|---|
| **Context distillation** (Snell, Klein & Zhong, arXiv:2209.15189 [S]) | Fine-tune the model to predict its own **context-conditioned** outputs without the context. Token-level KL. | Internalises instructions, in-context examples and scratch-pads. The primitive our `consolidate()` already uses, at the residual level instead of the logit level. |
| **Cartridges / self-study** (arXiv:2506.06266 [S]) | Generate synthetic conversations about a corpus; train a small **KV cache** with a context-distillation objective. | Matches ICL quality at **38.6× less memory** and **26.4× higher throughput** [S]. But it distils into *state*, not weights — this is a Tier-1.5 result, not a third tier. |
| **Synthetic continued pretraining / EntiGraph** (arXiv:2409.07431 [S]) | The "dreaming" step made concrete: extract entities from a small corpus, generate diverse texts connecting them, then continually pretrain. | **455M synthetic tokens** generated from the QuALITY corpus; Llama-3-8B continually pretrained on it can answer without the documents, and the gain **compounds with RAG**. |
| **SEAL** (arXiv:2506.10943 †, verified by R03 last session) | The model writes its own consolidation data ("self-edits") and applies them by finetuning; RL outer loop selects edits by downstream reward. | No-context SQuAD **32.7 → 47.0**, beating GPT-4.1-generated synthetic data (46.3) †. Mechanism is a full finetune per edit — not on-device. |
| **Memory Decoder** (arXiv:2508.09874 [S]) | Pretrain a **small parametric decoder to imitate a kNN retriever's output distribution**; interpolate with the base model at inference. Base parameters never change. | **−6.17 perplexity** average across biomedicine, finance and law [S]. The most direct precedent anywhere for "distil a non-parametric store into parameters", and it does so **without touching the base model** — an architecture option we should keep on the table. |
| **Self-Distillation Fine-Tuning** (arXiv:2601.19897 [S]; also SDFT arXiv:2402.13669 †) | Use the *demonstration-conditioned* model as its own teacher, making the update on-policy. | Higher new-task accuracy **and** substantially less forgetting than SFT; a single model accumulates skills sequentially "without performance regression" [S]. |
| **Self-Synthesized Rehearsal** (arXiv:2403.01244 [S]) | Generative replay for LLMs: the base model generates its own rehearsal instances in-context. | Comparable or better than real-data rehearsal, more data-efficient [S]. **Removes the need to ship pretraining data to the device.** |
| **2026 "sleep" line** (arXiv:2605.26099, arXiv:2606.03979 [S]) | Offline recurrent passes that consolidate context into fast weights with the attention cache cleared; and a Sleep/Dreaming framework distilling short-term fragile memory into long-term weights with replay and parameter expansion. | Single lab, no reproductions, no code found. Directionally identical to our design. Do not depend on their numbers. |
| **Nested Learning / HOPE** (arXiv:2512.24695 [S], NeurIPS'25) | Generalises CLS from two systems to a **continuum of memories updating at different frequencies**. | The right abstraction. Prophet's three tiers are a coarse, three-point discretisation of it. |
| **Position: Modular Memory** (arXiv:2603.01761 [S], ICML'26 spotlight) | The field's own statement that combining in-weight learning with in-context learning through modular memory is the missing piece. | Confirms the thesis is mainstream by mid-2026 and still unbuilt. |

### 2.4 Verdict on the working thesis

> *"We have built the fast sparse store. We have NOT built the slow distillation into
> weights. Without it, the ledger grows without bound and never becomes fluent knowledge."*

**Two thirds right, and the wrong third is the expensive one.**

- ✅ **Right that the slow system is missing.** Nobody in the R03 design writes weights.
  Everything above that crosses the gap does something we do not do.
- ✅ **Right that a lookup table is not a skill.** §5 makes that measurable, and the
  evidence that it is a real failure is strong (§5.1).
- ❌ **Wrong that the ledger "grows without bound".** It does not: `n_slots` is fixed at
  65,536, keys are frozen, and the per-user state is a bounded sparse delta. What grows
  without bound is the *number of episodes competing for a fixed number of slots* — a
  capacity-pressure problem, not an unbounded-growth problem. The fix for capacity pressure
  is eviction and abstraction, not necessarily weights.
- ❌ **Wrong that the destination should be weights, for facts.** §1.2: the trunk holds
  ≈57 MB of extractable knowledge *in total* at 229M params, all of it already used. The
  ledger holds 65,536 × 1536 values = **50.3 MB at int4** [derived] **at zero marginal
  FLOPs per token**. Moving a fact from ledger to trunk trades a free byte for an expensive
  one and evicts something else. **Facts must stay in the ledger permanently.** The third
  tier's job is not storage.

And one piece of hard 2026 evidence against the naive thesis, which must be confronted
rather than filed under related work:

> **Consolidation actively destroys utility when run indiscriminately.** In
> arXiv:2605.12978 [S], memory utility **rises then declines, and can fall below the
> no-memory baseline** as consolidation proceeds; on ScienceWorld and WebShop, distilled
> memory degrades with scale **while raw trajectories remain robust**; and even when
> consolidating from *ground-truth solutions*, a frontier model failed **54% of ARC-AGI
> problems it had previously solved without memory** [S]. The authors trace the regression
> to the **consolidation step itself**, and find an episodic-only control that just keeps
> the raw trajectories remains competitive with every consolidator tested.

That result is the single strongest argument that the third tier can be a **bad idea**. It
is why §4 gates consolidation on recurrence, consistency and verification, and why the
merge gate can reject a pass wholesale.

---

## 3. What works today, quantified

### 3.1 Replay: how much, and what it costs

| Setting | Replay needed | Result | Source |
|---|---|---|---|
| Weak shift, English→English (Pile → SlimPajama), 405M **and** 10B | **5%** | LR re-warm + re-decay + 5% replay ≈ **matches full retraining from scratch** on final loss and LM benchmarks, at a fraction of the compute | arXiv:2403.08763 [S] |
| Strong shift, English→German, same scales | **25%** | same conclusion; "the stronger shift requires more replay to mitigate forgetting to the same extent" | arXiv:2403.08763 [S] |
| Any replay at all | **1%** | "Even the lowest tested replay of 1% significantly reduces forgetting on Pile"; 1/5/10% cost **little downstream performance** vs 0% | arXiv:2403.08763 [S] |
| Extreme replay | **50%** | adapts *significantly worse* to the new data, yet still attains or surpasses the final average validation of training on the combined datasets | arXiv:2403.08763 [S] |
| Finetuning (not pretraining) | **1%** injected pretraining data | "shields the model from forgetting the pre-training set" | Apple, arXiv:2502.06042 [S] |
| Continual pretraining at 100B tokens/language, Llama family | replay **and** gradient alignment (Reptile/MER-style meta-updates) | both yield "more stable learning without forgetting" | arXiv:2508.01908 [S] |

**The compute price of replay, exactly** [derived]: at replay fraction ρ, reaching a fixed
number of *new* tokens costs `1/(1−ρ)` total tokens. ρ=0.01 → +1.0%; ρ=0.05 → **+5.3%**;
ρ=0.25 → **+33.3%**; ρ=0.50 → +100%. The 5% operating point is close to free. **This is the
best cost/benefit ratio in the entire continual-learning literature** and it is the reason
§4 spends its complexity budget elsewhere.

**Our own replay measurement** [ours]: consolidating a second batch of episodes degraded
recall of the first from 0.000 → **0.229**; with 25% replay, 0.000 → **0.145** — a 37%
reduction. Consistent in direction with the literature; the absolute level is far from
solved, and we have no data point between 0% and 25%. **W3-E3 fills that in.**

### 3.2 Update support: the single most important number in this document

| Method | New knowledge learned | Prior knowledge lost (NQ) |
|---|---|---|
| Full finetuning | baseline | **89%** |
| LoRA | same | **71%** |
| **Sparse memory finetuning** (top-T slots by session-vs-background usage) | same | **11%** |

arXiv:2510.15103 [S]. **Sparsity of the update's support, not its magnitude or rank,** is
what buys retention. This is why Prophet's Tier 2 is what it is, and it is the empirical
anchor of the whole memory design.

### 3.3 Schedules and recipes

- **LR re-warming + re-decaying** is the other half of Ibrahim's recipe; without it
  continual pretraining under-adapts (arXiv:2403.08763 [S]).
- **Infinite / WSD-style schedules** let you stop at any point in the constant phase with a
  short anneal — structurally the right shape for interruptible Colab sessions and for a
  repeated sleep schedule (arXiv:2503.02844 [S]). Prophet already runs WSD
  (`prophet/train/schedule.py`).
- **Data-blend design matters more than the algorithm**: NVIDIA's recipe reports **+9%
  average accuracy** over naive continued pretraining on a well-trained 15B model, and
  claims validity from 100B to 1T continued-training tokens (arXiv:2407.07263 [S]).

### 3.4 What does *not* work: weight editing at scale

- Editing degrades general abilities: performance on unrelated tasks trends **downward with
  the number of edits**; one method (KN) collapsed to **near zero on all tasks after a
  single edit**; and models are "not robust to weight perturbations even if **less than 1%**
  of parameters are edited" (arXiv:2401.04700 [S]).
- Sequential editing drives model collapse (arXiv:2406.11263 [S]).
- Edits do not propagate: injecting one fact leaves its logical consequences un-updated,
  which the RippleEdits benchmark (5K edits) shows current methods largely fail
  (arXiv:2307.12976 [S]).

**Conclusion for §4:** direct parameter editing of specific facts is off the table. Any
weight-touching tier must be (i) distillation-shaped, not edit-shaped, (ii) gated by a
held-out acceptance test, and (iii) reversible.

### 3.5 Sparsity, modularity, merging — briefly, because R03 covered them

- **LoRA**: learns less, forgets less †; but intruder dimensions make it a poor continual
  substrate (arXiv:2410.21228 [S]) and forgetting still follows a power law in tuned
  parameters (arXiv:2401.05605 [S]).
- **Merging as consolidation** (task arithmetic arXiv:2212.04089 †, TIES arXiv:2306.01708 †,
  DARE arXiv:2311.03099 †): DARE's finding that **90–99% of a finetuning delta can be dropped
  and rescaled with little loss** † is what makes a per-user weight delta fit in megabytes.
  Merging is a *fusion* operator, not a consolidation operator — it has no notion of which
  knowledge is worth keeping. Use it for storage compression of Δ, not as the third tier.
- **MoE as natural modularity**: attractive in principle; ruled out for the memory path by
  R05's finding that the iPhone's ANE cannot do per-token expert gathers.
- **Modular latent memory** (arXiv:2605.28889 [S]) distils each context into its own LoRA
  adapter with retrieval + routing + a self-gate. Elegant; it converts our capacity problem
  into an adapter-management problem and multiplies inference-time complexity. Not for a
  phone.

---

## 4. The missing third tier

### 4.1 The specification, restated from the evidence

From §1.2 (capacity), §1.5 (interference vs capacity), §2.1 (CLS-updated), §2.4 (ledger is
bounded; facts belong in it), §3.4 (editing fails) and §2.4's negative result
(consolidation harms when ungated), the third tier's job description is forced:

> **Tier 3 does not store knowledge. It learns how to *reach* and *use* what Tier 2 stored,
> and it distils only regularities that recur across episodes, are consistent with the
> trunk, and have been verified.**

The missing piece is not weights-in-general. It is **generalisation of retrieval**: the
step that turns *"the ledger contains it"* into *"the ledger is reachable from a query the
system has never seen"*. That is what makes a lookup table into a skill, and it is
measurable (§5).

### 4.2 The ladder: three rungs of increasing power and increasing risk

Each rung must earn the next by improving the skill ratio σ (§5). Rungs 1 and 2 preserve
properties the current design depends on; rung 3 spends them.

| Rung | What it changes | Backprop? | Runs on iPhone? | Trunk touched? | Risk |
|---|---|---|---|---|---|
| **2.5a — generative rehearsal into the ledger** | ledger `values` | **no** | **yes** | no | low |
| **2.5b — addressing consolidation** | ledger `query`, `query_norm` (1.57M params) | yes, tiny | no (Mac/5090) | no — model is **bit-identical with memory off** | low–medium |
| **3 — the cortex pass** | ~3.2M trunk params + the above | yes | no | **yes** | medium–high |

**This ladder is itself the recommendation.** Rung 2.5a is nearly free and should ship
regardless. Rung 3 should not be built until §5's measurement says rung 2.5b has run out.

#### Rung 2.5a — generative rehearsal into the ledger (backprop-free)

The cheapest possible answer to "memory that only retrieves". Before writing an episode,
**generate V variations of it with the model itself** (paraphrases, question forms,
implications, a negation, a two-hop consequence), and consolidate **all of them** with the
existing closed-form write. The ledger then holds a *neighbourhood* rather than a point.

This is EntiGraph (arXiv:2409.07431 [S]) and Self-Synthesized Rehearsal
(arXiv:2403.01244 [S]) applied to a memory layer, and it is justified by the strongest
single result in the memory-versus-skill literature: **knowledge must be augmented at
write time to be extractable at read time** — bioS single-form pretraining yields **9.7%**
QA extraction accuracy; multi5+permute augmentation yields **96.6%**
(arXiv:2309.14316 [S]).

Implementation cost: **zero new machinery.** `consolidate()` already takes a list of
`Episode`s; rung 2.5a widens the list. The only new code is the generator and the
episode-level replay accounting.

Costs to watch: V× slot consumption (65,536 slots is not infinite), and self-generated
errors entering permanent memory (mitigated by provenance gating, §6).

#### Rung 2.5b — addressing consolidation (the actual third tier, in its safe form)

Train **only** the ledger's query path — `ProductKeyMemory.query` (a
`d_model × n_heads·memory_dim` linear, **1536 × 1024 = 1.57M params** at the 500M probe
config [derived]) and `query_norm` (512 params) — so that *variations of a consolidated
episode address the slots that hold it*.

Keys stay frozen — that invariant is load-bearing and is not being relaxed. But note the
hazard this rung introduces, which is not in any paper and is the sharpest technical point
in this document:

> **Training `W_q` re-addresses every association already written.** It is the same failure
> mode as unfreezing the keys, arriving through a different door, and it is **invisible in
> the loss**: the new material fits beautifully while old recalls silently point at the
> wrong slots.

The defence is an explicit addressing-preservation term. Let `a(h) ∈ Δ^{H·k}` be the
top-k weight vector produced by `address()` and `I(h)` its slot indices. For a replay set
of previously consolidated probes `{h_j}` with **stored** `(I_j, a_j)`:

```
L_keep = Σ_j  D_KL( a_j ‖ a_θ(h_j) restricted to I_j )  +  μ · (1 − Jaccard(I_θ(h_j), I_j))
```

The second term is non-differentiable and enters as a monitored constraint (reject the
update if mean Jaccard falls below 0.8), not as a gradient.

The training objective for the rung itself:

```
L_addr =  Σ_e Σ_{v ∈ variations(e)}  ‖ m_θ(h⁻(x_e^v)) − t̄_e ‖²        # reach the episode's slots from any variation
L      =  L_addr + γ · L_keep,        γ = 1.0
```

where `t̄_e` is the episode-level consolidated target (the mean of `λ(h⁺−h⁻)` over the
episode's variations) and `m_θ` is the ledger read with the trained query path. `values`
are **frozen** during this rung — we are learning the index, not the content.

Why this is the safe form: with the ledger disabled the model is **bit-identical** to the
base checkpoint. Rung 2.5b cannot forget anything the trunk knows, by construction. The
only thing it can damage is the ledger's own addressing, and `L_keep` plus the Jaccard
constraint is a direct, cheap defence.

#### Rung 3 — the cortex pass

Only now do we touch the trunk. Four decisions.

**(a) What gets distilled — the eligibility gate.** An episode cluster is eligible iff:

1. **Recurrence** — its slot support has been written by **≥ m distinct episodes**
   (default `m = 3`). Implements "structure comes from interleaving, not from episodes"
   (§2.1). A single episode never becomes a weight. `ProductKeyMemory.write_counts` already
   gives the per-slot counter; the per-episode support sets need to be logged alongside.
2. **Internal generalisation** — split the cluster's variations into fit/holdout; the
   cluster is eligible only if the ledger, written from the fit half, already reduces loss
   on the **holdout** half. If the memory does not generalise inside its own tier, moving it
   into weights will not fix that — it will spend trunk capacity on one fact.
3. **Consistency with the trunk** — the teacher/student KL gap is below a ceiling. A cluster
   that contradicts the trunk wholesale is an *exception*; per CLS-updated (§2.1) exceptions
   consolidate slowly or never. Exceptions stay in the ledger permanently. **This is a
   feature, not a limitation.**
4. **Verified provenance** — user-authored or checked. The repo already has this concept:
   `DepthEpisode.verified` with `require_verified=True` by default.

**(b) Into which parameters.** Three subsets, ~3.2M parameters total at d=1536
(**1.4% of a 229M model, 0.64% of 500M** [derived]) — deliberately at the flat end of the
forgetting power law (§1.6):

| Subset | What | Params (d=1536) [derived] | Why |
|---|---|---|---|
| **A** | ledger `query` + `query_norm` (rung 2.5b) | 1.57M | learns class-level addressing |
| **B** | sparse **rows** of the FFN down-projection in the **two blocks following each ledger insertion point**, selected top-T by session-vs-background activation (TF-IDF, per arXiv:2510.15103) | T=1024 rows × 1536 = 1.57M | this is where "read" becomes "used" — integration, not storage |
| **C** | all RMSNorm gains in the trunk | 16 × 2 × 1536 ≈ 49k | the cheapest possible fix for the *task-alignment* component of forgetting (arXiv:2501.13453 [S]) |

**Explicitly excluded:** anything inside the recurrent core (§1.6 — perturbations are
applied `k` times and scale with the user's depth dial), embeddings, and the frozen
sub-keys. Bottom ⌊L/3⌋ blocks are frozen outright, per the "Freezing" result in
arXiv:2501.13453 [S] — free, and it is the highest-value/lowest-cost intervention found in
the entire 2025–26 literature.

**(c) The objective.**

```
L_total = L_distil + β·L_anchor + γ·L_keep      (β = 1.0, γ = 1.0)
```

**L_distil — the memory→weights term.** Token-level forward KL from a memory-on teacher to
a memory-off student, over the generated variation probes:

```
teacher:  p_T(·|x_<t) = softmax( f_{θ0}(x ; ledger ON, evidence in context, warm Tier-1 state) )
student:  p_S(·|x_<t) = softmax( f_θ  (x ; ledger OFF, no evidence, cold state) )

L_distil = E_x Σ_t KL( p_T(·|x_<t) ‖ p_S(·|x_<t) )
```

This is context distillation (arXiv:2209.15189 [S]) with the *memory* in the teacher's
context slot rather than a prompt. Full-distribution KL, not hard labels — consistent with
the project's finding that distillation transfers far more per token than next-token
prediction (`docs/00_PROBLEM_LANDSCAPE.md` §6). Teacher logits are computed **once** and
cached as top-64 log-probs (`64 × 6 bytes/token`; 4.1M probe tokens ⇒ **1.5 GB** on disk
[derived]).

*Optional, ablate:* the on-policy variant — sample continuations from the **student**,
score them under the teacher (arXiv:2601.19897 [S]). Removes the off-policy distribution
gap that is SDFT's stated cause of forgetting. Costs one extra generation pass.

**L_anchor — don't move where you shouldn't.** Self-distillation to the *pre-pass* model on
a generic replay stream:

```
L_anchor = E_{x ~ replay} Σ_t KL( p_{θ0}(·|x_<t) ‖ p_θ(·|x_<t) )
```

This is Learning-without-Forgetting (arXiv:1606.09282 [S]) at LLM scale, and it is
strictly stronger than replaying the LM loss because it constrains the whole output
distribution rather than one label per position. Replay fraction **ρ = 0.25 by tokens**,
because a personal corpus is a *strong* distribution shift and §3.1's strong-shift number
is 25%. Sweep {0, 0.05, 0.25} in W3-E3 — the literature's weak-shift answer is 5%, and if
5% suffices here it saves 28% of the pass.

**L_keep — protect Tier 2's address space.** As in rung 2.5b, but now it matters *more*:
the ledger is addressed by `h⁻`, which is a function of θ. **Moving the trunk invalidates
addressing even if `W_q` is frozen.** This term is not optional in rung 3.

**Post-pass ledger repair (no backprop).** After the weight update, re-run the existing
closed-form `ledger.write()` on the replay probes with `target = read_before_update` —
pulling the ledger's read back onto its previous function under the new trunk. Two forward
passes and a scatter-add, ~2% of pass cost. The primitive already exists.

**(d) The merge gate — the answer to arXiv:2605.12978.** The delta Δθ is held **separate
from θ0** and merged only if **all** hold on a held-out battery:

| Gate | Threshold |
|---|---|
| General BPB cost | `ΔBPB_general ≤ +0.005 bits/byte` (absolute) |
| Skill gain (§5) | `σ_holdout ≥ σ_before + 0.10` |
| Ledger integrity | `recall_error` increase `≤ 0.05`; mean addressing Jaccard `≥ 0.8` |
| Safety | no increase in the injected-instruction behaviour-change rate (§6) |

Otherwise **discard Δθ, keep the ledger, log the rejection**. A rejected pass is a normal
outcome, not a bug: it is the mechanism by which we avoid the 54%-ARC-regression failure.

**Storage and reversibility.** Δθ stored as int8 + per-row scale, DARE-pruned at 90%
(arXiv:2311.03099 †): 3.2M params → 320k surviving values × (1 byte + 4-byte index) ≈
**1.6 MB per user** [derived]. Reversible (un-merge), erasable (drop the file), auditable
(which pass wrote which rows).

### 4.3 When it runs

Never after every session — that is precisely the schedule that degraded utility in
arXiv:2605.12978 [S]. Trigger a cortex pass when **all** of:

1. ≥ `N_ep` (default 2,000) new consolidated episodes since the last pass;
2. ledger write-entropy has *risen* (`occupancy()` already reports it) — new material is
   diverse, not repetitive;
3. the recurrence gate finds ≥ `K` (default 50) eligible clusters;
4. the device is idle and charging (rungs 2.5b/3 are a training job; rung 2.5a is not).

### 4.4 Compute — this must run in hours, and it does

A100 at 312 TFLOP/s bf16, MFU 0.35 ⇒ **1.09e14 FLOP/s effective** (the project's own
constants, `prophet/scaling.py`). Pass cost = generation + teacher forward (once, cached) +
student forward/backward × epochs, with `1/(1−ρ)` replay overhead:

```
FLOPs ≈ 4·N·D_p            (teacher, evidence-extended context, once)
      + 6·N·(D_p + D_r)·E  (student fwd+bwd, E epochs)
```

| Scenario | N | probe tokens `D_p` | ρ | epochs | Compute | Generation | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| mini, 2k episodes × 8 variations × 256 tok | 229M | 4.1M | 0.25 | 3 | 2.6e16 → 240 s | ~205 s | **0.13 A100-h** |
| 350M, 10k episodes × 8 × 256 | 350M | 20.5M | 0.25 | 3 | 2.0e17 → 1836 s | ~1025 s | **0.8 A100-h** |
| 500M, 50M probe tokens (heavy) | 500M | 50M | 0.25 | 3 | 7.0e17 → 6420 s | ~3333 s | **2.7 A100-h** |

All [derived]. **A cortex pass costs ~0.1–3 A100-hours — 0.04%–1% of the 300-hour project
budget per pass.** Cost is not the reason to be cautious about the third tier. Damage is.

**The property we lose.** Rungs 2.5b and 3 require backprop, so **they cannot run on the
iPhone.** The current design's headline claim — "no gradient through the backbone at any
point, which is what makes this runnable on a phone" — survives only for rung 2.5a. Rungs
2.5b/3 run on the user's Mac/5090 or on our servers, exporting a 1.6 MB delta. That is a
real cost of the third tier and it must be stated in `docs/06_MEMORY.md` if the tier ships.

### 4.5 PyTorch sketch

Proposed as `prophet/memory/cortex.py`. **Not created** — this report is the only file this
track wrote. The sketch follows the conventions already in `consolidate.py`: the ledger is
applied externally to the hidden state (`hidden + lam * ledger(hidden)`), logits via
`model._project`, exactly as `depth_agreement()` does today.

```python
"""Tier 3: distilling memory into weights. The slow half of a complementary system.

Tier 2 stores. This turns stored content into reachable competence: it trains a small,
explicitly enumerated parameter subset so that (a) variations of a consolidated episode
address the slots holding it, and (b) the read is integrated rather than merely present.

It deliberately does NOT move knowledge into the trunk. At ~2 bits per parameter a 229M
model holds ~57MB of extractable knowledge, all of it already spent; the ledger holds
50MB at int4 for zero marginal FLOPs. Facts stay in the ledger. Skills move.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prophet.memory.ledger import ProductKeyMemory
from prophet.memory.consolidate import Episode


@dataclass
class CortexConfig:
    lam: float = 1.0
    replay_ratio: float = 0.25      # strong-shift number from 2403.08763; sweep {0,.05,.25}
    beta_anchor: float = 1.0        # weight on self-distillation to the pre-pass model
    gamma_keep: float = 1.0         # weight on ledger-addressing preservation
    epochs: int = 3
    lr: float = 1e-4
    top_rows: int = 1024            # sparse FFN rows, TF-IDF selected (cf. 2510.15103)
    freeze_bottom_frac: float = 1/3 # cf. 2501.13453 "Freezing"
    min_episodes_per_cluster: int = 3
    max_bpb_cost: float = 0.005     # merge gate
    min_sigma_gain: float = 0.10    # merge gate (see section 5)
    min_jaccard: float = 0.80       # merge gate: ledger addressing must survive


def trainable_subset(model: nn.Module, ledger: ProductKeyMemory,
                     cfg: CortexConfig, hot_rows: Tensor) -> list[nn.Parameter]:
    """A: ledger query path. B: sparse FFN rows after the ledger. C: norm gains.

    Nothing inside the recurrent core: a weight-shared core applies every perturbation
    ``k`` times, so an adapter there has an interference budget that moves with the
    user's depth dial. That constraint is ours, not the literature's.
    """
    params = [ledger.query.weight, *ledger.query_norm.parameters()]           # A
    for name, p in model.named_parameters():                                  # C
        if name.endswith("norm.weight") and not _in_core(name):
            params.append(p)
    for p in _ffn_down_rows_after_ledger(model, hot_rows):                    # B
        params.append(p)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    return params


@torch.no_grad()
def teacher_logprobs(model, ledger, probes, cfg, top: int = 64):
    """Memory-on teacher, computed once and cached as top-k log-probs (~6 bytes/token)."""
    out = []
    for x in probes:
        h = model(x.tokens, return_mtp=False).hidden
        h = h + cfg.lam * ledger(h)                       # the memory the student must absorb
        lp = F.log_softmax(model._project(h).float(), dim=-1)
        v, i = lp.topk(top, dim=-1)
        out.append((i, v - torch.logsumexp(v, -1, keepdim=True)))  # renormalise over top-k
    return out


def cortex_pass(model, ledger, clusters, replay_text, addr_probes, cfg=CortexConfig()):
    """One sleep. Returns a delta that is merged only if the gate passes."""
    theta0 = copy.deepcopy(model).eval().requires_grad_(False)   # the anchor
    probes = [v for c in clusters if len(c.episodes) >= cfg.min_episodes_per_cluster
                for v in c.variations]                          # recurrence gate
    cached = teacher_logprobs(model, ledger, probes, cfg)
    stored_addr = [ledger.address(h) for h in addr_probes]       # (idx, w) before the pass

    params = trainable_subset(model, ledger, cfg, _tfidf_rows(ledger))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    for _ in range(cfg.epochs):
        for batch, (t_idx, t_lp) in _interleave(probes, cached, replay_text,
                                                cfg.replay_ratio):
            # (1) memory -> weights: student runs with NO memory and NO evidence
            h = model(batch.tokens, return_mtp=False).hidden
            s_lp = F.log_softmax(model._project(h).float(), -1)
            l_distil = -(t_lp.exp() * s_lp.gather(-1, t_idx)).sum(-1).mean()

            # (2) anchor: stay the model you were, on generic data
            with torch.no_grad():
                a_lp = F.log_softmax(theta0(batch.replay, return_mtp=False).logits.float(), -1)
            r_lp = F.log_softmax(model(batch.replay, return_mtp=False).logits.float(), -1)
            l_anchor = F.kl_div(r_lp, a_lp, log_target=True, reduction="batchmean")

            # (3) keep the ledger reachable. Moving theta re-addresses everything already
            #     written, and nothing in the loss curve would tell you.
            l_keep = 0.0
            for h_j, (idx_j, w_j) in zip(addr_probes, stored_addr):
                _, w_new = ledger.address(h_j)
                l_keep = l_keep + F.mse_loss(w_new, w_j)

            (l_distil + cfg.beta_anchor * l_anchor + cfg.gamma_keep * l_keep).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

    # (4) backprop-free ledger repair: pull the read back onto its pre-pass function
    with torch.no_grad():
        for h_j, before in zip(addr_probes, [ledger(h) for h in addr_probes]):
            ledger.write(h_j, before)

    delta = {k: (v - dict(theta0.named_parameters())[k]).detach()
             for k, v in model.named_parameters() if v.requires_grad}
    return delta if _gate_passes(model, theta0, ledger, cfg) else None   # else: keep Tier 2
```

### 4.6 Is the third tier a bad idea? — the honest answer

**In the form the working thesis proposes: yes, it is a bad idea.** "Drain the ledger into
weights so it stops growing" fails on four counts, three of them quantified:

1. **Capacity.** 2 bits/param ⇒ ~57 MB of extractable knowledge in a 229M trunk, already
   spent. The ledger holds 50.3 MB at int4 **at zero marginal FLOPs**. Moving a fact from
   ledger to trunk trades a free byte for an expensive one and evicts something else
   [derived from arXiv:2404.05405 [S]].
2. **Evidence.** The only 2026 study of repeated LLM memory consolidation found utility
   rising and then falling **below the no-memory baseline**, with an episodic-only control
   remaining competitive (arXiv:2605.12978 [S]).
3. **Property loss.** It ends "no backprop, ever" — the claim that makes on-device
   personalisation possible. The phone cannot run it.
4. **The premise is wrong.** The ledger does not grow without bound; `n_slots` is fixed and
   the per-user delta is capped. The real problem is *capacity pressure among episodes*,
   whose first-line fixes are eviction and abstraction, not weights.

**In the restricted form of §4.2: no, it is a good idea, and it is cheap.** Rung 2.5a costs
one generation pass and no new machinery. Rung 2.5b touches 1.57M parameters, cannot
damage the trunk by construction, and attacks the exact failure the track cares about
(retrieval that does not generalise). Rung 3 costs 0.13–2.7 A100-hours and is gated,
bounded and reversible.

**Kill conditions, pre-registered:**

- If R03's **E2 gate fails** (context-cleared recall does not beat RAG at equal context
  budget), Tier 3 is dead — there is nothing worth consolidating. Tier 3 is strictly
  downstream of E2 and must not be funded before it.
- If **W3-E0 finds σ already ≥ 0.5** for the existing ledger, the premise is wrong and
  nothing here needs building.
- If **W3-E1 (rung 2.5b) does not raise σ by ≥ 0.15**, do not proceed to rung 3: if
  learning the index does not generalise retrieval, learning the trunk will not either, and
  rung 3 only adds the ability to break things.
- If **W3-E4 reproduces the rise-then-fall curve** of arXiv:2605.12978 under our own gating,
  ship rung 2.5a only and publish the negative result.

---

## 5. Memory versus skill

> *"A memory that only ever retrieves is not learning."* — the part of the brief that
> matters most, and the one thing in this document that is genuinely ours to define.

### 5.1 Why this is a real failure and not a philosophical worry

Three independent literatures say a stored fact is not a usable fact:

- **Storage ≠ extraction.** Pretraining on single-form biographies gives **9.7%** QA
  extraction; the *same knowledge* with paraphrase/permutation augmentation gives **96.6%**
  (arXiv:2309.14316 [S]). A ledger written from one surface form is the 9.7% case by
  construction.
- **Extraction ≠ manipulation.** Models that retrieve a stored attribute perfectly fail at
  the simplest classification or comparison over it without chain-of-thought at *both*
  train and inference time, and inverse search is **virtually 0%** regardless of prompt
  (arXiv:2309.14402 [S]).
- **Edits do not propagate.** An injected fact leaves its consequences un-updated; the
  RippleEdits benchmark (5K edits) shows current methods largely fail on them
  (arXiv:2307.12976 [S]).

### 5.2 The measurement: the skill ratio σ

Build **generative families**. A family `C` is a rule plus instances: an entity with a
schema of attributes and their implications; a unit-conversion rule; a small grammar; a
modular-arithmetic modulus; a synthetic API whose calls follow a pattern. Split each family
into consolidated instances `S` and **held-out instances of the same family** `H`, disjoint.

Everything is measured in **bits per byte**, not accuracy — mandatory at our scale, because
R11 established that most benchmarks are at chance below 500M parameters.

```
g_recall   = BPB_off(S) − BPB_on(S)          # did it store what we wrote?
g_transfer = BPB_off(H) − BPB_on(H)          # does it help on cases we never wrote?
leak       = BPB_off(G) − BPB_on(G)          # G = general validation. Should be ≈ 0.

                g_transfer
       σ  =  ───────────────
              max(g_recall, ε)
```

`on/off` means: ledger read enabled/disabled (Tiers 2, 2.5), or after/before the pass with
memory disabled in both (Tier 3). Both give a σ on the same scale, so the rungs are
directly comparable — which is the point.

| σ | Interpretation |
|---|---|
| ≈ 0 | **Lookup table.** Content is reachable only from the exact form written. This is the failure the brief names. |
| 0.2–0.6 | Partial induction: some class structure, form-sensitive. |
| ≈ 1 | The system learned the **rule** as well as it learned the instances. |
| > 1 | Genuine abstraction — transfer exceeds recall. Not expected; would be a result. |

**σ(n), and the consolidation threshold.** Plot σ against the number of consolidated
instances `n` per family. A lookup table's σ(n) is flat near zero at any n. A system that
induces rules shows σ rising with n. Report **n½ = the smallest n with σ ≥ 0.5**. This is
the direct analogue of the diversity threshold in the Physics-of-LM result and it is the
number that says whether Prophet's memory can ever become competence.

### 5.3 Localising the failure: two decompositions

σ alone says *whether*, not *where*. Two cheap diagnostics say where:

1. **Address recall** `J = Jaccard( top-k slots(h⁻(H)), top-k slots(h⁻(S)) )`. If `J ≈ 0`,
   held-out variations do not even reach the right slots — the failure is **addressing**,
   and rung 2.5b is the fix.
2. **Oracle-addressing gain.** Recompute `g_transfer` while *forcing* the slot set from `S`.
   If oracle addressing recovers the transfer gain, the content generalises and only the
   index does not (fix: rung 2.5b). If it does not, the stored content is instance-specific
   (fix: rung 2.5a — write variations, not points), and no amount of weight training will
   help.

This decomposition is what converts "a memory that only retrieves" from a complaint into a
two-way diagnosis with a different repair on each branch.

### 5.4 Controls, because σ is easy to fake

| Control | Guards against |
|---|---|
| **C1** no-memory baseline | trivially attributing base-model competence to memory |
| **C2 "dream-only"** — run the variation generator, write **nothing** | the generation step, not the memory, producing the gain |
| **C3 counterfactual family** — consolidate a *false* variant | memorisation masquerading as rules: a memoriser shows zero transfer to the true family; a system that induced a rule shows **negative** transfer (interference). Negative transfer here is *evidence of learning*. |
| **C4 unrelated-family leakage** | writes that improve everything (a scale/norm artefact) |
| **C5 RAG at equal context budget** | the R03 protocol: A5 must beat A2 at **0 context tokens** |
| **C6 retention** — re-measure σ after 5/10/20 unrelated cycles | the interesting prediction: **rules should survive cycles that erase entries.** If σ decays as fast as g_recall, nothing was abstracted. |

### 5.5 What already exists in the repo

`recall_error()` and the sibling track's `depth_transfer_error()` are the **residual-space**
analogues of `g_recall` and `g_transfer` — σ can be computed today, for free, as
`1 − depth_transfer_error(H) / (1 − depth_transfer_error(S))`-shaped ratio on residual norms.
Use it as the cheap inner-loop proxy, but **report σ in BPB**: the repo's own
`depth_agreement()` docstring makes the right point — residual measurements can improve
without the model's output changing. Both numbers, always, with BPB as the one that decides.

---

## 6. Long-horizon failure modes

| # | Failure | Evidence | Our exposure | Mitigation |
|---|---|---|---|---|
| 1 | **Consolidation degrades utility.** Memory usefulness rises, then falls below no-memory; even ground-truth-derived consolidation caused a **54% regression** on previously-solved ARC problems; the episodic-only control stayed competitive | arXiv:2605.12978 [S] | **Direct.** This is our design being described. | Recurrence + verification gate (§4.2a); pass-level **merge gate** with rejection; a permanent "consolidation off" arm in every evaluation; never fire after every session |
| 2 | **Capability erosion under self-evolution** across workflow, skill, model *and* memory channels | arXiv:2605.09315 [S] | Direct, at deployment horizon | W3-E4's 20-cycle protocol with BWT reported per tier; kill-switch on Δ |
| 3 | **Ledger becomes a lookup table** — memorised answers that never generalise | arXiv:2309.14316, 2309.14402, 2307.12976 [S] | **Direct and probably already true** | σ measured continuously (§5); rung 2.5a writes variations, not points |
| 4 | **Addressing invalidation.** Training `W_q` (rung 2.5b) or the trunk (rung 3) silently re-addresses everything already written | none — **ours, newly identified** | Direct, and **invisible in the loss** | `L_keep` term; Jaccard ≥ 0.8 merge gate; keys stay frozen forever |
| 5 | **Memory poisoning.** >80% attack success at <0.1% poison rate, <1% benign impact, no model training required | AgentPoison, arXiv:2407.12784 [S] | Tier 2 is offline-written, so lower than an agent's RAG store — but **Tier 3 raises the stakes**: poison in weights is far harder to erase than poison in slots | Provenance gating (user-authored / verified only); `web` namespace never consolidated; single-episode material never distilled; retrieved memory treated as data, never instructions; Δ is a separate, deletable file |
| 6 | **Model collapse from self-generated data.** Recursive training on generated content erases distribution tails, irreversibly | Shumailov et al., *Nature* 631:755–759, 2024 [S]; "curse of recursion" arXiv:2305.17493 † | Rung 2.5a and the dreaming step **are** recursive generation | Every pass anchored on **real** data (`L_anchor`, ρ≥0.05); generated variations never exceed a fixed fraction of pass tokens; monitor ledger write-entropy for tail loss (`occupancy()` already reports it) |
| 7 | **Drift.** Repeated closed-form writes with per-slot EWC and decay < 1 move value norms and collapse write entropy | design analysis [ours] | Direct | Track a **ledger half-life**: cycles until `recall_error` on a fixed probe set doubles. Report it in W3-E4. Entropy collapse does not show up in any loss curve |
| 8 | **Weight-edit collapse.** General abilities trend down with edit count; <1% of parameters perturbed already hurts; sequential edits collapse the model | arXiv:2401.04700, 2406.11263 [S] | Rung 3 only | Distillation-shaped updates, never edit-shaped; ≤1.4% of params; bottom third frozen; merge gate on general BPB |
| 9 | **Stale and contradictory facts** | — | Direct | The closed-form write is a delta rule: it erases what the slot already predicts before writing, so contradiction handling is intrinsic. Add an explicit `erase()` (Larimar-style †) for "forget this" |
| 10 | **Evaluation illusion** — reporting long-context wins as memory | arXiv:2410.10813 † and R03 §3.4 | Direct | Every claim uses **write → clear context → read**; the repo's `recall_error` is already defined this way |
| 11 | **Plasticity exhaustion.** Each protected pass consumes rank; §1.2 | arXiv:1910.07104, 2103.09762 [S] | Rung 3, long horizon | With 3.2M trainable params and ~10⁴ constrained probe directions per pass, the free subspace supports **~300 passes** before saturation [derived, order-of-magnitude — the constraints are soft, not hard projections]. Plan a periodic **re-base**: fold Δ into θ0, recompute the anchor, reset the protection set |

---

## 7. Ablation plan

All experiments run **on top of the existing `prophet/memory/` code**, at 50–350M
parameters, **< 6 A100-hours each**, using an already-pretrained checkpoint (these are
consolidation experiments, not pretraining runs — which is why they are cheap). Costs are
[derived] with the project's own constants (312 TFLOP/s × 0.35 MFU).

**Precondition:** R03's **E2 gate** (write → clear context → read, vs RAG at equal context
budget) must pass first. If E2 fails there is nothing to consolidate and the whole of W3 is
void. W3 asks for **13.5 A100-hours**, to be scheduled after E2, not alongside it.

| ID | Question | Setup | Cost | Success / kill |
|---|---|---|---|---|
| **W3-E0** | **Is the ledger actually a lookup table?** | 160M checkpoint, 40 synthetic families × 20 instances. Consolidate `S`, measure σ, J, oracle-addressing gain on `H`. No new code beyond the σ harness. | **0.5 h** | Establishes the baseline. **If σ ≥ 0.5 already, cancel W3-E1..E3** — the premise is false and 12 hours are saved. |
| **W3-E1** | Does **writing variations** (rung 2.5a) create class-level retrieval, backprop-free? | Same, with V ∈ {0, 2, 8} self-generated variations per episode written through the existing `consolidate()`. Report σ(V), slots consumed, and contamination from wrong variations. | **1.0 h** | Ship rung 2.5a if σ rises ≥ 0.15 at V=8 for ≤ 4× slot cost. This is the cheapest possible win and the test of arXiv:2309.14316 at our scale. |
| **W3-E2** | Does **addressing consolidation** (rung 2.5b) work, and does `L_keep` prevent it destroying Tier 2? | Train `query` + `query_norm` (1.57M params) with `L_addr + γ L_keep`, γ ∈ {0, 1}. Report σ, `recall_error` on 5,000 previously written probes, addressing Jaccard. | **1.5 h** | **Two results wanted:** σ gain ≥ 0.15, *and* γ=0 visibly destroying old recall while γ=1 does not. The second is the evidence for failure mode #4, which nobody has documented. |
| **W3-E3** | The **cortex pass**: does distilling into 3.2M trunk params beat rung 2.5b, and at what cost? | 160M. Arms: subsets {A, A+B, A+B+C} × replay ρ ∈ {0, 0.05, 0.25} × anchor {on, off} × on-policy {off, on}. Report σ, general BPB cost, forgetting matrix, merge-gate pass rate. | **4.0 h** | Ship only if σ ≥ 0.5 **and** general BPB cost ≤ 0.005 bits/byte. Also settles our own replay curve between 0% and 25%, which the literature does not give us for this setting. |
| **W3-E4** | **Twenty cycles.** Does the rise-then-fall curve of arXiv:2605.12978 reproduce under our gating? | 20 consecutive cycles over disjoint families. Arms: (i) episodic-only control (**no consolidation** — the arm that beat every consolidator in that paper), (ii) Tier 2 only, (iii) +2.5a, (iv) +2.5b, (v) full rung 3 with merge gate. Report ACC/BWT/FWT in BPB, σ retention, ledger write-entropy, ledger half-life. | **3.0 h** | **The decisive experiment of this track.** If the episodic-only control wins, ship rung 2.5a alone and publish the negative result — that is a real contribution. |
| **W3-E5** | **Poisoning and provenance.** | Inject 1% contradictory facts, 1% instruction-injection strings, 3% near-duplicate spam. Measure post-consolidation clean-family σ, injected-instruction behaviour-change rate after context clearing, and merge-gate catch rate. | **1.0 h** | Clean σ drop < 0.05; instruction-injection success 0%; gate catches ≥ 90% of poisoned passes. |
| **W3-E6** | **Depth consolidation transfer** — does the sibling track's `consolidate_depth()` produce skill or memoisation? | **350M** (R04: recurrence only wins above ~350M). Consolidate `k=16 → k=2` on `n` instances of an algorithmic family; measure σ and `depth_agreement()` on **held-out instances**. | **2.5 h** | σ ≥ 0.4 would be the strongest evidence in this project that memory can become skill — a cheap pass reproducing expensive computation on problems it never saw. σ ≈ 0 means depth consolidation is a cache, and should be described as one. |

**Total: 13.5 A100-hours.** Order: **E0 → E1 → E2 → E4 → E3 → E5 → E6**. Note the ordering
choice: **E4 (twenty cycles) runs before E3 (the cortex pass)**, on rungs 2.5a/2.5b only.
If long-horizon consolidation is already destructive at the cheap rungs, the expensive rung
must not be built. This inverts the natural "build then stress-test" order deliberately,
because arXiv:2605.12978 says the stress test is where consolidation dies.

---

## 8. References

Verification key: **[S]** = title/ID confirmed via search-result summary this session, paper
not read (all primary hosts blocked, §0); **†** = prior knowledge, ID unverified this session.

**Mechanism of forgetting**
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks* (EWC). arXiv:1612.00796 †
- Farajtabar, Azizan, Mott, Li. *Orthogonal Gradient Descent for Continual Learning.* arXiv:1910.07104 [S], AISTATS 2020
- Saha, Garg, Roy. *Gradient Projection Memory for Continual Learning.* arXiv:2103.09762 [S], ICLR 2021
- Doan, Bennani, Mazoure, Rabusseau, Alquier. *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix.* arXiv:2010.04003 [S], AISTATS 2021
- Riemer et al. *Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference* (MER). arXiv:1810.11910 [S]
- Lopez-Paz, Ranzato. *Gradient Episodic Memory for Continual Learning.* arXiv:1706.08840 †
- Mirzadeh, Farajtabar, Pascanu, Ghasemzadeh. *Understanding the Role of Training Regimes in Continual Learning.* arXiv:2006.06958 [S], NeurIPS 2020 — **note: commonly mis-cited as 2010.04495**
- Mirzadeh et al. *Linear Mode Connectivity in Multitask and Continual Learning.* arXiv:2010.04495 [S]
- Ramasesh, Dyer, Raghu. *Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics.* arXiv:2007.07400 [S], ICLR 2021
- Ramasesh, Lewkowycz, Dyer. *Effect of scale on catastrophic forgetting in neural networks.* ICLR 2022 [S] (OpenReview `GhVS8_yPeEa`)
- Evron, Moroshko, Ward, Srebro, Soudry. *How catastrophic can catastrophic forgetting be in linear regression?* arXiv:2205.09588 [S], COLT 2022
- Zheng, Cai, Qiu, Ma. *Spurious Forgetting in Continual Learning of Language Models.* arXiv:2501.13453 [S], ICLR 2025
- Jin, Ren. *Demystifying Language Model Forgetting with Low-rank Example Associations.* arXiv:2406.14026 [S], ICML 2025
- Kalajdzievski. *Scaling Laws for Forgetting When Fine-Tuning Large Language Models.* arXiv:2401.05605 [S]
- Shuttleworth et al. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* arXiv:2410.21228 [S]
- Luo et al. *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning.* arXiv:2308.08747 [S]
- *Mechanistic Analysis of Catastrophic Forgetting in LLMs During Continual Fine-tuning.* arXiv:2601.18699 [S] — single-source, 2026
- *Geometry Conflict: Explaining and Controlling Forgetting in LLM Continual Post-Training.* arXiv:2605.09608 [S] — single-source, 2026

**Complementary learning systems**
- McClelland, McNaughton, O'Reilly. *Why there are complementary learning systems in the hippocampus and neocortex.* Psychological Review 102(3):419–457, 1995 †
- Kumaran, Hassabis, McClelland. *What Learning Systems do Intelligent Agents Need? CLS Theory Updated.* Trends in Cognitive Sciences 20(7):512–534, 2016 [S]
- Tse et al. *Schemas and memory consolidation.* Science 316:76–82, 2007 †
- Shin, Lee, Kim, Kim. *Continual Learning with Deep Generative Replay.* arXiv:1705.08690 [S], NIPS 2017
- Arani, Sarfraz, Zonooz. *Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System* (CLS-ER). arXiv:2201.12604 [S], ICLR 2022
- Tadros, Krishnan, Ramyaa, Bazhenov. *Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks.* Nature Communications 13:7742, 2022 [S]
- Li, Hoiem. *Learning without Forgetting.* arXiv:1606.09282 [S]
- *Wake-Sleep Consolidated Learning.* arXiv:2401.08623 [S]
- *Semi-parametric Memory Consolidation: Towards Brain-like Deep Continual Learning.* arXiv:2504.14727 [S]
- Behrouz, Razaviyayn, Zhong, Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures* (HOPE). arXiv:2512.24695 [S], NeurIPS 2025
- Dorovatas et al. *Position: Modular Memory is the Key to Continual Learning Agents.* arXiv:2603.01761 [S], ICML 2026 spotlight
- *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* arXiv:2606.03979 [S] — single lab, unreplicated
- *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference.* arXiv:2605.26099 [S] — single lab, unreplicated

**Memory → weights (the third tier's precedents)**
- Snell, Klein, Zhong. *Learning by Distilling Context.* arXiv:2209.15189 [S]
- Eyuboglu et al. *Cartridges: Lightweight and general-purpose long context representations via self-study.* arXiv:2506.06266 [S]
- Yang et al. *Synthetic continued pretraining* (EntiGraph). arXiv:2409.07431 [S]
- Zweiger et al. *Self-Adapting Language Models* (SEAL). arXiv:2506.10943 † (verified by R03 in a prior session)
- *Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models.* arXiv:2508.09874 [S]
- *Self-Distillation Enables Continual Learning* (SDFT, on-policy). arXiv:2601.19897 [S]
- Yang et al. *Self-Distillation Bridges Distribution Gap in Language Model Fine-tuning.* arXiv:2402.13669 †
- Huang et al. *Mitigating Catastrophic Forgetting in LLMs with Self-Synthesized Rehearsal.* arXiv:2403.01244 [S]
- Zheng et al. *Context Distillation as Latent Memory Management.* arXiv:2605.28889 [S]
- Mu, Li, Goodman. *Learning to Compress Prompts with Gist Tokens.* arXiv:2304.08467 †

**Continual pretraining, quantified**
- Ibrahim, Thérien et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* arXiv:2403.08763 [S] — **the replay numbers**
- Béthune et al. *Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection.* arXiv:2502.06042 [S], ICML 2025 — **1% injection**
- Parmar et al. *Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models.* arXiv:2407.07263 [S], ICLR 2025
- *Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate Schedule for Continual Pre-training.* arXiv:2503.02844 [S]
- Abbes et al. *Revisiting Replay and Gradient Alignment for Continual Pre-Training of LLMs.* arXiv:2508.01908 [S], CoLLAs 2026
- Lin et al. *Continual Learning via Sparse Memory Finetuning.* arXiv:2510.15103 [S] — **11% vs 71% vs 89%**
- Biderman et al. *LoRA Learns Less and Forgets Less.* arXiv:2405.09673 †
- Wang et al. *Orthogonal Subspace Learning for Language Model Continual Learning* (O-LoRA). arXiv:2310.14152 †
- Ilharco et al. *Editing Models with Task Arithmetic.* arXiv:2212.04089 †; Yadav et al. *TIES-Merging.* arXiv:2306.01708 †; Yu et al. *Language Models are Super Mario* (DARE). arXiv:2311.03099 †

**Memory versus skill, and long-horizon failure**
- Allen-Zhu, Li. *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction.* arXiv:2309.14316 [S], ICML 2024 — **9.7% → 96.6%**
- Allen-Zhu, Li. *Physics of Language Models: Part 3.2, Knowledge Manipulation.* arXiv:2309.14402 [S] — **inverse search ≈ 0%**
- Allen-Zhu, Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* arXiv:2404.05405 [S], ICLR 2025 — **2 bits/param, int8 included**
- Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models* (RippleEdits). arXiv:2307.12976 [S]
- Gu et al. *Model Editing Harms General Abilities of LLMs: Regularization to the Rescue.* arXiv:2401.04700 [S], EMNLP 2024
- *Understanding the Collapse of LLMs in Model Editing.* arXiv:2406.11263 [S]
- Zhang et al. *Useful Memories Become Faulty When Continuously Updated by LLMs.* arXiv:2605.12978 [S] — **the strongest argument against the third tier**
- Yu et al. *Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation.* arXiv:2605.09315 [S]
- Chen, Xiang et al. *AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.* arXiv:2407.12784 [S], NeurIPS 2024 — **>80% ASR at <0.1% poison rate**
- Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI models collapse when trained on recursively generated data.* Nature 631(8022):755–759, 2024 [S]; *The Curse of Recursion.* arXiv:2305.17493 †
- Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* arXiv:2410.10813 †

**In-repository**
- `prophet/memory/ledger.py` — product-key ledger, closed-form write, frozen keys, trust region, EWC-lite, occupancy entropy
- `prophet/memory/consolidate.py` — context-axis consolidation (`consolidate`, `recall_error`) and depth-axis consolidation (`consolidate_depth`, `depth_transfer_error`, `depth_agreement`, added by the sibling depth track during this session)
- `prophet/memory/session.py` — Tier 1 serialisation with model fingerprinting
- `docs/06_MEMORY.md` — measured results [ours]; `docs/research/R03_memory_continual_learning.md` — the two-tier design this track extends
