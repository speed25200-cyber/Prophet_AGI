# W2 — The Transformer Expressivity Wall: What Our Architecture Escapes, and What It Only Rearranges

**Track:** W2 · **Status:** research complete, decision-ready · **Date:** 2026-09-03
**Scope:** circuit-complexity limits of fixed-depth transformers; what depth-recurrence
(looping a weight-shared core) provably buys; what bounded-state mixers (gated DeltaNet)
provably give up; and an honest audit of Prophet's stack against both.

---

## Summary — four findings, in order of what they cost us

1. **Looping at constant `k` buys no complexity class at all.** A `c`-layer core looped `k`
   times is a depth-`c·k` network; constant `k` means constant depth means uniform TC⁰.
   The class only changes when `k` grows with the input — `Θ(log n)` → TC¹, `Θ(logᵈ n)` →
   TCᵈ, polylog + padding → NC ([2505.18948](https://arxiv.org/abs/2505.18948)). Prophet
   samples `k ~ log-uniform[1,8]` **independently of the input** and defaults to `k = 4`.
   As configured, our central architectural bet is a constant-factor depth multiplier.
   (Track W1 reached this independently; §2.6 says what W2 adds.)

2. **Our gated DeltaNet is in the provably weakest class of bounded-state mixers, and the
   fix is one character.** `layers.py:462` sets `β = sigmoid(·) ∈ (0,1)`, so the transition
   `α(I − βkkᵀ)` has eigenvalues `{α, α(1−β)}` — all strictly positive. Grazzi et al.
   ([2411.12537](https://arxiv.org/abs/2411.12537)) prove positive-eigenvalue linear RNNs
   **cannot express parity**. I ran it on our own recurrence (§6.0): with `β ∈ (0,1)`,
   parity at 4× the training length is **0.44–0.54 — chance**, and a second layer does not
   help. With `β = 2·sigmoid(·)`, same seeds and budget: **0.92–0.95**. Zero parameters,
   zero FLOPs, one character. It is the *only* change in the whole architecture with a
   proof attached that it lifts us above a same-parameter dense transformer.

3. **Attention rescues recall, not state tracking — our docs conflate the two.** Attention
   is TC⁰-computable, so no number of full-attention layers can add serial-computation
   power. What they do add is exact recall, and the ratio matters: our effective
   linear:full ratio *degrades from 1:1 at `k=1` to 8:1 at `k=8`*, crossing outside the
   3:1–6:1 band that [2507.06457](https://arxiv.org/abs/2507.06457) measured as safe. **The
   depth dial and the recall budget are the same dial, pulling opposite ways** — never
   written down before now.

4. **The shipped config does not implement the documented architecture.** In
   `configs/prophet_500m_probe.json` the prelude and coda are pure GDN, the *only*
   full-attention layer is **inside the looped core**, and `nope_layers` is empty — so
   Decision D1's KV-invariant is violated and there is no SWA and no NoPE anywhere (§5.1).
   Fix before any ablation runs, or every W2 result is confounded.

**Single sharpest test:** W2-A4 (multi-key associative recall vs state size vs `k`). It is
the one experiment that can show the depth dial does nothing for the failure mode our own
R02 already flagged (89.8 % single-needle → 37.8 % multi-needle), and that turning `k` up
makes it worse. Cost: ~1 A100-hour.

---

## 0. Provenance and how to read the citations

**Egress was restricted for this session.** The following hosts returned `403` from the
organisation's egress proxy and could not be fetched, by `curl` or by the fetch tool:

```
arxiv.org        export.arxiv.org   ar5iv.labs.arxiv.org   huggingface.co
aclanthology.org openreview.net     papers.neurips.cc      proceedings.neurips.cc
api.semanticscholar.org  www.semanticscholar.org  direct.mit.edu  www.alphaxiv.org
```

Only `github.com` was fetchable. Web *search* worked. Therefore **no paper in this report
was read in full**. Every claim about a paper's content comes from search-engine
summaries of that paper's abstract/landing page. Marking convention:

| Mark | Meaning |
|---|---|
| `[S]` | Sourced from a search-result summary of the paper this session. Title, arXiv ID and headline claim are reliable; **exact constants, table values and theorem preconditions are not re-verified.** |
| `[V]` | Verified against retrievable text (a GitHub README) or against this repository's own source code. |
| `[E]` | Produced by an experiment run during this session; script and numbers in §6.0. |
| `[T]` | Textbook complexity theory (Barrington, Håstad, Reingold, Hesse). Not from a search; standard results. |

**Rule for the project:** no `[S]` constant may gate a compute spend before someone with
arXiv access re-reads the theorem statement. The *qualitative* separations below are
robust (they are stated identically across many independent summaries); the *numbers* are
not.

---

## 1. What a fixed-depth transformer cannot compute

### 1.1 The theorems only bind a specific model of computation

Every result in this section is about a **constant-depth, polynomial-width transformer
with bounded arithmetic precision**. Change any of those three and the result changes.
This matters because our architecture changes exactly one of them (depth) and it is easy
to claim more than the theorems allow.

| Model assumption | Upper bound | Source |
|---|---|---|
| Depth `O(1)`, precision `O(log n)` bits, FFNs computable in linear space | ⊆ logspace-uniform **TC⁰** | Merrill & Sabharwal, *The Parallelism Tradeoff*, [2207.00729](https://arxiv.org/abs/2207.00729), TACL 11:531–545 `[S]` |
| Same, expressed as a logic | Every log-precision transformer is a sentence of **FO(M)** — first-order logic with majority quantifiers, which characterises uniform TC⁰ | Merrill & Sabharwal, [2210.02671](https://arxiv.org/abs/2210.02671), NeurIPS 2023 `[S]` |
| Depth `O(1)`, **constant-bit** precision | ⊆ **AC⁰** (a *proper* subset of TC⁰) | Li, Liu, Zhou & Ma, [2402.12875](https://arxiv.org/abs/2402.12875), ICLR 2024 `[S]` |
| Finite precision, single head attends to `O(1)` tokens | ⊆ a generalisation of first-order logic; strictly weaker still (cannot express uniform attention) | Chiang et al., via [2210.02671](https://arxiv.org/abs/2210.02671) `[S]` |
| Average-hard attention, masked pre-norm, **polynomial padding** | **exactly** FO-uniform TC⁰ (matching lower bound) | Merrill & Sabharwal, [2505.18948](https://arxiv.org/abs/2505.18948), NeurIPS 2025 `[S]` |

The last row is the important one and is under-appreciated in our docs. Until 2025, TC⁰
was only an *upper* bound — nobody had shown transformers reach all of it. 2505.18948
closes the gap: with polynomial padding, transformers recognise TC⁰ **exactly**. So TC⁰
is not a loose over-approximation we might beat by being clever with the same depth. It
is the actual class. Anything outside TC⁰ requires changing depth, precision, or adding
serial decoding steps.

The intuition Merrill & Sabharwal give — the *parallelism tradeoff* — is worth stating in
our own terms: **any architecture whose forward pass is as parallelisable as a transformer
inherits the same ceiling.** This is a statement about parallel time, not about attention.
It applies to our stack too, and §3 shows exactly which part of our stack evades it and
which part does not.

### 1.2 The concrete problem classes, and where the wall actually sits

`[T]` for the complexity classifications; `[S]` for the transformer-specific claims.

| Problem | Complexity | Fixed-depth log-precision transformer? | Notes |
|---|---|---|---|
| Two-operand binary addition | AC⁰ `[T]` | **Yes**, in principle | Carry-lookahead is constant-depth unbounded-fan-in. **The carry chain is not a complexity barrier.** |
| Iterated addition of *n* numbers; integer multiplication; division | TC⁰ `[T]` | **Yes**, in principle | Hesse: division ∈ uniform TC⁰. Again not a complexity barrier. |
| PARITY / MOD-2 | TC⁰, ∉ AC⁰ `[T]` (Håstad) | **Yes** — explicit construction exists | Chiang & Cholak [2202.12172](https://arxiv.org/abs/2202.12172) build a transformer recognising PARITY with perfect accuracy, using layernorm to drive cross-entropy → 0 `[S]`. But see §1.3. |
| PARITY under **constant-bit** precision | ∉ AC⁰ `[T]` | **No** | Direct corollary of [2402.12875](https://arxiv.org/abs/2402.12875) `[S]`. **Relevant to int4 deployment — see F8.** |
| PARITY under **NoPE** | — | **No** | Transformers with no positional encoding "cannot express the very simple counting property of PARITY" — [2505.11199](https://arxiv.org/abs/2505.11199) `[S]`. |
| Word problem for S₅ / A₅ (permutation composition) | **NC¹-complete** `[T]` (Barrington 1986) | **No**, unless TC⁰ = NC¹ | The canonical hard state-tracking task. |
| Recognising an arbitrary regular language | NC¹-complete in general `[T]` | **No** in general; **yes** when the syntactic monoid is *solvable* | Liu et al. [2210.10749](https://arxiv.org/abs/2210.10749): `O(1)`-depth shortcuts for solvable groups, `O(log T)` for all automata `[S]`. |
| Boolean formula value problem | NC¹-complete `[T]` | **No**, unless TC⁰ = NC¹ | |
| Undirected s-t connectivity | L-complete `[T]` (Reingold) | **No**, unless TC⁰ = L | The substrate of multi-hop reasoning. |
| Directed graph reachability | NL-complete `[T]` | **No**, unless TC⁰ = NL | |
| Solving linear equalities; membership in an arbitrary CFG with ε-productions | P-complete `[T]` | **No**, unless L = P | Stated explicitly as a corollary in [2207.00729](https://arxiv.org/abs/2207.00729) `[S]`. |
| Circuit value problem; iterated composition of arbitrary functions | P-complete `[T]` | **No**, unless L = P | |

**The honest reading of this table for Prophet.** Two of the four failure modes our
`00_PROBLEM_LANDSCAPE.md` blames on "fixed depth" — arithmetic carry chains and parity —
are **not** complexity-class failures. Both live inside TC⁰. When a transformer fails at
long multiplication it is failing at *learning* and at *positional bookkeeping*, not at
*expressivity*. Conflating the two leads to the wrong fix: we would buy depth when we
needed a better digit representation.

The genuine, provable, fixed-depth walls are exactly three:

1. **NC¹-hard serial composition** — S₅, non-solvable automata, formula evaluation, and
   everything that reduces to them (entity tracking, chess-state-from-moves, executing
   code, "who holds the object now").
2. **L/NL-hard reachability** — multi-hop graph search, transitive closure, pointer
   chasing over long chains.
3. **P-hard iteration** — anything genuinely sequential with no parallel shortcut.

### 1.3 The *learnability* wall is lower than the expressivity wall, and matters more to us

Hahn, *Theoretical Limitations of Self-Attention*, [1906.06755](https://arxiv.org/abs/1906.06755):
self-attention cannot model periodic finite-state languages (PARITY) nor hierarchical
structure (DYCK) unless the number of layers or heads grows with input length `[S]`.
Chiang & Cholak later showed a construction *does* exist for PARITY `[S]`, so Hahn's
result is best read as being about robustness and about restricted attention regimes
rather than about raw expressibility.

Hahn & Rofin, *Why are Sensitive Functions Hard for Transformers?*,
[2402.09963](https://arxiv.org/abs/2402.09963), ACL 2024, gives the sharper and, for us,
more consequential statement: **the loss landscape is constrained by input-space
sensitivity. Transformers whose output is sensitive to many parts of the input occupy
isolated points in parameter space**, producing a systematic generalisation bias toward
low-sensitivity, low-degree functions `[S]`. This explains PARITY's persistent
length-generalisation failure despite being expressible.

Delétang et al., *Neural Networks and the Chomsky Hierarchy*,
[2207.02098](https://arxiv.org/abs/2207.02098), ICLR 2023: 20,910 models across 15 tasks.
RNNs and transformers fail to generalise on non-regular tasks; LSTMs handle regular and
counter languages; only structured-memory architectures climb higher `[S]`.

Zhou et al., *What Algorithms can Transformers Learn?*,
[2310.16028](https://arxiv.org/abs/2310.16028), ICLR 2024: the **RASP-generalisation
conjecture** — a transformer length-generalises on a task iff the task admits a short
RASP-L program that works at all lengths; and transformers tend to learn the *shortest*
RASP-L program fitting the training set `[S]`. This is the best practical predictor we
have, and it is the one we should use to design §7's task list.

Dziri et al., *Faith and Fate*, [2305.18654](https://arxiv.org/abs/2305.18654),
NeurIPS 2023: transformers reduce multi-step compositional reasoning to **linearised
subgraph matching**; GPT-4 reaches only 59 % on multi-digit × multi-digit multiplication,
and success tracks how much of the required computation graph was seen in training `[S]`.

### 1.4 The escape hatch we have deliberately declined

Chain of thought is the *proven* way out, and its price is exactly what our design refuses
to pay.

| Decoding budget | Reachable class | Source |
|---|---|---|
| constant intermediate tokens | TC⁰ (no gain) | Merrill & Sabharwal [2310.07923](https://arxiv.org/abs/2310.07923), ICLR 2024 `[S]` |
| `Θ(log n)` tokens | "pushes the limits only slightly" — between TC⁰ and L | same `[S]` |
| `Θ(n)` tokens | context-sensitive languages | same `[S]` |
| `poly(n)` tokens | up to P | same `[S]`; and [2402.12875](https://arxiv.org/abs/2402.12875) `[S]` |

`00_PROBLEM_LANDSCAPE.md` §4 rejects CoT on latency and KV-cache grounds. That is a sound
*product* decision. It is worth writing down that it is also a decision to give up the
only escape route with an unconditional proof behind it, and to bet on a route (§2) whose
proof has a precondition we currently do not satisfy.

---

## 2. What looping actually buys

This is the load-bearing question. The answer has two halves and they point in opposite
directions.

### 2.1 At constant *k*, looping changes nothing about the complexity class

A core of `c` layers applied `k` times with tied weights is, unrolled, a network of depth
`c·k`. If `k` is a constant — chosen at inference, but not a function of the input length
— then `c·k` is a constant, and every theorem in §1.1 applies unchanged. Weight tying can
only make the circuit family *more* uniform, never larger.

Prophet at `k = 4` has effective depth 20 `[V]` (`python -m prophet.budget
configs/prophet_500m_probe.json` reports `parameterised_depth=8 effective_depth=20`). At
`k = 8`, depth 36. Both are constants. **Both are in uniform TC⁰.**

So the sentence in `00_PROBLEM_LANDSCAPE.md` §4 — *"Un bloc récurrent partagé bouclé k fois
multiplie la profondeur effective sans multiplier les paramètres"* — is true and useful,
but it must not be read as *"and therefore escapes the fixed-depth wall."* It does not.
A 20-layer looped model and a 20-layer dense model sit in exactly the same complexity
class, and both sit in the same class as a 4-layer model.

### 2.2 At *k* growing with *n*, looping changes the class — and the result is exact

This is the strongest result in the whole track, and it was published in 2025:

> **Merrill & Sabharwal, [2505.18948](https://arxiv.org/abs/2505.18948) (NeurIPS 2025).**
> Padded transformers with `O(logᵈ n)` looping on inputs of length `n` recognise
> **exactly** the class FO-uniform **TCᵈ**. With polylogarithmic looping and polynomial
> padding, they recognise **exactly** FO-uniform **NC** — "the best that could be expected
> without losing parallelism." `[S]`

Read that as a dial with labelled detents:

| Loop schedule | Class | What it unlocks |
|---|---|---|
| `k = O(1)` | TC⁰ | nothing new |
| `k = Θ(log n)` | TC¹ | regular languages incl. S₅; graph connectivity |
| `k = Θ(log² n)` | TC² | |
| `k = polylog(n)` (+ poly padding) | NC | everything with a genuine parallel algorithm |

Independently, **Merrill & Sabharwal, [2503.03961](https://arxiv.org/abs/2503.03961)**
(*A Little Depth Goes a Long Way*) shows that **highly uniform transformers of depth
`Θ(log n)` can express (i) recognition of regular languages — i.e. state tracking — and
(ii) graph connectivity, and that both are outside fixed-depth transformers under standard
conjectures** `[S]`. Same threshold from a different direction.

And **Xu & Sato, [2509.25239](https://arxiv.org/abs/2509.25239)** (ICML 2026) formally
compares latent thought (looping) against CoT: **looping admits parallel computation and
establishes a separation in its favour in the polylogarithmic regime** for DAG evaluation,
while CoT retains an advantage for approximate counting and sampling via stochastic
decoding `[S]`. So looping is not a poor man's CoT — in the regime where it works, it is
asymptotically *better* than CoT.

Universal Transformers, [1807.03819](https://arxiv.org/abs/1807.03819), is the ancestor of
this line and its Turing-completeness claim is often mis-cited. It requires **(a)
input-dependent, unbounded iteration count** (the ACT halting mechanism) and **(b)
unbounded memory/precision** `[S]`. Neither is a property of a fixed `k` dial. Do not
write "Turing complete" anywhere in Prophet's documentation.

### 2.3 Prophet's dial is not wired to the input

Here is the gap, stated plainly.

```python
# prophet/config.py, RecurrentCoreConfig        [V]
train_loop_min: int = 1
train_loop_max: int = 8
train_loop_dist: Literal["uniform","log_uniform","poisson"] = "log_uniform"
default_loop_k: int = 4
halting: Literal["none","ponder","entropy"] = "none"
```

`k` is sampled **log-uniformly from [1, 8], independently of the input** `[V]`, and at
inference it is a constant supplied by the caller. Nothing in training ever presents the
model with the correlation `k ≈ f(n)` or `k ≈ f(difficulty)`. Under §2.2's theorems, this
is precisely the regime where looping buys **zero** additional expressive class.

The contrast with the literature is stark:

- **Fan, Du, Ramchandran & Lee, [2409.15647](https://arxiv.org/abs/2409.15647)** (ICLR
  2025), *Looped Transformers for Length Generalization*: it is the **adaptive** number of
  steps that produces length generalisation. Trained on parity up to 20 digits, the model
  generalises "near perfectly" past 40 digits by adapting the loop count at inference `[S]`.
  Their framing — `n`-RASP-L, tasks defined as `n` iterations of a length-generalisable
  RASP-L operation — is exactly the training protocol we are missing.
- **Bae et al., Mixture-of-Recursions, [2507.10524](https://arxiv.org/abs/2507.10524)**
  (NeurIPS 2025): lightweight routers assign **per-token** recursion depth; attention at
  depth `d` runs only over tokens still active, and only their KV is cached `[S]`. This is
  the input-dependent halting that `halting: "none"` leaves switched off.

**Recommendation R-W2-1 (highest value, cheapest):** make `k` a function of the input.
Two options, both config-level:
(a) a length schedule `k = clip(a + b·log₂ n, 1, k_max)` used at *training and inference*;
(b) enable the existing `halting: "ponder" | "entropy"` path and train it.
Without one of these, our central architectural bet — "depth as a runtime dial" — buys
parameter efficiency but no new capability class, and §6 predicts the specific tasks where
that will show.

### 2.4 What looping buys that is *not* a complexity class — and is still worth having

Nothing above says looping is a bad idea. It says the justification in our docs is the
wrong one. The real, well-evidenced benefits:

| Benefit | Evidence |
|---|---|
| **Depth per parameter.** A `k`-layer block looped `L` times "nearly matches" a `kL`-layer non-looped model on addition, `p`-hop induction and maths, and is much better than the `k`-layer model. | Saunshi et al., *Reasoning with Latent Thoughts*, [2502.17416](https://arxiv.org/abs/2502.17416), ICLR 2025 `[S]` |
| **Weight sharing is a good inductive bias, not just a compression.** A looped SSM with `k` parameters iterated `L` times matches or *beats* an untied SSM with `k·L` parameters across 4 architectures and 6 benchmarks — despite the looped model being a strict subset of the untied hypothesis space. The authors conclude the gain is optimisation, not expressivity. | Farsang et al., [2605.16048](https://arxiv.org/abs/2605.16048) `[S]` |
| **It works at tiny scale on hard puzzles.** HRM (27M params, ~1000 examples): 40.3 % ARC-AGI-1, 55 % Sudoku-Extreme, 74.5 % Maze-Hard. TRM (7M params, 2 layers): 45 % ARC-AGI-1, 8 % ARC-AGI-2, 87.4 % Sudoku-Extreme, 85.3 % Maze-Hard. | [2506.21734](https://arxiv.org/abs/2506.21734), [2510.04871](https://arxiv.org/abs/2510.04871) `[S]` |
| **…but for the reason we should care about.** The ARC Prize Foundation's independent ablation found the *hierarchical* structure contributed little versus a same-size transformer; **the under-documented outer refinement loop drove the gains, and mattered most at training time.** Cross-task transfer was limited — most performance came from memorising the evaluation tasks' augmentations. | [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis) `[S]` |
| **It scales to language, once.** Huginn-0125: 3.5B params, 800B tokens, recurrent-depth block iterated a randomly sampled number of times during training, unrolled to arbitrary depth at test time. Apache-2.0. | Geiping et al., [2502.05171](https://arxiv.org/abs/2502.05171) `[S]` |
| **Looped transformers can be programmed.** A looped 13-layer transformer emulates a small instruction-set computer — constant parameters, iterative algorithms. | Giannou et al., [2301.13196](https://arxiv.org/abs/2301.13196) `[S]` |

One caution and one free improvement from the newest work:

- **DeepLoop, [2607.13491](https://arxiv.org/abs/2607.13491)** `[S]`: residual scaling for
  looped models must depend on *how* depth is realised, not just nominal unrolled depth;
  the exponent must move from 1/4 to 1/2 as loop count grows, with `α = (2N)^{1/2}`,
  `β = (8N)^{-1/2}` for unrolled depth `N`. Prophet already scales by
  `(2·effective_depth)^{-1/2}` using `train_loop_max` `[V]`
  (`prophet/modeling/model.py`), which is the right *shape* but differs from their `β` by
  a factor of 2. A one-line ablation.
- The ARC Prize finding — the loop matters **most at training time** — argues against the
  framing of `k` as purely an inference-time dial. If the benefit is a training-time
  refinement signal, then `train_loop_max = 8` with `truncated_backprop_steps = 4` is
  cutting the benefit in half (see §5.2, item 6).

### 2.5 §2 verdict

> **Looping buys depth per parameter, a better optimisation landscape, and a
> memory-for-compute dial. It buys a strictly larger complexity class only when the loop
> count grows with the input — `Θ(log n)` for TC¹, polylog for NC — and only if the model
> is trained in that regime. Prophet trains `k ~ log-uniform[1,8]` independent of input
> and defaults to `k = 4`. As configured, Prophet's loop is a constant-factor depth
> multiplier and nothing more.**

### 2.6 Independent convergence with track W1

While this report was being written, track **W1** added §2ter to `docs/01_ARCHITECTURE.md`
reaching the same conclusion from a different direction — *"boucler `k` fois avec `k`
constant laisse la profondeur bornée par une constante : aucun changement de classe de
complexité"* — and raised trained halting to decision **D4b, "Requis"** `[V]`. Two tracks
that did not coordinate arrived at the same verdict on the project's central architectural
bet. That is worth recording as evidence, not as duplication.

W2 adds three things W1's note does not contain, and they are the actionable parts:

1. **The thresholds are known exactly**, not just "input-dependent depth is needed":
   `Θ(log n)` loops → TC¹, `Θ(logᵈ n)` → TCᵈ, polylog + poly padding → NC
   ([2505.18948](https://arxiv.org/abs/2505.18948)). A halting mechanism that averages
   `k ≈ 4` regardless of `n` satisfies D4b's letter and none of its purpose.
2. **A length-tied schedule is a cheaper first test than trained halting.**
   `k = clip(a + b·log₂ n, 1, k_max)` needs no halting head, no auxiliary loss, and no new
   failure mode, and [2409.15647](https://arxiv.org/abs/2409.15647) shows adaptivity alone
   is what produces the length generalisation. Test it first (W2-A6); adopt trained halting
   only if the schedule works and per-token granularity is then worth the complexity.
3. **`truncated_backprop_steps` silently caps the benefit.** D4b makes `k` input-dependent;
   truncation to the last 3–4 iterations (`design_search.py` passes 3, the shipped config
   says 4 `[V]`) means the model still cannot *learn* a step whose credit assignment spans
   more than 4 iterations, whatever `k` is at inference. D4b without W2-A7 is half a fix.

---

## 3. What bounded-state mixers give up

### 3.1 The literature splits bounded-state models into two classes; our docs treat them as one

This distinction is the single most important technical content in this report.

**Class A — diagonal state transitions** (S4, S6/Mamba, Mamba-2, GLA, RWKV-4/5/6, plain
linear attention). The state update is `S_t = diag(a_t) ⊙ S_{t-1} + …`. A product of
diagonal matrices is a prefix *sum* of logs — computable in constant parallel depth.

> **Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models*,
> [2404.08819](https://arxiv.org/abs/2404.08819) (ICML 2024).** Linear and Mamba-style SSMs
> **cannot express computation outside TC⁰**, and therefore cannot express inherently
> sequential problems such as permutation composition that RNNs express trivially.
> Empirically: transformer, S4 and Mamba all need depth increasing monotonically with
> sequence length on non-commutative groups; a single-layer RNN — and a single-layer
> *input-dependent* S4 — learn the word problem at arbitrary length. `[S]`

Their conclusion is worth quoting because it is aimed straight at our marketing copy:
arguments that SSMs beat transformers because they are "more recurrent" or "track state"
are **misguided** `[S]`.

**Class B — non-diagonal (identity-minus-rank-1 / generalised-Householder) transitions**
(DeltaNet, Gated DeltaNet, RWKV-7, TTT-Linear, Titans). The state update is
`S_t = (I − β_t k_t k_tᵀ) S_{t-1} + …`. Products of such matrices generate **non-abelian**
groups, and this is a genuinely different regime that 2404.08819 does not cover.

> **Peng et al., RWKV-7 "Goose", [2503.14456](https://arxiv.org/abs/2503.14456).**
> RWKV-7's dynamic state evolution "surpasses the fundamental TC⁰ limitation of the
> attention / linear-attention paradigm." It **solves an S₅ state-tracking problem known
> to be in NC¹ using a single layer, and recognises all regular languages with a constant
> number of layers** — exceeding transformers under the conjecture TC⁰ ≠ NC¹. `[S]`

> **Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative
> Eigenvalues*, [2411.12537](https://arxiv.org/abs/2411.12537) (ICLR 2025 Oral).**
> (i) LRNNs whose state-transition matrices have **only positive eigenvalues cannot solve
> parity**; the failure of Mamba on parity stems from restricting diagonal entries to
> `[0,1]`. (ii) **Non-triangular** matrices are needed to count modulo 3. (iii) LRNNs
> **can learn any regular language** when the transitions are products of
> `identity − vector outer product` matrices with eigenvalues in `[−1, 1]`.
> Extending Mamba's and DeltaNet's eigenvalue range to include negatives enables parity
> and improves state tracking with **no added training or inference cost**. `[V]` (README
> fetched from `github.com/automl/unlocking_state_tracking`) + `[S]`

> **Siems et al., *DeltaProduct*, [2502.10297](https://arxiv.org/abs/2502.10297).**
> Taking `n_h` micro-steps per token gives diagonal-plus-rank-`n_h` transitions formed as
> products of `n_h` generalised Householder transformations. **A linear RNN whose
> transitions are a product of `n_h` generalised Householders solves any group word problem
> over permutations of at most `n_h + 1` elements in one layer.** Empirically: S₃ needs
> `n_h = 2`; **S₅ needs `n_h = 4`**; S₄ and A₅ extrapolate at `n_h = 2` because they embed
> in SO(3,ℝ) (two reflections give a 3-D rotation). **DeltaNet (`n_h = 1`) cannot learn S₄
> or S₅ even with 5 layers.** All experiments used the extended `[−1,1]` eigenvalue range,
> because **DeltaProduct models failed to learn with the standard `[0,1]` range.** Trained
> on ≤128 products, extrapolation measured to 512. `[S]`

So: **replacing softmax attention with a fixed-size-state mixer does not automatically
make the model weaker.** With the right transition structure it makes it *strictly
stronger* on serial computation and *strictly weaker* on recall. The question is which
side of the diagonal/non-diagonal line — and which eigenvalue range — we are on.

### 3.2 Prophet's gated DeltaNet is on the weak side of that line. This is a bug, not a design choice.

From `prophet/modeling/layers.py` `[V]`:

```python
# GatedDeltaNet.forward, lines 461-462
alpha = torch.sigmoid(self.a_proj(x).float())   # (b, s, h)   ->  α ∈ (0, 1)
beta  = torch.sigmoid(self.b_proj(x).float())   # (b, s, h)   ->  β ∈ (0, 1)
...
# GatedDeltaNet._scan, line 507-509 (keys are L2-normalised, line 459)
S = a * S + bt * (vt - a * (S @ kt)) @ kt.transpose(-1, -2)
#   ==  α_t · S_{t-1} (I − β_t k_t k_tᵀ)  +  β_t v_t k_tᵀ ,   ‖k_t‖ = 1
```

The per-step state-transition matrix is `M_t = α_t (I − β_t k_t k_tᵀ)` with `‖k_t‖₂ = 1`.
Its spectrum is exact and elementary:

```
eigenvalues(M_t) = { α_t          with multiplicity d_k − 1 ,
                     α_t(1 − β_t) with multiplicity 1        }

α_t ∈ (0,1)  and  β_t ∈ (0,1)   ⟹   every eigenvalue lies in (0, 1) — strictly positive.
```

By Grazzi et al.'s result (i), **a linear RNN whose transitions have only positive
eigenvalues cannot solve parity.** Prophet's recurrent core, as implemented, is in that
class. Not "finds parity hard to learn" — the result is representational.

Two independent things are wrong, and they compound:

| Defect | Consequence | Fix | Cost |
|---|---|---|---|
| `β = sigmoid(·) ∈ (0,1)` | eigenvalues strictly positive → **cannot express parity**; DeltaProduct reports its models "failed to learn" in this range at all | `β = 2·sigmoid(·) ∈ (0,2)`, so `α(1−β) ∈ (−α, α)` | **one character**; 0 params, 0 FLOPs |
| `n_h = 1` (one Householder per token) | caps single-layer group word problems at permutations of `n_h+1 = 2` elements; **S₄ and S₅ unreachable even at 5 layers** | `n_h = 2` for S₄/A₅, `n_h = 4` for S₅ | `n_h ×` the recurrence cost of GDN layers only |

**Caveat to check before shipping the `β` fix `[V]`:** the fused path calls
`fla.ops.gated_delta_rule.chunk_gated_delta_rule(q,k,v, g=alpha.log(), beta=beta, …)`
(`layers.py:465`). Whether that kernel's WY-representation chunking is numerically valid
for `β > 1` must be verified against the kernel, not assumed. `HAS_FLA` is `False` in this
container, so the reference scan (which is unconditionally valid) is what ran in §6.0.

**Recommendation R-W2-2:** add `beta_max: float = 1.0` and `n_householder: int = 1` to
`MixerConfig` as explicit switches (CLAUDE.md rule 3). **§6.0 already ran the `beta_max`
arm on this repository's own recurrence and it separated completely** — chance versus
0.92–0.95 at four times the training length, three seeds each, identical budgets. This is
the highest return-on-effort change available anywhere in the architecture, and the only
thing in Prophet's stack with a *proof* attached that it lifts us above what a
same-parameter dense transformer can do. W2-A1 exists to confirm it at model scale and on
the fused kernel, not to discover whether it works.

### 3.3 What bounded state gives up regardless of eigenvalues: exact recall

Expressivity of *serial computation* and capacity for *exact memory* are orthogonal, and
the delta rule buys the first at the cost of the second.

- **Jelassi et al., [2402.01032](https://arxiv.org/abs/2402.01032)** (*Repeat After Me*,
  ICML 2024): a two-layer transformer can copy strings of length **exponential** in its
  size; generalised SSMs are **fundamentally limited by their fixed-size latent state**.
  Empirically transformers dominate on copying and on retrieval, both in synthetic tasks
  and among pretrained LLMs. `[S]`
- **Arora et al., Zoology, [2312.04927](https://arxiv.org/abs/2312.04927)**: on
  multi-query associative recall (MQAR), accuracy is governed by **recurrent state size**;
  within each architecture class, larger state ⇒ higher accuracy, almost monotonically. `[S]`
- **Arora et al., Just Read Twice, [2407.05483](https://arxiv.org/abs/2407.05483)**:
  repeating the *context* in the prompt yields **+11.0 ± 1.3 points averaged over 16
  recurrent LMs and 6 ICL tasks**, because it removes the dependence on data order that a
  causal fixed-state model suffers. `[S]`

**Prophet's numbers.** With `linear_heads = 8`, `linear_head_dim = 128`,
`linear_expand = 2.0` `[V]`, one GDN layer's state is `h × d_v × d_k = 8 × 256 × 128 =
262 144` floats = **512 KiB in bf16**, constant in context length. But the *information*
bound is tighter than the byte count: with L2-normalised keys, the delta rule's state is a
linear map from a `d_k = 128`-dimensional key space. **At most 128 linearly independent
keys per head can be retrieved without interference — about 1 024 exactly-recoverable
associations per GDN layer.** A full-attention layer at 32 k context holds 32 768 keys per
head: **256× more per head**, and it grows with the context while ours does not.

That is the trade, quantified. It is a good trade for a phone. It is not a free one, and
it predicts the failure in F4.

`01_ARCHITECTURE.md` §4 already flags the symptom from track R02 — the best 2026 linear
mixer reaches **89.8 % single-needle but 37.8 % multi-needle** `[V]` (repo-internal
figure) — and prescribes "more global layers, not a wider window" as the fix. §3.3 gives
the *reason* for the asymmetry: single-needle needs one association to survive; multi-needle
needs `m` associations to coexist in a rank-`d_k` state. The prescribed fix is correct, and
§5.1 shows the `k` dial works directly against it.

**A subtlety about our loop and JRT.** Re-reading the context is what closes the recall
gap for recurrent models `[S]`. Prophet's loop re-processes *hidden states* with
`inject_input_each_step` re-adding the prelude output `[V]` — it does **not** re-run the
sequence mixer over the token stream with a fresh state and a second look at the raw
prompt. It is therefore not obvious that looping inherits JRT's +11 points, and F3
predicts it does not. This is directly testable (W2-A4) and is one of the cheapest
high-information experiments in the plan.

### 3.4 Testing our assumption: does a minority of full attention rescue this?

`docs/01_ARCHITECTURE.md` §4 rests on the assumption that periodic full-attention layers
repair what bounded-state layers give up. **The literature says this assumption is right
for one thing and wrong for another, and our docs do not distinguish them.**

**(a) For recall: correct, and quantified — but our ratio is at the edge of the safe band.**

> **Wang et al., *A Systematic Analysis of Hybrid Linear Attention*,
> [2507.06457](https://arxiv.org/abs/2507.06457).** 72 models trained and released — 36 at
> 340M/20B tokens and 36 at 1.3B/100B tokens, 6 linear-attention variants × 5 hybridisation
> ratios. **Language-modelling loss is stable across ratios, but recall improves sharply
> with more full-attention layers, particularly below a 3:1 ratio.** Recommendation:
> HGRN-2 or **GatedDeltaNet with a linear-to-full ratio between 3:1 and 6:1**. `[S]`

Our nominal 3:1 sits exactly on the aggressive boundary of their recommended band — the
point past which their curve says recall starts falling off. And §5.1 shows our *effective*
ratio is much worse than 3:1 and gets worse as `k` grows.

**(b) For state tracking: incorrect. Attention cannot rescue it, because attention is the
thing that is limited.**

This follows from §1.1: a constant number of softmax-attention layers at log precision is
computable in uniform TC⁰. Adding TC⁰-computable layers to a stack cannot lift that stack
above whatever its other layers already reach. **Attention layers therefore contribute
nothing to serial-computation power that the stack does not already have.** If Prophet ever
exceeds TC⁰, the excess comes entirely from the delta-rule transition matrices of the GDN
core (§3.1 Class B) — never from the attention layers. Concretely: if `n_h = 1` and
`β ∈ (0,1)` make the core unable to express PARITY, no number of full-attention layers in
the prelude or coda repairs that, because a fixed-depth attention stack cannot express
PARITY under constant-bit precision either and can only do so at log precision via a
construction that is famously unlearnable (§1.3).

So the sentence in `01_ARCHITECTURE.md` §4 — attention restores *"le rappel exact"* — is
precisely and only true. The mistake would be to extend it to state tracking, which the
project's framing in `00_PROBLEM_LANDSCAPE.md` §2/§4 implicitly does.

**(c) There *is* a real hybrid advantage. It is a different one, and it is better than the
one we claim.**

> **[2605.16640](https://arxiv.org/abs/2605.16640)**, *Provably Shorter Scratchpads in
> Hybrid DeltaNet-Attention Decoders*: for a **parity-conditioned retrieval** task, a
> Qwen3-Next-style hybrid of Gated DeltaNet + gated attention needs only an **`O(1)`
> scratchpad** under constant precision, while **no such solution exists for pure Gated
> DeltaNet** and **pure gated attention requires at least a polynomial scratchpad**. `[S]`

> **[2603.08859](https://arxiv.org/abs/2603.08859)**, *Expressivity-Efficiency Tradeoffs
> for Hybrid Sequence Models*: any pure transformer or pure SSM solving certain tasks needs
> either many parameters or a large working memory; for **selective copying** and
> **associative recall**, small hybrids provably achieve both. `[S]`

The correct claim for our docs is: *hybridisation is well-founded because attention and
bounded-state recurrence supply different primitives — prefix-visible exact lookup versus
ordered in-place state update — and several tasks provably need both.* Not: *attention
patches the linear layers' weaknesses.*

---

## 4. State tracking: the sharpest test case

### 4.1 Why S₅ is the right probe

The word problem for a finite group `G`: given `g₁ g₂ … g_n ∈ G`, is the product the
identity? Barrington (1986) `[T]`: for any non-solvable `G` — S₅ and A₅ being the smallest
— this is **NC¹-complete**. It is the cleanest available separator between "constant
parallel depth" and "genuinely serial", it has no length-generalisation ambiguity, and
data generation is one line of code.

Liu et al., [2210.10749](https://arxiv.org/abs/2210.10749) `[S]`, gives the exact rule for
how much depth a task needs, in terms of the *algebra* of the automaton rather than its
surface form:

| Automaton's transformation semigroup | Depth needed by a transformer |
|---|---|
| Solvable group (ℤ_p counters, S₃, S₄, dihedral) | **`O(1)`** — shortcuts exist and are learnable by SGD |
| Any automaton | **`O(log T)`** |
| Non-solvable (A₅, S₅) | `Θ(log T)`; no `O(1)` shortcut unless TC⁰ = NC¹ |

The practical translation: **counting mod anything and tracking a solvable state machine
are within a fixed-depth model's reach; tracking a non-solvable one is not.** Real tasks
that are S₅-like: executing code with mutable aliased references, board-game state from a
move list, permutation/shuffle tracking, "who is holding which object after these
swaps", coreference through a long chain of reassignments.

### 4.2 The current scoreboard, one line per architecture

| Architecture | Parity | S₅ / A₅ word problem | Source |
|---|---|---|---|
| Fixed-depth transformer | expressible in TC⁰ but not learnable/length-generalisable; **not** expressible at constant-bit precision or under NoPE | **No** (unless TC⁰ = NC¹) | [2202.12172](https://arxiv.org/abs/2202.12172), [2402.09963](https://arxiv.org/abs/2402.09963), [2402.12875](https://arxiv.org/abs/2402.12875), [2505.11199](https://arxiv.org/abs/2505.11199) `[S]` |
| Log-depth transformer (`Θ(log n)`) | yes | **Yes** — regular languages expressible | [2503.03961](https://arxiv.org/abs/2503.03961) `[S]` |
| True RNN (non-linear, sequential) | yes, 1 layer | **Yes, 1 layer** — at arbitrary length | [2404.08819](https://arxiv.org/abs/2404.08819) `[S]` |
| Mamba / diagonal SSM | **No** (eigenvalues in `[0,1]`) | **No** — in TC⁰ | [2404.08819](https://arxiv.org/abs/2404.08819), [2411.12537](https://arxiv.org/abs/2411.12537) `[S]` |
| Input-dependent S4 (IDS4) | — | **Yes, 1 layer** | [2404.08819](https://arxiv.org/abs/2404.08819) `[S]` |
| DeltaNet, eigenvalues `[0,1]` | **No** | **No** (fails S₄/S₅ even at 5 layers) | [2411.12537](https://arxiv.org/abs/2411.12537), [2502.10297](https://arxiv.org/abs/2502.10297) `[S]` |
| DeltaNet, eigenvalues `[−1,1]` | **Yes** | S₂ only (`n_h = 1`) | [2411.12537](https://arxiv.org/abs/2411.12537), [2502.10297](https://arxiv.org/abs/2502.10297) `[S]` |
| DeltaProduct `n_h = 2`, `[−1,1]` | Yes | S₃, S₄, A₅ (extrapolates to 512) | [2502.10297](https://arxiv.org/abs/2502.10297) `[S]` |
| DeltaProduct `n_h = 4`, `[−1,1]` | Yes | **S₅** | [2502.10297](https://arxiv.org/abs/2502.10297) `[S]` |
| RWKV-7 | Yes | **S₅ in one layer**; all regular languages in constant layers | [2503.14456](https://arxiv.org/abs/2503.14456) `[S]` |
| **Prophet as implemented** | **No** — `β ∈ (0,1)` ⇒ eigenvalues strictly positive | **No** — `n_h = 1` and eigenvalues positive | `[V]` code + `[E]` §6.0 |

That last row is the finding of this track.

### 4.3 Iterated function composition is the same problem wearing a different hat

The brief asks about "iterated function composition" as a separate capability. It is not
separate — it is the general case of §4.1, and the mapping is exact:

| Surface task | Algebraic object | Class | Depth needed |
|---|---|---|---|
| Compose `n` permutations of 3 or 4 elements | S₃, S₄ — solvable | in TC⁰ | `O(1)` |
| Compose `n` permutations of ≥ 5 elements | S₅, A₅ — non-solvable | NC¹-complete | `Θ(log n)` |
| Compose `n` arbitrary functions `[m] → [m]` | full transformation monoid `T_m` | NC¹-complete for `m ≥ 5`; contains S₅ | `Θ(log n)` |
| Compose `n` arbitrary functions with unbounded `m` | reduces to circuit value | P-complete | serial |
| `p`-hop induction (follow a pointer `p` times) | pointer chasing | `O(log p)` depth *with attention* | see §4.4 |

The practical consequence for Prophet: **the threshold for "hard" is `m = 5`, and it is a
cliff, not a slope.** A model that tracks 4 interchangeable objects perfectly may be at
chance on 5. This is a testable, cheap, and unusually sharp prediction, and it is why
W2-A3 sweeps S₃/S₄/A₅/S₅ rather than just "a state-tracking task".

### 4.4 The adjacent failure: search

- **Sanford, Hsu & Telgarsky, [2402.09268](https://arxiv.org/abs/2402.09268)** (ICML 2024):
  a constant number of attention layers simulates a constant number of MPC communication
  rounds; consequently transformers solve `k`-hop pointer chasing in **`O(log k)` depth**
  by pointer doubling, and logarithmic depth suffices for tasks that sub-quadratic
  approximations cannot do efficiently. `[S]`
- **Saparov et al., *Transformers Struggle to Learn to Search*,
  [2412.04703](https://arxiv.org/abs/2412.04703)** (ICLR 2025): with the right training
  distribution small transformers *do* learn to search — they compute, at every vertex, the
  set reachable in `≤ d` steps, each layer extending the sets, giving reach exponential in
  depth. But this **degrades as graph size grows**, and LLMs do not do it robustly. `[S]`

The pointer-doubling algorithm requires **content-addressed random access across the whole
prefix** — i.e. attention. Prophet's looped core has none (in the canonical stack).
Prediction F6 follows.

---

## 5. The verdict on Prophet's stack

### 5.1 First, what the stack actually is — the code does not match the document

**This section reports a discrepancy that must be fixed before any W2 ablation runs, or
every result will be confounded.**

`docs/01_ARCHITECTURE.md` §2 and §4 specify: prelude and coda = `SWA(2048)` then
`full-attention (NoPE)`; looped core = `GDN only`, so that KV memory is independent of `k`.
`scripts/design_search.py` builds exactly that `[V]`:

```python
mixer   = MixerConfig(pattern=["swa","full_attn"], sliding_window=2048,
                      attention_sink_tokens=1, nope_layers=(1,))
recurrent = RecurrentCoreConfig(core_pattern=["gdn"], ...)
```

`configs/prophet_500m_probe.json` — the only non-smoke config in the repository — **does
not.** Resolving it through `ProphetConfig.section_layout()` `[V]`:

```
prelude 0  gdn        <- doc says swa
prelude 1  gdn        <- doc says full_attn (NoPE)
core    0  gdn
core    1  gdn
core    2  gdn
core    3  full_attn  <- ATTENTION INSIDE THE LOOP: violates Decision D1
coda    0  gdn        <- doc says swa
coda    1  gdn        <- doc says full_attn (NoPE)
```

Cause: `core_pattern`, `prelude_pattern` and `coda_pattern` are all `null`, so all three
sections fall back to `mixer.pattern = ["gdn","gdn","gdn","full_attn"]`, which is indexed
**within each section** `[V]`. Consequences:

1. **Decision D1 is violated.** The single full-attention layer is in the looped core, so
   KV cache scales with `k` — the exact "deployment trap" §2 of the architecture document
   says we avoided. The passing test cited there must be using a different config.
2. **There is no SWA layer and no NoPE layer at all** (`nope_layers: []`).
3. There is **no attention in the prelude or coda** — so the "exact recall before the final
   read" property of the coda does not exist in this config.

Second, even for the *correct* stack, the mixer census is a function of `k`, and our
headline "75 % GDN / 12 % SWA / 12 % full attention" is true at exactly one value of `k`.
For `prophet-main` (`p4 c4 ×k o4`, i.e. prelude 4, core 4, coda 4) `[V]` from
`01_ARCHITECTURE.md` §3:

| `k` | effective depth | GDN | SWA | full attn | GDN : full | (GDN+SWA) : full |
|---:|---:|---:|---:|---:|---|---|
| 1 | 12 | 33.3 % | 33.3 % | 33.3 % | 1 : 1 | 2 : 1 |
| 2 | 16 | 50.0 % | 25.0 % | 25.0 % | 2 : 1 | 3 : 1 |
| 4 | 24 | 66.7 % | 16.7 % | 16.7 % | 4 : 1 | 5 : 1 |
| **6** | **32** | **75.0 %** | **12.5 %** | **12.5 %** | **6 : 1** | **7 : 1** |
| 8 | 40 | 80.0 % | 10.0 % | 10.0 % | 8 : 1 | 9 : 1 |

(The census quoted in this track's brief — 75 % GDN / 12 % SWA / 12 % full attention — is
the `k = 6` row, and only that row.)

Against Wang et al.'s recommended band of 3:1 to 6:1 for GatedDeltaNet hybrids `[S]`:
counting GDN alone, `k = 4` is fine, `k = 6` is at the boundary, **`k = 8` is outside it**;
counting SWA as a sub-quadratic mixer too — which is the more faithful reading of their
setup — we are outside the band from `k = 4` onward. The dial we sell as
"more depth on a 5090" is simultaneously a dial that **dilutes the attention fraction and
degrades recall**. Nobody has ever written that down in our docs, and it is a genuine,
quantified design tension: the two things `k` controls pull in opposite directions.

### 5.2 The claim-by-claim audit

| # | Claim in our documents | Verdict | Basis |
|---|---|---|---|
| 1 | "Looping `k` times multiplies effective depth without multiplying parameters" (`00`, §4) | **True** | Arithmetic; and Saunshi [2502.17416](https://arxiv.org/abs/2502.17416) shows the depth is *useful* |
| 2 | Implied: "and therefore escapes the fixed-depth reasoning wall" | **False at constant `k`** | Constant `k` ⇒ constant depth ⇒ uniform TC⁰ (§2.1). Needs `k = Θ(log n)` (§2.2) |
| 3 | "`k` becomes a dial set at runtime — small on iPhone, large on 5090" (`00`, §4) | **True but two-edged** | It is a dial for depth-limited tasks; it is *not* a dial for memory-limited tasks, and it *worsens* the attention ratio (§5.1) |
| 4 | "A minority of full-attention layers restores exact recall" (`01`, §4) | **True**, with a ratio caveat | [2507.06457](https://arxiv.org/abs/2507.06457): recall falls off below 3:1; our effective ratio hits 8:1 at `k=8` |
| 5 | Implied: attention rescues what bounded-state layers give up, generally | **False for state tracking** | Attention is itself TC⁰; TC⁰ ∘ TC⁰ = TC⁰ (§3.4b) |
| 6 | "Bounded-state recurrence gives us state tracking transformers lack" | **False as implemented** | `β ∈ (0,1)` ⇒ strictly positive eigenvalues ⇒ provably cannot express parity ([2411.12537](https://arxiv.org/abs/2411.12537)); `n_h = 1` ⇒ caps at S₂ ([2502.10297](https://arxiv.org/abs/2502.10297)). Confirmed empirically in §6.0 |
| 7 | "NoPE on the global layers gives length extrapolation" (`01`, §4) | **Partly true, and switched off** | [2305.19466](https://arxiv.org/abs/2305.19466) supports it `[S]`; but NoPE transformers provably **cannot express PARITY** ([2505.11199](https://arxiv.org/abs/2505.11199)) and `nope_layers: []` in the shipped config |
| 8 | "Effective depth 20–40 ≈ a 20–40-layer transformer" | **Weaker than that** | `truncated_backprop_steps = 4` in the shipped config, `3` in `design_search.py` `[V]`: at `k = 8` (32 core-layer applications) gradients reach only the last 3–4 iterations. The model cannot *learn* an algorithm whose credit assignment spans the loop — precisely the thing looping is meant to buy |
| 9 | Random state init at train / zeros at eval "forces the loop to converge to an attractor independent of the starting point" (`01`, §5) | **Correct — and in tension with the goal** | `state_init` seeds the **residual stream** `h`, not the GDN state matrix `[V]` (`model.py`: `h = torch.randn_like(x) * init_std`). Training the loop to give the same answer from any start is explicit pressure toward a *fixed point*. That is exactly right for **iterative refinement** (denoise-toward-an-answer, which is what HRM/TRM do) and exactly wrong for **iterative computation** whose intermediate states must stay distinguishable step by step — a counter, a carry chain, a permutation accumulator. We are optimising for one and marketing the other. Testable: F9 |
| 10 | "`k` gives us reasoning depth at no memory cost" | **True only for the canonical stack** | In the shipped config the loop contains a full-attention layer, so KV grows with `k` (§5.1) |

### 5.3 So: what can Prophet solve that a same-parameter dense transformer cannot?

**As currently implemented: nothing demonstrable, and on one axis we are strictly worse.**
Precisely:

- Constant `k` gives constant depth, so the *looping* contributes no class change (§2.1).
- The attention layers are TC⁰-computable, so they contribute no serial-computation power
  beyond the rest of the stack (§3.4b).
- The GDN core is the only component that could exceed TC⁰ — and with `β ∈ (0,1)` it
  provably **cannot express PARITY**, a problem that is *inside* TC⁰, and it **measurably
  does not** (§6.0: chance at 2× and 4× the training length, at one layer and at two). So
  on the state-tracking axis the core does not even cover TC⁰, let alone exceed it.
  (Strictly:
  "cannot express PARITY" does not by itself prove containment in TC⁰, so I do not claim
  the stack is *in* TC⁰; I claim it has no demonstrated capability outside TC⁰ and a
  demonstrated *gap* inside it. That is worse, not better, than the honest baseline.)

Prophet's *real* current advantages are orthogonal to expressivity and should be described
that way:

- **Constant memory per token** for 67–89 % of the stack — an engineering property, not an
  expressivity property.
- **Depth per parameter** and **a better optimisation landscape** from weight sharing
  ([2502.17416](https://arxiv.org/abs/2502.17416), [2605.16048](https://arxiv.org/abs/2605.16048)).
- **A compute/quality dial at inference** — real, and rare.

**With two config changes, one genuine class separation becomes available**, and it is the
only one on the table:

> Set `β ∈ (0, 2)` and `n_h ≥ 2`. Then the delta-rule core has non-diagonal transitions
> with eigenvalues in `[−1,1]`, and — following RWKV-7 `[S]` and DeltaProduct `[S]` — the
> stack plausibly recognises regular languages that **no fixed-depth transformer of any
> parameter count can recognise**, under TC⁰ ≠ NC¹.

That is a defensible research claim for a paper and a real capability for entity tracking
and code semantics. **It is currently disabled by a `sigmoid`.** Nothing else in Prophet's
architecture has a comparable ratio of provable capability to implementation cost.

### 5.4 What we have given up, itemised

1. **Exact multi-key recall.** ~128 independent keys per head per GDN layer versus 32 768
   at 32 k context for attention (§3.3). Partially bought back by the attention layers,
   whose share *shrinks* as `k` grows.
2. **The CoT escape hatch**, deliberately (§1.4) — the only route with an unconditional
   proof.
3. **Long-horizon loop credit assignment**, via `truncated_backprop_steps = 4`.
4. **Input-adaptive depth**, via `halting: "none"` and input-independent `k` sampling.
5. **Sensitive-function robustness under int4**, since constant-bit precision drops the
   whole stack to AC⁰ ([2402.12875](https://arxiv.org/abs/2402.12875) `[S]`) — see F8.

---

## 6. Concrete failure predictions

Each is falsifiable, names its mechanism, and can be tested for well under one A100-hour.

### 6.0 F1 is already confirmed, on this repository's own recurrence

I re-implemented `prophet/modeling/layers.py::GatedDeltaNet` exactly — L2-normalised keys,
`α = sigmoid(a_proj(x))` with bias init 3.0, depthwise causal conv of kernel 4 with SiLU,
RMSNorm output, the same `S ← αS(I − βkkᵀ) + βvkᵀ` sequential scan — inside a standard
pre-norm residual block with a SwiGLU-shaped FFN. **Exactly one thing varies:**

```python
beta = torch.sigmoid(self.b_proj(x))                # beta_max = 1.0  — Prophet today
beta = 2.0 * torch.sigmoid(self.b_proj(x))          # beta_max = 2.0  — the one-character fix
```

Task: cumulative parity, `y_t = XOR(x_1..x_t)`, trained at length `T = 24`, scored on the
**final-token** bit at `T`, `2T`, `4T` (chance = 0.5). AdamW + OneCycle, 1 500 steps,
batch 64, `d_model = 64`, 4 heads × `d_k = 16`, `d_v = 32`, CPU, `HAS_FLA = False` so the
reference scan is what ran. Script: `<scratchpad>/parity_gdn.py`. `[E]`

| `beta_max` | eigenvalues of `α(I−βkkᵀ)` | layers | params | `T=24` (train) | `T=48` (2×) | `T=96` (4×) |
|---:|---|---:|---:|---:|---:|---:|
| **1.0** (Prophet) | `(0, 1)` — strictly positive | 1 | 43 k | 0.932 | **0.531** | **0.443** |
| **1.0** (Prophet) | `(0, 1)` — strictly positive | 1 | 43 k | 0.917 | **0.534** | **0.518** |
| **1.0** (Prophet) | `(0, 1)` — strictly positive | 2 | 86 k | 0.958 | **0.565** | **0.539** |
| **2.0** (fix) | `(−1, 1)` — negatives allowed | 1 | 43 k | 1.000 | **0.997** | **0.951** |
| **2.0** (fix) | `(−1, 1)` — negatives allowed | 1 | 43 k | 1.000 | **0.997** | **0.935** |
| **2.0** (fix) | `(−1, 1)` — negatives allowed | 2 | 86 k | 1.000 | **0.995** | **0.922** |

**The separation is total.** With `β ∈ (0,1)` the model reaches 92–96 % *at the training
length* — a length-specific approximation, not parity — and then collapses to **chance
(0.44–0.57)** at 2× and 4×. Adding a second layer does not help (0.539 at 4×), which is the
signature of a representational limit rather than a capacity limit. With `β ∈ (0,2)` the
same model, same seeds, same budget, is **perfect in-distribution and 92–95 % at four times
the training length**.

Caveats stated plainly: this is a 43k–86k-parameter probe on one task, three seeds per arm
at one training length, on CPU. It does not prove anything about a 370M model. But it
matches Grazzi et al.'s theorem ([2411.12537](https://arxiv.org/abs/2411.12537)) exactly, it
isolates a single character of our source, both arms had identical budgets and seeds, and
the effect size (chance → 0.95 at 4× length) leaves no room for an optimisation artifact
explanation. **Treat F1 as confirmed at small scale and make W2-A1 the first experiment
of the track.**

### 6.1 The prediction list

| # | Prediction | Mechanism | Cheapest test |
|---|---|---|---|
| **F1** | **PARITY fails at every `k` and every depth.** With `β ∈ (0,1)`, accuracy at 4× training length stays near chance; `β ∈ (0,2)` exceeds 92 %. Increasing depth does **not** fix it — 1 layer 0.44/0.52, 2 layers 0.54. | Positive-eigenvalue LRNNs cannot express parity ([2411.12537](https://arxiv.org/abs/2411.12537)) | W2-A1 — **already confirmed at 43k–86k params, §6.0** |
| **F2** | **Counting mod 3 fails for both `β` ranges.** Needs complex eigenvalues / non-triangular transitions, which one Householder with real `β` does not give. | [2411.12537](https://arxiv.org/abs/2411.12537): "non-triangular matrices are needed to count modulo 3" `[S]` | W2-A2 |
| **F3** | **S₅ composition fails at every `k`, every `β`.** Near chance beyond training length. Adding a full-attention layer does not fix it. Only `n_h ≥ 4` does. | `n_h = 1` ⇒ permutations of ≤ 2 elements; DeltaNet fails S₄/S₅ at 5 layers ([2502.10297](https://arxiv.org/abs/2502.10297)); attention is TC⁰ | W2-A3 |
| **F4** | **Multi-key recall has a knee, and `k` does not move it.** MQAR accuracy collapses somewhere near `d_k`–`h·d_k` distinct key-value pairs (predict a knee in 128–1024), and `∂accuracy/∂k ≈ 0` while `∂accuracy/∂d_k > 0`. | Recall is state-size-bound, not depth-bound ([2312.04927](https://arxiv.org/abs/2312.04927), [2402.01032](https://arxiv.org/abs/2402.01032)) | W2-A4 — **the sharpest single discriminator** |
| **F5** | **Turning `k` up makes long-context recall *worse*** relative to an untied baseline of the same effective depth, because the full-attention share drops from 33 % (`k=1`) to 10 % (`k=8`). Expect the crossover near `k = 6–8`. | §5.1 census × [2507.06457](https://arxiv.org/abs/2507.06457)'s ratio curve | W2-A4, sweep `k` at fixed context |
| **F6** | **Graph reachability needs depth linear in diameter, not logarithmic.** Solvable diameter ≈ `(prelude + core·k + coda)/2`, so `k ≈ d / core_layers`. A dense transformer with attention at every layer will need only `O(log d)`. | Pointer doubling needs prefix-wide content-addressed access ([2402.09268](https://arxiv.org/abs/2402.09268)); the canonical looped core has no attention | W2-A5 |
| **F7** | **Accuracy on iterative tasks plateaus at `k ≈ tbptt + 1 ≈ 5`**, and the plateau *moves* when `truncated_backprop_steps` is raised. | Gradients reach only the last 4 iterations `[V]` | W2-A7 |
| **F8** | **int4 quantisation costs far more on sensitive tasks than on perplexity.** Predict `ΔBPB < 0.02` but `Δparity`, `Δexact-recall` ≫ that, disproportionately. | Constant-bit precision ⇒ AC⁰ ⊊ TC⁰ ([2402.12875](https://arxiv.org/abs/2402.12875)); parity ∉ AC⁰ | W2-A8 |
| **F9** | **`k` saturates: logits at `k = 4` and `k = 8` become nearly identical** (predict mean per-token logit L2 distance falling by > 4× from `k=1→2` to `k=4→8`), because the loop is trained to converge to a start-independent attractor. | `state_init = "randn"` + convergence objective vs. iterative computation (§5.2 item 9) | W2-A6, measure `‖logits_k − logits_{k+1}‖` |
| **F10** | **The shipped probe config's KV cache grows ~linearly in `k`**, contradicting the invariant in `01_ARCHITECTURE.md` §2. | `core_pattern: null` ⇒ `full_attn` inside the loop `[V]` (§5.1) | Run the existing `test_attention_cache_does_not_grow_with_recurrence_depth` against `configs/prophet_500m_probe.json` |
| **F11** | **Length generalisation fails on every iterative task** even where the task has a short `n`-RASP-L program, because `k` is never correlated with `n` at training time. | [2409.15647](https://arxiv.org/abs/2409.15647): adaptivity is the mechanism | W2-A6 |

---

## 7. Ablation plan

### 7.0 Compute budget arithmetic

A100-80GB, bf16 dense peak ≈ 312 TFLOP/s. At a pessimistic **25 % MFU for small models
with a Python-heavy recurrent scan**, 2 hours = `2 × 3600 × 312e12 × 0.25 ≈ 5.6e17` FLOPs.
Using `6ND` with an effective-depth multiplier `m` folded into `N`:

| Model | `N` (active) | `m` (eff. depth / param depth) | Tokens affordable in 2 h |
|---|---:|---:|---:|
| 8M-param synthetic model | 8e6 | 4 | `5.6e17 / (6·8e6·4)` ≈ **2.9e9** |
| 30M-param synthetic model | 3e7 | 4 | ≈ **7.8e8** |
| 50M-param synthetic model | 5e7 | 8 | ≈ **2.3e8** |

Every task below needs `< 5e8` tokens of synthetic data. **All eight ablations fit
comfortably inside the 2-A100-hour-each budget; realistically the whole suite is
4–8 A100-hours**, i.e. **1.3–2.7 % of the 300-hour project budget** for the experiment that
decides whether our central architectural bet is real.

**On the 350M floor.** `01_ARCHITECTURE.md` §5 `[ABLATION A1]` warns that MoR
underperforms vanilla at 135M and only wins from 360M, so small ablations risk false
negatives. **That warning applies to language-modelling quality, not to algorithmic
expressivity.** The entire literature in §3–§4 runs these probes at 1–30M parameters
(TRM: 7M, 2 layers; HRM: 27M; DeltaProduct's group word problems: small synthetic models).
A parity or S₅ failure at 20M is an *architectural* result, not a scale artifact. The two
constraints do not conflict; W2's suite is not a substitute for A1.

### 7.1 The suite

Run in this order; each gates the next.

| ID | Task | Arms | Metric & pass bar | Gates | Cost |
|---|---|---|---|---|---|
| **W2-A1** | **PARITY / cumulative XOR** at Prophet scale (≥ 20M, full stack, real tokenizer). Train ≤ 32, test 32/64/128/256. **§6.0 already ran the 43k–86k version and it passed decisively** — this is the scale-up confirmation, not the discovery. | `beta_max ∈ {1.0, 2.0}` × `k ∈ {1,4,8}` × 3 seeds = 18 runs; both fused and reference kernels | Final-token accuracy at 4× train length. **Pass: `beta_max=2` > 90 %; `beta_max=1` < 60 %.** | If pass → adopt `beta_max=2.0` as default, project-wide. If `beta_max=2` fails *at scale* while passing at 43k, suspect the fused kernel (§7.3.3) before suspecting the theory. | ~0.3 h |
| **W2-A2** | **Modular arithmetic mod 3 and mod 5**, with and without brackets (Chomsky-hierarchy protocol of [2207.02098](https://arxiv.org/abs/2207.02098)). | `beta_max ∈ {1,2}` × `n_h ∈ {1,2}` × `k ∈ {1,4}` | Accuracy at 4× length. **Expect all `n_h=1` arms to fail mod 3 (F2).** | Whether `n_h > 1` is required for the *simplest* real state tracking, not just for S₅. | ~0.4 h |
| **W2-A3** | **S₅ and A₅ word problems.** Train on ≤ 128 products, extrapolate to 512 (DeltaProduct protocol). | `n_h ∈ {1,2,4}` × `beta_max ∈ {1,2}` × `{GDN-only core, core + 1 full-attn}` × `k ∈ {1,4,8}` | Accuracy at length 512. **Pass: `n_h=4, beta_max=2` > 90 %; expect `n_h=1` at chance for every `k`.** | The go/no-go on "our stack does hard state tracking". Also directly tests §3.4(b): the `+1 full-attn` arm should give **no** improvement. | ~1.5 h |
| **W2-A4** | **MQAR / multi-key associative recall** ([2312.04927](https://arxiv.org/abs/2312.04927) protocol) + multi-needle retrieval. | #KV pairs ∈ {8,32,128,512,2048} × `linear_head_dim ∈ {32,64,128}` × `k ∈ {1,2,4,8}` × linear:full ∈ {1:1, 3:1, 8:1, ∞} | Locate the knee. **Pass criteria for the *hypotheses*: `∂acc/∂k ≈ 0` (F4) and accuracy falling monotonically as linear:full rises past 3:1 (F5).** | Whether `k` is a dial for recall (predict: no) and what the minimum viable attention share is. **Highest information per GPU-hour in the suite.** | ~1.0 h |
| **W2-A5** | **Directed graph reachability**, controlled diameter `d ∈ {2,4,8,16,32}`, ≤ 64 nodes. | `{GDN-only core, core with 1 attention layer, dense transformer baseline}` × `k ∈ {1,2,4,8}` | Minimum `k` for > 90 % at each `d`. **Discriminator: is `k*(d)` linear (F6) or logarithmic?** | Whether the loop can do multi-hop search at all without attention inside it. Decides `[ABLATION A-KV]` (R04's shared-KV proposal). | ~1.0 h |
| **W2-A6** | **Adaptive depth.** Addition, copy, and parity, at mixed lengths. | `k`-schedule ∈ {log-uniform[1,8] (current), `k = ceil(log₂ n)`, `k ∝ n`, `halting="ponder"`} | Accuracy at 4× training length; plus F9's `‖logits_k − logits_{k+1}‖` curve. **Pass: a length-tied schedule beats log-uniform by > 20 points at 4× length.** | **The go/no-go on the "depth as a runtime dial" thesis** (§2.3). If length-tied `k` wins, change `train_loop_dist` and the inference API. | ~1.0 h |
| **W2-A7** | **TBPTT sweep** on the W2-A1/A6 winners. | `truncated_backprop_steps ∈ {2,4,8,full}` at `k = 8` | Does the accuracy-vs-`k` plateau move? **Pass for F7: plateau moves with tbptt.** | Whether our memory-saving truncation costs the capability the loop was bought for. | ~0.5 h |
| **W2-A8** | **int4 sensitivity.** Post-training-quantise the A1 and A4 winners. | fp16 vs int4-g64 | `ΔBPB` on held-out text vs `Δparity` and `Δrecall`. **Pass for F8: the sensitive-task delta is > 5× the BPB-implied delta.** | Whether the R08 quantisation plan must carve out sensitive capabilities. | ~0.2 h |

### 7.2 Config switches this requires

Per CLAUDE.md rule 3, every one of these is a switch, not a fork. Additions needed to
`prophet/config.py`:

```python
# MixerConfig
beta_max: float = 1.0          # write-strength ceiling; 2.0 admits negative eigenvalues
n_householder: int = 1         # DeltaProduct micro-steps per token

# RecurrentCoreConfig
loop_k_schedule: Literal["fixed","log_uniform","length_tied","ponder"] = "log_uniform"
loop_k_length_coeff: tuple[float,float] = (0.0, 1.0)   # k = clip(a + b*log2(n), 1, k_max)
```

### 7.3 Preconditions — do these before running anything

1. **Fix `configs/prophet_500m_probe.json`**: set `core_pattern: ["gdn"]`,
   `prelude_pattern`/`coda_pattern` to `["swa","full_attn"]`, and `nope_layers` to the
   index of the global layer. Otherwise every W2 result is confounded by an
   attention layer inside the loop (§5.1).
2. **Re-run `test_attention_cache_does_not_grow_with_recurrence_depth` against the shipped
   config**, not only against the canonical one (F10).
3. **Check `fla.ops.gated_delta_rule.chunk_gated_delta_rule` for `β > 1`** before trusting
   any fused-kernel W2-A1 number (§3.2).

---

## 8. References

Ordered by the section that uses them. `[S]` = search-summary sourced only (arXiv, ACL
Anthology, OpenReview, HuggingFace, Semantic Scholar and NeurIPS proceedings were all
blocked by this session's egress proxy — see §0). `[V]` = verified text or this
repository's source. `[T]` = standard complexity theory.

**§1 — Fixed-depth limits**

- Merrill & Sabharwal, *The Parallelism Tradeoff: Limitations of Log-Precision Transformers*, [arXiv:2207.00729](https://arxiv.org/abs/2207.00729), TACL 11:531–545 (2023). `[S]`
- Merrill & Sabharwal, *A Logic for Expressing Log-Precision Transformers*, [arXiv:2210.02671](https://arxiv.org/abs/2210.02671), NeurIPS 2023. `[S]`
- Merrill & Sabharwal, *Exact Expressive Power of Transformers with Padding*, [arXiv:2505.18948](https://arxiv.org/abs/2505.18948), NeurIPS 2025. `[S]`
- Li, Liu, Zhou & Ma, *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*, [arXiv:2402.12875](https://arxiv.org/abs/2402.12875), ICLR 2024. `[S]`
- Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought*, [arXiv:2310.07923](https://arxiv.org/abs/2310.07923), ICLR 2024. `[S]`
- Hahn, *Theoretical Limitations of Self-Attention in Neural Sequence Models*, [arXiv:1906.06755](https://arxiv.org/abs/1906.06755), TACL 2020. `[S]`
- Chiang & Cholak, *Overcoming a Theoretical Limitation of Self-Attention*, [arXiv:2202.12172](https://arxiv.org/abs/2202.12172), ACL 2022. `[S]`
- Hahn & Rofin, *Why are Sensitive Functions Hard for Transformers?*, [arXiv:2402.09963](https://arxiv.org/abs/2402.09963), ACL 2024. `[S]`
- Strobl, Merrill, Weiss, Chiang & Angluin, *What Formal Languages Can Transformers Express? A Survey*, [arXiv:2311.00208](https://arxiv.org/abs/2311.00208), TACL 12:543–561 (2024). `[S]`
- Bhattamishra, Ahuja & Goyal, *On the Ability and Limitations of Transformers to Recognize Formal Languages*, [arXiv:2009.11264](https://arxiv.org/abs/2009.11264), EMNLP 2020. `[S]`
- Dziri et al., *Faith and Fate: Limits of Transformers on Compositionality*, [arXiv:2305.18654](https://arxiv.org/abs/2305.18654), NeurIPS 2023. `[S]`
- Barrington (1986), bounded-width branching programs / S₅ word problem NC¹-completeness; Håstad (1986), PARITY ∉ AC⁰; Reingold (2008), USTCON ∈ L; Hesse (2001), division ∈ uniform TC⁰. `[T]`

**§2 — Looping and depth**

- Merrill & Sabharwal, *A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers*, [arXiv:2503.03961](https://arxiv.org/abs/2503.03961). `[S]`
- Dehghani, Gouws, Vinyals, Uszkoreit & Kaiser, *Universal Transformers*, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819), ICLR 2019. `[S]`
- Saunshi et al., *Reasoning with Latent Thoughts: On the Power of Looped Transformers*, [arXiv:2502.17416](https://arxiv.org/abs/2502.17416), ICLR 2025. `[S]`
- Fan, Du, Ramchandran & Lee, *Looped Transformers for Length Generalization*, [arXiv:2409.15647](https://arxiv.org/abs/2409.15647), ICLR 2025. `[S]`
- Giannou et al., *Looped Transformers as Programmable Computers*, [arXiv:2301.13196](https://arxiv.org/abs/2301.13196), ICML 2023. `[S]`
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn-0125), [arXiv:2502.05171](https://arxiv.org/abs/2502.05171). `[S]`
- Bae et al., *Mixture-of-Recursions*, [arXiv:2507.10524](https://arxiv.org/abs/2507.10524), NeurIPS 2025. `[S]`
- Xu & Sato, *A Formal Comparison Between Chain of Thought and Latent Thought*, [arXiv:2509.25239](https://arxiv.org/abs/2509.25239), ICML 2026. `[S]`
- Wang et al. (Sapient), *Hierarchical Reasoning Model*, [arXiv:2506.21734](https://arxiv.org/abs/2506.21734). `[S]`
- Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks* (TRM), [arXiv:2510.04871](https://arxiv.org/abs/2510.04871). `[S]`
- ARC Prize Foundation, *The Hidden Drivers of HRM's Performance on ARC-AGI*, [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis). `[S]`
- Farsang, Hasani, Rus & Grosu, *Looped SSMs: Depth-Recurrence and Input Reshaping*, [arXiv:2605.16048](https://arxiv.org/abs/2605.16048). `[S]`
- Li & Zhang, *DeepLoop: Depth Scaling for Looped Transformers*, [arXiv:2607.13491](https://arxiv.org/abs/2607.13491). `[S]` — post-dates this agent's knowledge cutoff; found via search only.

**§3–§4 — Bounded state and state tracking**

- Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models*, [arXiv:2404.08819](https://arxiv.org/abs/2404.08819), ICML 2024. `[S]`
- Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, [arXiv:2411.12537](https://arxiv.org/abs/2411.12537), ICLR 2025 (Oral). `[V]` (README) + `[S]`
- Siems et al., *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*, [arXiv:2502.10297](https://arxiv.org/abs/2502.10297). `[S]`
- Peng et al., *RWKV-7 "Goose" with Expressive Dynamic State Evolution*, [arXiv:2503.14456](https://arxiv.org/abs/2503.14456). `[S]`
- Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), ICLR 2025. `[S]`
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). `[S]`
- Jelassi, Brandfonbrener, Kakade & Malach, *Repeat After Me: Transformers are Better than State Space Models at Copying*, [arXiv:2402.01032](https://arxiv.org/abs/2402.01032), ICML 2024. `[S]`
- Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, [arXiv:2312.04927](https://arxiv.org/abs/2312.04927), ICLR 2024. `[S]`
- Arora et al., *Just Read Twice: Closing the Recall Gap for Recurrent Language Models*, [arXiv:2407.05483](https://arxiv.org/abs/2407.05483). `[S]`
- Wang, Zhu et al., *A Systematic Analysis of Hybrid Linear Attention*, [arXiv:2507.06457](https://arxiv.org/abs/2507.06457). `[S]`
- *Provably Shorter Scratchpads in Hybrid DeltaNet-Attention Decoders*, [arXiv:2605.16640](https://arxiv.org/abs/2605.16640). `[S]` — post-cutoff; search only.
- *Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models*, [arXiv:2603.08859](https://arxiv.org/abs/2603.08859). `[S]` — post-cutoff; search only.
- Liu, Ash, Goel, Krishnamurthy & Zhang, *Transformers Learn Shortcuts to Automata*, [arXiv:2210.10749](https://arxiv.org/abs/2210.10749), ICLR 2023. `[S]`
- Delétang et al., *Neural Networks and the Chomsky Hierarchy*, [arXiv:2207.02098](https://arxiv.org/abs/2207.02098), ICLR 2023. `[S]`
- Sanford, Hsu & Telgarsky, *Transformers, Parallel Computation, and Logarithmic Depth*, [arXiv:2402.09268](https://arxiv.org/abs/2402.09268), ICML 2024. `[S]`
- Saparov et al., *Transformers Struggle to Learn to Search*, [arXiv:2412.04703](https://arxiv.org/abs/2412.04703), ICLR 2025. `[S]`

**§5 — Positional encoding**

- Zhou et al., *What Algorithms can Transformers Learn? A Study in Length Generalization* (RASP-L), [arXiv:2310.16028](https://arxiv.org/abs/2310.16028), ICLR 2024. `[S]`
- Kazemnejad, Padhi et al., *The Impact of Positional Encoding on Length Generalization in Transformers*, [arXiv:2305.19466](https://arxiv.org/abs/2305.19466), NeurIPS 2023. `[S]`
- *NoPE: The Counting Power of Transformers with No Positional Encodings*, [arXiv:2505.11199](https://arxiv.org/abs/2505.11199). `[S]`
- McLeish et al., *Transformers Can Do Arithmetic with the Right Embeddings* (Abacus), [arXiv:2405.17399](https://arxiv.org/abs/2405.17399). `[S]`

**Repository sources `[V]`**

- `prophet/modeling/layers.py` — `GatedDeltaNet` (lines 354–520): `α`/`β` parameterisation, delta-rule scan, FLA call.
- `prophet/config.py` — `MixerConfig`, `RecurrentCoreConfig`, `layer_mixer`, `section_layout`.
- `prophet/modeling/model.py` — section construction, residual scaling by effective depth, `sample_loop_k`.
- `configs/prophet_500m_probe.json`, `scripts/design_search.py`, `docs/01_ARCHITECTURE.md`.
- `python -m prophet.budget configs/prophet_500m_probe.json`.

---

## 9. Appendix — positional encoding as a symptom (supporting §1, §5)

The brief asks why RoPE/NoPE tricks are needed at all and what that reveals. The short
answer: **self-attention is permutation-equivariant, so position is not a property of the
mechanism but a feature we bolt on, and every bolt-on is a different bet about how
position should extrapolate.**

- Kazemnejad et al. `[2305.19466]`: across APE, T5-relative, ALiBi, RoPE and NoPE, the
  explicit schemes are **not** well suited to downstream length generalisation, while
  **NoPE outperforms all of them at no extra computation**; NoPE can *represent* both
  absolute and relative PE, and under SGD converges to something resembling T5-relative
  attention. `[S]` Causal masking alone already breaks permutation symmetry, which is why
  NoPE works at all in a decoder.
- But NoPE is not free: `[2505.11199]` shows NoPE transformers **cannot express PARITY**,
  while being able to express integer solution sets of multivariate Diophantine equations
  under average-hard attention — a strange and revealing capability profile. `[S]`
- McLeish et al. `[2405.17399]`: arithmetic failure is largely **positional bookkeeping** —
  an embedding encoding each digit's index relative to the number's start restores it, and
  *only then* do input injection and recurrent layers add further gains. `[S]` Note the
  ordering: the positional fix is a **precondition** for the recurrence to pay off.
- Zhou et al. `[2310.16028]`: length generalisation follows from the existence of a short
  length-independent RASP-L program. Positional encoding determines which programs are
  short. `[S]`

**What this reveals.** Positional encoding is a hack in the precise sense that it patches
an architectural symmetry the task does not have. Recurrent layers *do* genuinely fix part
of it — an ordered state update carries position intrinsically, which is why several
shipped hybrids drop positional encodings on layers adjacent to recurrent ones — but they
move the failure rather than removing it: bounded state means position is remembered only
as far as the state's capacity allows, so the failure mode migrates from
"cannot extrapolate positions" to "cannot retain distinctions past the state's capacity"
(§3.3, F4). The honest summary for Prophet is that **NoPE + recurrence buys length
extrapolation and pays for it in exact recall**, which is the same trade as everything else
in §3, arriving through a different door.

**One inconsistency to fix while we are here `[V]`:** `MixerConfig.attention_sink_tokens`
defaults to `4`, `scripts/design_search.py` passes `1`, and
`docs/01_ARCHITECTURE.md` §4 says "1 per layer". Windowed attention without adequate sinks
collapses in streaming. Pick one value, justify it, and make the tests assert it.
