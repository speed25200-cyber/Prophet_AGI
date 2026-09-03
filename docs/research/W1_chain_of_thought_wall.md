# W1 — The Chain-of-Thought Wall: a 1-bit Feedback Channel Wrapped Around an Unbounded Tape

> **Track W1.** What chain-of-thought *is*, mechanically and formally, and therefore what a
> weight-shared recurrent core can and cannot replace.
>
> **Verdict in one line.** The information-bottleneck framing survives — but it is a bottleneck
> on *one specific link* (the top-of-stack → layer-0 feedback path), not on the model's memory,
> and the correct capacity figure is ~1–2 bits per step, not 15. The error-correction half of the
> hypothesis is empirically confirmed and mechanistically wrong: the stabiliser is **per-step
> training supervision**, not inference-time projection onto the token manifold. And the function
> our recurrent core fundamentally cannot replace is not depth — it is the **scratchpad**, which
> is now a theorem, not an intuition.

---

## 0. Provenance — read before trusting any number

The egress proxy in this session blocked **every** domain reachable by `WebFetch`, including
`arxiv.org`, `ar5iv`, `huggingface.co`, `openreview.net`, `semanticscholar.org`,
`neurips.cc` and `anthropic.com`. Direct `curl` to those hosts returns `403 CONNECT tunnel
failed` (organisation egress policy). **No paper PDF was read in full during this track.**

`WebSearch` worked. Every external figure below therefore comes from a search-engine summary
of an abstract or paper body, not from the source document. Markers:

| Marker | Meaning |
|---|---|
| `[S]` | Confirmed this session via web-search summary. Abstract-level confidence. **Re-verify before spending compute.** |
| `[M]` | From model memory, not confirmed this session. **Treat as a hypothesis, not a fact.** |
| `[C]` | Computed in this document from first principles. Check the arithmetic, not the source. |
| `[R]` | From this repository (`prophet/`, `docs/`). Locally verifiable. |

Two figures below are **internally inconsistent across secondary sources** and are flagged in
place (§3, the Anthropic hint-verbalisation sub-breakdown).

---

## 1. What CoT is, mechanically

### 1.1 The arithmetic in the brief, checked

The brief's arithmetic is correct as stated:

| Quantity | Value | Note |
|---|---:|---|
| `d_model = 2048`, bf16 container | 2048 × 16 = **32 768 bits** | `[C]` |
| Token from a 32 768-entry vocabulary | log₂(32768) = **15.000 bits** | `[C]` |
| Ratio | **2 184.5 : 1** | `[C]` — the brief's "roughly 2000:1" |

It is also, almost verbatim, the opening paragraph of the field's own survey. *A Survey on Latent
Reasoning* (arXiv:2507.06203) states: "Explicit reasoning transmits discrete tokens (≈15 bits
each), whereas latent reasoning exchanges full 2560-dimensional FP16 hidden states (≈40 960 bits
each), revealing a ~2.7 × 10³-fold bandwidth gap" `[S]`. And this repository already implements
the measurement — `prophet/analysis/bandwidth.py` `[R]` opens with the same 2048/32768/15/2000:1
calculation and then correctly notes that *both* numbers are wrong.

**So the framing is not novel.** That is worth saying plainly: W1's value cannot be the framing.
It has to be in getting the numbers right and in the design consequence, which the survey
literature gets wrong.

Prophet's own configurations `[C]` from `docs/01_ARCHITECTURE.md` `[R]`:

| Model | d_model | Vocab | Container bits | log₂ V | Nominal ratio |
|---|---:|---:|---:|---:|---:|
| prophet-mini | 1280 | 32 768 | 20 480 | 15.000 | **1 365 :1** |
| prophet-main (donor vocab) | 1536 | 151 936 | 24 576 | 17.213 | **1 428 :1** |

### 1.2 Both numbers are wrong, in the same direction

**The numerator (token side) is far too large.** A token does not carry log₂|V| bits; it carries
its surprisal, and in expectation the entropy of the distribution it was sampled from.
Measurements on reasoning traces: mean per-token entropy by role is **1.03–1.98 bits** against a
vocabulary maximum of 17.2 bits `[S]` (arXiv:2604.26355); approximately **20 % of tokens are
high-entropy "forking" tokens** at semantic/logical choice points, while the majority are
generated at very low entropy `[S]` (arXiv:2506.01939). Independently, mutual-information probing
of reasoning traces finds MI concentrated in sparse peaks at reflection tokens — "Hmm", "Wait",
"Therefore" `[S]` (arXiv:2506.02867), and counterfactual resampling finds that only planning and
uncertainty-management sentences carry high causal weight `[S]` (arXiv:2506.19143, *Thought
Anchors*).

**Working figure: the realised CoT channel runs at ≈ 1.2 bits/token, not 15.** The bottleneck is
an order of magnitude tighter than the brief claims.

**The denominator (state side) is also far too large.** 32 768 bits is a container size. The
residual stream is strongly anisotropic; the number of directions carrying variance is a small
fraction of `d`. `prophet/analysis/bandwidth.py` `[R]` measures this with the participation ratio
and charges a deliberately conservative **4 bits per effective dimension** (justified by the fact
that models run acceptably at 8-bit activations). If the effective rank is 5–15 % of `d` — the
range I would bet on but which **has not been measured here** — then for d=1536 the usable state
is roughly **310–920 bits**, not 24 576.

**Corrected ratio: ~250× to ~750×, not ~2000×.** The qualitative claim survives both corrections.
The magnitude falls by 3–9×. Measuring it is ablation **W1-A0** (§7) and costs about one GPU-hour.

### 1.3 The correction that actually matters: nothing is discarded

The brief says "99.95 % of the computed state is discarded at every step." **This is false as
stated, and the way in which it is false changes the design.**

Consider what a decoder does at step *t*. It computes, at every layer, a residual stream and a
K/V pair. Concretely for Prophet-main `[C]`, with 4 attention layers (2 prelude, 2 coda),
`n_kv_heads = 2`, `head_dim = 128`, bf16:

```
4 layers × 2 (K,V) × 2 kv-heads × 128 dims × 2 bytes = 4 096 bytes = 4 KiB/token = 32 768 bits
```

which reproduces exactly the 4 KiB/token figure R02 measured `[R]`. Those 32 768 bits are **not
discarded**. They persist in the cache and every subsequent position can read them by content.
So per emitted token the model *writes* ~32 768 bits into a persistent, addressable buffer and
*emits* ≤17.2 bits. The brief's 2000:1 ratio is real — but it is the ratio between the **buffer
write** and the **feedback**, not between "computed" and "kept".

Worse for the naive framing, and better for the analysis: **those 32 768 KV bits carry no
information at all**, in the Shannon sense.

> **Proposition 1 (per-step information ceiling).** Let the model be a deterministic function of
> its token prefix, and let `x_t ~ p(· | x_<t)` be the sampled token. Every activation at step
> *t* — residual streams, K/V entries, MoE routes, the recurrent state — is a deterministic
> function of `x_≤t`. By the data-processing inequality, conditioned on the prefix, the only new
> random variable produced at step *t* is `x_t`. Hence the total information an *n*-step chain of
> thought injects into the process, beyond what the prompt deterministically implies, is exactly
> `Σ_t H(x_t | x_<t)`. `[C]`

**Corollary.** A 1 000-token reasoning trace at ~1.2 bits/token carries **≈ 1.2 kbit ≈ 150 bytes**
of genuine decision content. That is the entire non-deterministic content of the model's
reasoning. Everything else in that trace is recomputable from the prompt.

This is corroborated from the adversarial side: the *usable covert* capacity of a CoT channel
under a paraphrase defence is measured at **≤ 3 bits per kilobyte of text**, and steganographic
schemes reach only ~0.2 bits/token `[S]`. If a *deliberate* encoder can only push 0.2 bits/token
past a paraphraser, the incidental channel is not secretly wide.

### 1.4 So where is the bottleneck, precisely?

In a decoder, position *t*'s layer-*l* activations are readable by position *t' > t* only at
layer *l+1* or above. An inter-position dependency chain therefore descends at most one layer per
hop and is **capped at L hops**, no matter how many positions exist. This is exactly why filler
tokens do not buy serial reasoning: Pfau et al. characterise the problems filler tokens help with
in terms of bounded quantifier depth of a first-order formula, i.e. *parallel* problems, and note
that filler tokens "need not provide information about the intermediate computational steps"
`[S]` (arXiv:2404.15758). Filler positions carry **full-width hidden states** — bandwidth is not
their limitation. Chaining is.

The one path in a transformer that resets the layer budget — that takes the *result* of a
computation and re-enters it at layer 0 — is the emitted token.

> **The mechanical definition of CoT.**
> CoT is the only unbounded-depth feedback path in a transformer. Its capacity is
> `H(x_t | x_<t) ≈ 1–2 bits` per step. Everything else the model computed is preserved in the KV
> buffer and remains readable — but only sideways, through attention, and only for L hops.

Both design consequences follow immediately, and they are *opposite*:

- **Widening the bottleneck** means widening the *feedback* path (loop the core, or feed the
  hidden state back). Prophet's recurrent core does exactly this. It adds **no memory**.
- **Extending the tape** means adding persistent, addressable cells. Only emitting tokens (or an
  explicit buffer) does this. Prophet's loop does **not** do it, and D1 — "the looped core
  contains only bounded-state mixers" `[R]` — is precisely the decision that forbids it.

The KV-cache win in §2 of the architecture document and the scratchpad loss are **the same
decision**. That is the finding of this track.

---

## 2. The formal picture

All classes below are for uniform circuit families; "log-precision" means O(log n)-bit values.

### 2.1 Without chain of thought

| Setting | Class | Source |
|---|---|---|
| Constant-depth, log-precision transformer | ⊆ uniform **TC⁰** | Merrill & Sabharwal, arXiv:2207.00729 / 2210.10749 `[M]` |
| Constant-depth, **constant-bit** precision | ⊆ **AC⁰**, a proper subset of TC⁰ | Li et al., arXiv:2402.12875 `[S]` |
| Averaging-hard-attention, masked-pre-norm, **polynomial padding** | **= FO-uniform TC⁰** exactly | Merrill & Sabharwal, arXiv:2505.18948 `[S]` |

The last line matters: adding a polynomial number of *blank* positions changes nothing. Extra
memory cells alone do not buy expressivity.

### 2.2 With chain of thought

Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought*
(arXiv:2310.07923) `[S]`, with `t(n)` intermediate decoding steps:

| `t(n)` | Result |
|---|---|
| Θ(log n) | "Pushes the limits of standard transformers only slightly"; upper-bounded within **L** |
| Θ(n), with projected pre-norm | Recognises **all regular languages** — a clear new ability under standard conjectures; contained in the context-sensitive languages |
| poly(n), generalised pre-norm | Recognises **exactly P** — the first exact characterisation of a transformer class by a standard complexity class |

Two independent constructions give the lower bounds:

- **Li, Liu, Zhou & Ma** (arXiv:2402.12875) `[S]`: with T(n) CoT steps, a constant-depth
  constant-precision decoder simulates Boolean circuits of size T(n); poly steps → **P/poly**.
  Their framing: "CoT empowers the model with the ability to perform inherently serial
  computation, which is otherwise lacking in transformers."
- **Feng, Zhang, Gu, Ye, He & Wang** (arXiv:2305.15408) `[S]`: bounded-depth transformers cannot
  directly emit correct answers for arithmetic-expression evaluation or linear-equation solving
  **unless model size grows super-polynomially in n**; constant-size autoregressive transformers
  solve both by generating CoT derivations, and also solve dynamic-programming problems.

### 2.3 With looping / padding — the parallelism ceiling

Merrill & Sabharwal, *Exact Expressive Power of Transformers with Padding* (arXiv:2505.18948)
`[S]`:

> Padded transformers with **O(logᵈ n) looping recognise exactly TCᵈ**. With **polylogarithmic
> looping, padded transformers converge to NC** — "the best that could be expected without losing
> parallelism (unless NC = P)."

This is the clean statement of the separation:

> **Separation 1 (depth).** Recurrence buys *parallel* depth and tops out at **NC**. CoT with
> polynomial steps reaches **P**. Under NC ≠ P, no polylogarithmic amount of recurrence replaces
> polynomial-length CoT.

Xu & Sato give the complementary positive result twice — *To CoT or To Loop?* (arXiv:2505.19245)
and *A Formal Comparison Between Chain of Thought and Latent Thought* (arXiv:2509.25239, ICML
2026) `[S]`: looped/latent thought **efficiently simulates parallel computation** formalised as
DAG evaluation and "yields separations beyond the polylogarithmic regime", while CoT with
**stochastic** decoding excels at approximate inference and sampling over self-reducible
structures. Latent depth is not weaker everywhere; it is weaker exactly where seriality is
irreducible, and *stronger* where parallelism is available.

### 2.4 With looping and a *bounded* state — the memory separation

This is the 2026 result that decides §5.

**Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation**
(arXiv:2605.30757) `[S]` compares three regimes — compressed latent loop, full sequence-state
loop, and CoT scratchpad:

> "A compressed loop is limited by the size of its recurrent state. Running the loop longer adds
> computation but does not by itself create a growing scratchpad, so a loop with a small
> recurrent state remains a small-space reasoner even when run for many steps. Under a standard
> complexity assumption, such loops **cannot decide problems that are P-complete under logspace
> reductions**, whereas polynomial-length chain of thought can."

> **Separation 2 (memory).** A loop whose state is bounded is a bounded-space machine. Time does
> not buy space. Only writing cells buys space.

And a matching quantitative version for the KV side: *How Much Cache Does Reasoning Need?*
(arXiv:2604.17935) `[S]` studies k-hop pointer chasing under a shared KV cache of size *s*,
attention dimension *m*, *H* heads, *p*-bit precision, and proves an unconditional upper bound
`L = O(min(k, ⌈k/s⌉ log s) · log n /(mp))` with a matching `L = Ω(max(⌈k/s⌉, log n/(Hmp)))`.
Depth and cache trade off explicitly: **halving the cache roughly doubles the required depth for
multi-hop reasoning.**

### 2.5 Somebody has already done the information-theoretic framing

The brief asks whether anyone is doing the information-theoretic framing properly. **Yes** —
*The Information Bottleneck of Chain-of-Thought and How Latent CoT Overcomes It* (OpenReview
`cCIdxLoLJ5`, Oct 2025) `[S]`:

> "Although each forward pass can activate a vast amount of neurons, the information the model
> writes down is limited to a single token […] Each token only conveys O(log |V|) bits […] for
> some natural problems, such as pointer chasing and computing parity, either 1-layer
> transformers or constant-layer finite-precision transformers require a rather long CoT to
> solve […] allowing the Transformer to write high-dimensional embeddings to the CoT
> significantly reduces the CoT length."

That is the brief's hypothesis, published, with theorems. Its result is a **length** result, not
a capability result: latent CoT shortens the chain; it does not lift the class. Which is exactly
consistent with §2.4 and is the honest form of the claim.

### 2.6 The corollary Prophet cannot dodge

Every separation above is asymptotic in *n*. **Prophet's `k` is a constant** — `default_loop_k =
4`, `train_loop_max = 8` `[R]` — independent of input length.

> A constant number of loop iterations multiplies depth by a constant and **does not change the
> complexity class at all**. Under the formal literature, Prophet's recurrent core earns exactly
> zero asymptotic credit.

Its entire justification is empirical: better loss per stored parameter (Saunshi et al.,
arXiv:2502.17416 `[S]`: a k-layer block looped L times "nearly matches the performance of a
kL-layer non-looped model"). That is a real and sufficient justification — R04's A1 is the right
experiment — but the track must stop describing the loop as buying reasoning *power*. It buys
reasoning *density per parameter*. If we ever want the class, `k` has to grow with problem size,
which means **halting is not a nicety, it is the only route from constant to n-dependent depth**.

---

## 3. The faithfulness problem, quantified

### 3.1 The measurements

| Study | Intervention | Number |
|---|---|---|
| Turpin et al., arXiv:2305.04388 `[S]` | Reorder few-shot options so the answer is always "(A)"; other biasing features | Accuracy drops **by up to 36 %** across 13 BIG-Bench Hard tasks (GPT-3.5, Claude 1.0). Models "systematically fail to mention" the bias in the explanation. |
| Anthropic, arXiv:2505.05410 `[S]` | Insert a hint, check whether the CoT admits using it | Averaged over hint types, **Claude 3.7 Sonnet verbalises the hint 25 %** of the time; **DeepSeek R1, 39 %**. |
| — same, sub-breakdown | "More concerning" hint categories | **Sources disagree.** One secondary source reports Claude 41 % / R1 19 %; another reports Claude 20 % / R1 29 %. **Do not cite the sub-breakdown until the paper is read.** `[S, inconsistent]` |
| Lanham et al., arXiv:2307.13702 `[S]` | Early answering, adding mistakes, paraphrasing, filler substitution | Sensitivity to CoT perturbation is **highest on AQuA/LogiQA** (logical reasoning), **lowest on ARC/OpenBookQA** (crystallised knowledge). Faithfulness **peaks near 13 B and decreases from 13 B → 175 B** — inverse scaling. CoT's gain comes neither from added test-time compute alone nor from the phrasing. |
| Arcuschin et al., arXiv:2503.08679 `[S]` | Natural, non-adversarial prompts; ask "Is X > Y?" and "Is Y > X?" | Models produce coherent arguments for **both**. "Implicit Post-Hoc Rationalization" at rates **up to 13 %** in production models; frontier reasoning models including R1 are not exempt. |
| Huang et al., arXiv:2310.01798 `[S]` | Intrinsic self-correction, no external feedback | On GSM8K, GPT-3.5 **corrects 7.6 %** of its wrong answers and **breaks 8.8 %** of its right ones. Net negative. |
| Overthinking, arXiv:2604.10739 / 2506.04210 `[S]` | Vary the thinking budget | Inverted-U: **82.2 % → 87.3 %** as thinking tokens go 385 → 1 100, then **87.3 % → 70.3 %** at 15 980 tokens. Incorrect answers correlate with longer chains. |

### 3.2 The caveat that cuts the other way

*The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies*
(arXiv:2605.10799) `[S]` finds that measured sensitivity to a corrupted chain is partly an
artefact: chains embed explicit answer tokens, and the model is reading those. **Corruption
studies therefore over-estimate faithfulness.** The true causal dependence on the reasoning
content is lower than the published numbers suggest, not higher.

Similarly, *LLM Reasoning Is Latent, Not the Chain of Thought* (arXiv:2604.15726) `[S]` argues
that the primary object is the latent-state trajectory and that surface CoT is "only a partial
interface", with a third hypothesis on the table (H0) that most apparent gains are explained by
**generic serial compute** rather than by any privileged representational object.

### 3.3 What this means for Prophet

Three consequences, all budget-relevant:

1. **Faithfulness is not a property we can lose, because we never had it.** Between 61 % and 75 %
   of decisive influences go unverbalised in frontier reasoning models. Replacing part of the
   trace with latent steps therefore costs far less interpretability than it appears to.
2. **Do not spend budget making the trace faithful.** Spend it making the trace **short**. The
   inverted-U result says the marginal reasoning token is often *negative* value past ~1–2 k
   tokens, which for a device-targeted model is a gift: we were never going to afford 16 k
   thinking tokens at 90–180 tok/s on an iPhone `[R]`.
3. **But keep the anchors in tokens.** Not for faithfulness — for monitorability and for accuracy
   (§4.4). If the anchors are free, take them.

---

## 4. Latent reasoning: what works and what breaks

### 4.1 The table

| Method | arXiv | Verdict | Numbers `[S]` | Failure mode |
|---|---|---|---|---|
| **Scratchpad** (Nye et al.) | 2112.00114 | Works, foundational | Long addition, polynomial eval, program execution — from "cannot" to "can" | — |
| **Pause tokens** (Goyal et al.) | 2310.02226 | Conditionally | 1 B model, gains on **8 of 9** tasks: **+18 % EM SQuAD**, +8 % CommonsenseQA, +1 % GSM8K | Only works if pause tokens are in **both** pretraining and finetuning; "zero-delay inference catastrophe" — the model breaks entirely if you omit pauses at inference |
| **Filler tokens** (Pfau, Merrill, Bowman) | 2404.15758 | Narrow | Solves two hard algorithmic tasks unreachable without intermediate tokens | "Learning to use filler tokens is difficult and requires specific, dense supervision to converge." Class limited to bounded quantifier depth — **no serial composition** |
| **Quiet-STaR** (Zelikman et al.) | 2403.09629 | Works, expensive | Zero-shot **GSM8K 5.9 → 10.9 %**, **CommonsenseQA 36.3 → 47.2 %** | Samples a rationale per token during training |
| **COCONUT** (Hao et al.) | 2412.06769 | Partially | **GSM8K 34.1 %** with 6 continuous thoughts vs **16.5 %** no-CoT; performance rises c=1→2 then **drops at c=3** | Multi-stage curriculum essential; degrades at scale (below) |
| **Huginn** (Geiping et al.) | 2502.05171 | Works at 3.5 B / 800 B tokens | **ARC-C 27.99 (r=4) → 38.23 (r=32)**; ARC-E 69.91, MMLU 31.38 at r=32 | Uneven saturation: **ARC-E moves 0.42 pts from r=16→32 and SciQ drops**; "scaling recurrence depth fails to match explicit reasoning" on GSM8K |
| **Looped transformers** (Saunshi et al.) | 2502.17416 | Works for reasoning | k-layer looped L times ≈ kL-layer non-looped on addition, p-hop induction, math | Explicitly **not** for memorisation |
| **Soft Tokens / Soft Thinking** (Butt et al.) | 2509.19170 | Partial | **pass@1 parity** with hard CoT, **pass@32 gains**, better robustness. Authors' own summary: "**train soft, infer hard**" | No pass@1 gain |
| **CCoT** (Cheng & Van Durme) | 2412.13171 | Works | Contentful, variable-length continuous contemplation tokens; adaptive on demand | Needs a teacher CoT to compress |
| **Token Assorted** (Su et al., ICML 2025) | 2502.03275 | Works | **+4.2 %** Math (Llama-3.2-1B), **+4.1 %** GSM8K (3B), **+13.3 %** Fresh-Gaokao (8B), at **−17 % trace length** | Needs a VQ-VAE and gold traces |
| **HybridCoT** (ICLR 2026) | OpenReview `4mfGbMzTwu` | Works | **94 % of full-text-CoT accuracy at ~50 % compute (1.97× speedup)** on AIME/MATH | **"Retaining math tokens is essential; without these symbolic anchors, the model suffers from the same hallucination issues as fully latent approaches."** |
| **LOTUS** | 2606.31779 | Works | Looped padded transformer + **per-latent-position cross-entropy against the gold CoT-step token** | States plainly: "existing latent CoT methods underperform explicit CoT beyond 1 B parameters, **and the gap widens with scale**" |
| **RiM** (Aichberger & Hochreiter) | 2605.30343 | Works | Fixed **memory blocks** processed in one forward pass; matches or exceeds latent baselines without autoregressive thought generation | Two-stage curriculum required: ground the blocks on explicit steps, then discard step supervision |
| **Thoughtbubbles** (Stanford NLP) | 2510.00219 | Works, unsupervised | Forks/deletes residual streams mid-network; learned in **pretraining with LM loss only**; beats standard decoder LMs on perplexity and zero-shot evals **at half the training budget** | Not yet reproduced at our exact scale |
| **UT + memory tokens** | 2604.21999 | Works | Sudoku-Extreme: **T=0 always fails; T=8 reliably succeeds (57.4 ± 0.7 % exact match); T=8–32 plateau; T=64 dilutes.** Mean halt falls 11.6 (T=8) → 8.3 (T=64) | A "router initialisation trap" kills the majority of ACT runs |

### 4.2 The negative results — read these before committing compute

Four independent 2025–2026 papers say the latent tokens are frequently **not doing the work**:

- **The Illusion of Superposition?** (arXiv:2604.06374) `[S]`. Logit-lens and entity-level
  probing. Only models **trained from scratch** show signs of superposition. For training-free
  Soft Thinking, off-the-shelf LLMs process superposed inputs essentially identically to discrete
  tokens: "entropy profiles match, KL divergences approach zero, cosine similarities exceed
  0.99." For fine-tuned COCONUT: **"a model achieves 96.6 % accuracy without any latent tokens,
  and entity-level probing reveals no step-by-step reasoning during latent computation."**
- **Do Latent Tokens Think?** (arXiv:2512.21711) `[S]`. COCONUT tokens are "uninterpretable
  placeholders", show **minimal sensitivity to steering**, and drive **strong shortcut
  dependence** on answer patterns and contextual cues.
- **Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer** (arXiv:2507.02199) `[S]`.
  Logit Lens and Coda Lens on Huginn-3.5B: "limited evidence of interpretable latent CoT",
  significant probing inconsistency across recurrent blocks.
- **How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?**
  (arXiv:2602.22441, ICLR 2026) `[S]`. **"Pervasive shortcut behaviour, where methods achieve
  high accuracy without relying on latent reasoning."**

**Methodological consequence for our ablations:** every latent-reasoning result must be reported
alongside a *latent-ablated control* — the same checkpoint evaluated with zero latent steps. If
the control matches, the latent machinery is decorative. This is now a required column in §7.

### 4.3 Why latent reasoning breaks — three mechanisms, none of which is "the manifold"

1. **Gradient/representation dual collapse.** *What Makes Effective Supervision in Latent
   Chain-of-Thought: An Information-Theoretic Analysis* (arXiv:2606.20075) `[S]` diagnoses
   "gradient attenuation along the optimization path and representational drift in the latent
   space", and decomposes the fix into **trajectory supervision** (dense per-step signal) and
   **space supervision** (preserve the latent manifold's semantic structure). Notably:
   "rigid geometric compression can collapse the reasoning space, whereas **generative
   reconstruction provides a more flexible semantic anchor**." That is a direct argument
   *against* hard projection and *for* a decodability loss.
2. **Feature collapse from recycling hidden states into untied embeddings.** COCONUT "degrades
   sharply at the 8 B scale (**50.3 % → 41.5 %** average), falling below even explicit CoT […]
   larger models with **untied embedding weights** suffer severely when hidden states are
   directly recycled as inputs" `[S]` (arXiv:2602.10229). **Flag for Prophet-main**: the donor
   conversion inherits the donor's embedding-tying decision; if it is untied, any hidden-state
   recycling scheme inherits this failure mode.
3. **The exploration–execution trade-off.** *Capabilities and Fundamental Limits of Latent
   Chain-of-Thought* (arXiv:2602.01148) `[S]` frames the observed pattern — latent CoT excelling
   at exploration (**ProsQA 97.0 %**) and failing at computation (**GSM8K 34.1 %**) — as a
   theorem: "high certainty enables precise execution but inhibits exploration, while low
   certainty facilitates search but causes error accumulation." They introduce a **Symbolic
   Index** quantifying decisional commitment, and **prove curriculum learning is necessary**
   because direct training fails from distributional mismatch.

### 4.4 Verdict on the brief's error-correction hypothesis

> **"The same discretisation is also error correction. Projecting onto the token manifold snaps
> the state back to something the model has seen."**

**The phenomenon is confirmed. The mechanism is probably wrong.**

Confirmed: HybridCoT's ablation is as direct as evidence gets — remove the symbolic anchor tokens
and "the model suffers from the same hallucination issues as fully latent approaches" `[S]`. And
2602.01148 gives the error-accumulation half a formal statement.

Probably wrong, in this sense: three of the strongest latent methods stabilise **without any
inference-time projection**. LOTUS supervises each latent position against the gold CoT token and
then runs continuously. RiM grounds memory blocks on explicit steps in stage 1 and **discards the
step supervision in stage 2**. 2606.20075 finds generative reconstruction beats rigid geometric
compression. Soft Tokens finds "train soft, infer hard" — the discretisation helping at
*inference*, but only after RL training that never discretised.

> **Refined hypothesis (W1's version).** Discreteness is a *sufficient* stabiliser, not a
> necessary one. What a latent step actually needs is a **decodability constraint** — that the
> state be *mappable back* to the token manifold — and that constraint is cheapest to impose in
> the **loss**, not in the forward pass. Snapping at inference throws away the width you paid
> for; supervising in training keeps it.

This is a falsifiable difference and ablation **W1-A3** (§7) separates the two with one dial.

---

## 5. Depth versus scratchpad

*The section the brief cares most about. The claim is right, and it is stronger than stated.*

### 5.1 CoT does three things, not two

The brief proposes two functions: serial depth, and a re-readable external buffer. The evidence
supports **three**, and the third is what the latent-reasoning literature keeps rediscovering.

**F1 — Vertical feedback (serial depth).** The emitted token is re-embedded at layer 0. This is
the *only* mechanism that resets the depth budget; without it, cross-position dependency chains
are capped at L hops (§1.4, and Pfau et al.'s characterisation). Formal weight: Li et al.
(T steps → circuits of size T), Merrill & Sabharwal (poly steps = P).

**F2 — Persistent, content-addressable, unbounded memory (the scratchpad).** Each step appends a
KV record readable by content at any later step, and the buffer grows without bound. Formal
weight: arXiv:2605.30757 (bounded-state loops cannot decide P-complete problems under logspace
reductions), arXiv:2604.17935 (depth Ω(⌈k/s⌉·…) for k-hop chasing under cache size s).

**F3 — Discretisation (commitment, error control, monitorability).** No formal weight. Empirical
weight: HybridCoT's anchor ablation, 2602.01148's exploration–execution theorem.

### 5.2 Which of the three can Prophet's core provide?

| Function | Prophet's recurrent core (`prelude → GDN core × k → coda`) | Why |
|---|---|---|
| **F1 serial depth** | **Yes — up to a constant factor.** k=4 gives ×4 depth for ×1 parameters. | But k is constant in n, so the class is unchanged (§2.6). Only halting can make k grow with difficulty. |
| **F2 scratchpad** | **No.** | D1 `[R]` puts *only bounded-state mixers* in the loop, for KV-memory reasons. arXiv:2605.30757: time does not buy space. The core has no writable cells. |
| **F3 discretisation** | **Removed.** | The loop is continuous by construction. |

The architecture's own measurement makes this concrete `[R]`
(`test_attention_cache_does_not_grow_with_recurrence_depth`):

| | k=1 | k=8 |
|---|---:|---:|
| Attention cache bytes | 10 240 | **10 240** |
| Recurrent state bytes | 11 264 | 90 112 |
| Recurrent state at 8× context | — | **unchanged** |

"Unchanged with context" is exactly the property that makes it *not a scratchpad*. The design
succeeded at what it aimed for and, by the same stroke, forfeited F2.

> **Verdict on the brief's claim.** *Correct, and now provable.* A fixed-size recurrent state
> cannot replace the scratchpad — arXiv:2605.30757 (May 2026) states it as a theorem rather than
> an intuition, and arXiv:2604.21999 confirms it empirically at small scale (**T=0 memory tokens:
> every configuration fails; T=8: 57.4 % exact match on Sudoku-Extreme**).

### 5.3 Three refinements that change what to build

**(a) The two functions are not equally scarce.** F1 is cheap (loop the core; Prophet already
has it) and *is not the binding constraint*. F2 is expensive and missing from the loop — but it
is **already present in Prophet whenever it emits a token**, because the prelude/coda KV cache
*is* the scratchpad. So the design question is not "can recurrence replace the scratchpad?" (no,
provably) but "**how much reasoning can we do per emitted token?**" That reframing is the whole
product argument: we are not removing CoT, we are compressing it.

**(b) Extra positions alone buy nothing.** arXiv:2505.18948 `[S]`: transformers with *polynomial
padding* recognise exactly FO-uniform TC⁰ — the same class as without padding. Blank cells are
not a scratchpad. **A scratchpad is cells + a feedback path that writes to them.** Filler-token
results say the same thing empirically. So any mechanism we build must have both, or it is
padding.

**(c) A re-readable buffer is only worth having if something can act on what it reads.** The
self-correction evidence is brutal: GPT-3.5 fixes 7.6 % and breaks 8.8 % on GSM8K without
external feedback `[S]`; small models specifically "need strong verifiers to self-correct"
(arXiv:2404.17140) `[S]`; multi-agent debate is no better than self-consistency at matched
samples `[S]`. Prophet is a small model. **Adding a scratchpad will not, by itself, buy
self-correction.** It buys working memory. The verifier has to come from R09's confidence head or
from RL with verifiable rewards (R10) — and that coupling should be stated in the roadmap rather
than assumed.

### 5.4 Has anyone separated the two functions experimentally?

Partially, and only recently.

- **arXiv:2605.30757** separates them *theoretically*: three memory regimes (compressed latent
  loop / full sequence-state loop / CoT scratchpad) with the P-complete separation between the
  first and the third `[S]`.
- **arXiv:2604.21999** separates them *empirically and quantitatively*: memory tokens and ponder
  depth "substitute as resources at fixed accuracy" — mean halt depth falls **11.6 → 8.3** as
  memory grows T=8 → 64 `[S]`. That is a measured exchange rate between the two functions, on one
  task. It is the closest thing to a direct test of the brief's claim in the literature.
- **arXiv:2604.17935** gives the trade-off a closed form for k-hop pointer chasing `[S]`.

What is missing — and what W1-A1 (§7) should produce — is a **task suite deliberately factored
into depth-bound and memory-bound halves**, evaluated on the same architecture with the same
budget. Nobody has published that. It is cheap. It is the experiment that decides Prophet's
recurrent bet on the reasoning axis rather than the loss axis.

---

## 6. Implications for Prophet

### 6.1 Six decisions that follow from §1–5

1. **Do not attempt to replace CoT.** Under NC ≠ P a bounded-state loop provably cannot
   (§2.4). The target is a **compression ratio**: equal accuracy at 2–4× fewer emitted tokens.
   External reference point: HybridCoT achieves **94 % of full-CoT accuracy at ~50 % compute**
   `[S]`. If we hit that at 350 M with a runtime-tunable dial, that is a shippable claim.
2. **Stop crediting the loop with expressivity.** Constant k changes no complexity class. Justify
   it on loss-per-parameter (R04 A1/A2). Correspondingly, **promote halting from "nice to have"
   to the only route to n-dependent depth** — and budget for the "router initialisation trap"
   that killed the majority of ACT runs in arXiv:2604.21999 `[S]`.
3. **D1 was right for memory and wrong for reasoning.** The fix is *not* to put attention back
   in the loop (that reintroduces the k× KV blow-up the architecture correctly avoided). The fix
   is a **small, fixed-size, content-addressable buffer whose cost is independent of context
   length**.
4. **Keep discrete anchors in the emitted stream.** HybridCoT's ablation says the anchors are
   needed for accuracy anyway `[S]`; monitorability comes free with them. This also protects
   R09's abstention story, which needs a legible trace.
5. **Impose decodability in the loss, not in the forward pass.** LOTUS, RiM and 2606.20075 all
   converge on per-step supervision as the stabiliser `[S]`. Make hard snapping an *ablatable
   dial*, not a design commitment (W1-A3).
6. **Check embedding tying before any hidden-state recycling.** `tie_word_embeddings = True` for
   prophet-mini `[R]`; prophet-main inherits the donor's choice. COCONUT's 8 B collapse
   (50.3 → 41.5) is attributed to untied embeddings + recycled hidden states `[S]`.

### 6.2 The mechanism to build: the **Anchored Ponder Buffer** (APB)

*One mechanism. It supplies F2 — the function the core provably lacks — at bounded cost, and it
makes F3 a dial so the brief's hypothesis can be tested rather than assumed.*

**Definition.** T learned latent slots (T = 8–16) of reduced width `d_s` that

- live **inside** the recurrent core and are updated once per loop iteration by a bounded
  attention (slots query the sequence; the sequence queries the slots);
- **persist across decoded tokens**, carried in `ProphetCache` alongside the GDN state — so they
  are a genuine working memory, not just extra depth;
- cost **O(T·d_s)** memory and **O(T·d_s)** FLOPs per token per iteration, **independent of
  context length** — the KV thesis of R02 is untouched;
- are anchored during training by a **decodability loss**: after loop iteration *i*, a shared
  read-out must predict the *i*-th gold CoT-step token from the slot summary (weight λ, annealed
  to 0 in a second stage, à la RiM `[S]`);
- may optionally, at inference, be **soft-snapped** onto the embedding table every *k*-th
  iteration with strength α and temperature τ. **α = 0 is pure latent; α = 1, τ → 0 is a hard
  token.** This single dial is the brief's "widen the bottleneck but keep the error correction",
  made falsifiable.

**Why this and not something else.**

- It supplies the *missing* function. Loops and filler tokens supply F1, which we already have.
- It has the strongest small-scale evidence of any candidate: arXiv:2604.21999 (T=0 always fails,
  T=8 reliably succeeds at **57.4 %**, T=64 dilutes) and arXiv:2605.30343 (RiM) `[S]`.
- It reuses the delta-rule / product-key primitive already in `prophet/memory/ledger.py` `[R]` —
  the same primitive used a third time, which is the project's stated aesthetic.
- It is fully reversible: `recurrent.ponder_slots = 0` recovers the exact current baseline
  (CLAUDE.md rule 3).

**Cost, computed** `[C]`, for prophet-main (d = 1536, d_s = 384, T = 16, bf16):

| Item | Formula | Value |
|---|---|---|
| Slot table | T · d_s | 6 144 params |
| Projections (read q, read out, write q, write out) | 4 · d · d_s | **2.36 M params** = **0.64 %** of 369 M active |
| FLOPs / token / iteration | 4 · (2 · d · d_s) + 4 · T · d_s | ≈ **4.8 MFLOP** |
| One core iteration, for comparison | 2N × (core_layers / parameterised_layers) = 2 × 369 M × 4/12 | ≈ **246 MFLOP** `[C]` |
| APB overhead per iteration | 4.8 / 246 | ≈ **2 %** |
| Runtime state per sequence | T · d_s · 2 bytes | **12 KiB** — vs 32 MiB of KV at 8 k context |
| iPhone budget impact | 12 KiB against a 3–5 GB budget | negligible |

**Honest limitation, stated up front.** T·d_s is still a *bounded* state. **APB does not change
the complexity class**; arXiv:2605.30757's separation still applies to it. Its claim is purely
empirical: fewer emitted tokens for the same accuracy, and a measurable capacity threshold in T.
If W1-A2 shows the capacity cliff sits at 4–8 items, we will know precisely what class of
problems still requires the token tape.

### 6.3 PyTorch sketch

Written against the actual structures in `prophet/modeling/model.py` `[R]` — the `run(section,
iteration, h)` closure, `ProphetCache`, and the `inject_input_each_step` / truncated-BPTT loop.

```python
# prophet/modeling/ponder.py  (sketch — omits init details, dtype plumbing, flash kernels)
from __future__ import annotations

import torch
from torch import Tensor, nn

from prophet.modeling.layers import RMSNorm


class AnchoredPonderBuffer(nn.Module):
    """A bounded, content-addressable scratchpad shared by every loop iteration.

    The recurrent core supplies serial depth but no memory: its state is a fixed matrix,
    so running it longer adds computation and not space (arXiv:2605.30757). This module
    adds the space, at a cost that does not grow with context length -- which is the only
    reason it is admissible under R02's KV budget.

    Three things must all be true or this is just padding (arXiv:2505.18948):
      1. the slots are *written* by the model (not blank positions),
      2. the slots are *read* by later computation,
      3. the slots *persist* across decoded tokens.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_slots: int = 16,
        slot_dim: int = 384,
        n_heads: int = 4,
        anchor_vocab: int | None = None,
    ) -> None:
        super().__init__()
        self.n_slots, self.slot_dim, self.n_heads = n_slots, slot_dim, n_heads
        self.head_dim = slot_dim // n_heads

        # Learned initial contents. Deterministic at inference (cf. eval_state_init).
        self.slot_init = nn.Parameter(torch.zeros(n_slots, slot_dim))

        self.norm_in = RMSNorm(d_model)
        self.norm_slot = RMSNorm(slot_dim)

        # write path: slots attend to the sequence
        self.w_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * slot_dim, bias=False)
        # read path: sequence attends to the slots
        self.r_q = nn.Linear(d_model, slot_dim, bias=False)
        self.r_out = nn.Linear(slot_dim, d_model, bias=False)
        self.r_out.weight.data.zero_()  # start inert: k=0 recovers the baseline exactly

        # F3 as *supervision*: the slot summary must be decodable to the gold CoT step.
        self.anchor_head = (
            nn.Linear(slot_dim, anchor_vocab, bias=False) if anchor_vocab else None
        )

    def _mha(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        b, nq, _ = q.shape
        nk = k.shape[1]
        shape = lambda t, n: t.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            shape(q, nq), shape(k, nk), shape(v, nk)
        )
        return out.transpose(1, 2).reshape(b, nq, self.slot_dim)

    def forward(
        self,
        h: Tensor,                     # (b, s, d_model) -- core state this iteration
        slots: Tensor | None,          # (b, T, slot_dim) -- carried across tokens AND iters
        *,
        write_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        b = h.shape[0]
        if slots is None:
            slots = self.slot_init.unsqueeze(0).expand(b, -1, -1).contiguous()

        x = self.norm_in(h)

        # -- write: slots read the sequence, delta-rule update (same primitive as the ledger)
        kv = self.w_kv(x)
        k, v = kv.chunk(2, dim=-1)
        if write_mask is not None:                    # causal write during prefill
            k = k.masked_fill(~write_mask.unsqueeze(-1), float("-inf"))
        delta = self._mha(self.w_q(self.norm_slot(slots)), k, v)
        slots = slots + delta                          # residual write; decay optional

        # -- read: every position queries the buffer
        read = self._mha(self.r_q(x), slots, slots)
        h = h + self.r_out(read)

        # -- anchor logits for the auxiliary decodability loss (training only)
        anchor_logits = (
            self.anchor_head(self.norm_slot(slots).mean(dim=1))
            if self.anchor_head is not None
            else None
        )
        return h, slots, anchor_logits

    @torch.no_grad()
    def soft_snap(self, slots: Tensor, embed: Tensor, *, alpha: float, tau: float) -> Tensor:
        """Optional inference-time projection toward the token manifold.

        alpha = 0  -> pure latent (no bottleneck).
        alpha = 1, tau -> 0  -> a hard token (the full 15-bit bottleneck).
        The brief's hypothesis is that intermediate alpha beats both ends. W1-A3 tests it.
        If accuracy is flat in alpha but rises with the anchor-loss weight, the hypothesis
        is wrong about the mechanism and right about the phenomenon.
        """
        if alpha <= 0.0:
            return slots
        e = embed[: self.n_slots * 0 + embed.shape[0]]      # (V, d_s) after down-projection
        p = torch.softmax(slots @ e.t() / tau, dim=-1)
        return (1.0 - alpha) * slots + alpha * (p @ e)
```

Integration into `ProphetModel.forward` — three lines inside the existing loop `[R]`:

```python
            slots = cache.ponder_slots if cache is not None else None
            for i in range(k):
                grad_on = i >= first_grad_iter
                with contextlib.nullcontext() if grad_on else torch.no_grad():
                    step_in = h + injected if r.inject_input_each_step else h
                    h = run("core", i, step_in)
                    if self.ponder is not None:
                        h, slots, a_logits = self.ponder(h, slots)
                        if a_logits is not None:
                            anchor_logits.append(a_logits)     # -> aux CE vs gold CoT step i
                        if not self.training and r.snap_every and (i + 1) % r.snap_every == 0:
                            slots = self.ponder.soft_snap(
                                slots, self.embed_ds, alpha=r.snap_alpha, tau=r.snap_tau
                            )
                if not grad_on:
                    h, slots = h.detach(), slots.detach()
            if cache is not None:
                cache.ponder_slots = slots          # persists across decoded tokens
```

Config additions (all default to the current behaviour, per CLAUDE.md rule 3):

```python
    # RecurrentCoreConfig
    ponder_slots: int = 0            # 0 disables the buffer entirely -> exact baseline
    ponder_slot_dim: int = 384
    ponder_heads: int = 4
    anchor_loss_weight: float = 0.0  # lambda: decodability supervision on the slots
    anchor_anneal_steps: int = 0     # RiM-style stage 2: decay lambda to 0
    snap_every: int = 0              # 0 = pure latent; k = snap once per k iterations
    snap_alpha: float = 0.0
    snap_tau: float = 1.0
```

### 6.4 The decode-time arithmetic that makes this worth doing

Prophet's device targets are **bandwidth-bound at decode**, not compute-bound `[R]`. Per emitted
token, the model must reread all active weights plus the KV cache. Per *latent* iteration it
rereads only the core's weights and adds **nothing** to the KV cache.

Rough exchange rate for prophet-main `[C]`, with 12 parameterised layers of which 4 are the
shared core:

| Operation | Weight bytes reread | KV bytes added | Sampling step |
|---|---:|---:|---|
| One emitted CoT token (k=4) | full active set | **4 096** | yes |
| One extra latent iteration | ≈ core share of the active set | **0** | no |
| One extra APB iteration | + 2.36 M params ≈ 1.2 MB bf16 / 0.6 MB int4 | **0** | no |

So four latent iterations cost roughly one token's worth of weight traffic and **zero** KV
growth. Over a 1 000-token trace, halving the emitted tokens saves **2 MiB of KV** and 500
sampling steps — on an iPhone at 90–180 tok/s `[R]` that is 3–6 seconds of wall clock per answer.
**The product case for W1 is latency and KV, not accuracy.** The accuracy case is defensive: do
not lose more than 5 %.

---

## 7. Ablation plan

**Shared setup.** 130 M-class (d=768, 12 heads / 4 kv, SwiGLU 2048, vocab 32 768 tied, seq 2 048)
for the synthetic experiments; 350 M (d=1024) for anything that decides a go/no-go, per D4's
documented MoR crossover warning `[R]`. Budgets assume 35 % MFU; synthetic-task runs use 300–500 M
tokens, natural-data runs 3 B.

**Two rules that apply to every run below**, both forced by §4.2:

- **Latent-ablated control.** Every arm using latent machinery is also evaluated with that
  machinery zeroed. If the control matches within noise, the arm is decorative and is reported as
  a failure, not a success.
- **Shortcut probe.** For every synthetic task, report accuracy on a *label-shuffled contextual
  cue* variant. Latent methods show "pervasive shortcut behaviour" `[S]`; we will not discover it
  at 350 M by accident.

### 7.1 The suite

**DepthBench-D (depth-bound, memory-light):** p-hop induction (p = 2…8), parity over k bits,
iterated function application, 2–5-step templated symbolic word problems. Needs serial depth,
O(1) working set.

**DepthBench-M (memory-bound, depth-light):** k-key associative recall (k = 2…64), list
reversal, sorting, keys-finding maze (per arXiv:2502.03275's setup `[S]`). Needs O(k) working
memory, shallow serial depth.

This factorisation does not exist in the literature (§5.4) and is the core deliverable of W1
beyond the report.

### 7.2 Experiments

| ID | Question | Arms | Budget | Kill / pass criterion |
|---|---|---|---|---|
| **W1-A0** | **Is the bottleneck framing quantitatively real?** Measure, do not assert. | Run `prophet.analysis.bandwidth` `[R]` over CoT traces from an available checkpoint (or the donor) at k ∈ {1,2,4,8}. Report effective rank, dimension utilisation, mean token entropy in bits, measured ratio. | **~1 GPU-h** | **Report-only, but decisive for the framing.** If the measured ratio < 20×, the bottleneck is quantitatively unimportant and W1 stops arguing from it. Second output: does effective rank *rise* with k? If not, **the loop is not widening the channel** and §6.2's premise weakens. |
| **W1-A1** | **Depth vs scratchpad — the separation experiment.** | 5 arms at 130 M: `D16` dense · `L(2,3,2)@r=4` · `L@4 + APB(T=16)` · `L@4 + 8 filler tokens` · `D16 + explicit CoT`. Both benches. | 5 × 1.5 h = **7.5 h** | **Prediction under test:** loop ≈ dense+CoT on **DepthBench-D**; loop **≪** CoT on **DepthBench-M**; loop+APB recovers most of the M-gap up to ~T items then falls off a cliff. **PASS** if the D/M gap between `L@4` and `D16+CoT` differs by ≥ 15 pts (i.e. the two functions really do separate). **If loop ≈ CoT on *both*, §5 is wrong** and the scratchpad claim is destroyed. |
| **W1-A2** | **Capacity law of the buffer.** How many items does T slots hold? | T ∈ {0, 4, 8, 16, 32, 64} at 130 M on DepthBench-M. | 6 × 1 h = **6 h** | External reference: T=0 fails / T=8 works / T=64 dilutes `[S]`. **PASS** if recall(k) tracks T with a visible threshold. Deliverable: the chosen T, and the item-capacity number that tells us *which* problems still need the token tape. |
| **W1-A3** | **The anchoring dial — tests the brief's hypothesis directly.** | 130 M, templated multi-step arithmetic with gold CoT. Grid: α ∈ {0, 0.25, 0.5, 1.0} × λ ∈ {0, 0.3}. | 8 × 45 min = **6 h** | **If accuracy rises with α at λ=0** → the projection itself error-corrects: **the brief is right about the mechanism**, and we ship the snap. **If accuracy is flat in α but rises with λ** → discretisation is a *training-signal* effect: **the brief is wrong about the mechanism**, and we ship the auxiliary loss and drop the snap. Either outcome is a shipped decision. |
| **W1-A4** | **Token compression ratio — the product metric.** | 350 M, SFT on templated CoT. (k, T) ∈ {(1,0) plain CoT, (4,0), (4,16), (8,16)}. Report accuracy vs **emitted tokens**, not vs FLOPs. | 4 × 5 h = **20 h** | **PASS** if ≥ 95 % of plain-CoT accuracy at ≤ 50 % emitted tokens. External reference: HybridCoT's 94 % @ ~50 % `[S]`. **This is the number that goes on the model card.** |
| **W1-A5** | **Error accumulation per step.** | Free evaluation on A1/A3 checkpoints: P(correct) vs number of reasoning steps; fit per-step success `p`. | **0 h** | Prediction: latent has lower `p` and steeper decay; λ > 0 recovers it. Quantifies the "error correction" claim as a number rather than an adjective. |
| **W1-A6** | **Does the trace stay monitorable?** | Decode APB slots through the LM head; top-1 agreement with the gold CoT step. | **0 h** | If agreement < 20 %, the "interpretable anchor" claim is void and must not appear in any external write-up. Note §3: the token trace is only 25–39 % faithful anyway — the bar is *relative*, not absolute. |
| **W1-A7** | *(Conditional on A1 being marginal.)* Scale guard. | Repeat A1's three decisive arms at 350 M. | 3 × 6 h = **18 h** | Guards against the documented 135 M → 360 M crossover `[R]`. Do **not** kill the track on a 130 M negative alone. |
| **W1-A8** | *(Cheap, high option value.)* Unsupervised alternative. | Thoughtbubbles-style residual forking (arXiv:2510.00219) at 130 M, LM loss only, vs `L@4` at matched FLOPs. | 2 × 3 h = **6 h** | The only candidate mechanism requiring **no CoT data at all** — which matters enormously given R10's licence minefield `[R]`. **PASS** if perplexity beats `L@4` at matched FLOPs; if so, escalate to a full track. |

**Staging.**

- **Stage 0 — is the premise true? (≈ 15 A100-h):** W1-A0 + W1-A1 + W1-A3. Answers: is the
  bottleneck real, do depth and scratchpad separate, and is the brief's error-correction
  mechanism right. **If A1 shows no D/M separation, W1 concludes "build nothing" and returns
  ~50 h to other tracks.**
- **Stage 1 — size it (≈ 12 h):** W1-A2 + W1-A8.
- **Stage 2 — prove the product (≈ 20 h):** W1-A4, plus A5/A6 for free.
- **Stage 3 — conditional (18 h):** W1-A7.
- **Total 47–65 A100-h.** Stage 0 alone is 15 h and is the only part that must run before any
  architectural commitment.

**Track-level kill criterion.** If W1-A1 shows the loop matching explicit CoT on *both* benches,
or W1-A0 measures a channel ratio under 20×, then the entire "widen the bottleneck" premise is
unsupported at our scale, and the correct action is to spend the budget on R06 (data) instead.
Say so in the report; do not quietly rescope.

---

## 8. References

Verification markers as defined in §0. **No PDF was read in full this session** — every `[S]`
is abstract-level confidence from a search summary.

### Formal expressivity — the complexity classes

- Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought* —
  **arXiv:2310.07923** (ICLR 2024) `[S]`. log steps ≈ L; linear steps ⊇ regular languages; poly
  steps **= P** exactly.
- Merrill & Sabharwal, *Exact Expressive Power of Transformers with Padding* —
  **arXiv:2505.18948** (NeurIPS 2025) `[S]`. Poly padding **= FO-uniform TC⁰**; O(logᵈ n) looping
  **= TCᵈ**; polylog looping **→ NC**.
- Merrill & Sabharwal, *The Parallelism Tradeoff: Limitations of Log-Precision Transformers* —
  **arXiv:2207.00729**; *A Logic for Expressing Log-Precision Transformers* —
  **arXiv:2210.10749** `[M]`.
- Li, Liu, Zhou & Ma, *Chain of Thought Empowers Transformers to Solve Inherently Serial
  Problems* — **arXiv:2402.12875** (ICLR 2024) `[S]`. Constant-bit precision without CoT ⊆ **AC⁰**;
  T CoT steps → circuits of size T.
- Feng, Zhang, Gu, Ye, He & Wang, *Towards Revealing the Mystery behind Chain of Thought: A
  Theoretical Perspective* — **arXiv:2305.15408** (NeurIPS 2023) `[S]`. Super-polynomial size
  needed without CoT for arithmetic evaluation / linear equations; constant size with CoT.
- Xu & Sato, *To CoT or To Loop? A Formal Comparison Between Chain-of-Thought and Looped
  Transformers* — **arXiv:2505.19245** `[S]`.
- Xu & Sato, *A Formal Comparison Between Chain of Thought and Latent Thought* —
  **arXiv:2509.25239** (ICML 2026) `[S]`.
- **Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation** —
  **arXiv:2605.30757** `[S]`. **The load-bearing citation for §5**: bounded-state loops cannot
  decide P-complete problems under logspace reductions.
- *How Much Cache Does Reasoning Need? Depth–Cache Tradeoffs in KV-Compressed Transformers* —
  **arXiv:2604.17935** `[S]`. Depth/cache trade-off for k-hop pointer chasing.
- *Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning* —
  **arXiv:2604.21999** `[S]`. **The load-bearing empirical citation for §5**: T=0 fails, T=8
  succeeds (57.4 % on Sudoku-Extreme), memory ↔ depth substitute at fixed accuracy.
- Sanford, Hsu & Telgarsky, *Transformers, Parallel Computation, and Logarithmic Depth* —
  **arXiv:2402.09268** `[M]`.
- Giannou et al., *Looped Transformers as Programmable Computers* — **arXiv:2301.13196** `[M]`.
- Merrill, Weiss et al., *A Formal Hierarchy of RNN Architectures* — **arXiv:2004.08500** `[M]`.
  Bounded-state recurrence ↔ finite automata; the reason a GDN loop is a small-space machine.

### The information-theoretic framing

- *The Information Bottleneck of Chain-of-Thought and How Latent CoT Overcomes It* —
  **OpenReview `cCIdxLoLJ5`** (Oct 2025) `[S]`. Each token conveys O(log|V|) bits; pointer chasing
  and parity require long CoT; high-dimensional writes shorten it. **The prior art for the
  brief's hypothesis.**
- *A Survey on Latent Reasoning* — **arXiv:2507.06203** `[S]`. Source of the "15 bits vs 40 960
  bits, 2.7 × 10³ gap" framing.
- *Capabilities and Fundamental Limits of Latent Chain-of-Thought* — **arXiv:2602.01148** `[S]`.
  Exploration–execution trade-off; Symbolic Index; curriculum learning proved necessary.
- *What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic
  Analysis* — **arXiv:2606.20075** `[S]`. Dual collapse (gradient attenuation + representational
  drift); trajectory vs space supervision.
- *Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks
  in LLM Reasoning* — **arXiv:2506.02867** `[S]`. HSIC-estimated MI peaks at "Hmm/Wait/Therefore".
- *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning* —
  **arXiv:2506.01939** `[S]`. ~20 % of tokens are high-entropy forks.
- *Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens* —
  **arXiv:2604.26355** `[S]`. Mean per-token entropy **1.03–1.98 bits** against a 17.2-bit
  vocabulary maximum.
- Roger & Greenblatt, *Preventing Language Models From Hiding Their Reasoning* —
  **arXiv:2310.18512** `[M]`; paraphrase defence bounds encoded content at **≤ 3 bits/KB** `[S]`.
- *Large language models can learn and generalize steganographic chain-of-thought under process
  supervision* — **arXiv:2506.01926** `[S]`.
- `prophet/analysis/bandwidth.py` `[R]`. This project's own instrumentation; already implements
  effective-rank and token-entropy corrections. **W1-A0 runs it.**

### Faithfulness

- Turpin, Michael, Perez & Bowman, *Language Models Don't Always Say What They Think* —
  **arXiv:2305.04388** (NeurIPS 2023) `[S]`. Up to **36 %** accuracy drop under unverbalised bias.
- Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning* — **arXiv:2307.13702**
  `[S]`. Early answering / adding mistakes / paraphrasing / filler; faithfulness peaks ~13 B and
  **decreases** to 175 B.
- Chen et al. (Anthropic), *Reasoning Models Don't Always Say What They Think* —
  **arXiv:2505.05410** `[S]`. Hint verbalisation: **Claude 3.7 Sonnet 25 %**, **DeepSeek R1 39 %**.
  Sub-breakdown by hint category is **inconsistent across secondary sources** — verify.
- Arcuschin, Janiak, Krzyzanowski, Rajamanoharan, Nanda & Conmy, *Chain-of-Thought Reasoning In
  The Wild Is Not Always Faithful* — **arXiv:2503.08679** `[S]`. Implicit Post-Hoc
  Rationalization up to **13 %**.
- *Thought Anchors: Which LLM Reasoning Steps Matter?* — **arXiv:2506.19143** `[S]`. Counterfactual
  resampling, receiver-head analysis, attention suppression; planning and uncertainty-management
  sentences dominate.
- *The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies* —
  **arXiv:2605.10799** `[S]`. **Corruption studies over-estimate faithfulness.**
- *LLM Reasoning Is Latent, Not the Chain of Thought* — **arXiv:2604.15726** `[S]`.
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* — **arXiv:2310.01798**
  (ICLR 2024) `[S]`. GPT-3.5 on GSM8K: fixes **7.6 %**, breaks **8.8 %**.
- *Small Language Models Need Strong Verifiers to Self-Correct Reasoning* — **arXiv:2404.17140**
  `[S]`. Directly relevant: Prophet is a small model.
- *When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling* — **arXiv:2604.10739**
  `[S]`; *Does Thinking More always Help?* — **arXiv:2506.04210** `[S]`. Inverted-U:
  **82.2 → 87.3 → 70.3 %** at 385 / 1 100 / 15 980 thinking tokens.

### Latent and hybrid reasoning — methods

- Nye et al., *Show Your Work: Scratchpads for Intermediate Computation with Language Models* —
  **arXiv:2112.00114** `[S]`. The origin of the scratchpad framing.
- Goyal et al., *Think before you speak: Training Language Models With Pause Tokens* —
  **arXiv:2310.02226** (ICLR 2024) `[S]`. +18 % EM SQuAD at 1 B; zero-delay inference catastrophe.
- Pfau, Merrill & Bowman, *Let's Think Dot by Dot: Hidden Computation in Transformer Language
  Models* — **arXiv:2404.15758** (COLM 2024) `[S]`. Filler tokens; bounded quantifier depth; needs
  dense supervision.
- Zelikman et al., *Quiet-STaR* — **arXiv:2403.09629** `[S]`. GSM8K 5.9 → 10.9 %.
- Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)* —
  **arXiv:2412.06769** `[S]`. GSM8K 34.1 % with 6 thoughts; c=3 degrades.
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth
  Approach* (Huginn-3.5B) — **arXiv:2502.05171** (NeurIPS 2025) `[S]`. ARC-C 27.99 → 38.23 across
  r=4 → 32; uneven saturation.
- Saunshi, Dikkala, Li, Kumar & Reddi, *Reasoning with Latent Thoughts: On the Power of Looped
  Transformers* — **arXiv:2502.17416** (ICLR 2025) `[S]`. (k ⊗ L) ≈ kL layers for reasoning, not
  for memorisation.
- McLeish et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted
  Recurrence* — **arXiv:2511.07384** `[S]`. **Directly validates D10's donor-conversion path**:
  looping a block of a pretrained LLM and training in looped mode beats post-training the
  original at matched compute, on mathematics.
- Cheng & Van Durme, *Compressed Chain of Thought: Efficient Reasoning Through Dense
  Representations* — **arXiv:2412.13171** `[S]`.
- Su et al., *Token Assorted: Mixing Latent and Text Tokens for Improved Language Model
  Reasoning* — **arXiv:2502.03275** (ICML 2025) `[S]`. +4.2 / +4.1 / +13.3 pts at **−17 %** trace
  length.
- Butt, Kwiatkowski, Labiad, Kempe & Ollivier, *Soft Tokens, Hard Truths* — **arXiv:2509.19170**
  `[S]`. pass@1 parity, pass@32 gains; "train soft, infer hard".
- *HybridCoT: Interleaving Latent and Text Chain-of-Thought for Efficient Reasoning* —
  **OpenReview `4mfGbMzTwu`** (ICLR 2026; NeurIPS 2025 ER workshop) `[S]`. **94 % of full-CoT
  accuracy at ~50 % compute; removing symbolic anchors reintroduces hallucination.** The closest
  published version of the brief's proposed design.
- *Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers* (LOTUS) —
  **arXiv:2606.31779** `[S]`. Per-latent-position CE against gold CoT tokens; notes latent CoT
  underperforms explicit CoT beyond 1 B **and the gap widens with scale**.
- Aichberger & Hochreiter, *Unlocking the Working Memory of Large Language Models for Latent
  Reasoning* (RiM) — **arXiv:2605.30343** `[S]`. Fixed memory blocks in one forward pass;
  two-stage curriculum. **Closest published relative of §6.2's APB.**
- Liu, Murty, Manning & Csordás, *Thoughtbubbles: an Unsupervised Method for Parallel Thinking in
  Latent Space* — **arXiv:2510.00219** `[S]`. Learned in pretraining with LM loss only; beats
  baselines at **half** the training budget. **The only candidate needing no CoT data — see
  W1-A8.**
- *Latent Thoughts Tuning* — **arXiv:2602.10229** `[S]`. Documents COCONUT's **50.3 → 41.5 %**
  collapse at 8 B and attributes it to untied embeddings + recycled hidden states.
- *Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought* —
  **arXiv:2509.23365** `[S]`.
- Deng, Choi & Shieber, *From Explicit CoT to Implicit CoT* — **arXiv:2405.14838** `[M]`.

### Latent reasoning — the negative results

- *The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models* —
  **arXiv:2604.06374** `[S]`. **COCONUT reaches 96.6 % accuracy with zero latent tokens**; Soft
  Thinking is indistinguishable from discrete (KL → 0, cos > 0.99).
- *Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought* —
  **arXiv:2512.21711** `[S]`. Latent tokens are uninterpretable placeholders; strong shortcut
  dependence.
- *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer* — **arXiv:2507.02199**
  `[S]`. Limited evidence of interpretable latent CoT in Huginn.
- *How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?* —
  **arXiv:2602.22441** (ICLR 2026) `[S]`. **Pervasive shortcut behaviour.**

### Internal

- `docs/00_PROBLEM_LANDSCAPE.md` §4 `[R]` — the reasoning lock this track serves.
- `docs/01_ARCHITECTURE.md` §2 (D1), §5 (D4) `[R]` — the decisions §5 re-examines.
- `docs/research/R04_reasoning_test_time_compute.md` `[R]` — the recurrence track. **W1 does not
  duplicate it**: R04 answers "does looping work?", W1 answers "what does looping *not* replace?"
  R04's A9 (latent depth vs token CoT at matched wall-clock) and W1-A1 (depth vs scratchpad
  benches) should be run as one experiment.
- `prophet/analysis/bandwidth.py`, `prophet/memory/ledger.py`, `prophet/modeling/model.py` `[R]`.

---

## Appendix A — the brief's hypothesis, adjudicated

| Claim in the brief | Verdict | Basis |
|---|---|---|
| "A hidden state of d=2048 at bf16 carries ~32 000 bits" | **Arithmetically right, informationally wrong.** Container size, not content. Effective rank is a small fraction of d. | §1.2; `prophet/analysis/bandwidth.py` `[R]` |
| "A token carries at most 15 bits" | **Right as a ceiling, ~12× too generous as a rate.** Realised ≈ 1.0–2.0 bits. | arXiv:2604.26355, 2506.01939 `[S]` |
| "Roughly 2000:1 compression" | **Right for the nominal ratio; ~250–750:1 after both corrections.** Still large. | §1.2 `[C]` |
| "99.95 % of the computed state is discarded at every step" | **False.** The state is preserved in the KV buffer and remains readable. What is lost is the *conditioning*: only the token re-enters at layer 0. | §1.3, Proposition 1 `[C]` |
| "CoT's cost is not just latency" | **Right, for a sharper reason than given.** The cost is that the only unbounded-depth feedback path in a transformer runs at ~1 bit/step. | §1.4 |
| "Discretisation is also error correction" | **Phenomenon confirmed, mechanism probably wrong.** Anchors are needed (HybridCoT), but the stabiliser that works is per-step *supervision*, not inference-time projection. | §4.3, §4.4 `[S]` |
| "Continuous CoT is hard to train and degrades past a few steps" | **Confirmed, with numbers.** COCONUT c=3 drops; 8 B collapse 50.3 → 41.5; curriculum proved necessary. | §4.1, §4.3 `[S]` |
| "The right design is k latent steps between discrete anchors" | **Right — and already the 2026 consensus.** HybridCoT, Token Assorted, LOTUS, RiM all implement a version. **The novelty cannot be the design.** | §4.1 `[S]` |
| "Serial depth is only half of what CoT provides; the other half is a re-readable buffer a fixed-size recurrent state cannot replace" | **Right, and now a theorem.** Bounded-state loops cannot decide P-complete problems under logspace reductions. It is also the *most* actionable finding for Prophet, because D1 forfeited exactly this function. | §5, arXiv:2605.30757, 2604.21999 `[S]` |
| *(implicit)* "Recurrence buys reasoning power" | **False at constant k.** Constant looping changes no complexity class. The loop buys density per parameter, not capability. | §2.6 `[C]` |
