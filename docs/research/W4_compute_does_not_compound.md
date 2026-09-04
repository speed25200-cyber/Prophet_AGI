# W4 — Test-time compute does not compound: turning expensive reasoning into cheap memory

> Track W4. The claim under investigation: *a model that spends ten minutes solving a hard
> problem today knows nothing more about it tomorrow.* The industry's answer to hard
> problems is "spend more test-time compute", and that compute is discarded the moment the
> response ends.
>
> **Verdict in one line: the observation is real, it is already named, and at least four
> groups have attacked it — but almost nobody makes later inference *cheaper* at equal
> accuracy, and the one mechanism Prophet could contribute (a backprop-free write of a
> depth delta into an addressable ledger) is unpublished. It is also, on the project's own
> R04 numbers, aimed at the wrong axis. §6.1 explains why, and what to build instead.**

---

## 0. Provenance — read before trusting any number here

This environment's egress proxy **blocked every primary paper host**. Confirmed 403 at the
gateway for: `arxiv.org`, `huggingface.co`, `openreview.net`, `api.semanticscholar.org`,
`www.semanticscholar.org`, `aclanthology.org`, `proceedings.mlr.press`,
`proceedings.neurips.cc`, `www.alphaxiv.org`, `www.emergentmind.com`, `www.themoonlight.io`,
`medium.com`, `*.github.io`, `en.wikipedia.org`, `www.nature.com`. The only fetchable host
was `github.com`. Web *search* worked and returned synthesised snippets.

Every claim below therefore carries a marker:

| Mark | Meaning |
|---|---|
| `[C]` | **Computed here**, in this repo, with its own tools. Reproducible by rerunning the command given. |
| `[R]` | Taken from an existing Prophet research track (R01–R12), inheriting that track's own marking. |
| `[V]` | Verified by fetching a primary or near-primary source (in practice: a GitHub-hosted paper note). |
| `[S]` | **Search-snippet only.** The arXiv ID is as reported by the search index; the PDF was unreachable. Treat the number as indicative, not citable. |
| `[?]` | arXiv ID in the 2601–2608 range. These are 2026 preprints surfaced by the search index and **could not be checked at all**. Several are load-bearing for §4 and §8. Verify before spending compute. |

**No `[S]` or `[?]` number in this document may be used to justify an A100-hour.** The
`[C]` numbers and the R04-derived `[R]` numbers are the only ones that carry weight today.

---

## 1. The observation, and whether it is already named

### 1.1 Three claims, only one of which is true

The framing in the brief bundles three separate assertions. They have very different truth
values, and separating them is most of the work.

| # | Claim | Status |
|---|---|---|
| **A** | Test-time compute is discarded when the response ends. | **True and trivial.** Every frozen-weight deployment has this property by construction. |
| **B** | Nobody stores the result of expensive reasoning. | **False.** At least a dozen systems do. §2 tabulates them. |
| **C** | Nobody makes *later inference cheaper at equal accuracy* by storing it. | **Mostly true, with four real exceptions.** This is where the space is. |

Claim B is where the brief overreaches. The 2023–2026 agent-memory literature is almost
entirely about storing the results of reasoning: Reflexion stores verbal self-critiques,
Voyager stores executable skills, ExpeL stores extracted insights, AWM stores induced
workflows, ReasoningBank stores distilled strategies, Dynamic Cheatsheet stores reusable
snippets, LAG stores the KV cache of prior reasoning. The field is crowded.

What is *not* crowded is claim C. Nearly all of these systems **spend more tokens at test
time, not fewer**. They prepend retrieved memories to the prompt, which grows the context,
which costs prefill FLOPs and KV bytes on exactly the device where we cannot afford them.
They trade compute for accuracy in the same direction as chain-of-thought does. They are
not amortisation; they are a different way of spending.

### 1.2 It is named, and the name is "sleep-time compute"

The observation is not un-named. It is stated almost verbatim in the motivation of
**Sleep-time Compute** (Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez;
arXiv **2504.13171**) `[S]`: when the model is not answering, it still has the context and
is idling. The paper pre-computes over the context offline and reports **~5× less test-time
compute for the same accuracy** on Stateful GSM-Symbolic and Stateful AIME, and up to
+13 % / +18 % accuracy when sleep-time compute is scaled further `[S]`. It also states the
amortisation argument explicitly: the pre-processing cost is shared across all queries
about the same context.

Other names for adjacent parts of the same idea, all pre-existing:

- **Amortised inference** — the probabilistic-ML term of art. Hu et al., *Amortizing
  Intractable Inference in Large Language Models*, arXiv **2310.04363** (ICLR 2024) `[S]`,
  fine-tunes an LLM with GFlowNet objectives so that a single forward sample approximates
  a posterior that would otherwise require search. This is the correct formal framing of
  what W4 wants.
- **System-2 → System-1 distillation** — Yu, Xu, Weston, Kulikov, arXiv **2407.06023** `[S]`.
- **Context distillation** — Snell, Klein, Zhong, arXiv **2209.15189** `[S]`, which already
  in 2022 reported internalising a prompt with an **11× reduction in inference tokens**.
- **Expert iteration / policy distillation** — Anthony, Tian, Barber, arXiv **1705.08439**;
  AlphaZero's entire training loop.
- **Test-time training** — Akyürek et al., arXiv **2411.07279** `[V]`.
- **Episodic memory as the missing piece** — a 2025 position paper, arXiv **2502.06975** `[S]`,
  which argues precisely that session-specific observations, tool outputs and intermediate
  reasoning are discarded when the session ends.

### 1.3 What is actually unclaimed

Stripping away what exists, the residue that Prophet could own is narrow and mechanical:

> A **closed-form, gradient-free write** that stores the *difference between a deep and a
> shallow forward pass* in an addressable slot table, so that a later shallow pass
> retrieves it — with no backpropagation, no fine-tuning, no added context tokens, and a
> cost that runs on a phone.

Every ingredient exists separately. Context distillation supplies the target form. Product-
key memory (Lample et al.; *Memory Layers at Scale*, arXiv **2412.09764** `[S]`) supplies the
store. Huginn (arXiv **2502.05171**) `[R]` supplies the depth dial. R03's delta-rule write
supplies the closed form. Their composition — specifically, **distilling along the depth
axis rather than the context axis, with a backprop-free write** — I could not find published.

That is a real but small novelty claim. It is an engineering composition, not a new idea.
**If the brief's author believed the observation itself was un-named, it is named, and the
person who named it with a number attached is Kevin Lin's group at Letta/Berkeley.**

---

## 2. Prior art

### 2.1 Master table

The two columns that matter are the third and fourth. "Cheaper?" asks whether *later*
inference costs strictly less at equal accuracy — not whether the method is cheaper than
some other method, but whether the stored artifact reduces the cost of the next query.
"Generalises?" asks whether the artifact helps with problems the system has not seen.

| Method | ID | What it stores | Later inference cheaper? | Generalises? | Evidence |
|---|---|---|---|---|---|
| **STaR** | 2203.14465 | Fine-tuned weights, from self-generated rationales that reached the right answer | **No** — still emits CoT | Yes, within the task distribution | `[S]` |
| **ReST-EM** | 2312.06585 | Weights; EM-style generate/filter/finetune | No | Yes | `[S]` — "performance eventually saturates and then declines due to overfitting" |
| **Self-Rewarding LM** | 2401.10020 | Weights, via iterative DPO with LLM-as-judge | No | Partially | `[S]` — only 3 iterations ever run; length inflation observed |
| **V-STaR** | 2402.06457 | Weights **+ a DPO-trained verifier** | **No** — verifier costs extra at test time | Yes | `[S]` — +4 to +17 points over STaR; 7B beats 70B on GSM8K |
| **SEAL** | 2506.10943 | Weights, from model-authored "self-edits" trained by RL | No | Yes | `[S]` — catastrophic forgetting explicitly unresolved |
| **TTT for few-shot** | 2411.07279 | A **per-task LoRA**, discarded after the task | **No** — pays a fine-tuning pass *per task* | Not across tasks (per-task LoRA beats shared LoRA by 7 tasks) | `[V]` — ARC 8B: 45 %→53 %; BBH 50.5 %→57.8 % |
| **Reflexion** | 2303.11366 | Verbal self-critiques in an episodic buffer | No — replayed into context | Weakly | `[S]` |
| **Voyager** | 2305.16291 | Executable JS skills, indexed by description embedding | **Partly** — a retrieved skill replaces re-derivation | Yes, compositionally | `[S]` |
| **ExpeL** | 2308.10144 | Natural-language insights extracted from an experience pool | No | Yes | `[S]` — matches Reflexion's R3 (40 % HotpotQA) **without repeated attempts**, which *is* a compute saving |
| **Agent Workflow Memory** | 2409.07429 | Induced reusable workflows | **Partly** — "reduces the number of steps" on WebArena | Yes: +8.9 to +14.0 abs. as train/test gap widens | `[S]` — +24.6 % / +51.1 % rel. SR |
| **ReasoningBank** | 2509.25140 | Distilled strategies from successes **and** failures | **Partly** — −16.0 % interaction steps | Yes, better than raw-trajectory stores | `[S]` — up to +34.2 % SR |
| **Memp** | 2508.06433 | Procedural memory: step-level scripts + abstractions; transferable between models | Partly | Yes | `[S]` |
| **Dynamic Cheatsheet** | 2504.07952 | Self-curated snippets and code in a persistent text memory | **No** — grows the prompt | Yes, strongly | `[S]` — GPT-4o Game-of-24 10 %→99 %; Claude 3.5 AIME more than doubled |
| **Log-Augmented Generation** | 2505.14398 | **KV cache** of prior reasoning, for a selected token subset | **Partly** — cheaper than replaying text logs | Claims yes | `[S]` |
| **Cartridges / self-study** | 2506.06266 | A **trained KV cache** per corpus, via a context-distillation objective | **Yes** — matches ICL at **38.6× less memory, 26× throughput** | Within the corpus | `[S]` |
| **Sleep-time compute** | 2504.13171 | Pre-computed natural-language inferences about a context | **Yes — ~5× less test-time compute at equal accuracy** | Within the context, not the problem class | `[S]` |
| **System-2 → System-1 distillation** | 2407.06023 | Weights, distilled from System-2 outputs, dropping the intermediate tokens | **Yes** — "less inference cost than System 2"; distilled BSM uses fewer output tokens *and* beats BSM | Yes | `[S]` |
| **Context distillation** | 2209.15189 | Weights, internalising instructions + scratchpad | **Yes — 11× fewer inference tokens** | Yes | `[S]` |
| **Implicit CoT (stepwise internalisation)** | 2405.14838 | Weights; CoT steps removed progressively | **Yes** — no CoT emitted at all | Within the task | `[S]` — GPT-2 small solves 9×9 multiplication; Mistral-7B >50 % GSM8K without CoT |
| **CODI** | 2502.21074 | Weights; CoT compressed into continuous space by self-distillation | **Yes — 2.7–5.9× speedup, 3.1–8.2× compression** | Generalises to complex datasets per authors | `[S]` |
| **GFlowNet amortisation** | 2310.04363 | Weights; a sampler for an intractable posterior | **Yes** — one sample replaces search | Yes, data-efficiently | `[S]` |
| **Expert iteration / AlphaZero** | 1705.08439 | A policy network distilled from tree search | **Yes — this is the canonical case** | Yes | `[S]` |
| **Searchless chess** | 2402.04494 | A 270M transformer distilled from Stockfish annotations | **Yes — no search at all**; Lichess blitz **Elo 2895** | Yes, "highly non-trivial generalisation" to novel boards | `[S]` — needs 10M games / 15B annotated data points |
| **DreamCoder** | 2006.08381 | A **growing DSL library** + a neural search guide | Yes — later searches are shorter in the richer language | Yes, by construction | `[S]` |
| **RLAD** | 2510.02263 | Weights for an *abstraction generator* trained jointly with a solver | No (still test-time compute) but reallocates it better | Yes — "improves generalization to harder problems"; +44 % avg AIME 2025 | `[S]` |
| **Titans** | 2501.00663 | A neural long-term memory updated at test time by a surprise signal | No | Within the sequence | `[S]` |
| **Memory Layers at Scale** | 2412.09764 | Product-key slots, trained by gradient descent | Neutral (adds params, not FLOPs) | Yes, on facts: **>100 % factual-QA improvement** | `[S]` |
| **Prompt / semantic caching** | — | Verbatim or embedding-matched responses | **Yes, trivially** | **No** — exact-hit only | `[S]` — production hit rates 20–45 % |

### 2.2 The scoreboard that matters

Of ~27 systems, the ones that demonstrably make **later** inference cheaper at equal or
better accuracy are:

1. **Weight distillation of reasoning** — context distillation (11× fewer tokens),
   implicit CoT, CODI (2.7–5.9× speedup), System-2 distillation. All require gradient
   fine-tuning, all risk forgetting, none run on a phone.
2. **Trained KV artifacts** — Cartridges (38.6× memory, 26× throughput). Gradient training
   per corpus; amortises over a *corpus*, not a *problem class*.
3. **Offline pre-computation** — sleep-time compute (~5×). No training at all, but the
   artifact is natural-language text that must be re-read, so the saving is bounded by how
   much shorter the summary is than the reasoning it replaces.
4. **Search → policy distillation** — AlphaZero, ExIt, searchless chess. The cleanest and
   most complete example in all of ML, and the one with the harshest preconditions (§3.2).

Everything else in the table buys accuracy with tokens. That is a legitimate trade, but it
is not what W4 is about.

### 2.3 The negative result nobody quotes

**Are Online Skill and Memory Modules Always Worth Their Tokens?** (arXiv **2606.15017**)
`[?]` runs the control the agent-memory literature usually skips: it compares AWM, ASI and
ReasoningBank against a **token-matched vanilla baseline** that spends the same budget on
extra actor steps instead of on memory. Across three WebArena domains and three models,
"the vanilla baseline matches or surpasses all three augmentation methods in aggregate
success rate while often using fewer total tokens" `[?]`. It also reports that run-to-run
variance materially changes the ranking.

This is the single most important entry in §2 for our purposes. **If the memory's gain
disappears against an equal-compute control, the memory is not amortising anything — it is
a more expensive way to spend the same budget.** §7 makes the equal-compute control a
mandatory arm of the W4 experiment for exactly this reason.

---

## 3. The closest existing thing

### 3.1 System-2 → System-1 distillation (2407.06023)

This is the nearest published relative of what W4 wants, and it is worth being precise
about what it did.

**What it got right.**

- **Unsupervised gating.** No labels. The System-2 method is applied to unlabeled inputs
  and the output is filtered by an unsupervised quality signal (self-consistency over
  sampled outputs). This is the right shape for a deployed system: the ledger cannot wait
  for a human.
- **It targets the *output*, not the trace.** The student is trained to produce the System-2
  *answer* directly, with the intermediate tokens dropped. This side-steps the question of
  whether the verbalised trace is the computation — R04 already flags that it is not `[R]`.
- **The distilled version can beat the teacher.** Distilled Branch-Solve-Merge outperformed
  BSM itself and GPT-4-as-judge, with fewer output tokens and less position bias `[S]`.
  Distillation acted as a regulariser, not just a compressor.
- **It reports what failed to distil.** The paper is explicit that not every System-2
  method compiles down — the ones requiring genuinely serial deliberation resist. I could
  not retrieve the specific list (arXiv blocked), so **treat "which methods failed" as an
  open question to check on the PDF before designing around it.**

**What it needs that we lack.**

| Requirement | Theirs | Ours |
|---|---|---|
| Update mechanism | SFT on a 70B model | Closed-form write, no gradients |
| Data volume | A large unlabeled pool | One problem at a time, on-device |
| Where the knowledge lands | All weights | ~32–128 slots per token |
| Forgetting | Unaddressed | 11 % vs 89 % for sparse writes `[R, R03]` |
| Runs on a phone | No | The whole point |

The gap is not conceptual. It is that they can afford a fine-tuning run per consolidation
and we cannot. Our substitute — the closed-form ledger write — buys the "no gradients"
property at the cost of writing into a small subspace instead of the whole model.

### 3.2 AlphaZero-style policy distillation

The cleanest amortisation in machine learning: MCTS is expensive, the policy network is
cheap, and expert iteration alternates *policy improvement* (search produces better moves
than the raw net) with *distillation* (supervised learning pulls the net toward the search
output). Repeat until the net alone plays at the level the search used to.

The end point is worth stating because it is the existence proof for W4's thesis:
**DeepMind's searchless chess** (arXiv **2402.04494**) `[S]` trains a 270M transformer on
Stockfish-16 annotations and reaches **Lichess blitz Elo 2895 with no search at all** —
grandmaster strength as a single forward pass. Expensive deliberation *can* be compiled into
a cheap function.

**What it needs that we lack — all four are hard:**

1. **A perfect, free verifier.** Chess has a terminal reward. Prophet has one only in
   verifiable domains (§6.4).
2. **Volume.** 10M games, 15B annotated data points `[S]`. Prophet's consolidation events
   number in the hundreds per user per week.
3. **A stationary problem distribution.** Chess positions are drawn from one distribution
   forever. A user's queries are not.
4. **Gradient training of the whole network.** Distillation there is full backprop over
   millions of examples. Ours is a scatter-add into 32 slots.

Anthony et al.'s own ablation is a useful calibration `[S]`: a search guided by the learned
policy won **97 %** of games against baseline MCTS, whereas simply **doubling** vanilla
MCTS iterations won only **56 %**. The learned prior was worth far more than a 2× compute
increase. That is the shape of the prize — and also a reminder that it took a full
expert-iteration loop, not one write, to get there.

### 3.3 Sleep-time compute and Cartridges — the two that fit our constraints best

- **Sleep-time compute (2504.13171)** `[S]` is the right *policy* (do the expensive thinking
  when idle, on the context you already have) with the wrong *representation* (natural
  language, which must be re-read into the prompt). Its ~5× saving is bounded by the
  compression ratio of the summary.
- **Cartridges (2506.06266)** `[S]` is the right *representation* (a compact learned artifact
  loaded into the KV path, not into the prompt) with the wrong *update rule* (gradient
  training per corpus).

**Prophet's ledger is the missing combination: a compact artifact loaded into the compute
path, written in closed form.** That is a genuine and defensible position — but it is a
position about *how*, not about *what*, and the "what" question (§4) is where it can die.

---

## 4. Memorisation versus generalisation

This is the crux, and the literature has a clearer answer than I expected.

### 4.1 The ranking of representations

Four candidates for "what to store", ordered by the evidence for transfer to *neighbouring*
problems:

| Rank | Representation | Transfers? | Evidence |
|---|---|---|---|
| 4 (worst) | **The final answer** | Only to the identical problem | Semantic caching: production hit rates 20–45 % `[S]`, and a hit means the *same* question. This is a cache, not learning. |
| 3 | **The full reasoning trace** | Weakly, and it actively hurts hard cases | "Abstract procedural memories transfer more reliably than detailed trajectories, while negative transfer disproportionately harms the hard cases" — arXiv **2604.27003** `[?]`. ReasoningBank beats raw-trajectory stores by up to +34.2 % SR `[S]`. |
| 2 | **An abstracted rule / procedure / workflow** | **Yes, and this is the best-supported answer** | AWM: +8.9 to +14.0 abs. points *as the train/test gap widens* `[S]`. RLAD: abstractions beat extra solutions at large budgets, +44 % avg AIME 2025 `[S]`. *Notes to Self* (arXiv **2607.20372**) `[?]`: self-extracted abstractions match teacher-extracted ones and transfer across datasets and models. DreamCoder: the whole system is this claim `[S]`. |
| 1 (unknown) | **An intermediate hidden state** | **No published evidence either way** | This is what Prophet proposes to store. Nobody has measured its transfer. §4.3 explains why that is a problem and §4.4 gives the measurement. |

The literature's verdict is unambiguous: **abstraction is what transfers.** Storing
instances memorises; storing the *procedure extracted from* instances generalises. This is
also the oldest result in the room — DreamCoder's wake-sleep library learning (arXiv
**2006.08381**) `[S]` alternates solving with *abstraction consolidation*, and the abstraction
step is what makes later problems easier, not the solutions themselves.

### 4.2 The uncomfortable implication for Prophet

Prophet's ledger stores a **vector**. A vector is not an abstraction; it is a point. The
question is whether the *addressing* imposes the abstraction — whether two instances of the
same problem class map to overlapping slots.

This is not a philosophical question. It is a measurable geometric property of the query
projection and the product-key codebooks, and it decides everything:

- If same-class instances address **overlapping** slots → the write generalises. Writing
  instance *i* moves the read for instance *j*. That is learning.
- If they address **disjoint** slots → the write memorises. The ledger is a cache with
  extra steps, and §5's economics collapse to the semantic-cache case (hit only on
  repeats).

### 4.3 What I measured

I ran the probe. On the repo's toy model (`d_model=128`, untrained/random init,
`n_slots=4096`, `top_k=16`, `n_heads=2`), with two "problem classes" defined as a shared
16-token template prefix plus a random 8-token instance suffix `[C]`:

```
||h_{k=16} - h_{k=2}|| / ||h_{k=2}||        = 0.674
cos(h_{k=16}, h_{k=2})                      = 0.773

slots addressed: class-A train 2383 / 4096, class-A held-out 2401, class-B 2475
Jaccard(A-train, A-heldout)  [same class]   = 0.530
Jaccard(A-train, B)          [other class]  = 0.493
```

Two findings, both important:

1. **The depth delta is large.** The k=16 state differs from the k=2 state by 67 % of the
   latter's norm. There is a substantial signal to store — at least geometrically. (Whether
   it is *task-relevant* signal is §6.1's problem, and the answer there is bleaker.)

2. **The addressing does not separate classes.** Same-class Jaccard 0.530 versus
   other-class 0.493 — a **0.037 gap**, essentially chance. At random initialisation the
   product-key addressing routes by fine-grained token geometry, not by problem class. It
   is a *hash*, and it is behaving like one.

   The second number also exposes a capacity problem: **16 episodes of 24 tokens saturate
   58 % of a 4096-slot ledger** `[C]`. Slot pressure arrives far sooner than the "65 536
   slots" headline suggests, because every *token* addresses `n_heads × top_k` slots, not
   every *episode*. Scaling that ratio, a 65 536-slot ledger absorbs on the order of
   **250 episodes of this length** before comparable saturation.

Caveat, stated plainly: this is an **untrained** model. The measurement says nothing about
what a trained Prophet's representation would look like — a trained model's k=2 state should
cluster by problem type far more than a random one's. What it *does* say is that class-like
addressing is **not** a free property of product keys. It has to be engineered and then
measured. §6.3 gives the engineering; §7 gives the measurement.

### 4.4 The design consequence

Because a per-token hidden state addresses by token geometry, **addressing the ledger with
the raw per-token shallow state is the memorising choice.** A new instance of the same class
has different tokens, therefore different states, therefore different slots.

The fix is to make the address *coarser than the instance*:

- **Two-level addressing.** A problem-level key (mean-pooled prelude output over the prompt
  span, which is instance-varying but template-dominated) concatenated with a small
  per-token component. Write and read at the same granularity.
- **Train the query projection for class collapse.** The ledger's `query` linear map is
  currently trained only implicitly. Give it an explicit contrastive objective during the
  ablation: same-class instances close, different-class far. This is cheap (1.57 M params
  at `d_model=1536` `[C]`) and is the difference between a hash and an index.
- **Store at a bottleneck.** Write the delta at the **coda input**, not at the final hidden
  state. The coda still runs afterwards, so the ledger supplies a *hint* the network can
  reinterpret, rather than a final answer it must accept verbatim. This converts the store
  from "the answer" toward "the procedure", which is the representation the literature says
  transfers.

### 4.5 The honest position

There is no published evidence that a hidden-state delta generalises across instances of a
problem class. There is strong published evidence that *natural-language abstractions* do.
A responsible reading is:

> The safest high-value artifact to consolidate is an **abstraction in the model's own
> input space** (a rule, a lemma, a procedure) — because that is what the literature shows
> transfers, and because it is inspectable, editable and revocable. The latent-delta write
> is the interesting bet, not the safe one, and it must be measured against the abstraction
> baseline rather than against nothing.

§7 makes the abstraction baseline a required arm.

---

## 5. The economics

### 5.1 What "N times a normal query" actually is — measured, not assumed

The brief posits "a hard query costs N times a normal query". For Prophet, N is not 8 when
k goes from 2 to 16, because the prelude, coda and heads are paid once regardless of depth.

Run against `configs/prophet_500m_probe.json` (`d_model=1536`, prelude 2 / core 4 / coda 2,
248.5 M active params) `[C]`:

```bash
python - <<'PY'
from prophet.config import ProphetConfig
from prophet.budget import inference_profile
cfg = ProphetConfig.from_json("configs/prophet_500m_probe.json")
for k in (1,2,4,8,16,32):
    p = inference_profile(cfg, device="rtx5090", loop_k=k, context_len=4096)
    print(k, cfg.effective_depth(loop_k=k), p.flops_per_token/1e9, p.decode_tok_s)
PY
```

| k | effective depth | GFLOP / token | RTX 5090 tok/s | iPhone 17 Pro tok/s | cost relative to k=2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.698 | 11 875 | 596 | 0.74× |
| **2** | **12** | **0.947** | **8 261** | **415** | **1.00×** |
| 4 | 20 | 1.444 | 5 135 | 258 | 1.52× |
| 8 | 36 | 2.438 | 2 923 | 147 | 2.57× |
| **16** | **68** | **4.427** | **1 570** | **79** | **4.67×** |
| 32 | 132 | 8.403 | 815 | 41 | 8.87× |

**N = 4.67 in FLOPs, 5.26 in wall-clock** (the discrepancy is the KV term) `[C]`. An
independent check on the toy model gave a k=16/k=2 wall-clock ratio of **4.83×** `[C]`,
consistent. Note these figures moved between two runs during this session because the config
and budget modules were being edited concurrently by a sibling track — **re-run the snippet
in §9 before quoting them.**

This is *good news for the economics and bad news for the ambition*: deep passes are cheaper
than the naive 8× would suggest, so the saving per hit is correspondingly smaller.

### 5.2 What the ledger costs to consult

At `n_slots=65 536`, `memory_dim=256`, `top_k=32`, `n_heads=4`, `d_model=1536` `[C]`:

| Quantity | Value | As a fraction of the k=2 forward pass |
|---|---:|---:|
| Read FLOPs / token | 4.06 MFLOP | **0.43 %** |
| Gathered bytes / token (fp16) | 918 KB | ~0.6 % of weight traffic |
| Query-projection params | 1.57 M | 0.6 % of active params |
| **Stored values (fp16)** | **201.3 MB** | **127 % of the model's own weight footprint (158.4 MB)** |

The read is free. **The storage is not.** A 65 536-slot ledger at full `d_model` width is
larger than the model it augments, which is fatal for the iPhone target (3–5 GB applicative
budget, model 0.67 GB `[R, 01_ARCHITECTURE §3]`).

Mitigations, in order of preference:

| Change | Size | Note |
|---|---:|---|
| 65 536 slots, fp16, full width | 201.3 MB | baseline |
| 16 384 slots, fp16 | **50.3 MB** | on-device default; ~250 episodes capacity per §4.3 scaling |
| 16 384 slots, int8 | **25.2 MB** | values are residuals — small dynamic range, quantises well |
| Low-rank values (store `r=256`, up-project) | **~8 MB** | 6× further; adds a `256×1536` up-projection (0.4 M params) |

**Recommendation: 16 384 slots, int8 values, low-rank width on the iPhone variant; 65 536
slots fp16 on the 5090.** The ledger size must be a device-level configuration knob, exactly
like `loop_k`.

### 5.3 The break-even arithmetic

Let, in units of one k=2 query:

- `c_s = 1` — a shallow query.
- `N = 4.67` — a deep query `[C]`.
- `C` — marginal consolidation cost. The deep pass is already paid (it produced the answer),
  so the marginal cost is `passes × 1` shallow passes plus a scatter-add. At `passes=3`,
  **`C ≈ 3`**.
- `V` — verification cost, in the same units. This is the free variable and it dominates.
- `h` — hit rate: probability that a future query in the class is answered correctly at k=2
  because of the write.
- `M` — number of queries in the class over the artifact's lifetime.

Cost without consolidation: `M·N`.
Cost with: `N + C + V + (M−1)·[h·1 + (1−h)·N]`.

Saving per subsequent query is `h·(N−1) = 3.67h`. Break-even:

```
(M − 1) · h · (N − 1)  ≥  C + V
(M − 1) · h            ≥  (3 + V) / 3.67
```

| Verification regime | `V` | Required `(M−1)·h` | At `h = 0.35` (production semantic-cache floor `[S]`) | At `h = 0.70` (agent inner loop `[S]`) |
|---|---:|---:|---:|---:|
| **Free verifier** (unit test, compiler, calculator, `assert`) | 0 | 0.82 | **M ≥ 4** | **M ≥ 3** |
| **Cheap learned verifier** (one k=2 pass + a head) | ~1 | 1.09 | M ≥ 5 | M ≥ 3 |
| **Self-consistency, m = 8 deep samples** | 8 × 4.67 = **37.4** | **11.0** | **M ≥ 33** | **M ≥ 17** |
| **Self-consistency, m = 32** | 149.4 | 41.5 | M ≥ 120 | M ≥ 60 |

**This table is the main economic result of the track.** Two readings:

1. **With a free verifier, consolidation pays back on the third or fourth query in the class.** That
   is an outstanding return, and it is available exactly in the domains R06 already
   over-weights: maths and code, 26.7 % of the mix `[R, R06]`.
2. **Without one, verification is 90 %+ of the total cost** and the break-even moves out to
   tens of queries per class. This inverts the intuition in the brief: the expensive part
   is not the deep pass, it is *knowing the deep pass was right*.

Corollary that should be written into the design: **the consolidation policy should be
verifier-gated, not confidence-gated.** A confidence head is a cheap verifier with an
unknown false-positive rate, and §8.1 shows what a false positive costs.

### 5.4 Realistic reuse rates

The only reuse figures I could source are production caching statistics `[S]`:

| Workload | Reported hit rate |
|---|---|
| Customer support / analytics | 30–50 % |
| **Template-heavy agent inner loops** | **40–70 %** |
| Long-tail conversational agents | 10–25 % |
| Aggregate across production deployments | 20–45 % |

One widely-repeated claim — that developers re-ask AI the same *type* of question 67 % of
the time — traces to a blog post with no visible methodology and **should not be used** `[S]`.

These are *exact/semantic-match* hit rates, i.e. the memorisation case. Prophet's target is
strictly harder (class-level transfer) but has a strictly larger pool of candidate hits.
Without a measurement of our own, `h = 0.3` is the defensible planning assumption and
`h = 0.6` the optimistic one.

**On-device realism.** The favourable case in the brief — same codebase, same documents,
same domain, every day — is real and is the best case in the table above (template-heavy
inner loops, 40–70 %). Over a week of daily use, `M` for a codebase-shaped class is in the
hundreds, so `(M−1)·h` is in the tens-to-hundreds and **even the self-consistency regime
clears break-even**. The economics are favourable *when the class is stable and the user
returns*. They are unfavourable for one-shot queries, which is the same conclusion
sleep-time compute reached.

### 5.5 The saving is capped, and the cap is low

Even with `h = 1`, the ceiling is `N/1 = 4.67×`, and only on queries that would otherwise
have gone deep. Compare:

| Method | Reported saving | Comparable? |
|---|---:|---|
| Cartridges | 38.6× memory, 26× throughput `[S]` | Different axis (context, not depth) |
| Context distillation | 11× fewer tokens `[S]` | Token-space |
| CODI | 2.7–5.9× speedup `[S]` | Token-space |
| Sleep-time compute | ~5× test-time compute `[S]` | Closest analogue |
| **Prophet depth consolidation** | **≤ 4.67×, realistically 1.4–2.2×** | — |

At `h = 0.35` the expected per-query cost is `0.35·1 + 0.65·4.67 = 3.39` versus `4.67`, a
**27 % saving**; at `h = 0.70` it is `2.10` versus `4.67`, a **55 % saving** (1.4× and 2.2×
respectively) `[C]`. That is worth having on a battery-powered device, but it is not the
order-of-magnitude the framing implies. The larger prize is the *accuracy* one: a k=2 pass
answering like a k=16 pass on the classes the user actually cares about.

**Write the claim honestly: this is a quality-per-joule play, not a 10× play.**

---

## 6. The mechanism for Prophet

### 6.1 First: the brief points at the wrong axis, and R04 already knows it

Before any design, one number from our own R04 has to be confronted, because it can kill the
mechanism before the ledger is involved.

R04 §2.3, "the most important negative result in this literature" `[R]`:

> Huginn, GSM8K, **CoT suppressed**: `3.11 → 4.47 → 4.78 → 4.93 → 4.70 → 4.93 → 4.62 %`
> for r = 4, 8, 16, 32, 64, 128, 256. **Flat from r=16 onward.** With explicit CoT at r=32:
> **24.9 strict / 38.1 flexible**. *Latent depth buys ~1.8 points; the CoT prompt buys ~33
> points.*

W4's proposed mechanism stores `λ(h_{k=16} − h_{k=2})`. If Prophet's depth curve looks like
Huginn's, **that delta contains on the order of two accuracy points**, and consolidating it
perfectly yields a k=2 model two points better. The 4.7× compute the deep pass cost bought
almost nothing worth storing.

There is a countervailing result, also in R04 `[R]`: the iso-FLOP study *Reasoning with
Latent Thoughts* (arXiv **2502.17416**) reports GSM8K-style accuracy going from near-0 at 1
iteration to **34.8 strict / 42.1 flexible at 32 iterations**. The two results are not
strictly contradictory — Huginn's sweep is one model's latent-only decode, the other is a
controlled iso-parameter comparison in a model trained for looping — but they bracket a huge
uncertainty, and the bracket is the whole business case.

Two consequences, and they are the most actionable output of this track:

**(a) Gate 0 is mandatory and comes before everything else.** Measure Prophet's own
`accuracy(k)` curve on the target task family. If the k=2 → k=16 gap is under ~5 points,
**W4's depth variant is dead on arrival** and no amount of ledger engineering rescues it.
This costs one evaluation sweep, no training. It must be run before a single A100-hour is
spent on consolidation. §7.1 specifies it.

**(b) The higher-value axis is the context axis, and the code for it already exists and is
already tested.** The general form of the write is:

```
h⁺ = f(privileged input, expensive setting)
h⁻ = f(plain input,      cheap setting)
target = λ (h⁺ − h⁻)          written at address h⁻
```

The brief instantiates "expensive" as **depth** (`k=16` vs `k=2`). R04's numbers say the
larger signal is in **tokens**: `h⁺ = f(problem ‖ long chain-of-thought)` versus
`h⁻ = f(problem)`. That is worth ~33 points rather than ~1.8, and it is *exactly* the
operation `prophet.memory.consolidate.consolidate()` already implements and
`tests/test_memory.py` already validates (recall error `1.00 → 0.003` after the context is
erased `[R, 06_MEMORY §5]`).

> **Recommendation R1: build the CoT variant first.** Solve the problem once with a long
> chain of thought at whatever depth is affordable; verify the answer; consolidate the CoT
> as *context* with the existing `consolidate()`; then check whether a later cheap,
> CoT-free pass answers correctly. This targets the 33-point signal with proven code. The
> depth variant (`consolidate_depth`) is the more elegant idea and the weaker bet; run it
> as arm B of the same experiment, gated on Gate 0.

This also aligns W4 with **W1's finding** (already merged into `01_ARCHITECTURE.md` §2bis/2ter):
a bounded-state loop at *constant* k buys a constant factor, not a complexity class, and the
recurrent core replaced CoT's serial depth but **not** its re-readable scratchpad. If the
core cannot do what CoT does, then compressing CoT into memory is more valuable than
compressing the core into memory. W1 and W4 converge on the same conclusion from opposite
directions.

### 6.2 What exists in the repo already

`prophet/memory/consolidate.py` now contains a scaffold — `DepthEpisode`,
`consolidate_depth()`, `depth_transfer_error()`, `depth_agreement()` — implementing the
depth-axis write with a `require_verified` flag. It is a good skeleton. What it lacks:

| Missing | Why it matters |
|---|---|
| **No tests.** `tests/test_memory.py` references none of it. | Everything in `06_MEMORY.md` §5 is measured; this is not. |
| **Addresses by raw per-token shallow state.** | §4.4: this is the memorising choice. |
| **`verified` is a caller-supplied boolean.** | The verifier is the expensive part and the whole risk surface (§5.3, §8.1). There is no policy, no quarantine, no provenance. |
| **Writes at the final hidden state.** | Nothing runs afterwards, so the ledger supplies an answer rather than a hint (§4.4). |
| **`depth_agreement` uses `model._project`.** | Reaches through a private API; also, token agreement with the deep pass is not accuracy. |
| **No forgetting/eviction policy.** | §4.3: 16 episodes saturate 58 % of a 4096-slot ledger `[C]`. |

The design below is what those gaps should become.

### 6.3 The design

**Trigger — when to think hard.** Not every query. Gate on the confidence head (D9,
`09_HALLUCINATION`) plus, once trained, the `recurrent.halting` signal that W1 promoted:
run deep when the shallow pass is *uncertain* and the halting head does not converge. This
keeps the 4.7× on the small fraction of queries that need it, which is also what makes
`M`-per-class large relative to consolidations.

**What is written.** Four candidates, evaluated against §4's ranking:

| Candidate | Verdict |
|---|---|
| The final answer tokens | **Reject.** Rank 4. This is a semantic cache; build one separately if you want one, it is 20 lines and it does not need a model. |
| The full latent trajectory `h_1 … h_k` | **Reject.** k× the storage, rank-3 transfer, no evidence. |
| `λ(h_deep − h_shallow)` at the **coda input** | **Accept as arm B.** A hint the coda can reinterpret, not a verdict it must accept. |
| A natural-language **abstraction** distilled from the deep solve, consolidated as context | **Accept as arm A, and prefer it.** Rank 2, the only representation with published transfer evidence, inspectable and revocable. Uses `consolidate()` unchanged. |

**Where it is written — the address.** This is the change that decides memorisation vs
generalisation (§4.4). Replace the raw per-token shallow state with a two-level key:

```
a_problem = LayerNorm( mean_pool( prelude_out[prompt_span] ) )      # class-ish
a_token   = h_shallow[t]                                           # instance-ish
address   = W_a · concat[ a_problem , a_token ]                     # 2d -> d
```

`W_a` is trained with a contrastive objective during the ablation (same class close,
different class far), which is what turns product keys from a hash into an index. Cost:
`2·d²` = 4.7 M params at `d_model=1536`, ~1 % of the model `[C]`.

**The target.** Absolute, never incremental. R03's formulation `m(x) + λ(h⁺−h⁻)` is correct
for a small gradient step and **wrong for a closed-form solve** — the first version of the
consolidation module made recall *worse* every pass (`1.00 → 1.44`); with an absolute target
it went to `0.003` `[R, 06_MEMORY §4]`. `consolidate_depth` already gets this right; keep it.

**Write-time safeguards** (all already in `ProductKeyMemory`, all mandatory here):
per-slot trust region, EWC-lite decay by write count, optional value decay, occupancy
entropy monitoring, and replay at 25 % (which R03 measured as reducing forgetting by 37 %
`[R, 06_MEMORY §5]`).

**Eviction.** New, and required by §4.3's saturation number. Score each slot by
`net value per byte` — the framing in arXiv **2606.25115** `[?]` — combining hit count,
verified-outcome rate, and age. Evict the bottom quantile when occupancy exceeds a
threshold. Never evict a slot whose provenance is a free verifier.

### 6.4 Verification policy — the part that costs money

§5.3 showed verification dominates the budget. It is also the entire safety surface (§8).
The policy is tiered, and the tier is recorded per write as provenance.

| Tier | Verifier | Cost (k=2 units) | Admission | Domains |
|---|---|---:|---|---|
| **T0** | Ground truth: unit test, compiler, `python -c`, calculator, formal checker | ~0 | **Write immediately**, permanent | Code, arithmetic, symbolic maths, SQL, structured extraction |
| **T1** | Self-consistency, `m` deep samples agree | `m·5.4` | Write to **quarantine**; promote after *g* independent later agreements | Word problems, multi-hop QA |
| **T2** | Learned verifier / confidence head above threshold | ~1 | Quarantine only; **never promoted alone** | Everything else |
| **T3** | Nothing | 0 | **Refuse.** | — |

**Quarantine** is a second, small ledger read with a lower `λ`, whose contributions are
capped and whose slots are aged out aggressively. Promotion to the main ledger requires
either a T0 confirmation or `g ≥ 3` independent T1 agreements. This is the mechanism that
converts an unreliable verifier into a slow-but-safe one, and it is the direct answer to
§8.1's failure mode.

Two numbers that set the thresholds:

- Verifier-free scaling is provably worse: `Ω(H/√n)` suboptimality versus `O(H/n)` for a
  verifier-based method, "corroborated empirically at 3/8/32B" — arXiv **2502.12118** `[S]`.
- Majority voting is not a safe verifier at scale: an estimated **30 % false-positive rate**
  among samples selected by pass@256 in one analysis `[S]`, and programs-as-verifiers beat
  vanilla majority voting by up to **18 % on GSM8K** `[S]`. **Where a program can check the
  answer, use the program.** That is also the cheapest tier.

**Design rule, stated as a hard constraint like the licence guard in `prophet.data.mixture`:
`consolidate_depth`/`consolidate` must refuse a T3 episode by default and must record the
tier as provenance on every slot it touches.** The current `require_verified: bool` is not
enough — a boolean cannot be audited, revoked, or used for eviction.

### 6.5 PyTorch sketch

```python
# prophet/memory/amortise.py  (sketch — the pieces consolidate.py is missing)
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import torch
from torch import Tensor, nn

class Tier(IntEnum):
    GROUND_TRUTH = 0   # a program said yes
    CONSENSUS    = 1   # m deep samples agreed
    LEARNED      = 2   # a head said yes
    UNVERIFIED   = 3   # refuse

@dataclass
class Solve:
    """One expensive solve, and the evidence that it was right."""
    tokens: Tensor            # (1, n) prompt
    prompt_span: slice        # which positions are the problem statement
    rationale: Tensor | None  # (1, m) CoT or distilled abstraction — arm A
    tier: Tier
    agreements: int = 0       # independent T1 confirmations

class ProblemAddress(nn.Module):
    """Two-level key: a class-ish problem code plus an instance-ish token code.

    Addressing the ledger with the raw per-token state is the *memorising* choice: a new
    instance of the same class has different tokens, hence different slots. Pooling the
    prelude output over the problem statement gives a component that varies far less
    across instances of one template, which is what makes a write reach neighbours.
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mix = nn.Linear(2 * d, d, bias=False)

    def forward(self, prelude_out: Tensor, h_shallow: Tensor, span: slice) -> Tensor:
        a_problem = self.norm(prelude_out[:, span, :].mean(dim=1, keepdim=True))
        a_problem = a_problem.expand_as(h_shallow)
        return self.mix(torch.cat([a_problem, h_shallow], dim=-1))

    def contrastive_loss(self, addr: Tensor, class_id: Tensor, tau: float = 0.07) -> Tensor:
        """Same class together, different classes apart. This is what turns product keys
        from a hash into an index; §4.3 measured the hash behaviour (Jaccard gap 0.037)."""
        z = torch.nn.functional.normalize(addr.mean(dim=1), dim=-1)
        sim = (z @ z.t()) / tau
        pos = (class_id[:, None] == class_id[None, :]).float()
        pos.fill_diagonal_(0)
        log_p = sim - sim.logsumexp(dim=-1, keepdim=True)
        return -(pos * log_p).sum(-1).div(pos.sum(-1).clamp_min(1)).mean()

@torch.no_grad()
def amortise(model, addresser, main, quarantine, solves, *,
             deep_k=16, shallow_k=2, lam=1.0, passes=3,
             min_tier=Tier.CONSENSUS, promote_at=3):
    """Write what the expensive pass computed, addressed so a cheap pass can find it.

    Arm A (preferred): `rationale` is present -> the privileged input is the CoT, and the
    delta is a *context* effect, which R04's numbers say is worth ~33 points rather than
    ~1.8. Arm B: `rationale` is None -> the privileged setting is depth.
    """
    written = []
    for solve in solves:
        if solve.tier > min_tier:
            continue                                   # T3 never enters, by construction
        target_ledger = (main if (solve.tier == Tier.GROUND_TRUTH
                                  or solve.agreements >= promote_at) else quarantine)

        cheap = model(solve.tokens, loop_k=shallow_k, return_mtp=False)
        if solve.rationale is not None:                  # ---- arm A: context axis
            rich_ids = torch.cat([solve.tokens, solve.rationale], dim=1)
            n = solve.tokens.shape[1]
            rich = model(rich_ids, loop_k=shallow_k, return_mtp=False).hidden[:, :n, :]
        else:                                            # ---- arm B: depth axis
            rich = model(solve.tokens, loop_k=deep_k, return_mtp=False).hidden

        # Write at the coda *input*, so the coda can still reinterpret the hint rather
        # than being handed a verdict. `hidden` is the post-coda state; `core_out` is the
        # pre-coda one the model must expose for this to be possible.
        h_cheap = cheap.core_out
        addr = addresser(cheap.prelude_out, h_cheap, solve.prompt_span)
        target = lam * (rich - h_cheap)                  # absolute, never incremental

        for _ in range(passes):
            stats = target_ledger.write(addr, target)
        written.append((solve, stats, int(solve.tier)))
    return written
```

Three things the sketch requires from the model that it does not currently expose:
`prelude_out`, `core_out` (pre-coda), and a public projection. All three are cheap to add
and all three are needed for §7's measurements anyway.

### 6.6 Where the ledger read goes at inference

Read once, at the coda input, on the cheap path only:

```
h  = core(prelude(x), k)                 # k = 2 on device
h  = h + λ · ledger(address(prelude_out, h, span))     # 0.43 % of the forward FLOPs [C]
y  = lm_head(coda(h))
```

`λ` is a runtime dial like `k`. `λ = 0` restores the base model exactly, which satisfies the
project's reversibility rule (CLAUDE.md §3) and gives §7 its control arm for free.

---

## 7. Evaluation — the experiment that proves or kills it

### 7.1 Gate 0 — the cheap experiment that must run first

**Question.** Does Prophet's deep pass know anything the shallow pass does not?

**Protocol.** On a trained checkpoint (the A1 ablation checkpoint at ≥ 350M — R04 warns
that recursion under-performs vanilla at 135M and only wins from ~360M `[R]`), sweep
`k ∈ {1, 2, 4, 8, 16, 32}` on a held-out set of the target task family. Report accuracy and
**bits-per-byte on the solution span** (R11: under 500M, decide on BPB, not accuracy `[R]`).

**Kill criterion.**

| `accuracy(k=16) − accuracy(k=2)` | Decision |
|---|---|
| **< 3 points** (Huginn-shaped, flat) | **Arm B is dead.** Do not build depth consolidation. Go straight to arm A (CoT axis) or drop W4. |
| 3–8 points | Arm B is marginal; run it only after arm A succeeds. |
| **> 8 points** | Arm B is live. Proceed to §7.2 with both arms. |

**Cost:** one evaluation sweep. ~2.8e15 FLOPs ≈ **minutes of A100 time** `[C]`. There is no
excuse for running any other W4 experiment first.

### 7.2 The main experiment (E-W4)

**Task family.** It must satisfy three constraints simultaneously: (i) a **free T0
verifier**, (ii) **parametric instance families** so "held-out instance of the same class"
is well-defined, (iii) **measurable signal at ≤ 500M**. Two candidates:

- **Primary — templated arithmetic / GSM-Symbolic-style families.** A "class" is a template
  (e.g. two-step rate problem with named entities); instances vary the numbers and names.
  The verifier is a calculator. This is the cleanest possible test of *class* transfer, and
  small models have real signal on it.
- **Secondary — a synthetic "same codebase" family.** N functions in a held-out module, each
  with a docstring and a unit test. A class is "calls into this module correctly". The
  verifier is `pytest`. This is the realistic on-device scenario from §5.4 and it directly
  probes the product claim.

Explicitly **not** MMLU or GPQA: near chance at our scale `[R, R11]`.

**Protocol.**

1. Split each class into `train` (consolidated) and `held-out` (never consolidated), plus a
   `control` set from *disjoint* classes.
2. Solve the `train` instances expensively — arm A: k=4 with a long CoT; arm B: k=16 latent.
3. Verify with the T0 verifier. Record the tier. Discard failures (and keep them: §7.4).
4. Consolidate into the ledger.
5. **Erase the context and the CoT.** Evaluate at k=2 only.

**Primary metrics.**

| Metric | What it answers | Passing |
|---|---|---|
| `acc@k=2` on **held-out same-class**, before vs after | **Did anything generalise?** | **+5 points absolute** |
| `acc@k=2` on **consolidated instances**, before vs after | Did anything get stored at all? (a sanity check, not a result) | large, and *expected*; a big number here with a flat held-out number **is memorisation and is a failure** |
| `acc@k=2` on **control classes**, before vs after | Negative transfer | **≥ −1 point** |
| `acc@k=2 + ledger` vs `acc@k=16` | How much of the deep pass was recovered | report the fraction |

**Secondary metrics.** `depth_transfer_error` on held-out vs consolidated vs control (the
representation-level version of the same three questions); `depth_agreement`; ledger
`occupancy()` write-entropy; fraction of updates hitting the trust region.

### 7.3 The controls, which are the actual experiment

Any of these five killing the effect kills the track. §2.3's budget-matched study exists
precisely because the field skips them.

| # | Control | Why | Failure means |
|---|---|---|---|
| **C1** | **Equal compute.** Spend the consolidation + verification budget on running the *base* model at higher k or with more samples instead. | This is the control that destroyed the agent-memory gains in arXiv 2606.15017 `[?]`. | The ledger is a more expensive way to spend the same budget. |
| **C2** | **Equal parameters.** Same ledger, written with **shuffled targets** (right magnitude, wrong content). | The read adds 1.57 M query params + 201 MB of values `[C]`. Some gain could be capacity, not content. | The gain is an artifact of adding parameters. |
| **C3** | **Retrieval baseline at equal context budget.** Put the verified answer/abstraction in the prompt instead of the ledger. This is R03/E2's baseline `[R, 06_MEMORY §6]`. | If a 200-token retrieval does the same job, do that: it is inspectable, editable and needs no new machinery. | The ledger loses to RAG. Ship RAG. |
| **C4** | **Memorisation floor.** An exact-match semantic cache over the consolidated instances. | Establishes what pure caching already achieves (§5.4: 20–45 % hit `[S]`). | The ledger is a cache with extra steps. |
| **C5** | **Abstraction baseline (arm A′).** Consolidate a *natural-language abstraction* of the solution instead of a latent delta. | §4.1 says this is the representation with actual transfer evidence. | The latent delta is the wrong artifact; use text. |

**The one-line summary of the whole evaluation: a large number on consolidated instances
with a flat number on held-out instances is not a result, it is a cache. Report both, always,
side by side.**

### 7.4 The longitudinal arm — because the failure is not immediate

Everything above is a single round. The literature's most consistent finding is that
self-improvement loops **rise then collapse** (§8.2). So:

- Run **10 sequential consolidation rounds** on a stream of new classes.
- After each round, re-evaluate **round 1's** held-out set.
- Plot utility versus round. The expected shape from arXiv **2605.12978** `[?]` is a rise,
  a peak, then a fall **below the no-memory baseline**.
- Report the peak round and the crossing point. If the peak is at round 1–2, the mechanism
  needs an eviction/consolidation policy before it is deployable, not after.

Also required: consolidate a batch containing a **known-wrong** answer (a T2-tier admission
that a T0 verifier later refutes) and measure how much accuracy the class loses and whether
a corrective write repairs it. §8.1 says this is the scariest number available.

### 7.5 Cost

| Stage | FLOPs | A100-hours @ 35 % MFU |
|---|---:|---:|
| Gate 0 sweep (500 problems × 6 depths × 300 tokens) | 2.8e15 | **< 0.01** pure compute |
| Deep solves, 2 000 problems × 1 000 tokens @ k=16 | 8.9e15 | ~0.02 pure compute (~1 in practice: batch-1 decode is bandwidth-bound, MFU 1–5 %) |
| T0 verification (calculator / pytest) | ~0 | ~0 |
| T1 verification if T0 unavailable (m = 8) | 7.1e16 | **~1–8** |
| Addresser contrastive training (4.7 M params) | negligible | < 0.5 |
| All evaluation arms C1–C5 × 10 rounds | ~2e16 | ~1 |
| **Total, T0 domain** | | **~2–3 A100-hours** |
| **Total, T1 domain** | | **~10 A100-hours** |

`[C]`, given an existing trained checkpoint. **W4 is one of the cheapest decision gates in
the project** — under 1 % of the 300-hour budget — because it consumes a checkpoint rather
than producing one. Its cost profile is dominated by verification, which is exactly what
§5.3's arithmetic predicted.

---

## 8. Risks

### 8.1 Consolidating wrong answers — and the number that should worry you most

The failure is not "the memory is useless". It is **"the memory is confidently wrong and the
model stops recomputing"**. A ledger that returns a plausible delta suppresses the deep pass
that would have found the right answer. Accuracy on the class is then capped at `1 − p`
where `p` is the wrong-write rate, and no amount of test-time compute recovers it, because
test-time compute is precisely what the memory replaced.

The empirical evidence is worse than the theory:

> **Even when consolidating from ground-truth solutions, GPT-5.4 failed on 54 % of a set of
> ARC-AGI problems it had previously solved without memory** — arXiv **2605.12978** `[?]`.

Read that twice. The memories were derived from *correct* solutions, and consolidating them
destroyed the majority of previously-correct answers. The paper's general finding is that
"memory utility first rises, then degrades, and can fall below the no-memory baseline" `[?]`,
across ALFWorld, ScienceWorld, WebShop, AppWorld and Mind2Web.

**This is the single strongest argument in the whole report, and it argues against the
mechanism, not for it.** It must be verified on the primary source before W4 is funded — but
if it holds, the design implications are non-negotiable:

- **A verified-correct source is not sufficient.** The *consolidation operator itself*
  introduces the error. Prophet's closed-form write is a much narrower operator than an LLM
  rewriting a memory bank in text, which is grounds for hope, not for confidence.
- **Quarantine is mandatory** (§6.4), not an optimisation.
- **The no-memory path must remain reachable.** `λ = 0` must be a supported runtime state,
  and the confidence head must be able to *veto* the ledger read. A memory the model cannot
  ignore is a memory that can only hurt.

### 8.2 Drift, rise-and-collapse, and the self-improvement literature's consistent verdict

Every iterated self-improvement scheme in §2 that was run for enough rounds reports the same
shape:

| Result | Finding |
|---|---|
| ReST-EM (2312.06585) `[S]` | "Multiple iterations provide further improvements, though performance eventually **saturates and then declines due to overfitting**." |
| Self-Rewarding LM (2401.10020) `[S]` | Only **3 iterations** ever run; the authors flag scaling laws over more iterations as open, and observe length inflation (a reward-hacking signature). |
| Self-Improvement Reversal (2407.05013) `[S]` | Accuracy rises while **OOD generalisation falls**; iterative SFT and SFT-DPO push the model toward easier problems. Pass@1 hides it. |
| Sharpening (2412.01951) `[S]` | Self-improvement is the model **sharpening onto its own verifier**. It cannot add information the model does not already have. |
| Rise-and-collapse (2606.21090) `[?]` | pass@1 peaks within tens of gradient steps then falls, **sometimes to near zero**; KL- and EWC-style constraints do not prevent it. |
| Model collapse (Shumailov et al., *Nature* 631:755–759, 2024) `[S]` | Recursive training on self-generated data degenerates; keeping ~10 % real data slows but does not stop it. |

Prophet's mitigation is structurally different in one respect that matters: **we do not
fine-tune.** The trunk is provably untouched (`06_MEMORY §5`, test-verified `[R]`), so the
sharpening/collapse dynamics that operate on weights do not have a substrate here. The
danger relocates to the ledger — which is exactly what arXiv 2604.27003 `[?]` reports:
"relocating the continual-learning bottleneck from parameter updates to memory access",
with three named failure modes (retrieval pollution, context competition, memory dilution).

Concrete mitigations, all already present or specified: frozen keys (an association never
silently moves), per-slot trust region, EWC-lite step decay, value decay, replay at 25 %
(−37 % forgetting `[R]`), occupancy entropy (a collapse does **not** show in a loss curve),
and the eviction policy of §6.3.

### 8.3 Memorising instead of generalising

Covered in §4; restated here as a risk because it is the *likely* outcome, not the tail one:

- Measured: the untrained ledger's class separation is **0.037 of Jaccard**, i.e. chance `[C]`.
- The default `consolidate_depth` addresses by raw per-token state, which is the memorising
  choice by construction (§4.4).
- The failure is **invisible in every metric except the held-out one.** `depth_transfer_error`
  on consolidated episodes will look excellent while nothing generalises. This is why §7.2
  makes the consolidated-instance number explicitly *not* a result.

### 8.4 Poisoning

A durable, writable store reachable from user content is an attack surface with a persistence
property that prompt injection lacks: it survives every subsequent session.

| Attack | Reported effectiveness |
|---|---|
| **AgentPoison** (2407.12784) `[S]` | ≥ 80 % attack success at **< 0.1 % poison rate**, with < 1 % impact on benign queries, no fine-tuning required. |
| **MINJA** (memory injection) `[S]` | > 95 % injection success, ~70 % attack success under idealised conditions; effectiveness drops substantially when pre-existing legitimate memories are present. |
| Sleeper memory poisoning (2605.15338) `[?]` | Dormant entries that activate on a trigger. |

Prophet's exposure and defences:

1. **Structural.** The ledger is never written from a live conversation, only by a deliberate
   consolidation pass `[R, 06_MEMORY §7]`. This removes the direct injection path.
2. **But the trigger is content-driven.** If consolidation fires because the confidence head
   was low on attacker-supplied text, the attacker chooses what gets consolidated. The
   verification tier is the only thing standing between that and a durable write — which is
   another reason **T3 must be refused, not merely warned about.**
3. **The delta is opaque.** Unlike a text memory, a poisoned latent delta cannot be read,
   audited or diffed by a human. This is a genuine and unfixable disadvantage of arm B
   relative to arm A, and it should be weighed in the arm choice.
4. **Attribution.** Every slot needs provenance: tier, source hash, timestamp, hit count.
   Without it there is no revocation story, and "delete everything I learned from that
   document" is a reasonable user request that becomes impossible to honour.
5. **Model fingerprinting** already refuses a state produced by different weights `[R]`.
   Extend it to refuse an *imported* ledger by default — a shared ledger is a supply chain.

### 8.5 Risks specific to Prophet

| Risk | Note |
|---|---|
| **Δ(k) is flat** (§6.1) | The most likely killer. Gate 0 costs minutes; run it. |
| **Ledger larger than the model** | 201.3 MB vs 158.4 MB of weights at 65 536 slots `[C]`. Fixed by §5.2's sizing, but it must be a device knob. |
| **Slot saturation** | 16 episodes fill 58 % of 4 096 slots `[C]`. Eviction is not optional. |
| **The trunk never learns to use the ledger** | The read is added post-hoc to a model trained without it. The addresser and `λ` need at least a short adaptation phase, or the coda will treat the injected hint as noise. Budget for it. |
| **Interaction with W1** | A bounded-state core at constant k buys a constant factor, not a complexity class. Consolidating it therefore compresses a constant factor. Claim accordingly. |
| **Measurement trap** | `depth_transfer_error` and `depth_agreement` can both improve with no change in what a user sees. Only `acc@k=2` on held-out instances counts. |

---

## 9. References

Verification marks as defined in §0. **`[S]` and `[?]` entries were not retrieved; the arXiv
IDs are as reported by the search index and are unverified in this environment.**

### Amortisation and distillation of reasoning
- Yu, Xu, Weston, Kulikov. *Distilling System 2 into System 1.* arXiv **2407.06023** `[S]`
- Snell, Klein, Zhong. *Learning by Distilling Context.* arXiv **2209.15189** `[S]`
- Deng, Choi, Shieber. *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step.* arXiv **2405.14838** `[S]`
- Shen et al. *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation.* arXiv **2502.21074** `[S]`
- Hu et al. *Amortizing Intractable Inference in Large Language Models.* arXiv **2310.04363**, ICLR 2024 `[S]`
- Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez. *Sleep-time Compute: Beyond Inference Scaling at Test-time.* arXiv **2504.13171** `[S]`
- Eyuboglu et al. *Cartridges: Lightweight and general-purpose long context representations via self-study.* arXiv **2506.06266** `[S]`
- Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search.* arXiv **1705.08439** `[S]`
- Ruoss et al. *Amortized Planning with Large-Scale Transformers: A Case Study on Chess.* arXiv **2402.04494**, NeurIPS 2024 `[S]`

### Self-improvement, and its failure modes
- Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* arXiv **2203.14465** `[S]`
- Singh et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* (ReST-EM). arXiv **2312.06585** `[S]`
- Yuan, Pang et al. *Self-Rewarding Language Models.* arXiv **2401.10020** `[S]`
- Hosseini et al. *V-STaR: Training Verifiers for Self-Taught Reasoners.* arXiv **2402.06457** `[S]`
- Zweiger, Pari et al. *Self-Adapting Language Models* (SEAL). arXiv **2506.10943** `[S]`
- Wu, Li, Liu. *Progress or Regress? Self-Improvement Reversal in Post-training.* arXiv **2407.05013** `[S]`
- Huang et al. *Self-Improvement in Language Models: The Sharpening Mechanism.* arXiv **2412.01951** `[S]`
- *Self-Improvement Can Self-Regress: The Rise-and-Collapse Failure Mode of LLM Self-Training.* arXiv **2606.21090** `[?]`
- Shumailov et al. *AI models collapse when trained on recursively generated data.* Nature **631**:755–759 (2024) `[S]`
- Setlur, Rajaraman, Levine, Kumar. *Scaling Test-Time Compute Without Verification or RL is Suboptimal.* arXiv **2502.12118** `[S]`

### Test-time training and adaptation
- Akyürek, Damani, Zweiger, Qiu, Guo, Kim, Andreas. *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning.* arXiv **2411.07279** `[V]`
- Behrouz, Zhong et al. *Titans: Learning to Memorize at Test Time.* arXiv **2501.00663** `[S]`
- Suzgun et al. *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory.* arXiv **2504.07952** `[S]`

### Agent memory: what is stored, and whether it helps
- Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* arXiv **2303.11366** `[S]`
- Wang et al. *Voyager: An Open-Ended Embodied Agent with Large Language Models.* arXiv **2305.16291** `[S]`
- Zhao et al. *ExpeL: LLM Agents Are Experiential Learners.* arXiv **2308.10144** `[S]`
- Wang, Mao, Fried, Neubig. *Agent Workflow Memory.* arXiv **2409.07429** `[S]`
- *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory.* arXiv **2509.25140** `[S]`
- *Memp: Exploring Agent Procedural Memory.* arXiv **2508.06433** `[S]`
- *Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation.* arXiv **2505.14398** `[S]`
- *Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents.* arXiv **2502.06975** `[S]`
- Hajimiri et al. *Are Online Skill and Memory Modules Always Worth Their Tokens? A Budget-Constrained Study of Web Agents.* arXiv **2606.15017** `[?]` — **the negative control**
- Hu, Long, Wang. *When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents.* arXiv **2604.27003** `[?]`
- Zhang et al. *Useful Memories Become Faulty When Continuously Updated by LLMs.* arXiv **2605.12978** `[?]` — **the 54 % ARC-AGI regression**
- *Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory.* arXiv **2606.25115** `[?]`

### Abstraction, library learning, generalisation
- Ellis et al. *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning.* arXiv **2006.08381** `[S]`
- Qu et al. *RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems.* arXiv **2510.02263** `[S]`
- Liu, Li, Dubrawski. *Notes to Self: Can LLMs Benefit from Experiential Abstractions?* arXiv **2607.20372** `[?]`

### Memory architecture
- Berges et al. *Memory Layers at Scale.* arXiv **2412.09764**, ICLR 2025 `[S]`
- Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn). arXiv **2502.05171** `[R via R04]`
- *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* arXiv **2502.17416** `[R via R04]`

### Security
- Chen et al. *AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.* arXiv **2407.12784**, NeurIPS 2024 `[S]`
- *Memory Poisoning Attack and Defense on Memory Based LLM-Agents.* arXiv **2601.05504** `[?]`
- *Hidden in Memory: Sleeper Memory Poisoning in LLM Agents.* arXiv **2605.15338** `[?]`

### Internal
- `docs/research/R03_memory_continual_learning.md`, `docs/research/R04_reasoning_test_time_compute.md`, `docs/research/R11_evaluation.md`
- `docs/06_MEMORY.md`, `docs/01_ARCHITECTURE.md` §2bis/2ter (W1)
- `prophet/memory/ledger.py`, `prophet/memory/consolidate.py`, `prophet/budget.py`

### Reproducing the `[C]` numbers

```bash
cd /home/user/Prophet_AGI
# §5.1 depth cost table, §5.2 ledger cost
python -c "
from prophet.config import ProphetConfig; from prophet.budget import inference_profile
c=ProphetConfig.from_json('configs/prophet_500m_probe.json')
for k in (1,2,4,8,16,32):
    p=inference_profile(c,device='rtx5090',loop_k=k,context_len=4096)
    q=inference_profile(c,device='iphone17pro',loop_k=k,context_len=4096)
    print(k, c.effective_depth(loop_k=k), round(p.flops_per_token/1e9,3), round(p.decode_tok_s,1), round(q.decode_tok_s,1))
"
```

The §4.3 addressing probe (Jaccard 0.530 vs 0.493, `||h16−h2||/||h2|| = 0.674`) was run on
the `tiny_model()` fixture from `tests/test_memory.py` with two templated token classes; the
script lives in the session scratchpad, not in the repo. **It should be turned into a proper
test in `tests/test_memory.py` before arm B is built** — it is the cheapest possible early
warning that the addressing memorises.

---

## 10. Recommendation

1. **Run Gate 0** (§7.1). Minutes of compute. If Prophet's `acc(k=16) − acc(k=2)` is under
   3 points, arm B is dead and this is settled cheaply.
2. **Build arm A first** — consolidate a verified chain of thought (or an abstraction
   distilled from it) along the *context* axis with the existing, tested `consolidate()`.
   R04's own numbers say that is where ~33 points live, versus ~1.8 for latent depth.
3. **Fix the addressing before building arm B** (§6.4). Two-level key plus a contrastive
   objective. Without it, the ledger memorises, and §4.3 measured that it does.
4. **Make the verifier tier a hard gate with provenance**, not a boolean. T3 refused by
   construction, like the licence guard in `prophet.data.mixture`.
5. **Report the held-out number next to the consolidated number, always.** The consolidated
   number is not a result.
6. **Do not claim 10×.** The measured ceiling is 4.67×, the realistic expectation is
   1.3–2.2× on compute (27–55 % saving at `h` = 0.35–0.70) `[C]`, and the real prize is
   accuracy per joule on the classes a user returns to.
