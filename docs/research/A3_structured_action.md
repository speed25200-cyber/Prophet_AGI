# A3 — Actions as typed objects: what structured tool emission buys, and what it cannot

> Track A3 of the agentic extension. Question: should Prophet emit tool calls as free-text
> JSON, or as typed objects — tool identity from a learned head, arguments through
> grammar-constrained decoding and pointer-copy from context?
>
> **Provenance.** The egress proxy blocked `arxiv.org`, `export.arxiv.org`, `huggingface.co`,
> `hf-mirror.com`, `openreview.net`, `aclanthology.org`, `semanticscholar.org`, `alphaxiv.org`,
> `gorilla.cs.berkeley.edu`, `lmsys.org`, `blog.mlc.ai`, `ai.google.dev` and every secondary
> review site tried. What got through: GitHub raw (Gorilla repo including its `gh-pages`
> blog posts, llguidance, xLAM, ToolRL, Hammer, Toucan, llama.cpp), and search-engine
> snippets that quote the papers. Every number is tagged:
>
> - **[V]** read on a fetched page (GitHub-hosted source, or a snippet quoting the paper
>   verbatim with the figure);
> - **[S]** search-engine snippet only — the figure is quoted but the paper itself was not
>   opened; treat as *probably right, re-check before spending compute*;
> - **[P]** from memory, unverified;
> - **[C]** computed here (`prophet.budget`, `prophet.data.tokenizer`, arithmetic).
>
> As with R03/R09/R11, **no [S] or [P] figure should decide a compute expense** until the
> source has been re-read.

---

## 0. Verdict in one paragraph

The hypothesis is **right about speed and training signal, and only half right about
reliability**. At 1–4B parameters the dominant failure is not malformed output: it is
*not calling at all* (68 % of sampled failures across small models, 89 % for a 3B model
[S]) and *wrong argument values* (78.8 % of errors for a 72B model on multi-step calls [S]).
Unparseable output and hallucinated names — the class that grammar-constrained decoding
eliminates by construction — is a minority class that fine-tuned models already mostly
avoid. So "structured emission" as usually meant (constrained JSON) fixes the small
problem. What fixes the large problems is the *other* half of the design: an explicit
**no-call option in a selection head** (the omission failure becomes one supervised
decision instead of an absent token), and a **pointer-copy path for argument values**
(the value-error failure becomes "point at the right span" instead of "regenerate 18
digits correctly"). Speed is not in doubt: a call that costs ≥ 49 decode steps as JSON
under Prophet-Tok costs 6–8 steps structured [C], consistent with 3–6× end-to-end
speedups reported for parallel/compressed call decoding [S]. One part of the hypothesis
as stated is wrong for Prophet: *"tool id from a learned head"* in the ToolkenGPT /
Octopus sense (one vocabulary row per tool) buys speed by giving up open-set tools; every
tool must be trained in. Prophet's tools are open-set (MCP-style schemas arriving at run
time), so the head must be a **pointer over schema anchors in context**, not a fixed
vocabulary. That is the design in §7.

---

## 1. How tool calls fail

### 1.1 The taxonomy the benchmarks actually use

BFCL's evaluation (AST match against a ground-truth call, plus executable checks) defines
the error classes that every later benchmark inherits [V, BFCL v1 blog via `gh-pages`]:

| Class | Definition | Fixed by grammar? | Fixed by selection head? | Fixed by copy head? |
|---|---|---|---|---|
| Unparseable | output is not a valid call | **yes, by construction** | — | — |
| Hallucinated function / parameter name | name not in the provided tools | **yes** (grammar enumerates names) | yes | — |
| Wrong function | valid but wrong tool chosen | no | **partly** — one CE term, but the decision is still semantic | — |
| Missing parameter | required argument omitted | **yes** (schema `required`) | — | — |
| Wrong parameter value | plausible but wrong value | no | no | **partly** — for values that exist in context |
| No call when one was needed (omission) | model answers in prose | no | **yes** — explicit `no-call` option is supervised | — |
| Unnecessary call (irrelevance) | calls when it should not | no | **yes** | — |
| Wrong sequencing / stop early | multi-turn: skips a prerequisite call, stops before finishing | no | no | no |

BFCL v3 has 4,751 test cases [S]: expert-curated single-turn (simple 400, multiple 200,
parallel 200, parallel-multiple 200), live user-contributed (live_multiple 1,053,
live_irrelevance 882, plus smaller live categories) and 800 multi-turn cases (200 base +
200 missing-function + 200 missing-parameter + 200 long-context) [V blog for the 800;
S for the rest]. The v3 blog's qualitative error analysis of multi-turn failures names
three modes, none of them syntactic [V]: *implicit-action failures* (filling a tank
without checking the level first), *state-awareness deficiencies* (`mkdir alex` while
already in `alex/`), and *unnecessary planning* (re-authenticating when already
authenticated). BFCL v4 adds web-search and memory agentic categories and a
"format sensitivity" category that is measured but non-scoring [V README].

### 1.2 Quantified shares

| Source | Setting | Finding |
|---|---|---|
| *When Agents Fail to Act* (arXiv 2601.16280) [S] | 200 hand-inspected failures from qwen2.5:3b, qwen2.5:7b, Functionary-Small | **Omission 68 % / malformed 32 %** overall. qwen2.5:3b: **~89 % omission**. Functionary-Small (a function-calling fine-tune): 55 % / 45 %. Malformed = wrong tool name, invalid JSON, hallucinated parameters, not separated further in the snippet. |
| *ComplexFuncBench* (arXiv 2501.10132) [S] | multi-step, constrained, long-context calls; frontier models | Five classes: `func_error`, `param_missing`, `hallucination`, `value_error`, `stop_early`. **`value_error` dominates in every model — 78.8 % for Qwen2.5-72B.** `stop_early`: Claude-3.5-Sonnet 19.7 %, GPT-4o 21.0 %. The authors attribute value errors to *constrained value reasoning and long-context extraction*, i.e. failing to carry a value from an earlier tool response into the next call. |
| *Butterfly Effects in Toolchains* (arXiv 2507.15296, EMNLP-Findings 2025) [S] | parameter-filling failures under 15 input perturbations, three input sources | Five parameter-failure categories derived from the invocation chain. **Parameter-name hallucination is the only category attributed to the model itself; every other category is driven by the input sources** (user query, tool docs, prior tool responses). >50 % of failed cases show several patterns at once ("transfer effect"). Per-category percentages were not retrievable — **VERIFY**. |
| *The Reasoning Trap* (arXiv 2510.22977, ICLR 2026) [S] | SimpleToolHalluBench: (i) no tool available, (ii) only distractor tools | RL that raises task performance raises tool hallucination **proportionally**; the effect is training-method-agnostic and appears even when the RL task is mathematics. Directly relevant to Prophet's post-training plan (R10): a reasoning-RL phase will *increase* over-calling unless the no-call decision is supervised. |
| *Reducing Tool Hallucination via Reliability Alignment* (arXiv 2412.04141) [S] | tool-selection vs tool-usage (parameter) hallucination | Snippet: parameter hallucinations cut task success by >12 pts in most cases vs <8 pts for selection hallucinations. **[S, attribution to this paper not certain — VERIFY.]** |

The picture is consistent across four independent sources: **format is not the
bottleneck**. Small models fail to call; all models fail on values.

### 1.3 Small models versus frontier, in numbers

| Model | Params | Score | Note |
|---|---:|---|---|
| Llama-3.2-1B-Instruct | 1B | BFCL v2 **25.7** [V model card] | |
| Llama-3.2-3B-Instruct | 3B | BFCL v2 **67.0** [V model card] | 1B→3B is +41 pts: below ~2B the base capability is not there |
| Qwen3-0.6B | 0.6B | BFCL single-turn 58.2 (no-think) / 67.4 (think) [S, third-party paper] | thinking mode +9 pts |
| Qwen3-1.7B | 1.7B | BFCL v3 **56.6** [S, Qwen3 tech report] | mode not confirmed in snippet |
| Qwen3-4B | 4B | BFCL v3 **65.9** [S, Qwen3 tech report] | |
| STAR-0b6 (Qwen3-0.6B + distillation + similarity-RL) | 0.6B | BFCL v3 **51.70**, ACEBench 53.00 [S] | what a 0.6B model reaches with a dedicated recipe |
| xLAM-2-1b-fc-r | 1B | BFCL v3 multi-turn **43.12** [S, APIGen-MT paper] | beats o1 (36 %) and GPT-4o-FC (41 %) on multi-turn — data quality, not size |
| xLAM-2-3b-fc-r | 3B | multi-turn **56.00**, relevance 94.44 [S] | |
| Hammer-7b | 7B | BFCL (v2 era, 2024-09) overall **83.92** [S] | function masking (§3) |
| TinyAgent-1.1B | 1.1B | own 16-tool Mac benchmark **80.06 %** vs GPT-4-Turbo 79.08 % [S] | closed tool set + ToolRAG: small models are fine when the tool set is small and fixed |
| FunctionGemma-270M | 270M | ~58 % zero-shot → ~85 % after task fine-tune; BFCL irrelevance 70.6 [S, Google blog via Medium] | **licence: Gemma ToU → blocked for Prophet (§6)** |
| Qwen3-4B on τ²-bench | 4B | retail 0.491 / airline 0.365 / telecom 0.189 pass@1 [S] | RL on verified data lifts pass@1 28.1 → 41.5 [S] |
| GPT-4o on τ-airline | — | 35.2 % [S, τ-bench paper] | multi-turn with a simulated user is hard for everyone |
| BFCL v4 leader (2026-09) | — | Qwen3.7-Max 0.750 [S, llm-stats] | |

Two readings. (a) Prophet-mini (253M) sits *below* every row of this table; no emission
mechanism buys the missing knowledge, and A3 must be evaluated on the *gap it closes at
fixed capacity*, not on absolute BFCL. (b) The 1B rows that beat frontier models
(xLAM-2-1b on multi-turn, TinyAgent-1.1B) did it with **verified, executable training
data and a narrow tool set** — which is also the regime where a selection head and
pointer-copy are most natural.

### 1.4 The failure is visible inside the model before it happens

Three 2026 probing papers matter for what the confidence head should gate (§7.6):

- *A Few Neurons Reveal When LLMs Misuse Tools* (arXiv 2608.00218) [S]: across six
  Qwen3/Llama/Gemma models, **over-calling and missing a needed call are linearly
  detectable at the pre-generation prompt boundary with ROC-AUC 0.90–1.00**; call
  *validity* is detectable from the emitted call span with ROC-AUC 0.86–0.90.
- *Tool Calling is Linearly Readable and Steerable* (arXiv 2605.07990) [S]: 18 models,
  270M–27B. The chosen tool is linearly readable at 83–100 % on 4B+ instruction-tuned
  models (15-tool synthetic), 77–94 % on τ-bench airline. **Queries where the top-1/top-2
  tool margin is smallest produce 14–21× more wrong calls** (Gemma-3 12B/27B).
- *The Calls are Coming from Inside the Model* (arXiv 2608.27750) [S]: linear probes on
  18 tool-calling LLMs over BFCL; probes catch wrong-value-right-type errors that
  loggers miss; **effectiveness rises with model size**; probes generalise to unseen
  error types.

Consequence: a selection head is not adding information the model lacks — the decision is
already linearly present — it is *exposing* it as a supervised, thresholdable output. The
margin-of-selection finding is the direct justification for gating on it.

---

## 2. Constrained decoding: cost, tax, and compatibility with our tokenizer

### 2.1 Cost per token — solved

| Engine | Mechanism | Per-token overhead | Compile cost | Source |
|---|---|---|---|---|
| XGrammar (arXiv 2411.15100) | byte-level pushdown automaton; >99 % of vocabulary pre-classified as *context-independent* at compile time, only the rest checked at run time | **< 40 µs/token** (JSON schema, CFG-JSON), < 200 µs (XML, Python DSL); up to **100×** lower per-token latency than prior engines; up to **80× end-to-end** serving speedup (Llama-3.1, H100) | "often seconds, sometimes minutes" per llguidance's comparison [V] | [S] for the numbers |
| llguidance (Guidance) | byte-level token trie walked by a lexer/DFA; ~13 cycles per trie node | **~50 µs/token CPU for a 128k vocabulary**; <1 % of masks exceed 1 ms on JSONSchemaBench | negligible startup | [V GitHub README + `docs/toktrie.md`] |
| Outlines (arXiv 2307.09702) | regex → FSM, precomputed token-transition tables | low per token once compiled | **40 s to 10+ min** on complex schemas [S, blog]; large memory | [P] for the paper |
| llama.cpp GBNF | character-level grammar walked per token; supports `<[token-id]>` terminals | ms-scale; documented "performance gotchas" for `x? x? x?` repetition and deep stacks; reported hangs on >10k-token prompts [S issue] | none | [V grammars/README] |
| SGLang compressed FSM | merge singular-transition FSM edges → **jump-forward** several tokens per step, with retokenisation of the jumped segment | **negative**: 1.6× higher JSON-decode throughput than unconstrained [S] | regex compile | [S] |
| Grammar-constrained decoding, 2023 (arXiv 2305.13971) | character-level | 1–69 ms/token on CPU depending on task | — | [S] |

For Prophet, vocabulary 32,768 is 4× smaller than the 128k the llguidance figure was
measured on; the trie cost scales with total token bytes, so **expect ~10–20 µs/token
[C]** against decode steps of 3–8 ms on the phone (§5). Overhead is < 1 %. The
compile-time cost is what matters on device: a JSON-schema→automaton compile must be
cached per tool set, which is trivial (schemas change rarely within a session).

### 2.2 The "structured output tax" — real, but a *when* problem, not a *whether* problem

| Study | Finding | Tag |
|---|---|---|
| *Let Me Speak Freely?* (arXiv 2408.02442) | gpt-3.5-turbo, claude-3-haiku, gemini-1.5-flash, LLaMA-3-8B-Instruct, Gemma-2-9B: format-restricting instructions degrade reasoning tasks (GSM8K, Last Letter, Shuffled Objects); **stricter constraint → larger drop**; JSON-mode worst when the answer field precedes the reasoning field. Classification tasks unaffected or improved. Exact numbers not retrievable — VERIFY. | [S] |
| *Capacity, Not Format* (arXiv 2606.09410) | MATH-Hard: Sonnet JSON 88.7 ± 4.0 vs CoT 89.3 ± 1.7 — **no tax with spare capacity**. Haiku **−36.2 pp** (p < 1e-4), mostly truncation under a standard token budget; GPT-4o-mini **−28.0 pp** even with an extended budget — pure capacity competition. A *delayed-structure* ablation (reason freely, then format) recovers to 80–87 %. | [S] |
| DOMINO (arXiv 2403.06988) | The tax has a second, mechanical cause: **token misalignment**. Naive template constraining (Guidance, 2024) loses up to **11 pp**; GSM8K/Mistral-7B: unconstrained 0.415, Guidance 0.345, DOMINO (subword-aligned, with pre-computation) **0.418** — and 1.77–2.71× *faster* than unconstrained. | [S] |
| JSONSchemaBench (arXiv 2501.10868) | 10k real schemas, six engines. Coverage on easy (GlaiveAI) schemas > 86 % for all; on GitHub-Hard: Guidance 41 %, llama.cpp 39 %, XGrammar 28 %, Outlines 3 %, OpenAI 9 %, Gemini 0. **GSM8K accuracy with Guidance constraints: 80.1 % → 83.8 %** (constraint *helped*). Guidance has better time-per-output-token than XGrammar. | [S] |
| *Lost in Space* (arXiv 2502.14969) | Much of the reported GCD degradation is the grammar's token *choices*: +5–10 % from letting the grammar accept leading-whitespace tokens, largest gain on small models. | [S] |
| *Structure snowballing* (arXiv 2604.06066) | constrained decoding during *reflection* propagates early structural commitments — another instance of structure-before-reasoning. | [P] |

Synthesis: the tax appears when structure is imposed **before** or **instead of**
reasoning, in models near their capacity limit — which describes Prophet-mini exactly.
It disappears when (a) the model reasons first and the grammar applies only to the action
span, and (b) the engine is subword-aligned. Both are design choices, not costs. **Prophet
must never constrain the `<|think|>` span; the grammar switches on at `<|call|>` and off
at `<|/call|>`.**

### 2.3 Compatibility with Prophet-Tok v1

Prophet-Tok is a byte-fallback BPE with single-digit tokens, no merges across newlines,
explicit indentation tokens, 32,768 ids, and 256 reserved ids (`prophet/data/tokenizer.py`).
Each property interacts with grammar decoding:

| Property | Effect on constrained decoding | Action |
|---|---|---|
| **Byte fallback** (ids 0–255 are raw bytes) | Engines that walk a *byte-level* trie (llguidance, XGrammar) handle it natively: a multi-byte codepoint is a path of byte edges, and a lone continuation byte is only allowed where the automaton is mid-codepoint. Character-level engines (GBNF, early Outlines) must be told these tokens are bytes, or they will mask them inside strings and make non-ASCII string values unreachable. | Require a byte-level engine. Inside JSON strings, allow byte tokens only in UTF-8-valid positions. |
| **Single-digit tokens** | Numbers are the classic misalignment case (a token `123,` straddles number-end and separator). With one digit per token, **integer/number grammars are perfectly token-aligned by construction** — DOMINO's problem does not arise for numerics. The price: a 10-digit id costs 10 decode steps (§5). | Nothing for correctness; §4 handles the cost. |
| **Punctuation runs merge** (`":`, `"},`, `{"` are single tokens) | Multi-character syntax tokens straddle several grammar transitions. Trie-based engines advance the automaton byte-by-byte within the token and are exact; template-based engines are not (the 11 pp DOMINO loss). | Trie-based engine; **jump-forward** all syntax (§5). |
| **No merge across `\n`, indentation tokens** | Pretty-printed JSON would emit indentation tokens whose identity depends on nesting depth — a needless choice point. | Grammar emits **compact JSON** (no whitespace). |
| **Reserved ids** (256–511) | Must be masked everywhere inside a call, except the three terminals the grammar expects (`<|/call|>`, `<|copy|>`, `<|nocall|>`, §7.2). Leaking `<|eos|>` mid-call is a real failure mode in naive setups. | Mask set is part of the grammar. |
| **Leading-space tokens** (`" Paris"`) | Inside string values the model's natural continuation often has a leading space; the grammar must accept both (`Lost in Space`). | Accept both spellings of string content. |
| **Prompt boundary** | Ending the prefix at `"` forces the value to start with a token that has no leading space — byte-sampler (arXiv 2506.14123) shows exact BPE-consistent byte-level conditioning is possible but costly; the cheap fix is the previous row. | Same. |

Net: Prophet-Tok is *easier* to constrain than a 128k Llama/Qwen vocabulary (fewer tokens
to classify, digits aligned, no whitespace-run ambiguity), provided the engine is
byte-level. **No tokenizer change is needed.** For Prophet-main, which inherits the
donor's ~152k vocabulary (D10), the same engines apply at the ~50 µs figure.

---

## 3. Structured action heads

| Method | Mechanism | Cheaper at inference? | More reliable? | Open-set tools? | Citation |
|---|---|---|---|---|---|
| Toolformer | API calls inserted as text spans `[QA(...)]`; self-supervised filter keeps calls that reduce perplexity on later tokens; GPT-J 6.7B beats GPT-3 175B zero-shot; calculator invoked in 97.9 % of relevant cases | no (calls are text) | yes for *when to call* (the filter is a when-to-call signal) | yes | arXiv 2302.04761 [S] |
| Gorilla | LLaMA-7B, retriever-aware fine-tuning on APIBench (TorchHub 94 / TensorHub 696 / HF 925 APIs); AST sub-tree match; 67–94 % with oracle retriever; hallucination below GPT-4 | no | yes on hallucinated APIs, via retrieval + docs in context | yes | arXiv 2305.15334 [S] |
| ToolkenGPT | one new LM-head row ("toolken") per tool; frozen LLaMA-33B; toolken fires → switch to argument mode; 234 toolkens on KAMEL; beats ReAct/CoT on GSM8K-XL, FuncQA | selection = 1 step; arguments still text | yes on tool choice; no on arguments | **no** — every tool needs an embedding trained from demonstrations | arXiv 2305.11554, NeurIPS 2023 [S] |
| ToolGen | 47k tools as atomic vocabulary tokens in Llama-3-8B; 3 stages (tool memorisation → retrieval training → agent tuning); SoPR 54.19 / SoWR 49.70 on StableToolBench (G.T.), beats ToolRetriever, matches IterFeedback on NDCG | yes — **no retrieval, no schemas in context** | yes on retrieval (NDCG) | **no** — new tool = new token = retraining | arXiv 2410.03439, ICLR 2025 [S] |
| Octopus v2 | Gemma-2B; one functional token per API (`<nexa_i>`), 20 Android APIs, 100–1000 examples each; **99.524 %** accuracy; **0.38 s** per call; **35×** lower latency than Llama-7B+RAG; **context −95 %** | yes — the biggest reported win, from dropping schemas *and* the name tokens | yes on a closed set | **no** | arXiv 2404.01744 [S] |
| Hammer | function *masking*: randomly replace function/parameter names with meaningless strings during training so the model reads descriptions instead of memorising names; irrelevance data; Qwen2.5-coder 0.5–7B; Hammer-7b 83.92 (BFCL v2) | no | **yes on name-hallucination and irrelevance** | yes — that is the point | arXiv 2410.04587 [V README + S] |
| TOOLDEC | finite-state decoding that only ever emits existing tool names and type-valid arguments; syntax errors → 0, name hallucination → 0 on unseen tools | slightly (no retries) | yes, on exactly the classes in §1.1 marked "grammar" | yes | arXiv 2310.07075 [P] |
| SimpleTool / RealtimeTool | special tokens compress JSON syntax 4–6×; function name and each argument decoded by **parallel heads sharing the prefix KV**; latency = max(head) not sum; **3–6× end-to-end**, 61.2 ms P50 at 4B on a consumer GPU | **yes** | "competitive" accuracy (games, robotics, animation) | yes | arXiv 2603.00030, ICML 2026 [S] |
| ToolSpec | training-free speculative decoding: FSM fills deterministic schema tokens, drafts variable fields, reuses similar past calls as drafts; up to **4.2×** on API-Bank / ToolAlpaca / BFCLv2 with Qwen2.5-7B/14B | **yes** | neutral (lossless) | yes | arXiv 2604.13519 [S] |
| ToolRL | GRPO with reward = format + correctness decomposed into **tool name / parameter names / parameter values**; Qwen2.5-1.5B/3B/7B + Llama-3.2-3B; **+10 pts absolute over SFT** on the same 4k examples (ToolACE 2k + Hammer 1k + xLAM 1k) | no | yes | yes | arXiv 2504.13958 [V README + S] |
| ReTool | RL with a live code interpreter; 32B reaches AIME 67 % in 400 steps | no | n/a (code tools) | — | arXiv 2504.11536 [P] |
| Agent-R1 | step-level MDP RL framework for multi-turn tool agents | no | — | — | arXiv 2511.14460 [S] |
| When2Call | 15k SFT + 9k preference pairs on *whether* to call; RPO raises BFCL irrelevance while keeping call accuracy | no | **yes on omission/over-calling** | — | arXiv 2504.18851 [S] |
| STAR | distillation + similarity-guided RL for 0.6B; BFCL v3 51.70 | no | yes at tiny scale | — | arXiv 2602.03022 [S] |
| Linear probes (§1.4) | read tool choice / call validity from hidden states; gate or steer | n/a | yes as a gate | yes | 2605.07990, 2608.00218, 2608.27750 [S] |

**What is cheaper:** everything that removes decode steps — tool tokens (ToolkenGPT,
ToolGen, Octopus), syntax compression and parallel heads (SimpleTool), jump-forward and
speculation (SGLang, ToolSpec). Octopus's 35× is mostly the *context* reduction (no
schemas), not the emission itself; the emission part is worth ~5–10× (§5).

**What is more reliable:** (i) removing the name/syntax classes entirely (TOOLDEC,
grammars, function masking); (ii) supervising the *decision* to call (Toolformer's
filter, When2Call, the probes); (iii) decomposed rewards (ToolRL). Note that none of the
"more reliable" rows fix value errors — that is §4.

**The tension Prophet must resolve:** the cheapest mechanisms (fixed tool vocabulary)
are closed-set. ToolGen needs three training stages per tool set; Octopus needs 100–1000
examples per function. Prophet-main's tools are whatever the host application registers
(an MCP-style open set), and Prophet-mini's are a small fixed set on the phone. The design
therefore uses a **pointer over schema anchors** (works for any tool present in context,
zero-shot) as the primary mechanism and keeps **32 bound slot tokens** as a
ToolkenGPT-style *optional* fast path for a pinned on-device tool set — ablation A3-4
decides whether the slot path earns its ids.

---

## 4. Argument grounding: copy versus generate

### 4.1 Most arguments are copies

In BFCL/ToolBench-style calls the argument values are overwhelmingly one of: a literal
from the user turn (city, date, name), an identifier from an earlier tool result (order
id, file path, message id), or a schema enum. ComplexFuncBench's authors attribute their
dominant `value_error` class to "long-context parameter extraction" — carrying a value
from a prior response into the next call [S]. *Butterfly Effects* finds that every
parameter-failure category except name hallucination is driven by the **input sources**,
not the model [S]: the value was there and was not transported faithfully. The industrial
failure pattern is the same [S, arXiv 2605.11234]: a statistically plausible but wrong
identifier → empty result → confident wrong summary. A hallucinated *fact* can be
checked; a hallucinated *query parameter* silently returns the wrong data.

Generation and copying are different operations with different error profiles. Generating
`ORD-2024-0091837` under Prophet-Tok is **18 independent single-digit/char decisions
[C]** (§5, `id_lookup` example: 18 digit units), each a chance to drift; copying it is one
decision — *which span* — with a probability mass concentrated on a handful of candidates
that appeared verbatim in context.

### 4.2 What is known about copy mechanisms

| Work | Setting | Result | Tag |
|---|---|---|---|
| Pointer networks (Vinyals et al., arXiv 1506.03134) | output = positions of the input | the original mechanism; exact for sorting/convex-hull-type outputs | [P] |
| CopyNet (Gu et al., arXiv 1603.06393); pointer-generator (See et al., arXiv 1704.04368) | seq2seq with mixture of generate and copy distributions, gate p_gen | copy handles OOV and rare identifiers; hallucinated entities drop; risk is over-copying | [P] |
| Slot filling with joint pointer + attention (Zhao & Feng, ACL 2018) | spoken-language slot values | pointer wins on OOV slot values present in the utterance — the exact regime of tool arguments | [P] |
| Text-to-SQL value copying (RYANSQL; SV2-SQL) | WHERE-clause values | slot-filling + value-extraction modules separate *which column* from *which literal*; extraction from the question beats generation | [S] |
| PICARD (arXiv 2109.05093) | incremental parsing constrains SQL decoding | eliminates schema-invalid outputs without retraining | [P] |
| LargePiG (arXiv 2410.11366, WWW 2025) | training-free: turn an LLM into a pointer-generator by mixing its attention-derived copy distribution with its vocabulary distribution | lower relevance/factuality hallucination in query generation, better doc-QA | [S] |
| Copy-Paste prompting (arXiv 2510.00508) | explicit copy instruction | **+10.9 to +19.1 pts** contextual faithfulness over baselines | [S] |
| Flattened schemas (industry, unverified) | complex nested APIs | "40–60 % fewer parameter hallucinations" — a hint that *addressability* of the value is what matters | [U] |

No paper measures pointer-copy on function-call arguments specifically at 1–4B — this is
a gap and is ablation A3-3. The consistent prior across summarisation, slot filling and
text-to-SQL is: **when the target is verbatim in the input, copying beats generating on
exact-match, and the gap widens for rare strings and small models.**

### 4.3 What copying cannot do

Values that are *derived* (a date computed from "next Friday", a unit-converted quantity,
a regex the user described in words) must be generated; ComplexFuncBench's "constrained
value reasoning" errors are of this type and a copy head does not touch them. The gate
must therefore learn *when* to copy; a mis-gated copy is a new failure mode
(copying `"Paris"` when the user said "not Paris"). This is why the design keeps a
mixture with a learned gate rather than forcing copy for string arguments.

---

## 5. Latency arithmetic on-device

### 5.1 Decode budget

From `python -m prophet.budget` (bandwidth-bound ceilings; real kernels reach 50–70 %) [C]:

| Model | Device | k | Ceiling | Realistic (×0.5–0.7) | Step time |
|---|---|---:|---:|---:|---:|
| prophet-mini 253M | iPhone 17 Pro, 8k ctx | 2 | 472 tok/s | 235–330 tok/s | 3.0–4.3 ms |
| prophet-main 3.83B/408M | iPhone 17 Pro, 8k ctx | 2 | 289 tok/s | 145–200 tok/s | 5.0–6.9 ms |
| prophet-main | Mac Studio | 4 | 1,130 tok/s | 565–790 tok/s | 1.3–1.8 ms |
| prophet-main | RTX 5090, 32k ctx | 4 | 3,536 tok/s | 1,770–2,475 tok/s | 0.4–0.6 ms |

Measured reference points from R08 [V there]: Qwen3.5-2B 4-bit on iPhone 17 Pro via
MLX-Swift **61 tok/s**; LFM2.5-350M on the ANE **52 tok/s**; Llama-3 8B Q4 on an RTX 5090
~142 tok/s. Prophet's 145–200 tok/s on the phone for main is the *ceiling-derived*
figure; the task's "~100–300 tok/s" bracket is the right planning range.

### 5.2 What a call costs in tokens under Prophet-Tok

Pre-tokenisation units (`prophet.data.tokenizer.pre_tokenize`) are a **lower bound** on
token count — merges never cross a unit, rare identifiers split further [C]:

| Call | chars | units (≥ tokens) | of which digits | of which punctuation |
|---|---:|---:|---:|---:|
| Hermes-style `get_weather(city, unit)` | 99 | **32** | 0 | 20 |
| `edit_file(path, line, new_text)` | 151 | **49** | 3 | 25 |
| `get_order(order_id="ORD-2024-0091837", user_id=5583921)` | 113 | **53** | 18 | 22 |
| OpenAI-style `search_flights(...)` with escaped JSON string | 187 | **48** | 10 | 21 |

Two things stand out. Roughly **40–50 % of the tokens in a call are JSON syntax** — every
one of them is a decode step whose outcome is known before it is taken. And identifiers
are digit-heavy: the single-digit repair that helps arithmetic makes ids **expensive to
generate and cheap to copy**.

### 5.3 Structured emission, step by step

For `edit_file(path="/home/user/project/src/utils/parser.py", line=142, new_text="return None")`:

| Stage | Text JSON | Structured (§7) |
|---|---:|---:|
| decide to act, choose tool | ~3–6 tokens (`<tool_call>{"name":"edit_file"`) | **1 step**: `<|call|>` is emitted and the selection head fires on the same hidden state; the tool name is jump-forwarded |
| syntax + parameter names | ~20 tokens | **0 steps** (grammar singular transitions, prefilled) |
| `path` value | ~12 tokens | **1 step** (`<|copy|>`; start/end pointers fire in parallel) |
| `line` value | 3 tokens | 1 step (copy) or 3 (generate) |
| `new_text` value | ~3 tokens | ~3 tokens (generated — derived, not in context) |
| close | 2–4 tokens | 1 step (`<|/call|>`) |
| **decode steps** | **≥ 49** (realistically 55–60) | **7–9** |
| prefill of jump-forwarded tokens | — | ~35 tokens, batched: ≈ 10–20 ms on the phone [C] |
| wall clock, main on iPhone (150 tok/s) | **0.33–0.40 s** | **≈ 0.06–0.08 s** |
| wall clock, mini on iPhone (280 tok/s) | 0.18–0.21 s | ≈ 0.03–0.04 s |

Ratio: **6–8× fewer decode steps, ~5× wall clock** on the phone [C]. This agrees with the
independently reported 3–6× (SimpleTool, parallel heads + compressed syntax [S]) and
4.2× (ToolSpec, schema-aware speculation [S]); Octopus's 35× includes removing ~3k tokens
of schemas from the prompt, which §7 deliberately does *not* do for main (open-set tools)
but can do for mini's pinned slot path. A ten-call agentic task saves ~3 s on the phone;
at k=4 on the Mac/5090 the absolute times shrink but the ratio is unchanged — and the
recurrent core makes every saved decode step worth k core passes.

Prefill is the other side of the ledger: 20 schemas ≈ 3k tokens ≈ 0.5–1 s on the phone at
main's prefill rate [C, rough] — larger than the emission saving on a cold prompt. Schema
prefix must be KV-cached across turns (it is a fixed prefix; `ProphetCache` already
supports this) or, for mini, replaced by bound slots.

---

## 6. Datasets and licences

Read with R10's rule: **a licence string on a card is not a licence audit**, and a
permissive licence on data generated by a model whose terms bind derivatives is not
clean. Gemma-generated rows are blocked outright by `prophet.data.mixture`
(`BLOCKED_LICENSES["gemma"]`); OpenAI-generated rows were excluded from the release mix
by R10 (OpenHermes precedent). All `huggingface.co` cards were unreachable; licence fields
come from GitHub READMEs or search snippets quoting the card.

| HF id | Size | Licence | Generated by | Verdict for a permissive Prophet release |
|---|---:|---|---|---|
| `Agent-Ark/Toucan-1.5M` | 1.5M trajectories, 495 MCP servers, 2,000+ tools; multi-turn, parallel, real executions | **apache-2.0** [S, card + IBM blog] | **GPT-OSS-120B (Apache-2.0), Kimi-K2 (modified MIT), Qwen3-32B (Apache-2.0)** [S, paper] | **Tier A — use.** Cleanest large corpus. Kimi-K2's attribution clause triggers only above 100M MAU / $20M revenue [P]. Caveats stated on the card: community MCP servers, June–Sept 2025 time-bound responses. |
| `nvidia/Nemotron-Post-Training-Dataset-v2` (and `-v1`) | tool-calling subset ~400k [S] | **cc-by-4.0** (small ODC-BY / CC-BY-SA prompt subsets) [V via R10] | Qwen3-235B-A22B (Apache-2.0) [S]; single-turn prompts sampled from xlam-60k, glaive-v2, When2Call [S] | **Tier A — use.** Prompt provenance inherits CC-BY-4.0 / Apache-2.0: fine. |
| `nvidia/Nemotron-Agentic-v1` | tool sets from public datasets, simulated conversations | cc-by-4.0 [S] | Qwen3-235B-A22B-Thinking/Instruct-2507 [S] | **Tier A — use.** |
| `nvidia/When2Call` | 15k SFT + 9k preference + 3.95k test | **cc-by-4.0**, "ready for commercial use" [S, card] | Mixtral-8x22B (Apache-2.0) [S, paper]; prompts from APIGen simple/multiple | **Tier A — use** (see next row for the APIGen inheritance). Only corpus that supervises *not* calling. |
| `Team-ACE/ToolACE` | 11.3k dialogues, 26,507 APIs | **apache-2.0** [S, card] | multi-agent self-evolution; **backbone LLM not established** — VERIFY | Tier A if the generator is Apache/MIT, Tier B otherwise. Used by ToolRL (2k) and ToolACE-8B. |
| `Salesforce/xlam-function-calling-60k` | 60k single-turn, 3,673 executable APIs, 21 categories, execution-verified (95 % human-checked) | **Conflict: card says cc-by-4.0 [S, two independent snippets]; the xLAM GitHub README summary said CC-BY-NC-4.0 "research only" [V summary of README — may be the *model* licence bleeding into the summary].** | first 33,659 rows DeepSeek-V2-Chat, rest Mixtral-8x22B-Instruct [S] | **Hold until the card is read.** If cc-by-4.0: usable (DeepSeek-V2's licence has use-based restrictions on *the model*, and the output-derivative question is the same grey R10 accepted for DeepSeek-R1 traces — flag in `DATA_PROVENANCE.md`). If NC: blocked by `BLOCKED_LICENSES["cc-by-nc"]`. |
| `Salesforce/APIGen-MT-5k` | 5k multi-turn | same conflict as above | APIGen-MT pipeline (models not established) | **Hold.** |
| `MadeAgents/xlam-irrelevance-7.5k` | 7.5k irrelevance cases | derived from xlam-60k → inherits | — | Follows xlam-60k. |
| `glaiveai/glaive-function-calling-v2` | 113k | **apache-2.0** [S] | Glaive's own synthetic pipeline; **generator undisclosed** | **Tier B** — permissive licence, unknown teacher. Widely used (Granite, Hermes, Nemotron prompts). Usable for ablations; release-mix only if provenance policy accepts undisclosed teachers. |
| `NousResearch/hermes-function-calling-v1` | ~11.5k (func-calling, singleturn, json-mode, agentic subsets) [P for the count] | **apache-2.0** [S] | synthetic, led by @interstellarninja; **generator undisclosed** (GPT-4 era) | **Tier B.** Same as glaive. |
| `OpenBMB/ToolBench` (ToolLLM) | 126k instructions, 16k RapidAPI APIs [P] | **apache-2.0** [S] | **ChatGPT (gpt-3.5-turbo-16k)** [S] | **Tier B/C** — OpenAI-generated, same status as OpenHermes in R10: ablations only, not the release mix. Also noisy (StableToolBench exists because ~half of ToolBench queries were unsolvable). |
| `gorilla-llm/APIBench` | 1,645 APIs, ~17k instruction pairs [P] | **apache-2.0** ("all leaderboard statistics and data used to train the models") [V README] | GPT-4 self-instruct [S] | **Tier B/C** — OpenAI-generated. |
| BFCL test sets (`gorilla` repo) | 4,751 (v3) + v4 agentic | apache-2.0 [V] | expert + live users | **Evaluation only. Never in a training mix** — decontaminate against it (`prophet/data/decontaminate.py`). |
| `liminghao1630/API-Bank` | ~2k eval dialogues + training split | **MIT** [S] | GPT-4 assisted | Evaluation; training split Tier B/C. |
| `sierra-research/tau-bench`, `tau2-bench` | retail / airline / telecom | **MIT** [V GitHub] | hand-built | **Evaluation only** (multi-turn with simulated user). |
| `google/functiongemma-270m-it` and anything distilled from it | — | **Gemma Terms of Use** | — | **BLOCKED** (`BLOCKED_LICENSES["gemma"]`): §1.1(e) makes a model trained on Gemma outputs a Model Derivative. Do not use FunctionGemma as a teacher, a draft model, or a data generator. |

**Licence verdict.** A fully clean (Tier A) function-calling mix exists and is large:
**Toucan-1.5M + Nemotron-v2 tool-calling + Nemotron-Agentic-v1 + When2Call ≈ 2M
trajectories under Apache-2.0 / CC-BY-4.0 with Apache-2.0 teachers.** This is a better
position than R10 found for general instruction data. The two most-cited academic sets
(xLAM-60k, ToolACE) are *probably* usable but each has one unverified field (licence
string vs README; generator identity) that must be read on the card before download — the
`mixture` validator already refuses `unknown`, so this is enforced mechanically. Register
in `configs/data_mixture_v1.yaml` as: Toucan `apache-2.0`, Nemotron `cc-by-4.0`,
When2Call `cc-by-4.0`, ToolACE `unknown` (until verified), xlam-60k `unknown` (until
verified), glaive/hermes `apache-2.0` with a `teacher: undisclosed` note.

---

## 7. Design for Prophet

### 7.1 Principles

1. **Think free, act typed.** No constraint touches the `<|think|>…<|/think|>` span. The
   action grammar is active only between `<|call|>` and `<|/call|>` (§2.2).
2. **Open-set by default.** Tool identity is a *pointer over schema anchors present in
   context*, so any tool whose schema is in the prompt is callable without training.
   Fixed slot tokens are an optional fast path for pinned tool sets (mini on the phone).
3. **Every failure class in §1.1 gets its own supervised output**: no-call/over-call
   (selection head, `<|nocall|>`), wrong tool (selection head), syntax and names
   (grammar), missing parameter (grammar), wrong value (copy head + grammar types),
   confidence (existing head, retargeted). Sequencing errors remain a data problem.
4. **Reversible.** `heads.action_head=False` yields exactly today's model; a model trained
   with the heads can still emit text JSON (the LM loss keeps a small weight on syntax).
5. **Cache-neutral.** No new per-token cache. The copy pointer reuses the coda's NoPE
   global-attention key cache (position-free keys are what a content pointer wants).

### 7.2 Reserved token ids

`SPECIAL_TOKENS` currently names 16 ids (256–271); ids 272–511 are reserved and free
(`N_RESERVED = 256`, `prophet/data/tokenizer.py`). Append — never reorder — a block of
**6 control ids + 32 slot ids = 38**, leaving 202 for R12:

| id | name | role |
|---:|---|---|
| 272 | `<|tool_def|>` | opens one tool schema in the system prompt |
| 273 | `<|/tool_def|>` | closes it — **the anchor**: its hidden state, computed causally, summarises the schema that precedes it |
| 274 | `<|call|>` | the act decision; the selection head is read at this position |
| 275 | `<|/call|>` | closes the call; the confidence head is read here |
| 276 | `<|copy|>` | "the next value is a span of context"; start/end pointers are read at this position |
| 277 | `<|nocall|>` | emitted when the selection head picks the null option; trains omission/over-call explicitly |
| 278–309 | `<|slot_0|>` … `<|slot_31|>` | optional ToolkenGPT-style bound slots (§7.4, ablation A3-4) |

Tool results reuse the existing `<|tool|>` role token — no new ids. `<|idk|>` (R09)
remains the abstention for *answers*; `<|nocall|>` is the abstention for *actions*.

### 7.3 How the schema is presented

Text, in the system prompt, one block per tool, compact JSON Schema, in the order the
host registered them:

```
<|system|>…
<|tool_def|>{"name":"edit_file","description":"…","parameters":{"type":"object",
"properties":{"path":{"type":"string"},"line":{"type":"integer"},"new_text":{"type":"string"}},
"required":["path","line","new_text"]}}<|/tool_def|>
<|tool_def|>{"name":"get_order", …}<|/tool_def|>
```

Function masking (Hammer): during training, with p = 0.3 replace `name` and parameter
names by random identifiers *consistently* in the schema and in the target call, so the
model reads descriptions and cannot shortcut on memorised names [S for the technique].
The runtime compiles each schema to an automaton once and caches it keyed by the schema
hash; the schema prefix's KV state is cached across turns.

### 7.4 The heads

All three are read from the coda output `x` (pre-`norm_out`, exactly where
`confidence_head` reads) so that they share the trunk's representation and add no depth.

**Selection head** (read at the `<|call|>` position, or at every assistant position for
the no-call decision during training):

- `q = W_q · RMSNorm(x_t)`, `k_i = W_k · RMSNorm(x_{a_i})` for each anchor position `a_i`
  (input id == `<|/tool_def|>`), plus a learned null key `k_∅`.
- `logits = [q·k_∅, q·k_1, …, q·k_n] / √d_k`; softmax over `n+1` options.
- Params: `2·d·d_k` (+ `d_k`); at d=1536, d_k=128 → 0.39M [C]. FLOPs per step: `n·d_k`.
- Anchors are gathered once per prompt (they sit in the cached prefix); at decode time
  the head is `n+1` dot products.
- Optional slot path: if the host pins a tool set, each schema is also tagged with
  `<|slot_i|>` before `<|tool_def|>`, and the LM head may emit `<|slot_i|>` directly;
  slot embeddings are trained like any token. This is the closed-set fast path; it needs
  no anchors in context at all (Octopus-style, context −95 %) but only for tools seen in
  training. A3-4 decides whether it survives.

**Copy head** (read at the `<|copy|>` position):

- Two query projections `W_s, W_e` (start, end) against the **existing keys** of the coda's
  NoPE global-attention layer, one designated KV head: `s = (W_s·x_t)·K / √d_head`,
  `e = (W_e·x_t)·K / √d_head`, each a softmax over context positions ≤ t, with `e ≥ s`
  enforced by masking at decode. Both fire on the same step (SimpleTool's parallel-head
  principle). The span is retokenised, validated against the grammar's current state
  (type/enum/pattern); on violation the runtime falls back to generation for that value.
- Gate `g = σ(w_g · RMSNorm(x_t))` at every value-start position decides
  `<|copy|>` vs generate; at inference the grammar exposes `<|copy|>` as a legal token
  only at value-start, so the gate is simply the LM probability of `<|copy|>` and
  needs no separate parameter — the explicit gate is kept for the loss (below).
- Params: `2·d·d_head` = 0.39M at d=1536 [C]. **Extra cache: 0 bytes.**

**Confidence head** (exists; `heads.confidence_head`): retargeted at `<|/call|>` to
"this call is correct" (AST-match / execution label). It gates three things (§7.6).

### 7.5 Loss

With `−100` ignore labels on tokens the runtime jump-forwards (tool name, syntax,
parameter names) at weight `ω_syn = 0.1` — small, not zero, so text-JSON fallback stays
alive:

```
L = L_lm(non-jumped tokens) + ω_syn · L_lm(jumped tokens)
  + λ_sel  · CE(selection logits, target ∈ {∅, 1..n})     at <|call|> / decision positions
  + λ_ptr  · [CE(start) + CE(end)]                          at <|copy|> positions
  + λ_gate · BCE(gate, value-is-verbatim-in-context)       at value-start positions
  + λ_conf · BCE(confidence, call-is-correct)               at <|/call|>   (existing term)
```

Defaults: `λ_sel = 0.5, λ_ptr = 0.5, λ_gate = 0.1, λ_conf = 0.1` (existing). Targets are
derived offline from the datasets: the selection target is the index of the gold tool
among the presented schemas (∅ for When2Call negatives and BFCL-style irrelevance);
copy targets are the *last* verbatim occurrence of the gold value in context (ties → the
most recent tool result); gate target is 1 iff such an occurrence exists. This is
precisely ToolRL's reward decomposition (name / parameter names / values) turned into
per-term cross-entropies, which is what "cleaner training signal" means concretely:
a wrong tool is one wrong argmax over `n+1` options, not a slightly-wrong token string.

For RL later (R10's polishing phase): the same decomposition serves as the reward, and
the selection logits give a natural entropy term for the over-calling problem the
Reasoning Trap describes.

### 7.6 What the confidence head gates

Read at `<|/call|>` (call validity, AUROC 0.86–0.90 in probes [S]) and, through the
selection head's margin (top-1 − top-2, the 14–21× signal [S]), *before* the call:

| Signal | Threshold action |
|---|---|
| selection margin small **and** `∅` not chosen | emit `<|nocall|>` + ask a clarifying question (τ-bench's dominant failure is acting on ambiguity) |
| confidence at `<|/call|>` below τ₁ | re-decode the argument values with sampling (best-of-4 over values only — 4 × ~6 steps, still cheaper than one text call) |
| confidence below τ₂ (< τ₁) on the phone | hand the call to the Mac/5090 tier (same weights, larger k) — the runtime-depth dial is the escalation path |
| confidence below τ₂ with no larger tier | `<|nocall|>` and answer in prose with `<|idk|>` semantics |

The calibration study is R09's; A3 only adds the targets.

### 7.7 PyTorch sketch

Fits `ProphetModel` as `confidence_head` and `mtp_heads` are wired: constructed in
`__init__` under a config flag, read from the coda output `x`, returned on
`ProphetOutput`, scored in `prophet/train/loss.py`. Not wired here (rule: this report
modifies nothing but itself).

```python
# prophet/config.py — addition to HeadsConfig
@dataclass
class HeadsConfig:
    ...
    action_head: bool = False
    """Typed tool emission: selection pointer over schema anchors, span-copy pointer for
    argument values, explicit no-call option. Off => model is byte-identical to today."""
    action_dk: int = 128
    action_kv_head: int = 0          # which coda global-attention KV head the copy pointer reads
    n_tool_slots: int = 0            # 0 = pointer-only (open set); 32 = also bound slot tokens
    sel_loss_weight: float = 0.5
    ptr_loss_weight: float = 0.5
    gate_loss_weight: float = 0.1
    jumped_token_lm_weight: float = 0.1
```

```python
# prophet/modeling/action.py
from __future__ import annotations
import torch
from torch import Tensor, nn
from prophet.modeling.layers import RMSNorm


class ActionHeads(nn.Module):
    """Selection pointer + span-copy pointer + copy gate, read from the coda output.

    Nothing here owns a cache: anchors live in the prompt's cached hidden states and the
    copy pointer scores the coda global-attention layer's existing keys.
    """

    def __init__(self, d_model: int, d_k: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, eps)
        self.sel_q = nn.Linear(d_model, d_k, bias=False)
        self.sel_k = nn.Linear(d_model, d_k, bias=False)
        self.null_key = nn.Parameter(torch.zeros(d_k))     # the "do not call" option
        self.copy_start_q = nn.Linear(d_model, head_dim, bias=False)
        self.copy_end_q = nn.Linear(d_model, head_dim, bias=False)
        self.gate = nn.Linear(d_model, 1)
        self.d_k, self.head_dim = d_k, head_dim

    def select(self, x_t: Tensor, anchors: Tensor, anchor_mask: Tensor) -> Tensor:
        """x_t: (b, d) at the <|call|> position; anchors: (b, n, d) hidden states at the
        <|/tool_def|> positions; anchor_mask: (b, n) bool. Returns (b, n+1) logits,
        index 0 = no-call."""
        q = self.sel_q(self.norm(x_t))                                   # (b, dk)
        k = self.sel_k(self.norm(anchors))                               # (b, n, dk)
        k = torch.cat([self.null_key.expand(k.shape[0], 1, -1), k], 1)  # (b, n+1, dk)
        mask = torch.cat([anchor_mask.new_ones(k.shape[0], 1), anchor_mask], 1)
        logits = torch.einsum("bd,bnd->bn", q, k) / self.d_k ** 0.5
        return logits.masked_fill(~mask, float("-inf"))

    def copy(self, x_t: Tensor, keys: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """x_t: (b, d) at the <|copy|> position; keys: (b, L, head_dim) — one KV head of
        the coda NoPE layer, straight from AttentionCache; valid: (b, L) bool (<= t and
        not a special id). Returns start and end logits over L."""
        h = self.norm(x_t)
        s = torch.einsum("bd,bld->bl", self.copy_start_q(h), keys) / self.head_dim ** 0.5
        e = torch.einsum("bd,bld->bl", self.copy_end_q(h), keys) / self.head_dim ** 0.5
        neg = float("-inf")
        return s.masked_fill(~valid, neg), e.masked_fill(~valid, neg)

    def copy_gate(self, x: Tensor) -> Tensor:
        """(b, s, d) -> (b, s) logit of 'this value is verbatim in context'."""
        return self.gate(self.norm(x)).squeeze(-1)
```

```python
# prophet/modeling/model.py — wiring, mirrors confidence_head / mtp_heads
#   __init__ (after self.confidence_head):
self.action = (
    ActionHeads(d, cfg.heads.action_dk, cfg.head_dim, cfg.norm_eps)
    if cfg.heads.action_head else None
)
#   ProphetOutput gains:
#     sel_logits: Tensor | None       (b, n+1)     when call_positions given
#     copy_start: Tensor | None       (b, L)       when copy_positions given
#     copy_end:   Tensor | None       (b, L)
#     copy_gate:  Tensor | None       (b, s)
#   forward(...) gains keyword args
#     anchor_positions, call_positions, copy_positions: Tensor | None
#   and, after `confidence = ...`:
if self.action is not None:
    copy_gate = self.action.copy_gate(x)
    if anchor_positions is not None and call_positions is not None:
        anchors = x.gather(1, anchor_positions[..., None].expand(-1, -1, x.shape[-1]))
        x_call = x.gather(1, call_positions[..., None].expand(-1, -1, x.shape[-1]))[:, 0]
        sel_logits = self.action.select(x_call, anchors, anchor_positions >= 0)
    if copy_positions is not None:
        # keys of the designated KV head of the coda's NoPE layer; during training taken
        # from the layer's last K projection, at decode from its AttentionCache slot
        keys = self._coda_global_keys(cache)[:, self.cfg.heads.action_kv_head]   # (b, L, hd)
        x_copy = x.gather(1, copy_positions[..., None].expand(-1, -1, x.shape[-1]))[:, 0]
        copy_start, copy_end = self.action.copy(x_copy, keys, valid_mask)
```

```python
# prophet/train/loss.py — additional terms (same shape as the confidence term)
if output.sel_logits is not None and sel_targets is not None:
    sel = F.cross_entropy(output.sel_logits.float(), sel_targets)      # 0 = no-call
    total = total + sel_weight * sel
if output.copy_start is not None and copy_targets is not None:        # (b, 2) start,end
    ptr = (F.cross_entropy(output.copy_start.float(), copy_targets[:, 0])
           + F.cross_entropy(output.copy_end.float(), copy_targets[:, 1]))
    total = total + ptr_weight * ptr
if output.copy_gate is not None and gate_targets is not None:         # -100 where n/a
    m = gate_targets >= 0
    gate = F.binary_cross_entropy_with_logits(
        output.copy_gate.float()[m], gate_targets.float()[m])
    total = total + gate_weight * gate
```

```python
# runtime decode loop (serving path, sketch) — grammar on only inside the call
engine = GrammarEngine.compile(schemas, tokenizer)          # cached per schema hash
while True:
    out = model(next_id, cache=cache, loop_k=k, anchor_positions=anchors,
                call_positions=last_pos, copy_positions=last_pos)
    if in_call:
        mask = engine.token_mask(state)                     # ~10-20 us, byte-level trie
        if engine.only_one_path(state):                     # jump-forward syntax/names
            ids = engine.jump(state); cache.prefill(ids); continue
        logits = out.logits[:, -1].masked_fill(~mask, float("-inf"))
    next_id = sample(logits)
    if next_id == CALL:
        choice = out.sel_logits.argmax(-1)                  # 0 => emit <|nocall|>
        if choice == 0 or margin(out.sel_logits) < tau_margin: emit(NOCALL); break
        cache.prefill(engine.jump_to_tool(choice))          # name + '{' jumped, 0 steps
        in_call = True
    elif next_id == COPY:
        s, e = out.copy_start.argmax(-1), out.copy_end.argmax(-1)   # same step
        span = tokenizer.encode(context_text[s:e + 1])
        if not engine.accepts(state, span): fall_back_to_generation()
        else: cache.prefill(span)
    elif next_id == END_CALL:
        if sigmoid(out.confidence[:, -1]) < tau1: resample_values()
        in_call = False; execute()
```

Everything above is < 1M parameters per model (0.4 % of mini, 0.02 % of main), adds
zero cache, and is a no-op with `action_head=False`.

---

## 8. Ablation plan (each < 6 A100-hours)

Throughput from `prophet.budget` at 35 % MFU [C]: mini ≈ 172M tokens per A100-hour,
main ≈ 74M. A tool-calling SFT epoch over 50k Toucan trajectories × ~3k tokens = 150M
tokens ≈ **0.9 A100-h on mini**, so every ablation below is training-cheap; evaluation
(BFCL-style AST scoring with a local executor) is the cost to watch. Scale: mini (253M)
from a general checkpoint, because R04's warning about false negatives concerns the
*recurrence* bet, not output heads — heads are linear readouts and do not need ≥ 350M to
show an effect. Metrics are the §1.1 classes, scored separately, plus decode steps per
call; **BPB does not decide here** (R11's rule applies to trunk changes, not to action
emission).

| # | Question | Arms | Data | Cost | Go / no-go |
|---|---|---|---|---:|---|
| A3-0 | Baseline: text JSON, same data | SFT on Tier-A mix (Toucan 50k + When2Call 15k), text tool calls | held-out Toucan split (never BFCL), BFCL v3 single-turn + live_irrelevance as external check | 1.5 h | establishes the failure-class histogram for mini; expect omission-dominated (§1.2) |
| A3-1 | Is the structured-output tax real *for us*? | A3-0 weights, decode with grammar on/off, think-span never constrained | same | **0 h train**, ~1 h eval | grammar must give unparseable = 0 and **no drop** in value/tool accuracy; a drop ≥ 2 pts means token misalignment in the engine → fix the engine before anything else |
| A3-2 | Does a supervised no-call/selection decision beat implicit? | + selection head with `∅` (A3-0 + `action_head`, copy off) | + When2Call negatives | 2 h | omission + over-call rate down by ≥ 30 % relative vs A3-0 at equal call accuracy; wrong-tool rate not worse |
| A3-3 | Does copy beat generate on verbatim values? | + copy head; measure `value_error` split into *verbatim-in-context* vs *derived* | Toucan (many ids/paths), ComplexFuncBench-style long-context subset | 2 h | verbatim-value error ≤ ½ of A3-2's; derived-value error unchanged (if it worsens, the gate is over-copying → raise `λ_gate`); decode steps per call ≤ ⅓ |
| A3-4 | Bound slot tokens vs pointer-over-anchors on *unseen* tools | pointer-only vs pointer + 32 slots; test on tools held out of training | Toucan tool-level split | 2 h | slots may win on seen tools (Octopus regime); if they lose ≥ 3 pts on unseen tools, slots become mini-only and the ids stay reserved |
| A3-5 | Does the confidence head separate good from bad calls? | A3-3 weights; AUROC of `<|/call|>` confidence and of selection margin vs AST correctness | held-out | 0 h train, 1 h eval | AUROC ≥ 0.80 (probes report 0.86–0.90 at 4B+ [S]; mini may be lower); selective accuracy at 80 % coverage ≥ +5 pts over unselective |
| A3-6 | Function masking | A3-3 ± masking p = 0.3 | same | 2 h | ≥ +2 pts on unseen-tool split, no loss on seen |
| A3-7 | Latency on the real target | export A3-3 to the iPhone runtime; measure steps and ms per call, text vs typed, with jump-forward | 50 calls | 0 h train | ≥ 4× wall-clock on the phone at k=2, ≥ 6× fewer decode steps |
| A3-8 | Sanity at main scale | A3-3 recipe on the converted donor (0.97B, D10), 2 epochs of 50k trajectories | same | 4 h | replicate A3-2/A3-3 directions; this is the only main-scale run |

Total ≈ 17 A100-hours across eight ablations. Order: A3-0 → A3-1 (cheapest and it can
kill the engine choice) → A3-2 → A3-3 → A3-5, then A3-4/A3-6 in parallel, A3-7 and A3-8
last. Pre-register the go/no-go thresholds above before A3-0 runs.

**What would falsify the track.** If A3-2 does not move omission/over-call at mini scale,
the decision is not linearly exposed at 253M (the probe papers are all ≥ 4B) and the head
is decoration — keep the grammar (free) and drop the heads. If A3-3 shows the gate cannot
be learned (derived-value errors rise as verbatim errors fall), copy becomes a runtime
*option* the host enables per parameter (`"x-copy": true` in the schema) rather than a
learned mixture.

---

## 9. References

Tags: [V] read on a fetched page; [S] search snippet quoting the source; [P] from memory.
arXiv ids given; arxiv.org and huggingface.co were unreachable during this track.

**Failure analysis and benchmarks**
1. Berkeley Function Calling Leaderboard — v1 blog (`gh-pages/blogs/8_…`), v3 multi-turn blog (`gh-pages/blogs/13_…`), README (v4, Apache-2.0 data). `ShishirPatil/gorilla` [V].
2. *When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems*, arXiv 2601.16280 [S].
3. *ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario*, arXiv 2501.10132 [S].
4. *Butterfly Effects in Toolchains: A Comprehensive Analysis of Failed Parameter Filling in LLM Tool-Agent Systems*, arXiv 2507.15296, EMNLP-Findings 2025 [S].
5. *The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination*, arXiv 2510.22977, ICLR 2026 [S].
6. *Reducing Tool Hallucination via Reliability Alignment*, arXiv 2412.04141 [S].
7. *τ-bench*, arXiv 2406.12045; `sierra-research/tau-bench`, `tau2-bench` (MIT) [V licence, S results].
8. *API-Bank*, arXiv 2304.08244 (MIT) [S].
9. *ToolLLM / ToolBench*, arXiv 2307.16789 (Apache-2.0, ChatGPT-generated) [S].
10. *STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models*, arXiv 2602.03022 [S].
11. Qwen3 Technical Report, arXiv 2505.09388 (BFCL v3: 1.7B 56.6, 4B 65.9) [S]. Llama 3.2 model card (BFCL v2: 1B 25.7, 3B 67.0) [V snippet of `meta-llama/llama-models`].
12. *xLAM* arXiv 2409.03215; *APIGen* arXiv 2406.18518; *APIGen-MT* arXiv 2504.03601 [S]; `SalesforceAIResearch/xLAM` README [V].
13. *TinyAgent: Function Calling at the Edge*, arXiv 2409.00608 [S].
14. FunctionGemma model card / Google Developers blog (via Medium) [S].

**Probing / gating**
15. *A Few Neurons Reveal When LLMs Misuse Tools*, arXiv 2608.00218 [S].
16. *Tool Calling is Linearly Readable and Steerable in Language Models*, arXiv 2605.07990 [S].
17. *The Calls are Coming from Inside the Model: Probe-based Detection of Tool-Calling Errors*, arXiv 2608.27750 [S].
18. *LLM Agents Already Know When to Call Tools — Even Without Reasoning*, arXiv 2605.09252 [S].

**Constrained decoding**
19. *XGrammar*, arXiv 2411.15100; *XGrammar-2*, arXiv 2601.04426 [S]; `mlc-ai/xgrammar` README [V].
20. llguidance README and `docs/toktrie.md`, `guidance-ai/llguidance` [V].
21. *JSONSchemaBench*, arXiv 2501.10868; `epfl-dlab/jsonschemabench` [V README, S numbers].
22. *Let Me Speak Freely?*, arXiv 2408.02442 [S].
23. *Capacity, Not Format: Rethinking Structured Reasoning Failures*, arXiv 2606.09410 [S].
24. *Guiding LLMs the Right Way: Fast, Non-Invasive Constrained Generation* (DOMINO), arXiv 2403.06988 [S].
25. *Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding*, arXiv 2502.14969 [S].
26. *Sampling from Your Language Model One Byte at a Time*, arXiv 2506.14123 [S].
27. *Automata-based Constraints for Language Model Decoding*, arXiv 2407.08103 [P].
28. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*, arXiv 2305.13971 [S].
29. *Outlines: Efficient Guided Generation*, arXiv 2307.09702 [P].
30. SGLang, arXiv 2312.07104; LMSYS compressed-FSM blog (2024-02-05) [S].
31. llama.cpp `grammars/README.md` [V].
32. *From Hallucination to Structure Snowballing: The Alignment Tax of Constrained Decoding in LLM Reflection*, arXiv 2604.06066 [P].
33. *Accelerating Constrained Decoding with Token Space Compression* (CFGzip), arXiv 2605.29986 [S].

**Structured action heads**
34. *Toolformer*, arXiv 2302.04761 [S].
35. *Gorilla*, arXiv 2305.15334 [S].
36. *ToolkenGPT*, arXiv 2305.11554, NeurIPS 2023 [S]; `Ber666/ToolkenGPT` [V].
37. *ToolGen: Unified Tool Retrieval and Calling via Generation*, arXiv 2410.03439, ICLR 2025 [S].
38. *Octopus v2: On-device language model for super agent*, arXiv 2404.01744 [S].
39. *Hammer: Robust Function-Calling for On-Device Language Models via Function Masking*, arXiv 2410.04587 [S]; `MadeAgents/Hammer` README [V].
40. *Syntax Error-Free and Generalizable Tool Use for LLMs via Finite-State Decoding* (TOOLDEC), arXiv 2310.07075 [P].
41. *SimpleTool / RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling*, arXiv 2603.00030, ICML 2026 [S].
42. *ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding*, arXiv 2604.13519 [S].
43. *ToolRL: Reward is All Tool Learning Needs*, arXiv 2504.13958 [S]; `qiancheng0/ToolRL` README [V].
44. *ReTool*, arXiv 2504.11536 [P]. *Agent-R1*, arXiv 2511.14460 [S].
45. *When2Call: When (not) to Call Tools*, arXiv 2504.18851 [S].
46. *Chain-of-Tools*, arXiv 2503.16779 [S].

**Copy / pointer**
47. Vinyals et al., *Pointer Networks*, arXiv 1506.03134 [P]. Gu et al., *CopyNet*, arXiv 1603.06393 [P]. See et al., *Get To The Point* (pointer-generator), arXiv 1704.04368 [P].
48. Zhao & Feng, *Improving Slot Filling in SLU with Joint Pointer and Attention*, ACL 2018 [P].
49. *LargePiG: Your Large Language Model is Secretly a Pointer Generator*, arXiv 2410.11366, WWW 2025 [S].
50. *Copy-Paste to Mitigate Large Language Model Hallucinations*, arXiv 2510.00508 [S].
51. PICARD, arXiv 2109.05093 [P]. RYANSQL (Comput. Linguistics 2021); SV2-SQL (Multimedia Systems 2024) [S].

**Datasets**
52. *TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments*, arXiv 2510.01179, ICLR 2026; `Agent-Ark/Toucan-1.5M` (Apache-2.0; GPT-OSS-120B / Kimi-K2 / Qwen3-32B) [S]; `TheAgentArk/Toucan` [V].
53. *ToolACE: Winning the Points of LLM Function Calling*, arXiv 2409.00920, ICLR 2025; `Team-ACE/ToolACE` (Apache-2.0) [S].
54. `Salesforce/xlam-function-calling-60k` (card: cc-by-4.0 [S]; README summary: CC-BY-NC-4.0 [V-summary]) — **conflict, verify**. `Salesforce/APIGen-MT-5k` — same.
55. `glaiveai/glaive-function-calling-v2` (Apache-2.0, 113k) [S]. `NousResearch/hermes-function-calling-v1` (Apache-2.0) [S]; `NousResearch/Hermes-Function-Calling` [V].
56. `nvidia/Nemotron-Post-Training-Dataset-v1/-v2`, `nvidia/Nemotron-Agentic-v1` (CC-BY-4.0; Qwen3-235B teachers) [S, V via R10]. `nvidia/When2Call` (CC-BY-4.0; Mixtral-8x22B) [S].
57. `gorilla-llm/APIBench` (Apache-2.0) [V README]. `MadeAgents/xlam-irrelevance-7.5k` [V README mention].
58. Gemma Terms of Use §1.1(e) — via `prophet/data/mixture.py` and R10 [V].

**Internal**
59. `prophet/modeling/model.py` (head wiring), `prophet/train/loss.py`, `prophet/data/tokenizer.py` (reserved ids), `prophet/data/mixture.py` (licence guard), `prophet/budget.py` (decode ceilings), R08 (measured on-device tok/s), R09 (confidence head), R10 (licence audit method).
