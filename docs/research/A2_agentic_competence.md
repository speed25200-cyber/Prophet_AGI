# A2 — Agentic competence: where agents fail by cause, and what a 1B-active agent with a ledger can honestly do about it

> Track A2. Establishes what "agentic competence" consists of, decomposes the failures of
> 2024–2026 agents by mechanism, and derives an agent loop, a training recipe and an
> evaluation protocol for Prophet. Companion to `07_WALLS.md` (W1–W4) and `06_MEMORY.md`.
>
> **Provenance warning.** The egress proxy blocked `arxiv.org`, `alphaxiv.org`,
> `ar5iv.labs.arxiv.org`, `semanticscholar.org`, `huggingface.co`, `emergentmind.com`,
> `themoonlight.io`, `metr.org`, `sierra.ai`, `anthropic.com`, `gorilla.cs.berkeley.edu`,
> `snorkel.ai`, `swesmith.com`, `together.ai`, `andonlabs.com` and
> `syfi.cs.washington.edu`. Reachable: `github.com` READMEs and web-search result
> summaries. Every figure is therefore marked:
>
> | Mark | Meaning |
> |---|---|
> | **[V]** | read on a fetched page (GitHub README) |
> | **[S]** | taken from a web-search summary of the paper, not from the paper itself — treat as approximately right, verify before spending compute |
> | **[U]** | from memory; unverified |
> | **[C]** | computed here, from `prophet.budget` or from arithmetic shown inline |
>
> arXiv identifiers are given wherever known so that the verification pass can be done
> mechanically once the proxy allows it.

---

## 0. The thesis in four numbers

| Number | What it says | Source |
|---|---|---|
| **>60 % → <25 %** | GPT-4o on τ-bench retail: pass^1 to pass^8. An agent that "works 60 % of the time" reliably works on about a quarter of tasks. | τ-bench, 2406.12045 [S] |
| **719 min vs 70 min** | METR 50 %-horizon vs 80 %-horizon for the best frontier model (Jan 2026). Reliability costs a factor of **10** in horizon. Under independent per-step errors it would cost 3.1× (§4.1); the excess is self-conditioning. | METR TH1.1 [S]; ratio [C] |
| **294 : 1** | Input tokens to output tokens in real Claude Code / Codex sessions; 95.7 % of prompt tokens served from prefix cache; ~81 LLM steps per session. The agent workload is *reading*, not writing. | TraceLab, 2606.30560 [S] |
| **0 / 121** | Reflexion-style self-generated reflections that named the correct object in "frozen" ALFWorld environments. Unverified experience is worse than no experience. | Honest Lying, 2605.29463 [S] |

The consequence for Prophet: agentic competence at our scale is not a capability to be
pretrained; it is (i) a *loop* that removes the failure classes a harness can remove,
(ii) a *training recipe* that spends its few hours on consistency and recovery rather than
on coverage, and (iii) a *memory* that turns repeated task families from coin-flips into
near-certainties — the one axis on which a frozen model, however large, cannot follow.

---

## 1. Where agents fail, by cause

### 1.1 The evidence base

| Study | What was annotated | Scale |
|---|---|---|
| MAST — *Why Do Multi-Agent LLM Systems Fail?* (2503.13657) | 14 failure modes across 7 frameworks; κ = 0.88 between annotators | >1,600 traces [S] |
| *Understanding Code Agent Behaviour* (2511.00197, ICSE 2026) | OpenHands, SWE-agent, Prometheus trajectories on SWE-bench; failure labels | not stated in summary [S] |
| *Failure as a Process* (2607.09510) | Onset / evolution / point-of-no-return of failures; 7 models × 3 scaffolds on Terminal-Bench | 3,843 trajectories [S] |
| *How Coding Agents Fail Their Users* (2605.29442) | Developer–agent misalignment in real IDE/CLI sessions | 20,574 sessions, 1,639 repos [S] |
| AgentLens (2605.12925) | Process quality of *passing* SWE-bench trajectories | 2,614 trajectories, 8 backends [S] |
| HORIZON / *The Long-Horizon Task Mirage* (2604.11978) | Horizon-dependent failure composition, 4 domains; human–judge κ = 0.84 | 3,100+ trajectories [S] |
| OWL's GAIA error analysis (2505.23885) | Six error classes on GAIA failures | 52 failures [S] |
| OSWorld 2.0 (2606.29537) | Qualitative failure categories on 108 tasks, 150–500 step budgets | best agent ≈ 31 % [S] |
| *LLMs Get Lost in Multi-Turn Conversation* (2505.06120, ICLR 2026 oral) | Aptitude vs unreliability decomposition | 200k+ simulated conversations, 15 models [S] |
| *Beyond pass@1* (2603.29231) | Reliability decay vs task duration | 396 tasks, 10 models, 23,392 episodes [S] |
| τ-bench / τ²-bench (2406.12045, 2506.07982) | pass^k on retail / airline / telecom | 165 + tasks [V][S] |

### 1.2 The table

Shares are of *failures* unless stated. "Mechanism" is our reading of the root cause, not
the paper's phrasing. The last column names the Prophet mechanism that addresses the
cause — **direct**, **partial**, **harness** (solved outside the model), or **none**.

| # | Failure mode | Share, where measured | Root cause (mechanism) | Prophet mechanism |
|---|---|---|---|---|
| 1 | **Reasoning–action mismatch** — the action taken is not the one the reasoning selected | 13.2 % (MAST) [S]; only 0.5–4.8 % of *steps* in code agents, yet strongly predictive of failure (2511.00197) [S] | The action is sampled from free text conditioned on the thought; nothing structurally couples them. Small models have less spare capacity for the coupling. | **Partial.** Typed action grammar with constrained decoding *only inside the action span* (§7.2); confidence head scores the action span, not the thought. |
| 2 | **Step repetition / action looping** | 15.7 % (MAST) [S]; "pervasive action looping" is *the* SLM failure on SWE-bench (SWE-Protégé, 2602.22124) [S]; Vending-Bench "meltdown loops" from which models rarely recover (2502.15840) [S] | Self-conditioning: a repeated action in context raises the probability of repeating it again (induction); no state register says "already tried". | **Partial.** Bounded-state layers are a natural "already-done" register but do not guarantee it. The cheap fix is a harness loop detector on `hash(action, args)` that forces a `reflect` step, plus an RL penalty on repeats (SWE-Protégé) [S]. |
| 3 | **Premature termination / false completion** | 6.2 % + 12.4 % "unaware of termination conditions" (MAST) [S]; OSWorld 2.0: "submit as complete when they aren't" [S]; frontier CLI agents drop to 3/9 successes at 100-artifact goals (*Push Your Agent*, 2605.23574) [S] | `done` is an ordinary token trained by imitation of trajectories that *did* end; there is no calibrated estimate of "goal satisfied". | **Direct.** Confidence head as `P(goal satisfied)` gating the `done` action; below threshold → verification tool, not termination (§7.3). |
| 4 | **No / incorrect verification** | 8.2 % + 9.1 % = 17.3 % (MAST) [S]; 10.7 % of *passing* SWE-bench trajectories are "lucky passes" (regression cycles, blind retries, missing verification), 0.5–23.2 % by model (AgentLens) [S] | Verification costs tokens and yields no immediate reward; models cannot self-correct reasoning without an external signal (2310.01798 [U]). | **Harness + partial.** Verification is a tool (tests, diffs, rubric), not a thought. Confidence head decides *when* to pay for it. |
| 5 | **Context exhaustion / lost information** | Memory limits and forgetting = 27.5 % of design-level failures on long-horizon tasks (HORIZON) [S]; OSWorld 2.0 trajectories 30 → 300 steps, agents "lose information gathered early" [S]; NIAH accuracy −20–50 % from 10k to 100k+ tokens across 18 models (Chroma) [S]; compaction drops in-context constraints → violation 0 % → 30 % (up to 59 %) (Governance Decay, 2606.22528) [S] | Everything lives in an O(N) context whose salience decays with distance; compaction is a lossy, unverified rewrite. | **Partial, and the central bet.** Bounded state + pinned prefix + external notes (§4.4, §7.1). Bounded state is a summary, not a scratchpad (W1) — the notes file is the scratchpad. |
| 6 | **Inability to recover from a wrong action** | −39 % in multi-turn vs single-turn, of which unreliability +112 % vs aptitude −15 % (2505.06120) [S]; self-conditioning does not shrink with scale, thinking suppresses it (2509.09677) [S]; Agent-R recovery training +5.59 % and fewer loops (2501.11425) [S] | Errors in context are evidence for further errors; no mechanism rewinds the model's internal state to before the mistake. | **Direct.** Per-step snapshots of `SessionMemory` (12.6 MB at k = 4 for Prophet-main [C]) make rollback to the pre-error state O(1); Agent-R-style spliced recovery data at training (§8). |
| 7 | **Task-spec / constraint violation** | 11.8 % "disobey task spec" (MAST) [S]; >60 % of GPT-5 failures on SWE-agent are instruction-following (2511.00197) [S]; 91.49 % of visible misalignments need explicit user correction; constraint violations *grow* in share over time (2605.29442) [S] | Constraints are far from the decision point and are the first thing compaction drops. | **Partial.** Constraints live in the pinned, never-compacted prefix that the full-attention prelude/coda re-read every step; the recurrent core is re-injected with h₀ every iteration (D4). |
| 8 | **Fails to ask for clarification / acts prematurely** | 6.8 % (MAST) [S]; <50 % *consistent* pass on disambiguation tasks even for frontier reasoning models (CAR-bench, 2601.22027) [S]; OSWorld 2.0: "reluctant to ask" [S] | Training rewards plausible completion over honest uncertainty — the same guessing incentive R09 identified for facts. | **Direct.** Same confidence head, third threshold: `ask` when uncertainty is about the *user's* latent goal (§7.3). Trainable with a simulated user (SpeakRL 2512.13159, 2606.03135) [S]. |
| 9 | **Tool-call formatting / grounding brittleness** | Qwen3-1.7B: 8.38 % on BFCL multi-turn; Qwen3-4B: 16.88 % (prompt-based FC) [S]; "limited tool capability" = 32.69 % of GAIA failures (OWL) [S]; strict JSON mode degrades reasoning (2408.02442) [S] | Syntax is learned by imitation and competes with reasoning for the same tokens; small models lose that competition. | **Harness + MTP.** Grammar-constrained decoding on the action span only; free text elsewhere; MTP heads draft the boilerplate (1.07–1.46× decode, R08). |
| 10 | **Localisation / exploration failure** | Bug localisation is the primary bottleneck; failure trajectories are 12–82 % longer (2511.00197) [S]; "failure to reach all required pages" in ~90 % of WebArena failures [S] | Search in a large state space with no memory of the environment across episodes. | **Direct, conditional on W4.** Per-environment consolidated memory (repo map, tool quirks, working procedures). Skill/workflow memories give +24–54 % relative on web tasks (§5). |
| 11 | **Inconsistency** | pass^1 > 60 % → pass^8 < 25 % (τ-bench) [S]; SE reliability GDS 0.90 → 0.44 with task duration (2603.29231) [S] | Per-task success is a coin flip, not a global noise level (§3.1). | **Partial.** §3.3. |
| 12 | **Goal drift** | Frontier models inherit drift when conditioned on weaker agents' prefilled trajectories (2603.03258) [S]; Vending-Bench derailments do *not* correlate with context fill [S] | The goal has no privileged channel; whatever dominates the context dominates behaviour. | **Partial.** Goal in the pinned prefix + `inject_input_each_step`; not a guarantee. |
| 13 | **Memory poisoning / experience confabulation** | 0/121 correct self-reflections in frozen ALFWorld envs; programmatic extraction 0 → 86 % (2605.29463) [S]; error propagation and misaligned replay; selective add + delete +10 % absolute (2505.16067) [S]; repeated consolidation rises then falls *below* no-memory (2605.12978, via W3) | Self-diagnosis without ground truth; unverified writes persist. | **Direct.** Ledger is never written from a live episode; quarantine + verifier gate (06_MEMORY §2, §7.5). |
| 14 | **Inter-agent misalignment** | 36.9 % (MAST) [S] | Sub-agents lack each other's context. | **Avoided by construction.** Single-threaded loop (Cognition's "Don't build multi-agents" [S]). |

Three observations fall out of the table.

1. **Roughly half of measured failures are decision-quality failures at a few tokens**
   (rows 1, 3, 4, 8, 11): whether to act, stop, verify or ask. These are exactly the
   tokens a calibrated head and an input-dependent depth can afford to treat differently
   from the other 99 % of tokens. This is where our architecture is unusually positioned.
2. **Another third is context management** (rows 5, 7, 12): what stays readable, what
   is compressed, what is pinned. A bounded-state stack changes the economics of this
   (§6) but does not by itself solve it (§4).
3. **The remainder is environment knowledge** (rows 9, 10, 13): what the tools do,
   where things are in *this* repo, what worked last time. This is memory, and it is the
   only row where a small accumulating model can beat a large frozen one.

---

## 2. What competent agents do differently

### 2.1 The harness is half the score

- **The interface matters as much as the model.** SWE-agent's designed agent–computer
  interface reached 12.5 % on SWE-bench when shell-only agents "struggle to reliably take
  actions" (2405.15793) [S]. *Inside the Scaffold* (2604.03515) analyses 13 open
  scaffolds along 12 dimensions and finds context strategy "drives outcomes as much as
  the model" [S].
- **Single-threaded, continuous context.** Cognition: share full traces, not messages;
  default to one linear agent; add a dedicated *compressor* model for long tasks [S].
  Claude Code (as of mid-2025) spawns subtasks but does not parallelise work [S].
- **Structured hand-off instead of compaction.** Anthropic's long-running harness uses an
  initializer agent that writes a feature list and a progress file, git commits as
  checkpoints, and a fresh context rebuilt from the hand-off file for long jobs, because
  "compaction alone wasn't sufficient" [S]. Verbose tool output is truncated to a few
  summary lines; tools and instructions are revealed on demand [S].
- **Budget awareness.** Simply raising the tool-call budget does not raise performance;
  agents lack awareness of remaining budget (2511.17006) [S]. Adaptive per-step reasoning
  depth (CogRouter, ARES) is reported at 50–62 % fewer tokens at equal task performance
  [U — blog-level source].

### 2.2 The training recipes, with their cost

| System | Environment / data | Method | Model | Result | Compute |
|---|---|---|---|---|---|
| SWE-Gym (2412.21139) | 2.4k real tasks, 11 Python repos | SFT on **<500** successful trajectories; verifier for best-of-n | Qwen2.5-Coder-32B | +14 pts absolute on Verified; 32 % / 26 % Verified / Lite with verifier [V] | LoRA on **one H100**, 5 epochs [S] |
| SWE-smith (2504.21798 [U]) | 52k synthetic tasks, 26k trajectories | SFT on **5k** Claude-3.7 trajectories | Qwen2.5-Coder-32B → SWE-agent-LM-32B | **40.2 %** Verified [V] | not stated; SFT-scale |
| R2E-Gym (2504.07164) | procedural environments | SFT + hybrid verifiers | 32B | 34.4 % → 51 % with test-time scaling [U] | — |
| DeepSWE (Together/Agentica) | 4.5k R2E-Gym tasks | pure RL (GRPO variant) | Qwen3-32B | 42.2 % → 59 % with TTS [U] | **64 H100 × 6 days ≈ 9,200 H100-h** [S] |
| SWE-RL (2502.18449 [U]) | 11M PRs, rule-based similarity reward | RL, non-agentic | Llama-3-70B | 41.0 % Verified [U] | frontier-scale |
| Qwen3-Coder-Next (2603.00729) | **800k** agentic coding tasks, execution rewards | RL at scale | 80B-A3B, **Gated-DeltaNet hybrid** | **70.6 %** Verified with 3B active [S] | frontier-scale |
| Kimi K2 (2507.20534) | 3,000+ real MCP tools + 20,000 synthetic tools; rubric-graded simulated tasks | SFT + joint RL (RLVR + self-critique rubric) | 1T-A32B | — | frontier-scale |
| AFM / Chain-of-Agents (2508.13167) | multi-agent traces distilled | SFT → agentic RL | Qwen2.5 7B / 32B | 32B: GAIA 55.3, BrowseComp 11.1; RL adds +8.5 (7B) / +13.2 (32B) [S] | — |
| Nemotron-Terminal (2602.21193) | synthetic terminal tasks | SFT | 8B / 14B / 32B | 8B: 13.0 on Terminal-Bench 2 (5× base); 14B: 20.2 > GPT-OSS-120B 18.7 [S] | — |
| SWE-Protégé (2602.22124) | expert-augmented trajectories | SFT + RL penalising loops and useless expert calls | **7B** with sparse expert access | 17.0 % → **42.4 %** Verified; expert answers ~4 questions/task = 11 % of tokens; 8.2× cheaper than expert-only [S] | — |
| SynthAgent / *Mock Worlds* (2601.22511, ACL 2026) | teacher-synthesised tools, mock tool responses, LLM user simulator, rubric rewards | RL | small models | gains across 14 datasets [S] | — |
| RAGEN / StarPO-S (2504.20073) | Sokoban, FrozenLake, WebShop, … | multi-turn PPO/GRPO with trajectory filtering | 0.5–1.5B-class [U] | diagnoses the **Echo Trap**: entropy collapse to templated reasoning [V] | small |
| Training-Free GRPO (2510.08191) | few dozen samples | experience library as token prior, no weight update | DeepSeek-V3.1 | AIME24 +2.7, AIME25 +5.4, **$18** [S] | trivial |
| Darwin Gödel Machine (2505.22954) | agent edits its own scaffold code | evolutionary self-modification | frozen FM | SWE-bench 20 → 50 %, Polyglot 14.2 → 30.7 % [S] | **$22k / run, 2 weeks** [S] |
| SICA (2504.15228 [U]) | same idea | — | frozen FM | 17 → 53 % on a Verified subset [S] | — |

### 2.3 What the recipes have in common, and what is feasible on one A100

1. **Verifiable environments first.** Every recipe that moved a score has an executable
   environment with a reward that cannot be talked into: tests, database-state
   comparison (τ-bench), rubric over tool-call logs. Nothing in this table is trained on
   free-text preference alone.
2. **Expert-trajectory SFT captures most of the gain at negligible cost.** <500
   trajectories (SWE-Gym) or 5k (SWE-smith) move a 32B model by 14–32 points. At
   ~15k tokens per trajectory, 5k trajectories are 75M tokens: **~1–2 A100-hours** at
   the 40–80M tok/A100-h R10 measures for our sizes. Feasible, trivially.
3. **RL buys consistency and recovery, at 10–100× the SFT cost.** DeepSWE's 9,200
   H100-hours is 30× our entire budget. What *is* affordable is R10's box: LoRA r = 32,
   short per-turn completions, ~16–19 A100-h per 300 GRPO steps. Multi-turn agent RL
   fits that box only if per-turn outputs stay ≤2k tokens and episodes ≤20 turns.
4. **Rollout generation, not gradient steps, is the real cost — and on Colab it is a
   CPU cost.** A sandboxed test run takes 10–60 s; the GPU idles. Any agentic RL plan
   for us must be asynchronous (rollouts buffered ahead of updates) or the A100-hours
   are wasted waiting on Docker.
5. **The scaffold is a lever as large as the training.** DGM and SICA doubled scores
   with a *frozen* model by editing the scaffold. We should treat harness engineering as
   part of the deliverable, not as glue.
6. **Licensing is a live risk.** SWE-smith's 26k trajectories were generated by Claude
   3.7 Sonnet [V]; Anthropic's terms restrict using outputs to train competing models.
   xLAM/APIGen data is CC-BY-NC [U]. The `prophet.data.mixture` licence gate applies to
   agent trajectories exactly as it applies to text. Nemotron agentic datasets
   (CC-BY-4.0, generated by open models) and trajectories we generate ourselves with an
   Apache-2.0 teacher (Qwen3) are the safe path.

---

## 3. Consistency (pass^k)

### 3.1 The arithmetic of pass^k, and what the curve reveals

pass^k is the probability that *all* k i.i.d. attempts succeed: pass^k = E_task[p_i^k],
where p_i is the per-task success probability. If success were a uniform p = 0.6 on
every task, pass^8 would be 0.6⁸ = **1.7 %**. τ-bench observes ~25 % [S]. The only way to
reconcile the two is a **bimodal** p_i: a fraction f of tasks solved almost always and
the rest at roughly a coin flip. Solving f + (1−f)q = 0.60 and f + (1−f)q⁸ ≈ 0.25 gives
**f ≈ 0.24, q ≈ 0.47** [C]. In words: the agent reliably owns a quarter of the tasks and
flips a coin on the other three quarters. Inconsistency is per-task, not global noise.

*LLMs Get Lost* decomposes the same phenomenon differently and reaches the same place:
of the −39 % multi-turn drop, **unreliability rises 112 %** while aptitude falls only
15 % [S]. The model has not become less able; it has become a worse coin.

### 3.2 What drives the coin

| Driver | Evidence | Note for a 1B model |
|---|---|---|
| **Path dependence at a few decision tokens.** One early token flips the trajectory; errors then self-condition. | 2509.09677: self-conditioning is not reduced by scale, is reduced by thinking [S]; *Failure as a Process*: failures have an onset and a point of no return [S] | Small models have flatter action distributions at decision tokens; the flip probability is higher. |
| **Lucky passes.** | 10.7 % of passes are process-defective [S] | Inflate pass^1 without contributing to pass^k. |
| **Environment and user-simulator nondeterminism.** | τ-bench's simulated user; τ²-bench telecom pass^k falls faster than airline for the same pass^1 (claude-3.7: 49 % telecom) [S] | Not ours to fix, but our evaluation must hold the simulator fixed. |
| **Format lottery.** | Qwen3-1.7B at 8.38 % BFCL multi-turn [S] | Constrained decoding removes this term entirely. |
| **Duration.** | GDS drops 0.90 → 0.44 for SE tasks as duration grows (2603.29231) [S] | Compounding, §4.1. |

### 3.3 What reduces it — and the trade-off with R10

- **RLVR sharpens rather than expands.** RL on verifiable rewards raises pass@1 toward
  the base model's pass@k while leaving pass@k roughly unchanged (2504.13837 [U]); R10
  found on-policy distillation *preserves* pass@64 where RL narrows it. For chat and
  maths R10 correctly prefers distillation. **For agents, pass^k wants exactly the
  sharpening that RL provides**: we want the coin replaced by a decision. The recipe in
  §8 therefore puts a short RL stage *after* distillation, on agent tasks only, and
  measures pass^4, not pass@4.
- **Deterministic decisions, stochastic thoughts.** Sample in the `<think>` span (it is
  where exploration helps), decode the action span greedily under the grammar, and keep
  `eval_state_init="zeros"` (D5) so two runs on the same prefix are bit-identical. This
  removes every source of variance we own; what remains is the environment's.
- **Verify before irreversible actions.** τ-bench's hardest failures are irreversible
  database writes made on wrong arguments [S]. A confidence-gated verification step
  (re-read the policy, echo the arguments) before `write`-class tools converts a
  coin flip into a check.
- **Memory turns coin-flips into certainties on repeated families.** A consolidated,
  verified procedure for a task family moves p_i from ~0.5 to ~1 for that family; skill
  libraries as *executable APIs* are deterministic by construction (Voyager, SkillWeaver).
  This is the mechanism by which a small agent can have a *higher pass^k* than a larger
  frozen agent on a family it has seen — while having a lower pass^1 on the open set.
- **Consistency-aware confidence labels.** Run each training task n = 8 times; label a
  step's confidence target with the success rate of the sub-tree below it, not with the
  binary outcome of the one trajectory. The head then predicts "how much of a coin is
  this step", which is what the gate needs.

Honest bound: on open-set τ-bench-style tasks, a 1B-active model's per-task p_i will be
low *and* diffuse; pass^8 will be near zero. The metric we can move is pass^k on
families with consolidated memory, against the same model with `λ = 0` (memory off).

---

## 4. Long-horizon coherence

### 4.1 Compounding, and the 10× that independence does not explain

With independent per-step success p, the horizon at which success falls to s is
H(s) = ln s / ln p:

| p (per step) | H₅₀ | H₈₀ | H₅₀ / H₈₀ |
|---:|---:|---:|---:|
| 0.95 | 13.5 | 4.4 | 3.1 |
| 0.99 | 69 | 22 | 3.1 |
| 0.999 | 693 | 223 | 3.1 |
| 0.9999 | 6,931 | 2,231 | 3.1 |

[C]. Two things follow. First, *The Illusion of Diminishing Returns* (2509.09677) is
right that horizon is exponentially sensitive to per-step accuracy: +0.9 points of
per-step accuracy (0.99 → 0.999) buys **10× the horizon**. Second, the independence
ratio H₅₀/H₈₀ is fixed at ln 0.5 / ln 0.8 = 3.1 — yet METR measures 719 / 70 ≈ **10.3**
[S][C]. The observed reliability gap is 3.3× worse than independent errors predict. That
excess *is* self-conditioning: per-step accuracy falls as the context fills with the
agent's own mistakes. The same paper reports GPT-5-thinking executing >1,000 steps,
Claude-4-Sonnet 432, DeepSeek-V3 failing at two steps without CoT while R1 manages 200
[S] — and that thinking models are largely immune to self-conditioning.

For a 1B model the lesson is not "be more accurate per step" (we cannot buy that) but
**"do not carry your errors forward"**: rollback (§7.4), verification before
commitment, and a state representation that forgets verbatim mistakes.

### 4.2 Where context-window approaches break

- **Reading dominates.** 294 : 1 input-to-output; prefixes of 32k–256k tokens; a few
  hundred to a few thousand tokens appended per step [S]. The context is the tool
  outputs, and the tool outputs are mostly noise.
- **Rot is continuous, not a cliff.** −20–50 % on retrieval from 10k to 100k+ tokens on
  every one of 18 models; a single distractor lowers accuracy; shuffled haystacks are
  *easier* than coherent ones [S]. Long-horizon search agents "give up or answer
  prematurely" as context grows, and the effect is reproduced by pruning experiments
  (2606.29718) [S].
- **Compaction is lossy in the worst place.** Constraints that were obeyed 100 % while
  visible are violated 30 % of the time after summarisation (59 % for some models);
  when the constraint survives the summary, violation stays at 0 % [S]. Compaction also
  raises tool-call counts as the agent re-fetches what it lost [S].
- **Drift is not capacity.** Vending-Bench finds no correlation between derailment and
  context-window fill [S]. HORIZON attributes 27.5 % of design-level failures to memory
  limits and forgetting [S] — a large share, but not the majority. The rest is decision
  quality under a long, noisy conditioning.

### 4.3 Does bounded-state recurrence plus a persistent ledger help, or make drift worse?

**Evidence that bounded state is compatible with agentic competence.**

- Qwen3-Coder-Next — a **Gated-DeltaNet 3:1 hybrid**, i.e. our mixer stack — reaches
  70.6 % on SWE-bench Verified with 3B active parameters [S]. Nemotron 3 Nano (hybrid
  Mamba-Transformer, 30B-A3.5B) reports 38.76 SWE-bench (OpenHands), 49.04 τ²-bench
  average, 53.76 BFCL v4 [S]. Whatever bounded-state layers cost in exact recall, it
  does not prevent frontier-class agentic behaviour when hybridised with attention.
- MEM1 (2506.15841) trains an agent by RL to keep a **constant-size internal state**,
  discarding previous steps: on 16-objective multi-hop QA a 7B MEM1 improves performance
  3.5× while using 3.7× less memory than Qwen2.5-14B, and generalises beyond its
  training horizon [S]. A learned compressive state, trained end-to-end on the task, is
  not merely cheaper — it is *better* than accumulating context, because it cannot
  self-condition on verbatim errors it no longer holds.

**Evidence that it can make things worse.**

- A recurrent state is a summary (W1): it cannot re-read. Multi-needle recall collapses
  to 37.8 % in the best 2026 linear mixer (R02), and raising the depth dial *k* worsens
  the effective linear-to-attention ratio (W2). An agent that must recall the exact
  arguments of a tool call made 200 steps ago cannot rely on the state for them.
- Goal persistence through a purely recurrent channel decays geometrically with the
  gate α; in a KV-cache agent the goal is always one attention hop away. A pure-SSM
  agent would drift *more*, not less.

**Resolution: the hybrid resolves it, provided the split is designed rather than
accidental.**

| Role | Where it lives in Prophet | Why |
|---|---|---|
| Goal, constraints, tool schemas | **Pinned prefix**, read by the full-attention (NoPE) prelude/coda layers every step; never compacted; prefix-cached | Exact, re-readable, distance-independent (NoPE). Protects against rows 7 and 12 of §1. |
| Structured notes (plan, decisions, open questions, verified facts) | **Notes span**, ≤2k tokens, rewritten by the agent via a `note` tool; also pinned | The scratchpad W1 says a bounded loop lacks — provided externally, at O(1) per step. |
| Last W tool outputs, verbatim | **Attention window** (SWA 2048 + KV of recent full-attn tokens) | Exact recall where it is needed: the immediate past. |
| Everything older | **Recurrent state only**: KV of old tool outputs is *evicted* once they leave the window; their information survives as state + notes | Constant cost; verbatim errors are forgotten — self-conditioning is structurally damped. |
| Cross-episode knowledge | **Ledger** (product-key, closed-form write) + explicit skill files | §5. |

The eviction rule is the Prophet-native form of compaction: no summarisation pass, no
LLM call, no rewrite that can drop a constraint (constraints are pinned). It changes the
model's function, so it must be *trained*: sample random eviction of old tool-output KV
during agentic SFT/RL so the model learns to depend on state + notes for anything past
the window. This is an ablation, not a given — **[ABLATION A2-EVICT]**, §8.

### 4.4 Recovery: state snapshots instead of context surgery

Because tier-1 state is serialisable (`prophet/memory/session.py`), a snapshot at every
step costs a fixed 4 layers × k × 393,216 × 2 B = **3.1 MB × k** for Prophet-main
(12.6 MB at k = 4) [C] plus a pointer into the shared prefix KV. Rolling back to the
step before the first bad action — the intervention *Failure as a Process* says is what
"turns babysitting into a checkpoint you can place" [S] — is a file load, and the
resumed model has genuinely not seen the error. On a KV-only agent the same rollback is
possible by truncating the cache, but the *evidence* of the error remains in any summary
made since. Agent-R constructs recovery training data by exactly this splice (first error
step → adjacent correct sibling) and reports +5.59 % with fewer loops [S]; §8 reuses the
construction.

---

## 5. Learning from experience: the representation question

### 5.1 What has actually helped a future episode

| Representation | System | Measured gain | What it buys | What it costs |
|---|---|---|---|---|
| **Raw successful trajectory**, retrieved by similarity | ExpeL (2308.10144) | trajectory recall helps action-heavy ALFWorld (50 → 55 %), insights help knowledge-heavy HotpotQA [S] | exact reuse on near-duplicate tasks | memorisation; error propagation; stale on shift |
| **Verbal reflection on failure** | Reflexion (2303.11366) | +22 pts ALFWorld, +20 HotpotQA, HumanEval 91 % [S] | cheap in-episode retries | **confabulation**: 0/121 correct in frozen envs; fixed only by programmatic extraction (0 → 86 %) [S] |
| **Abstracted natural-language insight / rule** | ExpeL; ReasoningBank (2509.25140) | ReasoningBank: +8.3 pts absolute WebArena, −2.8 steps on SWE-bench-Verified, up to +20 % relative [S] | transfers across instances of a family | hard to audit; grows; contradictions accumulate |
| **Workflow / procedure with instance bindings abstracted out** | Agent Workflow Memory (2409.07429) | 35.6 % WebArena (SOTA at the time) [V]; +24.6 % / +51.1 % relative on WebArena / Mind2Web [U] | reusable sub-routines; fewer steps | needs induction step; brittle to UI change |
| **Executable skill / API, verified by execution** | Voyager (2305.16291 [U]); SkillWeaver (2504.07079) | Voyager: 3.3× unique items, tech-tree milestones up to 15.3× faster [V]; SkillWeaver +31.8 % relative WebArena, and *weaker agents gain +54.3 % from stronger agents' skills* [S] | deterministic, composable, transferable across models | environment-specific; needs a sandbox to verify |
| **Verified outcome / solution cache** | Dynamic Cheatsheet (2504.07952); Training-Free GRPO | Game of 24: 10 → 99 % once a Python solution was found and reused [S]; AIME +2.7 / +5.4 for $18 [S] | the largest gains per byte in the literature | only for tasks that recur nearly verbatim |
| **Weight update on own trajectories** | SWE-Gym self-improvement; DGM; SICA | Moatless 7B → 10 % Lite [V]; DGM 20 → 50 % at $22k/run [S] | permanent | forgetting (89 % for full FT, R03); expensive; unauditable |
| **Curated memory management** (selective add + delete) | 2505.16067 | +10 pts absolute over naive add-everything across 3 agents × 3 tasks [S] | — | the policy is the product |

### 5.2 The answer to the representation question

What helps a later episode is **(a) verified, (b) abstracted to the level of a
procedure with instance-specific bindings removed, (c) indexed by task-family features
rather than surface tokens, (d) small enough to audit, and (e) curated on write and on
delete**. Raw trajectories help only near-duplicates; free-form reflections are actively
harmful without a programmatic verifier; executable skills and verified solution caches
give the largest gains per byte and are the only representations that *raise pass^k*
(they remove the coin).

This is, point for point, W4's finding about our ledger: addressing by raw per-token
state — Jaccard 0.530 vs 0.493 between same-class and different-class instances, i.e.
chance — is the *raw trajectory* representation, and it memorises by construction.
Criterion (c) is the two-level contrastive key W4 specifies; criteria (a) and (e) are
the quarantine and the merge gate W3 specifies. Nothing in the agent literature
contradicts the memory design; it tells us which half of it to build first.

### 5.3 Mapping onto Prophet's two memories

The literature's winners are **explicit** (code, workflows, rules): verifiable,
composable, readable without decoding a latent. Our ledger is **implicit**: it stores
λ(h⁺ − h⁻), the effect of having had some context. Both are needed, and they have
different jobs:

| Memory | Stores | Written when | Read when | Gate |
|---|---|---|---|---|
| **Skill library** (files: procedure + code + the test that verified it + the environment fingerprint) | explicit procedures | after an episode whose outcome verifier passed *and* whose process was not a lucky pass (AgentLens-style check: verification happened before `done`) | retrieved by the harness at task start and when the loop detector fires; injected into the notes span | execution-verified |
| **Ledger** (`ProductKeyMemory`) | the consolidated effect of having the relevant skill/notes in context, on the *query* representation of the task family | offline, by `consolidate()` on (context = skill + notes, query = task prompt), after quarantine | implicitly, inside every forward pass | σ ≥ threshold on held-out instances of the family (W3), else the write is rolled back |

The ledger's job is not to store the trajectory. It is to make the model *behave as if it
had read its notes* on the next instance of the family — including when retrieval misses.
The skill library's job is to make the behaviour deterministic when retrieval hits.

Two warnings carry over unchanged. Consolidation "after every session" is the schedule
that produced rise-then-fall utility (2605.12978, via W3); consolidate per task family,
on a merge gate. And a `λ = 0` path — memory off, model unchanged — must remain reachable
in every evaluation (W4 D.5).

---

## 6. On-device agentic economics

### 6.1 What an agent step costs on our targets

All model numbers are `prophet.budget` outputs for `configs/prophet_main.json`
(3.83B total, 408M active per the revised estimator, int4_g64 weights, int8 KV) [C]. Decode figures are
bandwidth ceilings; real kernels reach 50–70 %. Prefill assumes 40 % of dense BF16
throughput, which short appends on a 0.4B-active model will not reach; halve it.

| Device | k | Context | Weights | KV + state | Decode ceiling | Prefill ceiling |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 1 | 32k | 1.92 GB | 70 MB | 6,182 tok/s | 34,600 tok/s |
| RTX 5090 | 4 | 32k | 1.92 GB | 70 MB | 3,536 tok/s | 25,900 tok/s |
| RTX 5090 | 8 | 32k | 1.92 GB | 70 MB | 2,251 tok/s | 19,400 tok/s |
| RTX 5090 | 4 | 128k | 1.92 GB | 262 MB | 2,530 tok/s | 10,400 tok/s |
| iPhone 17 Pro | 1 | 32k | 1.92 GB | 70 MB | 310 tok/s | 1,650 tok/s |
| iPhone 17 Pro | 4 | 32k | 1.92 GB | 70 MB | 178 tok/s | 1,230 tok/s |

Per-token KV+state cost: **2.2 KB/token at 32k, 2.1 KB at 128k** (int8) [C]. For
comparison, a dense Llama-3.2-1B (16 layers, 8 KV heads, d_head 64) costs 32 KB/token in
bf16 and Qwen3-1.7B (28 layers, 8 KV heads, d_head 128) ~114 KB/token [U for the
configs, C for the arithmetic]: 128k of context is **4.3 GB and ~15 GB** respectively,
against 262 MB for Prophet-main. The recurrent-state snapshot is 3.1 MB per loop
iteration (12.6 MB at k = 4) [C].

### 6.2 The shape of the loop, applied

TraceLab's real-session profile — ~81 steps, a 32k–256k prefix, a few hundred to a few
thousand tokens appended, ~200 decoded, 95.7 % prefix-cache hit [S] — gives the per-step
cost with prefix caching:

| | RTX 5090 (k = 4, 32k) | iPhone 17 Pro (k = 4, 32k) |
|---|---|---|
| Ingest 2k appended tokens | 2k / ~13k realistic ≈ **0.15 s** | 2k / ~600 ≈ **3.3 s** |
| Decode 200 tokens | 200 / ~2,000 ≈ **0.10 s** | 200 / ~100 ≈ **2.0 s** |
| Model time per step | ≈ 0.25 s | ≈ 5 s |
| Model time per 81-step session | ≈ 20 s | ≈ 7 min |
| Without prefix cache (re-prefill 64k each step) | 64k / 13k ≈ 5 s / step → 7 min / session | 64k / 600 ≈ 107 s / step → 2.4 h / session |

[C]. Three conclusions:

1. **Prefix caching is not an optimisation; it is the difference between feasible and
   not.** For the attention layers this is standard block-hashed KV reuse. For the
   recurrent layers it means checkpointing the state at step boundaries — which is
   exactly the `SessionMemory` snapshot of §4.4. Stateful serving that keeps the cache
   alive across tool calls is reported at 2.1× per turn on 6-turn and 4.2× on 35-turn
   workflows (2605.26289) [S]; on an M4 Pro, persisting Q4 KV to disk cuts time-to-first-
   token by up to 136× at 32k context, at −0.7 to +3.0 % perplexity (2603.04428) [S].
2. **Tool outputs must be truncated at the harness, as Anthropic's harness does.** A
   100k-token test log costs ~8 s of prefill on the 5090 and ~3 min on the phone.
3. **Tool-output ingestion should run at k = 1; decisions at k = 4–8.** FLOPs per
   token are 1.22 GFLOP at k = 1 vs 3.24 GFLOP at k = 4 (32k) [C] — ingesting the 294 :
   1 input stream shallowly and thinking deeply only on the ~200 decision tokens per
   step is a 2.7× saving on the dominant term, available *only* to an architecture whose
   depth is a per-token dial. This requires the recurrent state produced at k = 1 to be
   consumable at k = 4 — plausible given per-batch depth sampling, but currently untested
   at the per-segment level: **[ABLATION A2-MIXK]**.

### 6.3 Where a 32 GB GPU runs out first

- **Not memory, for Prophet.** Weights 1.9 GB + 262 MB per 128k session + 13 MB per
  state snapshot: ~100 concurrent 128k sessions fit in 30 GB. A dense 8B-class agent
  (~131 KB/token bf16 KV [C]) fills the same card with **one** 128k session plus its
  16 GB of weights.
- **Bandwidth, once sessions are batched.** Decode shares weight reads across sessions
  but each session's KV read is its own; at 128k × 2.1 KB = 262 MB per session per
  token, eight sessions read 2.1 GB per decoded token — more than the weights. The KV
  term, small per session, dominates at batch. Compact the window (§4.3) and the term
  shrinks to the pinned prefix + window.
- **The depth dial.** k = 8 halves decode throughput relative to k = 2. Spent
  uniformly it is a tax; spent by the halting head on decision tokens it is the
  cheapest reasoning we have.
- **On the phone, prefill.** 1.2k tok/s ceiling means every 1k tokens of tool output
  is a second of latency; the ingestion budget, not the decode budget, sets the
  interaction rhythm. The notes span (≤2k) and a hard tool-output cap (≤4k) are product
  requirements there.
- **MTP heads earn more on agents than on chat.** JSON, paths and repeated identifiers
  are highly predictable; self-speculative acceptance rates on action spans should
  exceed R08's 1.07–1.46× average. Measure it: **[A2-MTP]**.

---

## 7. Recommendation for Prophet: the agentic architecture

### 7.1 The loop

```
                 ┌─────────────── pinned prefix (attention, prefix-cached, never compacted) ───────────────┐
                 │ system · goal · constraints · tool schemas │ NOTES span (≤2k, rewritten by `note` tool) │
                 └────────────────────────────────────────────┴────────────────────────────────────────────┘
                                                      │
   tool output ──▶ ingest @ k=1 ─▶ recurrent state ───┼──▶ window (last W tool I/O, exact) ──▶ evict older KV
                                                      │
                                       ┌──────────────▼──────────────┐
                                       │  think span   (free text,   │  sampled, budgeted, halting sets k per token
                                       │  optional, ≤ T_think tokens)│
                                       ├─────────────────────────────┤
                                       │  action span  (grammar-     │  greedy, constrained; k = k_decide
                                       │  constrained: tool, args)   │
                                       └──────────────┬──────────────┘
                                                      │ confidence head reads here
                    ┌───────────────┬─────────────────┼──────────────────┬──────────────────┐
                    ▼               ▼                 ▼                  ▼                  ▼
              c ≥ τ_act        c < τ_act &        action == done     action == done     ambiguity
              → execute        irreversible       & c ≥ τ_done       & c < τ_done       about user
                               → verify first     → stop             → run verifier     → ask
```

**Structured vs free text.** The think span is free text (it is where exploration and
self-conditioning immunity come from, §4.1). The action span is a typed grammar —
`tool(name, args)` with JSON-schema-constrained arguments, plus the reserved actions
`note`, `verify`, `ask`, `done`, `rollback(step)`. Constrained decoding applies *only*
to the action span, so the reasoning penalty of strict-format modes (2408.02442) [S] is
not paid on the thought. Tool outputs are wrapped as a distinct modality id (the R12
`modality_ids` hook exists): the model can learn that these tokens are *observations*,
not its own prior outputs — the cheapest available defence against self-conditioning
on tool noise.

### 7.2 Where memory is read and written

| Moment | Read | Write |
|---|---|---|
| Task start | Harness retrieves ≤3 skills by embedding of (goal, environment fingerprint) into NOTES; ledger reads implicitly | none |
| Every step | Recurrent state (tier 1) carries the folded past; ledger is read inside the forward pass | `note` tool rewrites NOTES; state snapshot saved |
| Loop detector fires | Second skill retrieval keyed on the *stuck state* | none |
| Episode end | — | Trajectory + outcome + process labels go to **quarantine**; nothing touches the ledger |
| Consolidation (offline, per family, on a merge gate) | — | Skill library (procedure + test) if verified; `consolidate()` writes λ(h⁺−h⁻) to the ledger; rolled back unless σ_holdout ≥ σ_before + 0.10 (W3) |

### 7.3 Where the confidence head gates

One head, three thresholds, three questions. The head's target is the *sub-tree success
rate* below the step (§3.3), so it answers "how much of a coin is this?":

| Gate | Question | Below threshold |
|---|---|---|
| `τ_act` | Is the action I am about to take (arguments included) correct? Applied only to irreversible tool classes (write, submit, purchase, delete). | Emit `verify` first: re-read policy/spec, echo the arguments, run a dry-run tool if one exists. |
| `τ_done` | Is the goal satisfied? | Run the verifier tool (tests, checklist against NOTES); `done` is refused until the verifier or the head agrees. |
| `τ_ask` | Is my uncertainty about the *user's* intent rather than about the world? | Emit `ask`. Never a bare abstention — always a question plus the action that would resolve it (R09's product rule). |

Thresholds are set per tool class from the cost matrix (a false `verify` costs one step;
a false irreversible action costs the episode), exactly as R09 sets abstention
thresholds per benchmark. On 0/1-scored short tasks the gates should be *off*, for the
reason R09 gives.

### 7.4 How halting is used

`halt_threshold` is passed per segment, not per request: tool-output ingestion at
`loop_k=1`; think span under the learned halting distribution (`halting="ponder"`, cap
`train_loop_max`); action span at `k_decide` (4 on the 5090, 2 on the phone). Expected
depth is logged per step: an agent that ponders deeply on every `ls` is a bug the
halting prior is supposed to prevent (07_WALLS A.3).

### 7.5 Pseudocode

```python
# prophet/agent/loop.py — sketch, not code. Names match prophet.modeling.model /
# prophet.memory.* where they exist.

from prophet.modeling.model import ProphetModel, ProphetCache
from prophet.memory.session import extract_session, restore_session
from prophet.memory.ledger import ProductKeyMemory
from prophet.memory.consolidate import consolidate, Episode

IRREVERSIBLE = {"write_file", "git_commit", "submit", "purchase", "delete", "send"}

class AgentLoop:
    def __init__(self, model: ProphetModel, tools, skills, cfg):
        self.m, self.tools, self.skills, self.cfg = model, tools, skills, cfg
        self.snapshots = []            # (step, SessionMemory, cache_position)
        self.seen_actions = {}         # hash(tool, args) -> count

    # ---- one episode -------------------------------------------------------------
    def run(self, goal, env_fingerprint, max_steps=200):
        cache = ProphetCache()
        pinned = self.render_pinned(goal, self.tools.schemas)        # never compacted
        notes = self.skills.retrieve(goal, env_fingerprint, k=3)     # explicit memory in
        self.m(pinned + notes, cache=cache, loop_k=1)                # prefix, cached
        window = []                                                  # last W tool I/O

        for step in range(max_steps):
            self.snapshots.append((step, extract_session(self.m, cache), cache.position))

            # 1. think (free text, sampled, learned halting sets depth per token)
            think = self.decode(cache, span="think", loop_k=None,
                                halt_threshold=self.cfg.halt_threshold,
                                max_tokens=self.cfg.think_budget, greedy=False)

            # 2. act (grammar-constrained, greedy, fixed deep k)
            action, out = self.decode(cache, span="action", loop_k=self.cfg.k_decide,
                                      grammar=self.tools.grammar, greedy=True)
            conf = sigmoid(out.confidence[:, -1])                    # P(sub-tree succeeds)

            # 3. gates — the only place "agentic" decisions are made
            if action.name == "done":
                if conf < self.cfg.tau_done and not self.verifier_passed(cache):
                    action = self.tools.verify_action(notes)          # refuse to stop
                else:
                    return self.finish(cache, notes, verified=True)
            elif action.name in IRREVERSIBLE and conf < self.cfg.tau_act:
                action = self.tools.dry_run_or_echo(action)          # verify before commit
            elif action.name == "ask" or (conf < self.cfg.tau_ask and action.needs_user):
                return self.ask_user(action)                         # question + proposed action

            # 4. loop detector (harness, not model)
            h = hash((action.name, action.canonical_args()))
            self.seen_actions[h] = self.seen_actions.get(h, 0) + 1
            if self.seen_actions[h] > self.cfg.max_repeats:
                notes = self.skills.retrieve_stuck(notes, action)     # second retrieval
                action = self.tools.reflect_action()

            # 5. execute; ingest observation shallowly, as a distinct modality
            obs = self.tools.run(action, cap_tokens=self.cfg.tool_cap)  # truncate at harness
            if action.name == "rollback":
                step_to, mem, pos = self.snapshots[action.step]
                restore_session(self.m, cache, mem); cache.truncate(pos)
                continue
            self.m(obs.tokens, cache=cache, loop_k=1,
                   modality_ids=obs.modality_ids)                     # k=1 ingestion
            window.append(obs)
            if len(window) > self.cfg.window_steps:                   # state-carried compaction
                old = window.pop(0)
                cache.evict_attention_kv(old.span)                    # state keeps the gist
            if action.name == "note":
                notes = action.args["text"]; cache.rewrite_pinned_notes(notes)

        return self.finish(cache, notes, verified=False)

    # ---- after the episode: nothing is written live -------------------------------
    def close(self, trajectory, outcome):
        quarantine.add(trajectory, outcome,
                       process_ok=trajectory.verified_before_done and not trajectory.lucky)

def consolidate_family(model, ledger: ProductKeyMemory, family, holdout, skills):
    """Offline. Runs per task family, never per session (W3)."""
    verified = [t for t in quarantine.get(family) if t.outcome.passed and t.process_ok]
    if not verified:
        return
    proc = skills.induce(verified)                     # workflow with bindings abstracted out
    if skills.execute_test(proc):                      # executable skill: the proven representation
        skills.add(proc, family, env_fingerprint=family.env)
    episodes = [Episode(context=tokens(proc.text + t.notes), query=tokens(t.goal))
                for t in verified]
    before = skill_ratio(model, ledger, holdout)       # σ on held-out instances, context cleared
    snapshot = ledger.state_dict()
    consolidate(model, ledger, episodes, lam=1.0, passes=3,
                replay=quarantine.replay(), replay_ratio=0.25)
    if skill_ratio(model, ledger, holdout) < before + 0.10 or ledger_recall_drift() > 0.05:
        ledger.load_state_dict(snapshot)               # merge gate: the write is refused
```

What is deliberately *not* in the loop: a planner module, a critic model, a sub-agent,
a summariser. Each is a place where the literature finds a new failure class (MAST's
inter-agent 36.9 %; governance decay from summarisation). The loop has one model, one
grammar, one head, one state, and one offline consolidation step.

---

## 8. Training recipe on one A100

Hours are for Prophet-main (~0.4B active; 40–80M tokens per A100-hour per R10's
throughput table, scaled). The 300-hour plan in `05_ROADMAP.md` is already
oversubscribed 1.7×; the recipe below is presented **as a gated request**, sized so that
its first gate is cheap and its expensive stages die if the gate fails.

| Stage | What | Data / environment | Reward or target | A100-h |
|---|---|---|---|---:|
| **G-A (gate)** | Can a 0.4B-active model act at all after SFT? | SFT on 5k licence-clean trajectories (own generation with an Apache-2.0 teacher on SWE-smith-style synthetic bugs in Apache-2.0 repos + SynthAgent-style mock-tool tasks), converted to the §7.1 grammar | Pass if ≥ 5 % on a 100-task held-out SWE-smith slice **and** ≥ 30 % on 200 mock-tool tasks **and** ≥ 25 % BFCL-v3 multi-turn (vs 8.4 % for Qwen3-1.7B [S]). Fail → agentic RL is unfunded; ship the harness + memory on the SFT model. | **3** |
| S1 | Environments | ~2k synthetic SWE tasks (bug injection + tests), ~2k mock-tool tasks with a Qwen3-8B user simulator and rubrics, ~500 CLI tasks (CLI-Gym style) — all CPU, all sandboxed, generated offline | — | (CPU) |
| S2 | Rejection-sampling fine-tuning, 2 rounds | 8 rollouts × 2k tasks per round; keep passes that are not lucky (verification present before `done`) | outcome verifier + process check | 2 × (12 rollout + 2 SFT) = **28** |
| S3 | Recovery data (Agent-R splice) | From S2 failures: first-error step (verifier diff) → adjacent passing sibling; also random KV-eviction augmentation for **[A2-EVICT]** | imitation | **5** |
| S4 | Agentic GRPO, LoRA r = 32 | P = 32, G = 8, ≤ 20 turns, ≤ 2k tokens/turn, asynchronous rollouts (StarPO-S filtering; SNR-adaptive prompt selection) | verifiable outcome + grammar validity + loop penalty + wasted-verify penalty; **no** LLM-judge reward | **20** (≈ 300 steps at R10's 16–19 h) |
| S5 | Confidence head | Labels from S2/S4 rollouts: sub-tree success rate per step (n = 8) — no new rollouts | BCE; release gate AUROC ≥ 0.75 on held-out steps (R09's kill criterion at 0.70) | **3** |
| S6 | Halting / mixed-k | Per-segment k sampling (ingest k = 1, decide k ∈ [2, 8]) folded into S2–S4 | ponder loss already in `Trainer` | 0 |
| S7 | Memory ablation E2-agentic | 40 task families × 20 instances; consolidate 15, hold out 5; arms: λ = 0, skills only, ledger only, both; report pass^4 and σ | W3 merge gate | **8** |
| | **Total** | | | **≈ 67** (3 h before any commitment) |

Notes on feasibility and honesty:

- **Rollouts dominate and they are CPU-bound.** 16k rollouts × ~10k tokens = 160M
  generated tokens; batched decode on the A100 at ~5k tok/s aggregate is ~9 h, but each
  rollout also waits on a sandbox (10–60 s per test run). Without asynchronous rollout
  buffering the same work costs 3–5× the wall-clock and the same GPU-hours idle.
  Colab's CPU allowance, not the A100, is the binding constraint for S2–S4.
- **The Echo Trap is real at our size.** RAGEN documents entropy collapse to templated
  reasoning under vanilla multi-turn GRPO [V]; the trajectory filtering and reward-
  variance prompt selection of StarPO-S are not optional for a 0.4B-active policy.
- **Where the hours come from.** S0–S4 *are* post-training; the honest accounting is
  that agentic SFT/RFT displaces part of R10's 45 h (it replaces some chat SFT, not adds
  to it), and S4 + S7 (28 h) compete with the unfunded R04 depth ablations for any
  reserve that survives preemption. This is a decision for the project owner, and the
  G-A gate exists so that it can be taken on a 3-hour result.
- **What is not fundable.** Long-CoT agent RL at 8–32k completions (R10: 240–300
  A100-h per 300 steps); DeepSWE-scale RL (9,200 GPU-h); any recipe requiring a critic
  or a second model resident during training.

---

## 9. Evaluation

### 9.1 The harness

| Tier | Benchmark | What it measures for us | Cost per run |
|---|---|---|---|
| **1 — deterministic, local, minutes** | BFCL v3 multi-turn subset (200 human-curated trajectories) [S] | grammar validity, state tracking; the format lottery must read 0 % | minutes |
| 1 | Our mock-tool families (from S1), held-out instances | pass^k, k ∈ {1, 2, 4, 8}, with `λ = 0` and with memory; the *learning curve* below | minutes |
| **2 — hours** | τ²-bench retail / airline / telecom (2506.07982) with an open user simulator (Qwen3-8B) held fixed | pass^k on open-set tool-agent-user tasks; expect near zero pass^8 at our size — report it | hours |
| 2 | SWE-bench Verified (100-task slice) + SWE-smith held-out repos (contamination-resistant) | resolve rate; AgentLens-style process quality (lucky-pass rate) | hours, sandboxed |
| 2 | Terminal-Bench 2.0 (89 tasks) [S] | CLI competence; reference points: Nemotron-Terminal-8B 13.0, 14B 20.2 [S] | hours |
| **3 — days** | *Beyond pass@1* reliability protocol (2603.29231): RDC / GDS / meltdown onset over duration buckets [S] | horizon reliability; the H₅₀/H₈₀ ratio of §4.1 | days |
| 3 | SWE-Bench-CL (2507.00014): 8 chronological sequences, 273 tasks, forgetting / transfer metrics [S] | the only public benchmark shaped like our thesis | days |

Report everything with the `λ = 0` arm (memory off), the skill-library-only arm, and
the full arm. Report pass^k, not pass@k, everywhere except where the benchmark defines
otherwise. Decontaminate: SWE-bench repositories are in every pretraining corpus and in
the donor's; the SWE-smith held-out repos and our own synthetic families are the
defensible numbers.

### 9.2 The learning-curve benchmark — the number that is ours

For each of N task families (≥ 40; mock-tool, synthetic SWE, CLI), plot **pass^4 on
held-out instances against the number of consolidated episodes** (0, 1, 2, 4, 8, 16),
context cleared between episodes. A frozen model — ours at λ = 0, and a 10× larger
frozen baseline (Qwen3-8B / Nemotron 3 Nano) with the same harness and a plain retrieval
of past trajectories at equal context budget — is a flat line. The claim Prophet can
make, if it is true, is that the curve rises and **crosses the larger frozen model's
line at some n½**. Report n½ per family, σ (W3), and the fraction of families where the
crossing happens. Report also the families where it never happens; that is the boundary
of the claim.

### 9.3 What "beating the competition" would honestly mean at our size

| Claim | Honest? | Why |
|---|---|---|
| Beat Qwen3-Coder-Next (70.6 % Verified, 800k RL tasks) on SWE-bench | **No.** | 30–100× the compute per active parameter; open-set coverage is bought with tokens we do not have. |
| Beat same-size frozen models (Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B) on pass^4 for BFCL-multi-turn / mock-tool / Terminal-Bench, same harness | **Yes, and required.** | Reference points: 8.4 % / 16.9 % BFCL multi-turn for Qwen3-1.7B / 4B [S]. This is the G-A gate scaled up. |
| Beat a 10× larger frozen model **on task families after n consolidated episodes**, with context cleared | **Yes — this is the thesis, and it is falsifiable by §9.2.** | Skill transfer from stronger to weaker agents gives +54 % relative (SkillWeaver); verified solution caches give 10 → 99 % (Dynamic Cheatsheet). The frozen model cannot climb. |
| Higher reliability ratio (H₈₀ / H₅₀) than larger models at *equal* horizon | Plausible, unproven | Rollback + verification gates target self-conditioning directly; measure with the *Beyond pass@1* protocol. |
| Lower cost per successful episode on a 5090 / phone at 128k context | **Yes, by arithmetic** (§6) | 2.1 KB/token vs 32–131 KB/token; the dense competitor does not fit the session. |
| GAIA level 3, BrowseComp, open-world OSWorld | **No.** | World knowledge and perception; out of scope at ~2 bits/parameter. |

---

## 10. References

Marks: [V] page fetched, [S] web-search summary, [U] memory / unverified.

**Failure taxonomies and evaluations**
- Cemri, Pan, Yang et al. *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657 [S]
- *Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories.* arXiv:2511.00197 (ICSE 2026) [S]
- *Failure as a Process: An Anatomy of CLI Coding Agent Trajectories.* arXiv:2607.09510 [S]
- *How Coding Agents Fail Their Users: … 20,574 Real-World Sessions.* arXiv:2605.29442 [S]
- Sahoo et al. *AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation.* arXiv:2605.12925 [S]
- *The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break* (HORIZON). arXiv:2604.11978 [S]
- Yao et al. *τ-bench.* arXiv:2406.12045 [S]; GitHub README [V]. Barres et al. *τ²-bench.* arXiv:2506.07982 [S]
- Laban et al. *LLMs Get Lost in Multi-Turn Conversation.* arXiv:2505.06120 [S]
- *Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents.* arXiv:2603.29231 [S]
- Sinha et al. *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs.* arXiv:2509.09677 [S]
- Kwa et al. *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499; METR Time Horizon 1.1 (Jan 2026) [S]
- Backlund, Petersson. *Vending-Bench.* arXiv:2502.15840 [S]
- *Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals.* arXiv:2603.03258 [S]
- *Push Your Agent: Measuring and Enforcing Quantitative Goal Persistence.* arXiv:2605.23574 [S]
- *CAR-bench: Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty.* arXiv:2601.22027 [S]
- Scale AI. *SWE-Bench Pro.* arXiv:2509.16941 [S]. *OSWorld 2.0.* arXiv:2606.29537 [S]. *Terminal-Bench.* arXiv:2601.11868 [S]
- OWL (GAIA error analysis). arXiv:2505.23885 [S]. GAIA. arXiv:2311.12983 [U]
- BFCL v3/v4 (Gorilla). Leaderboard page blocked; small-model figures via *TinyLLM*, arXiv:2511.22138 [S]
- Chroma. *Context Rot* (technical report, 2025) [S]. *Diagnosing and Mitigating Context Rot in Long-horizon Search.* arXiv:2606.29718 [S]
- *Governance Decay: How Context Compaction Silently Erases Safety Constraints.* arXiv:2606.22528 [S]
- Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* arXiv:2310.01798 [U]
- Tam et al. *Let Me Speak Freely? Format Restrictions and LLM Performance.* arXiv:2408.02442 [S]

**Agents, harnesses, training**
- Yang et al. *SWE-agent: Agent-Computer Interfaces.* arXiv:2405.15793 [S]
- *Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures.* arXiv:2604.03515 [S]
- Anthropic Engineering. *Effective harnesses for long-running agents*; *Effective context engineering for AI agents* [S]. Cognition. *Don't Build Multi-Agents* [S]
- Pan et al. *SWE-Gym.* arXiv:2412.21139 [S]; GitHub [V]. Yang et al. *SWE-smith.* arXiv:2504.21798 [U]; GitHub [V]
- Jain et al. *R2E-Gym.* arXiv:2504.07164 [S]. DeepSWE (Together / Agentica) [U]. Wei et al. *SWE-RL.* arXiv:2502.18449 [U]
- Qwen. *Qwen3-Coder-Next Technical Report.* arXiv:2603.00729 [S]. NVIDIA. *Nemotron 3 Nano.* arXiv:2512.20848 [S]. *On Data Engineering for Scaling LLM Terminal Capabilities* (Nemotron-Terminal). arXiv:2602.21193 [S]
- Kimi Team. *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534 [S]
- Li et al. *Chain-of-Agents / Agent Foundation Models.* arXiv:2508.13167 [S]
- Kon et al. *SWE-Protégé.* arXiv:2602.22124 [S]
- *Mock Worlds, Real Skills* (SynthAgent). arXiv:2601.22511 [S]
- Wang et al. *RAGEN / StarPO.* arXiv:2504.20073; GitHub [V]
- Yuan et al. *Agent-R.* arXiv:2501.11425 [S]; GitHub [V]
- Cai et al. *Training-Free GRPO.* arXiv:2510.08191 [S]
- Zhang et al. *Darwin Gödel Machine.* arXiv:2505.22954 [S]. Robeyns et al. *A Self-Improving Coding Agent* (SICA). arXiv:2504.15228 [U]
- *Budget-Aware Tool Use Enables Effective Agent Scaling.* arXiv:2511.17006 [S]
- *Learning When to Act or Refuse* (MOSAIC). arXiv:2603.03205 [S]. *SpeakRL.* arXiv:2512.13159 [S]. *Uncertainty-Aware Clarification with Information Gain.* arXiv:2606.03135 [S]
- Yue et al. *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* arXiv:2504.13837 [U]
- Hammer (function masking for on-device tool calling). arXiv:2410.04587 [S]. xLAM. arXiv:2409.03215 [S]. APIGen-MT. arXiv:2504.03601 [S]

**Learning from experience**
- Shinn et al. *Reflexion.* arXiv:2303.11366 [S]. Zhao et al. *ExpeL.* arXiv:2308.10144 [S]
- Wang et al. *Voyager.* arXiv:2305.16291 [U]; GitHub [V]
- Wang, Neubig et al. *Agent Workflow Memory.* arXiv:2409.07429 [V for id]; GitHub [V]
- Zheng et al. *SkillWeaver.* arXiv:2504.07079 [S]
- Ouyang et al. *ReasoningBank.* arXiv:2509.25140 [S]
- Suzgun et al. *Dynamic Cheatsheet.* arXiv:2504.07952 [S]
- *How Memory Management Impacts LLM Agents: Experience-Following Behavior.* arXiv:2505.16067 (ACL 2026) [S]
- *Honest Lying: Understanding Memory Confabulation in Reflexive Agents.* arXiv:2605.29463 [S]
- Chen et al. *SWE-Exp.* arXiv:2507.23361 [S]. *SWE-Bench-CL.* arXiv:2507.00014 [S]
- Zhou et al. *MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents.* arXiv:2506.15841 [S]
- Repeated memory consolidation degrades utility. arXiv:2605.12978 (as cited by W3)

**Serving and on-device**
- Zhu et al. *TraceLab: Characterizing Coding Agent Workloads for LLM Serving.* arXiv:2606.30560 [S]
- Norgren. *Stateful Inference for Low-Latency Multi-Agent Tool Calling.* arXiv:2605.26289 [S]
- Shkolnikov. *Agent Memory Below the Prompt: Persistent Q4 KV Cache … on Edge Devices.* arXiv:2603.04428 [S]
- Zheng et al. *SGLang / RadixAttention.* arXiv:2312.07104 [S]

**Internal**
- `docs/07_WALLS.md`, `docs/06_MEMORY.md`, `docs/research/W1–W4`, `docs/research/R09`, `docs/research/R10`; `prophet.budget` outputs reproduced in §6 [C].
