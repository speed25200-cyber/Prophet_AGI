# Native R04 capability evaluation

Status: all six evaluations and their paired analysis are complete and verified
locally. No useful increase in accuracy from extra loops is established. The
preceding learned-reinjection language-loss screen fails all four frozen criteria;
this probe does not reverse that verdict. Its 28 targeted CPU/CUDA tests pass
without skips and prepared inputs reproduce the frozen manifest. Both CI runs
at evaluator revision `4632101` pass; the PR suite reports 873 passed and 22 skipped.

## Question and scope

Does a change in language prediction correspond to better answers, and do six
loops improve question answering relative to four? The existing Qwen recovery
evaluation cannot answer this for R04 because those models use a different
tokenizer and training trajectory.

This probe uses all 2,376 already frozen ARC-Easy test questions, an exploratory
Tier-1 task allowed by the project evaluation plan. It runs on three preselected
native checkpoints: the original step-4,096 parent and both completed step-512
reinjection arms, each at k4 and k6. All six runs are specified before any native
ARC score is viewed. The probe is informative even if the language-loss screen
fails, but cannot overturn that screen's preregistered outcome.

## Inputs and scoring

The [manifest](experiments/2026-09-20-arc-native-protocol/manifest.json) binds the
same pinned dataset revision, source order, prompts, choices, labels and question
hashes as the [donor/recovery evaluation](20_RECOVERY_CAPABILITY_EVAL.md).
Only tokenization changes. The original questions file has SHA256
`5011ec1abef00263b75101c484acdd4e8e2a1e51f8d6895059e87126ccdc6b3b`;
the native prepared file has SHA256
`6692f86fc175fe21ffd341309f5c3788164328b45214b1288cfa94d1d1f57ee8`.
Question text stays under ignored `data/`, outside Git.

There are 9,501 answer candidates and 305,468 candidate tokens. The longest
candidate has 167 tokens. Each prompt ends in `Answer:` and is jointly encoded
with a space plus the candidate answer; preparation rejects changed prompt-token
boundaries, truncation and empty answers. No chat template, BOS, EOS, few-shot
example or generated chain of thought is added.

Scoring sums FP32 causal negative log probabilities over answer tokens only.
Prompt and padding positions are excluded. Both raw and character-normalized
choice accuracy are reported, with first-source-choice tie handling, visible tie
counts, gold-answer CE and BPB. Comparisons use identical input and arithmetic
contracts; different tokenizers do not permit direct nats/token comparisons.

Candidates are grouped by length alone into four fixed right-padded shapes:

| Input length | Candidates | Batches of eight |
|---:|---:|---:|
| 32 | 6,219 | 778 |
| 64 | 2,878 | 360 |
| 128 | 392 | 49 |
| 256 | 12 | 2 |

This processes 437,760 padded positions rather than 2,433,024 for a single
256-token shape, an 82.0% reduction in processed positions, not a measured speedup.
Results are restored to original question/choice order. Every occupied shape is
checked against the single-candidate causal scorer on its first eight candidates
(32 candidates total), using fixed tolerances `atol=1e-4`, `rtol=1e-5`.
Any mismatch aborts the report. Local probability-oracle and learned-model tests
cover causal alignment, ignored positions, padding, source order and the final
partial batch; their success does not substitute for the full-size CUDA check.

## Execution limits and interpretation

The six inference-only runs use one A100 after the training queue finishes,
with a separate maximum of 1,800 seconds including setup and CUDA checks. This
is a bounded R11 evaluation allocation, not another training extension. Actual
Colab availability is checked before launch. Deadline failure means incomplete
evaluation; it must not become a score on a selected question prefix.

Model weights and prepared questions must be hashed before scoring. Only the
original published parent and the completed frozen `dce35ab` component runs are
accepted. CUDA uses strict deterministic settings, FP32 weights/logits, no
autocast and disabled PyTorch TF32 matmul/cuDNN; the Triton setting remains
`tf32x3` and is recorded. Each process records source, runtime, weight publication,
precision, timing and peak allocation.

The question-level report permits paired comparisons at k6 versus k4 and across
arms. Accuracy, normalized accuracy, answer BPB, disagreements and uncertainty
must all be reported. There is no success threshold chosen from these outcomes,
no early quality-based stop and no claim of cross-seed confidence.

The analysis preselects nine contrasts per metric: k6 minus k4 for each model;
learned minus fixed reinjection at each depth; and each adaptation minus the
original parent at each depth. It resamples 10,000 whole questions jointly
(PCG64 seed zero, linear quantiles). Gold CE/BPB use ratios of summed losses and
denominators, preserving answer length weighting. Accuracy differences use
percentage points, with both correctness disagreements and changed predictions
reported. The 95% intervals are descriptive and unadjusted for multiplicity.
The analyzer refuses partial sets, changed identities or arithmetic contracts,
and recomputes every ranking, tie count and aggregate before resampling.

The original overlap screen found no long lexical ARC matches but excluded short
items and did not test semantic overlap. This is not a benchmark-clean claim.
ARC tests basic multiple-choice science knowledge and inference, not open-ended
instruction following, persistent memory or autonomous work. Even a positive
result needs new seeds and genuinely held-out capability confirmation before
architectural adoption.

## Completed results

All six runs score every question and pass their 32-candidate batch-versus-single
oracle. The largest observed absolute oracle error is 0.000038147 nats, below the
absolute tolerance even without the relative allowance. No raw or normalized
choice ties occur. Uniform random choice has expected accuracy 25.0161% because
the candidate count is not exactly four for every question.

| Checkpoint | Loops | Correct / 2,376 | Raw accuracy | Character-normalized accuracy | Gold-answer BPB |
|---|---:|---:|---:|---:|---:|
| Original step 4,096 | 4 | 837 | 35.2273% | 33.5859% | 1.451176550 |
| Original step 4,096 | 6 | 764 | 32.1549% | 31.3131% | 1.589064729 |
| Fixed-sum adaptation | 4 | 840 | 35.3535% | 33.7542% | 1.454418138 |
| Fixed-sum adaptation | 6 | 824 | 34.6801% | 33.8384% | 1.463486293 |
| Learned-mix adaptation | 4 | 815 | 34.3013% | 33.6700% | 1.473074014 |
| Learned-mix adaptation | 6 | 801 | 33.7121% | 33.4175% | 1.499022519 |

Paired accuracy differences below are percentage points, left minus right.
Intervals are the preselected, **unadjusted descriptive 95% question intervals**;
they do not include training-seed variation or support architecture selection.

| Contrast | Raw difference [95% interval] | Normalized difference [95% interval] |
|---|---:|---:|
| Original k6 − k4 | −3.0724 [−4.3771, −1.7677] | −2.2727 [−3.4933, −1.0522] |
| Fixed sum k6 − k4 | −0.6734 [−1.3047, −0.0421] | +0.0842 [−0.4630, +0.6313] |
| Learned mix k6 − k4 | −0.5892 [−1.5572, +0.4209] | −0.2525 [−1.0943, +0.5471] |
| Learned − fixed, k4 | −1.0522 [−2.1475, +0.0842] | −0.0842 [−1.0522, +0.8838] |
| Learned − fixed, k6 | −0.9680 [−2.2306, +0.3367] | −0.4209 [−1.4731, +0.6313] |
| Fixed − original, k4 | +0.1263 [−0.9680, +1.2626] | +0.1684 [−0.7997, +1.1364] |
| Fixed − original, k6 | +2.5253 [+1.1364, +3.9983] | +2.5253 [+1.2205, +3.8721] |
| Learned − original, k4 | −0.9259 [−2.2306, +0.3788] | +0.0842 [−1.0522, +1.2205] |
| Learned − original, k6 | +1.5572 [+0.0421, +3.1987] | +2.1044 [+0.6734, +3.5354] |

The fixed-sum adaptation improves robustness at k6 relative to the original
fixed-depth checkpoint, but does not make k6 better than its own k4. Its k4 raw
advantage over the original is just three questions; the interval spans zero.
Learned reinjection shows no raw or normalized accuracy improvement over the
matched control, but the accuracy intervals also span zero: this task does not
establish a definite accuracy regression between those two trained components.
Its gold-answer loss is worse than the control at both depths; the paired BPB
deltas are +0.018656 [0.016058, 0.021228] at k4 and +0.035536
[0.032066, 0.039062] at k6.

All nine contrasts, prediction disagreements, correctness discordances and CE/BPB
intervals are retained in the [complete analysis](experiments/2026-09-20-arc-native-final/summary.json).
These results support neither useful additional inference computation nor adopting
the learned adapter. The language-loss failure remains binding. They also do not
measure instruction following, persistent learning, tool use or an assistant.

## Evidence and reproduction

The queue finishes in **772.77 seconds** under its 1,800-second limit, with all
eleven processes exiting zero. The [23-source export](experiments/2026-09-20-arc-native-final/export-manifest.json)
is 4,466,543 bytes, SHA256
`29babc8e07e4326c95d3858b8504c0331bf4f5d81bea7797ebb63b9d3f8c5816`.
The [local verifier](experiments/2026-09-20-arc-native-final/verify.py) checks every
original source hash, the fixed evaluation source, all question/token identities,
checkpoint publication identities and numerical contracts, then reproduces all
rankings, ties, aggregates and paired analyses. Its [receipt](experiments/2026-09-20-arc-native-final/verification.json)
does not claim a second GPU inference or tensor reload.

The training queue is saved before its ZIP and again after adding the export
receipt. Native evaluation copies that second save. The verifier checks identical
scientific fields and the exact export receipt, while separately validating the
0.080197-second increase in the recorded elapsed time; neither source is edited.

```bash
python scripts/prepare_arc_native_eval.py --items ORIGINAL_ARC --tokenizer PILOT/tokenizer.json --out FRESH_NATIVE_ARC
python scripts/eval_arc_native.py --run FROZEN_RUN --step 512 --loop-k 4 --items FRESH_NATIVE_ARC --tokenizer PILOT/tokenizer.json --out FRESH_REPORT.json
python scripts/summarize_arc_native.py --reports ALL_SIX_REPORTS --items FRESH_NATIVE_ARC --out FRESH_ANALYSIS.json
```

The original parent uses `--step 4096`; repeat each model at `--loop-k 6`.
The launch queue must enforce the total deadline. Individual commands refuse
existing reports and verify inputs, checkpoint provenance and the planned shapes.
The analyzer expects names `original4096-k4.json`, `fixed_sum-k4.json`,
`learned_mix-k4.json` and their k6 counterparts (optionally gzipped).
