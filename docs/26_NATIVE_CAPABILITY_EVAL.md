# Native R04 capability evaluation

Status: inputs and evaluator prepared; no native ARC-Easy score yet.
The learned-reinjection language-loss experiment continues under its existing
frozen protocol. This capability probe does not change that experiment's primary
criterion or choose an intermediate checkpoint.

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

The original overlap screen found no long lexical ARC matches but excluded short
items and did not test semantic overlap. This is not a benchmark-clean claim.
ARC tests basic multiple-choice science knowledge and inference, not open-ended
instruction following, persistent memory or autonomous work. Even a positive
result needs new seeds and genuinely held-out capability confirmation before
architectural adoption.

## Reproduction

```bash
python scripts/prepare_arc_native_eval.py --items ORIGINAL_ARC --tokenizer PILOT/tokenizer.json --out FRESH_NATIVE_ARC
python scripts/eval_arc_native.py --run FROZEN_RUN --step 512 --loop-k 4 --items FRESH_NATIVE_ARC --tokenizer PILOT/tokenizer.json --out FRESH_REPORT.json
```

The original parent uses `--step 4096`; repeat each model at `--loop-k 6`.
The launch queue must enforce the total deadline. Individual commands refuse
existing reports and verify inputs, checkpoint provenance and the planned shapes.
