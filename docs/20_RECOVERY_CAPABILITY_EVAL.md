# Recovery capability evaluation: frozen ARC-Easy protocol

The four-arm recovery pilot measures held-out language loss, which does not by
itself establish useful question answering. This next probe scores the unchanged
Qwen3-0.6B donor and all four exact recovered step-2,048 checkpoints on **all 2,376
ARC-Easy test questions**. It is exploratory Tier-1 evaluation, not an untouched
release benchmark, instruction-following test, agent evaluation or architecture
adoption criterion. The reserved Tier-2 sets remain unexecuted.

## Source and preparation

Dataset: Allen Institute for AI, ARC, `allenai/ai2_arc`, `ARC-Easy`, test split,
revision `210d026faf9955653af8916fad021475a3f00453`. The pinned source metadata
records CC BY-SA 4.0; attribution: Peter Clark et al., *Think you have Solved
Question Answering? Try ARC, the AI2 Reasoning Challenge* (2018).
[Dataset](https://huggingface.co/datasets/allenai/ai2_arc),
[license](https://creativecommons.org/licenses/by-sa/4.0/).
Question/choice text matches all 2,376 rows of the previously downloaded lexical
overlap screen, in order. That screen's limits still apply: short and semantic
overlaps are not excluded, and donor-pretraining contamination is unknown.

`scripts/prepare_arc_recovery_eval.py` pins the source revision, retains every
question and every choice in source order, and records the labels, source-row
hashes and exact joint-tokenization hashes. The prepared items remain outside Git;
only the manifest and source-parity proof are committed.

The frozen items file has SHA256
`5011ec1abef00263b75101c484acdd4e8e2a1e51f8d6895059e87126ccdc6b3b`.
There are 2,365 four-choice, seven three-choice and four five-choice questions.
All candidates fit: the longest joint prompt/answer has **167 tokens**, below the
512-token training window. Across all candidates there are **297,436 input tokens**
including the final target token; the forward scorer consumes all but that final
token. No questions or choices are truncated, filtered or selected using model scores.

## Fixed scoring contract

The raw prompt is `Question: {question}\nAnswer:`. Each candidate continues it
with one space followed by the original choice text. This follows the prompt and
choice framing of the primary [lm-evaluation-harness ARC task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_easy.yaml).
Character-normalized accuracy divides summed log likelihood by the original
choice's character count, as in its [multiple-choice implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py).
This is our recorded raw-tokenizer protocol; scores are not asserted identical to
another harness version, chat formatting or published model-card results.

- Zero-shot, no chat template, BOS, EOS or thinking tokens.
- Joint tokenization must preserve the complete prompt prefix. A mismatch is an
  error, not a silently shifted boundary. All actual candidates pass this check.
- Score only candidate continuation tokens, including their initial space, with
  causal next-token alignment. Prompt loss never enters the choice score.
- Report both summed-likelihood accuracy and character-normalized accuracy.
  Ties select the first source choice and are counted explicitly. Chance is the
  mean reciprocal choice count, not an assumed universal 25%.
- Also report reference-answer nats/token and bits/byte, including the continuation
  space in both its loss and byte count. This is conditional answer likelihood,
  not full-document BPB and not interchangeable with the recovery loss.
- One candidate per forward pass, full FP32, PyTorch TF32 disabled, no cache reuse.
  Recovered models use their trained fixed depth. No increased-depth result is
  inferred from this evaluation.

The CLI requires the same frozen items, tokenizer and donor identity for every
arm. Recovered checkpoints undergo the strict exact-evaluated-checkpoint audit;
the trained window and precision policy must match. Reports retain per-item scores,
predictions, gold indices, all candidate token identities, checkpoint audits,
runtime identity, source/scorer hashes, elapsed time and peak CUDA allocation.
Question and answer texts are omitted from report artifacts.

## Validation and execution status

Twelve focused tests pass. An independent Markov probability oracle detects prompt
loss inclusion and incorrect causal alignment. Other cases cover different choice
counts, normalization reversing a ranking, visible ties, boundary retokenization,
forbidden truncation, corrupted token identities, invalid labels, nonfinite logits
and reduced precision.

An actual pinned Qwen CPU FP32 integration check scores one question's four
candidates and compares each sum against the original Hugging Face model's loss
with prompt labels masked. The maximum absolute difference is **2.86103e-6 nats**.
This validates the scoring path on real donor weights; it is not a benchmark score.
Evidence: [preparation manifest and integration oracle](experiments/2026-09-20-arc-recovery-protocol/manifest.json),
[oracle](experiments/2026-09-20-arc-recovery-protocol/donor-scoring-oracle.json),
[source parity](experiments/2026-09-20-arc-recovery-protocol/source-parity.json).

The related local suite passes **62 tests with five CUDA skips**. Full CI at
`6d36dce` passes **793 CPU tests with 15 CUDA-only skips** in 84.47 seconds.
On Colab, all twelve scorer tests also pass and preparation reproduces the local
manifest and items SHA256 exactly. The seven-file preparation archive is 6,481
bytes, SHA256 `6d21a697a57fe756492aaf62f5d1202780e6140f4ef396d098336fcaaee369a0`.
Its tests, manifest and exact executed queue source were independently checked
after download; [Colab evidence](experiments/2026-09-20-arc-recovery-protocol/colab/queue-at-ready.json).

The complete five-model GPU evaluation is **finished**. All five process handles
returned zero, after all four training and GPU cache suites passed. Evaluation used
an isolated worktree pinned to `6d36dce`, with a 30-minute limit per model; training
remained pinned to `467d7c8`. All five reports have identical A100/FP32 runtime,
tokenizer, questions, answer ordering and scoring policy. CPU oracle numbers are
not mixed into this comparison.

## Complete ARC-Easy results

| Model | Correct / 2,376 | Raw accuracy | Character-normalized accuracy | Gold-answer BPB |
|---|---:|---:|---:|---:|
| Unchanged donor | 1,446 | 60.8586% | 55.9764% | 0.943759 |
| Hybrid CE | 773 | 32.5337% | 31.9865% | 1.635445 |
| Attention CE | 811 | 34.1330% | 31.9865% | 1.621070 |
| Hybrid KL | 834 | 35.1010% | 32.5758% | 1.585024 |
| Attention KL | 825 | 34.7222% | 32.5758% | 1.550828 |

There are no tied predictions in either scoring convention. Uniform-choice chance
is 25.0161%. All five arms score the same 10,748 gold-answer tokens and 56,324
gold-answer bytes. These are exploratory raw-prompt scores, not published harness
scores or proof of an instruction-following assistant.

The complete 16-member source archive is 3,719,057 bytes with SHA256
`ce0e4f64b8c5385cdaaf1853ab5de9afac44adc2eb695cf8466e5465a0726915`.
Its source hash was read in Colab and all member identities checked before local
extraction. The [independent analysis](experiments/2026-09-20-arc-recovery-final/independent-analysis.py)
reads the large per-item reports from lossless gzip files, which reproduce the
exact original JSON bytes and hashes, and
checks each source-row hash, gold index, choice length and candidate-token identity
against the original frozen items kept outside Git. It recalculates all predictions,
ties and aggregates from the recorded choice losses, verifies the executed source
hashes against Git, and matches all four checkpoint audits to the earlier endpoint
audits. This is independent aggregation and identity verification, **not a second
GPU forward evaluation**. [Complete verification and paired results](experiments/2026-09-20-arc-recovery-final/independent-verification.json).

For raw accuracy, paired-item bootstrap intervals (10,000 resamples, seed 0) give:

| Contrast | Difference, percentage points | Descriptive 95% interval |
|---|---:|---:|
| Hybrid CE − attention CE | −1.5993 | [−3.0724, −0.2104] |
| Hybrid KL − attention KL | +0.3788 | [−1.0943, +1.8098] |
| Hybrid KL − hybrid CE | +2.5673 | [+1.0522, +4.0404] |
| Attention KL − attention CE | +0.5892 | [−0.8838, +2.0623] |

Intervals are unadjusted across eight contrasts per scoring convention and do not
cover training-seed or contamination uncertainty. They do not select a winning
architecture. In particular, the nine-question lead of hybrid KL over attention KL
is weak evidence; normalized accuracy ties exactly. The attention KL model still
has the better gold-answer likelihood, so lower language loss and higher raw
multiple-choice accuracy do not give the same ordering.

Every recovered arm is far behind the unchanged donor. Even hybrid KL loses
25.7576 percentage points (paired interval [−27.9882, −23.4428]). The short recovery
budget has not restored donor capabilities. **Do not adopt the conversion or scale
this recipe on the strength of its improved text loss.** The next development
diagnostics isolate how much damage weight sharing causes before more GPU recovery.

```bash
python scripts/prepare_arc_recovery_eval.py \
  --source <pinned-qwen-source> --out <new-frozen-arc-directory>
python scripts/eval_arc_recovery.py --arm donor --device cuda \
  --source <pinned-qwen-source> --items <frozen-arc-directory> --out <new-donor-report>
python scripts/eval_arc_recovery.py --arm recovered --device cuda \
  --run <completed-recovery-run> --step 2048 \
  --source <pinned-qwen-source> --items <frozen-arc-directory> --out <new-arm-report>
```
