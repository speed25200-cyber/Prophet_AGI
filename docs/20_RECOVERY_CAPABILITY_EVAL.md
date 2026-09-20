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

The complete five-model GPU evaluation has **not run yet**. It must wait until
the four recovery arms and their post-training GPU cache jobs release the GPU.
All models will use the same device/precision policy; CPU oracle numbers are not
mixed into that comparison. No answer-quality conclusion follows from preparation.

```bash
python scripts/prepare_arc_recovery_eval.py \
  --source <pinned-qwen-source> --out <new-frozen-arc-directory>
python scripts/eval_arc_recovery.py --arm donor --device cuda \
  --source <pinned-qwen-source> --items <frozen-arc-directory> --out <new-donor-report>
python scripts/eval_arc_recovery.py --arm recovered --device cuda \
  --run <completed-recovery-run> --step 2048 \
  --source <pinned-qwen-source> --items <frozen-arc-directory> --out <new-arm-report>
```
