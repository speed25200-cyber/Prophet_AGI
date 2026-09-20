# Real-text pilot

The pilot now has a downloaded corpus and a trained 32,768-entry tokenizer. It is a
bounded English web sample for checking the training path and preparing comparisons.
It does not replace the project's final multilingual/code mixture.

## Source and separation

Source: [HuggingFaceFW/FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
`sample-10BT`, `train`, immutable revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The Hub reported ungated access and `odc-by`.
The published schema uses `int_score`, not `edu_score`; all three FineWeb recipe
entries have been corrected accordingly. Other source-specific filters still need
their own schema verification before the full mixture is used.

The preparation takes a capped prefix, filters short/oversized documents and scores
below 3, then removes normalized exact duplicates. A seeded hash of URL host/path
assigns validation membership, keeping versions of the same URL together. Before
writing training shards, it rejects any document sharing a normalized 13-word span
with validation. This is conservative against boilerplate but does not detect all
paraphrases. External benchmark sets have **not** been decontaminated for this pilot;
it cannot support downstream benchmark claims.

Observed preparation:

- 20,204 source rows inspected; 20,000 unique documents selected.
- 203 quality/length rejections and one normalized duplicate.
- 55 training documents removed for held-out span overlap.
- 19,569 training documents, 86,282,982 UTF-8 text bytes.
- 376 validation documents, 1,751,847 UTF-8 text bytes.

Corpus contents remain under ignored `data/fineweb-pilot-v1/`. The committed
[manifest](experiments/2026-09-19-pilot-corpus.json) contains provenance, filters,
counts and SHA256s, not corpus text.

## Tokenizer

The tokenizer learned 32,256 merges on the first 10,000 **training** documents
(43,095,993 UTF-8 bytes), leaving the existing 512 byte/reserved IDs unchanged.
Training took 55.11 seconds on this local run. The SHA256 is
`4d212345c25569add0ed6b2a397767062b9205f77b0fabc8f9a794107dffb1c2`.
The [metadata](experiments/2026-09-19-pilot-tokenizer.json) binds it to the training
shard fingerprint and exact implementation revision.

The BPE trainer now uses incremental pair counts and a reverse index of affected
words. Its frequency ordering, byte-order tie-breaks and non-overlapping merges are
unchanged. Sixteen randomized/multilingual cases compare the complete merge list
with an independent full-recount oracle, including reversed input order. The
existing byte fallback, digit, whitespace and control-token invariants also pass.

## Reproduce

```bash
pip install -e '.[train,dev]'
python scripts/prepare_pilot.py --out data/fineweb-pilot-v1
python scripts/train_tokenizer.py --data-root data/fineweb-pilot-v1/train \
  --out data/fineweb-pilot-v1/tokenizer.json --vocab-size 32768 --max-docs 10000
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/pilot_smoke.py
```

Preparation refuses to replace an existing corpus. Interrupted preparations leave a
`.building` directory rather than publishing a partial manifest. Use a fresh output
directory when repeating. The smoke verifies corpus and tokenizer hashes, trains a
small model, saves a checkpoint halfway, recreates the trainer, resumes, and scores
the same eight held-out document prefixes before and after. Its short run is a
pipeline check, not the required multi-seed R04 language-quality comparison.

Next quality work needs a declared token budget within the corpus's repetition cap,
matched shared/unshared training, multiple seeds, fixed held-out evaluation, and
persistent checkpoints for interrupted Colab sessions. Architecture adoption remains
open until those comparisons are measured.

## Failed smoke attempt retained in the record

The first smoke at revision `dbed68d` used width 256. It reached step 32 on real text
(LM loss 8.4345), then failed its resume assertion: the new smoke script had disabled
periodic checkpoints and had not explicitly saved at its interruption boundary.
`Trainer.train()` does not implicitly write a final checkpoint. No trained weights
from that attempt were retained, and it supplied no final held-out result.

The script now explicitly saves at the interruption and final boundaries. A CLI test
executes training, reload and report generation, then verifies the final saved step
and token count. The pipeline smoke uses width 64 to keep CPU validation inexpensive;
the measured R04 configurations are unchanged.

The corrected real-text smoke at `b9fe83c` completed 64 steps, with a checkpoint
written at step 32, restored into a new trainer, and a final checkpoint at step 64.
It consumed 16,384 tokens with zero non-finite skipped steps. Held-out cross entropy
on the fixed 1,016 scored tokens fell from 10.41136 to 8.60454 nats/token. The model
has 2,468,360 parameters; this only validates data ingestion, optimization, scoring
and persistence. See [the report](experiments/2026-09-19-real-pilot-smoke.json).

## Sized full comparison, not yet completed

An exact encoding pass counted **19,419,613 training tokens** and **393,416 validation
tokens**, including EOS per document, and verified lossless decoding of every
document. [Token counts](experiments/2026-09-19-pilot-token-counts.json) include the
tokenizer and shard fingerprints. A 100M-token run would exceed the project's
four-epoch limit on this corpus.

The [pilot recipe](../configs/data_mixture_pilot.yaml) instead sets 67,108,864 tokens,
or 4,096 updates at batch 8 / sequence 2048 and 3.456 nominal corpus repetitions.
For both R04 configurations on seeds 0, 1 and 2, the measured synthetic throughput
projects about **13.69 A100 hours total**, excluding data loading, evaluation,
checkpoint I/O and compilation. It is a planning estimate, not an executed result
or a promise that the full language model is trained in that time.

Before a long Colab run, establish persistent checkpoint storage and retain identical
source, tokenizer, learning rates, token budget and loss settings across resumes.
Run a short learning-rate/stability check first. A command for one planned arm is:

```bash
python scripts/train.py --config configs/prophet_r04_loop.json \
  --mixture configs/data_mixture_pilot.yaml --data-root data/fineweb-pilot-v1/train \
  --tokenizer data/fineweb-pilot-v1/tokenizer.json --tokens 67108864 \
  --batch-size 8 --seq-len 2048 --grad-accum 1 --loss-chunk-tokens 512 \
  --muon-lr 0.01 --adamw-lr 0.0003 --seed 0 --session-minutes 45 \
  --checkpoint-every 512 --checkpoint-dir checkpoints/r04-loop-seed0 --device cuda
```

The learning rates above are pilot starting values, not tuned results. Repeat the
same settings for the unshared arm and the other seeds. Use fixed held-out
document-level scores and record all failed or stopped runs. This early-learning
comparison alone cannot establish behavior after large-scale pretraining or the
project’s persistent-memory and agent capabilities.

## Full-document evaluation and resumable R04 sessions

The step-64 smoke checkpoint was additionally scored on **all 376 held-out
 documents**, with a context of 256 and one-token overlap between windows. Each
 target is scored exactly once, excluding each document's first token; EOS has zero
 payload bytes. Across 393,040 scored tokens and 1,750,592 payload bytes, CE falls
 from **10.408930 to 8.657214 nats/token** and BPB from **3.371572 to 2.804171**.
 These are whole-corpus measurements of the same 2.47M-parameter pipeline smoke,
 not results for either R04 arm. The [full report](experiments/2026-09-19-pilot-full-validation.json)
 retains per-document hashes and scores. Window positions reset; the evaluator
 preserves model mode and training RNG. It does not measure cross-window memory.

`run_r04_pilot.py` now defines the reproducible paired experiment and supersedes
 the manual training command above. It verifies the audited shard checksums and
 vocabulary, freezes the source revision and numerical environment, rejects
 incompatible resumes, and evaluates all held-out documents before training and
 after each bounded session. Its default session covers 128 steps of the fixed
 4,096-step schedule. Learning rates remain unvalidated starting values.

```bash
TRITON_F32_DEFAULT=tf32x3 python scripts/run_r04_pilot.py \
  --variant loop --seed 0 --out /content/drive/MyDrive/Prophet_AGI/R04/loop-seed0
```

Repeat the identical command to resume; use `plain` and seeds 0, 1, 2 for the other
 arms. A normal stop publishes a checksum-protected checkpoint. An exception during
 a batch leaves the previous checkpoint intact. Training history without a surviving
 checkpoint is rejected. Source changes require a separately recorded experiment.
 Trainer checkpoint format 3 freezes learning rates, schedule and objective settings;
 older checkpoint formats warn that their numerical settings cannot be verified.

Tokenizer JSON file bytes can differ between Windows and Linux. The runner checks
 the locally recorded file hash, the train-only source fingerprint, and a canonical
 UTF-8 JSON vocabulary hash (`sort_keys=True`, `ensure_ascii=True`):
 `7d8d36adb2b2bf9dac7a6060294641ea16c59ce20506e2294c556c7abd99537b`.
 This keeps cross-platform serialization differences from changing the experiment.

The first paired A100 sessions and an actual checkpoint continuation have now
completed. See [the R04 pilot results](14_R04_PILOT.md): both arms reached step 128,
and the shared arm resumed to step 160. The full 4,096-step, three-seed comparison
remains pending; the shared model is slightly worse at the measured common boundary.
