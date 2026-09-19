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
