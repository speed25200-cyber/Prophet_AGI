# R04: first matched real-text sessions

The shared and unshared R04 arms both completed 128 A100 training steps on real
text. The shared arm was then restarted from its Drive checkpoint and continued
for another 32 steps. This establishes stable training and persistence at the
intended parameter scale. **The shared model has not demonstrated a quality
advantage:** its held-out CE is slightly worse at the common step-128 boundary.

## Frozen experiment

Both arms use source revision `e5720d0b774977455b1920a6f67b5df078377f11`, seed 0,
batch 8, sequence 2048, chunked loss 512, full BPTT, and the audited FineWeb-Edu
pilot and 32,768-entry vocabulary. Both execute 20 blocks per token. The total
schedule is fixed at 4,096 steps; stopping a session does not shorten its anneal.
Muon peaks at 0.01 and AdamW at 0.0003, with 100 warm-up steps and clipping at 1.0.
Auxiliary heads and halting are disabled in both model configurations.

Hardware/software: A100-SXM4-40GB, PyTorch 2.11.0+cu128, CUDA 12.8, FLA 0.5.2,
Triton 3.6.0, `TRITON_F32_DEFAULT=tf32x3`. The eight-test GPU gate succeeded before
corpus preparation. Each arm additionally passed the strict reference-output gate:
maximum absolute error 0.000027031 for shared and 0.000434492 for unshared, below
0.002. Protocol files retain the exact configurations, fingerprints and versions.

## Common boundary: 128 updates, 2,097,152 tokens per arm

| Measurement | Shared, k=4 | Unshared, k=1 |
|---|---:|---:|
| Parameters | 374,688,512 | 920,675,072 |
| Initial held-out CE (nats/token) | 10.750804 | 10.765816 |
| Held-out CE at step 128 | 6.208639 | **6.202430** |
| Held-out bits/byte | 2.011050 | **2.009038** |
| Median logged step duration | 1.927 s | 2.108 s |
| Non-finite skipped steps | 0 | 0 |
| Checkpoint bytes, including optimizers | 3,233,902,603 | 7,605,974,203 |

Timing uses eight logged steps per arm, spaced 16 updates apart; it excludes
startup compilation, held-out evaluation and checkpoint I/O. These samples are
not an end-to-end runtime comparison or a statistically established speedup.

Evaluation covers all 376 held-out documents: 393,040 scored tokens and 1,750,592
payload bytes. Windows have context 2048 with one-token overlap, reset positions,
and no cross-document context. Each document's first token is unscored and EOS has
zero payload bytes.

The shared-minus-unshared CE difference is **+0.006209 nats/token**. A paired
whole-document bootstrap (10,000 draws, seed 0, corpus-weighted ratios, nearest-rank
percentile endpoints) gives **[+0.002213, +0.010399]** at 95%. This favors the
unshared model for these particular trained weights and documents. It conditions
on this training seed and **does not measure training-seed uncertainty**.

The shared arm stores 59.3% fewer parameters, but the project's R04 quality gate
remains open. This is only 128/4096 steps of one seed; the three-seed, full-budget
comparison and depth-generalization evaluation remain unexecuted. Continue the
predeclared comparison before adopting or killing recurrence. There are no
benchmark, reasoning, persistent-memory or AGI claims from this pilot.

## Actual checkpoint continuation

A separate process restored the shared model at step 128 and reached step 160
(2,621,440 cumulative tokens). Held-out CE then reached **6.080348** and BPB
**1.969494**. The step-128 evaluation is retained for the matched comparison.
All three retained checkpoints were checksum-verified, deserialized on CPU and
checked for finite model and optimizer tensors. Every audit reports zero skipped
non-finite steps. The latest shared checkpoint SHA256 is
`117c065f71a7594ee5ae69a997b6191383b6aae560bf73f68ab7448c7ad5f452`.

Weights stay under `MyDrive/Prophet_AGI/R04/{loop,plain}-seed0/checkpoints`.
The corpus and tokenizer are cached under `MyDrive/Prophet_AGI/R04/corpus-v1`.
The evidence archive was downloaded and verified locally (141,551 bytes, SHA256
`ba51950f7bcb7bf0080732d2860b5ca91eb641e63ddfab82d89730d91b4de238`).
Only numerical reports, hashes and logs are committed. Drive reported a 100 GiB
capacity and 35,362,328,576 free bytes after this pilot; retain storage headroom
when extending the remaining seeds. No checkpoints were removed.
`drive.flush_and_unmount()` completed successfully, and the A100 runtime was
released after the evidence was verified locally. The Colab notebook reports all
changes saved.

## Evidence and reproduction

The later [retrospective benchmark overlap screen](16_PILOT_BENCHMARK_OVERLAP.md)
records nine training documents with short lexical overlaps and incomplete
benchmark coverage. It does not modify this frozen corpus or establish that the
pilot is benchmark-clean.

- [Summary with paired uncertainty](experiments/2026-09-19-r04-stability-summary.json)
- [Shared reports and audits](experiments/2026-09-19-r04-stability/loop-seed0)
- [Unshared reports and audit](experiments/2026-09-19-r04-stability/plain-seed0)
- [Pinned Colab notebook](../notebooks/r04_pilot.ipynb)

```bash
python scripts/summarize_r04.py \
  --root docs/experiments/2026-09-19-r04-stability \
  --step 128 --seed 0 --out /tmp/r04-summary.json
```

For training continuation, check out the recorded experiment revision and repeat
`run_r04_pilot.py` with the same arm, seed and output folder. Session length may
change; the numerical protocol may not. Next sessions should progress toward the
fixed 4,096-step budget, with all stopped or failed runs retained in the record.
