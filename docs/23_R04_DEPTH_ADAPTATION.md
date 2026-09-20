# 23 — Does training across depths make additional loops useful?

**Status: final frozen-model depth sweep completed; matched adaptation driver
implemented, six CPU/CUDA tests and both real-shape memory gates passed. The first
training segment was stopped at step 54 after real-shape preflights showed different
gradients. Exact full-size reproducibility remains unresolved. No new architecture
or training policy is adopted.**

## Why this experiment

The donor-sharing initialization series ends with a negative
[activation-calibration result](22_ACTIVATION_WEIGHTED_SHARING.md). The native R04
model preserves much more of its unshared control's language quality at lower
parameter count, but useful extra inference depth remains unproven. Its original
training used exactly four loops, whereas the intended runtime depth dial needs
its own training and evaluation.

Huginn samples recurrence counts during training and backpropagates through a
bounded trailing set of iterations. Its published heavy-tailed distribution,
normalization, model scale and data budget differ from this pilot. The simple
bounded uniform distribution below isolates a training-depth intervention; it is
not a reproduction of Huginn or evidence of its reasoning results.
[Primary source, §3.2–3.3](https://arxiv.org/html/2502.05171v2).

## Frozen final checkpoint

The new sweep uses the step-4,096 shared checkpoint, exact original model source,
runtime, tokenizer and all 376 development documents. It checks the published
checkpoint hash and reproduces its saved k=4 CE exactly before testing other
depths. All five passes use 393,040 scored tokens and 1,750,592 payload bytes.

| Loops at inference | Nats/token | Bits/byte |
|---:|---:|---:|
| 1 | 4.806536396 | 1.556892354 |
| 2 | 4.229619707 | 1.370022411 |
| 4, trained depth | 3.855913359 | 1.248974632 |
| 6 | 4.424840339 | 1.433256617 |
| 8 | 4.479179774 | 1.450857784 |

More loops still hurt at the final checkpoint. This is language-loss sensitivity
of fixed-depth-trained weights, not a reasoning evaluation or a test of
variable-depth training. The queue completed in 158.82 seconds under its ten-minute
limit. Timing includes verification, loading and compilation and is not a decode
latency benchmark. Training weights and optimizer states were not modified.

The [export manifest](experiments/2026-09-20-r04-final-depth/export-manifest.json)
binds every source file; the 128,783-byte ZIP has SHA256
`24cfe9aa69413292f0934e7c08e3730cf608cca9498ffb4d0193a27b7ce58e38`.
The per-document report is stored as lossless gzip. An
[independent local recomputation](experiments/2026-09-20-r04-final-depth/verification.json)
checks all identities, denominators and aggregate scores. Every k=4 document
loss equals the previously published final evaluation exactly. Eight loops
increase BPB by **16.1639%**; the paired CE difference is +0.6232664 nats/token,
with unadjusted 95% document-bootstrap interval [0.5912429, 0.6583953]
(10,000 draws, seed zero). This conditions on the checkpoint and excludes
training-seed uncertainty; it is not a second GPU forward run.

## Matched adaptation protocol, fixed before training

The [generated configurations, budget/search/plan outputs and protocol](experiments/2026-09-20-r04-depth-adaptation-plan/protocol.json)
define two separate warm starts from the same final R04 weights and loader cursor:

| Setting | Control | Depth-trained arm |
|---|---|---|
| Training loops per microbatch | Always 4 | Uniform integer 2 through 6 |
| Expected loops | 4 | 4 |
| Backpropagation | All visited loops | All visited loops |
| Architecture / unique parameters | Same / 374,689,648 | Same / 374,689,648 |
| Additional steps | 512 | 512 |
| Batch / sequence length | 8 / 2,048 | 8 / 2,048 |
| Additional input tokens | 8,388,608 | 8,388,608 |

The existing loop sampling option performs the intervention. Both configurations
set the backpropagation cap to six so that no visited loop is detached. The
random recurrent state initialization, prelude, GDN core, coda and losses stay the
same. No auxiliary heads, learned stopping or memory component is introduced.
Expected depth is matched; actual sampled depth counts and elapsed time must be
reported rather than claiming exact matched FLOPs.

Each arm starts a new optimizer and 512-step WSD schedule: Muon 0.001, AdamW
0.00003, weight decay 0.1, gradient clipping 1.0, existing minimum 100-step warmup
and 18% decay. This is a warm start, **not** exact continuation of the previous
optimizer. The original loader cursor continues in both arms. Within each arm,
interruptions must restore model, optimizer, loader and CPU/CUDA RNG exactly.
Both warm starts reset RNG to the parent's seed (zero), after model construction;
they do not inherit the old optimizer's RNG position. Their own subsequent
checkpoints preserve RNG without resetting it on resume.

The cumulative per-model input budget is 75,497,472 tokens, or 3.8877 corpus
passes, below the existing four-pass limit. The corpus is the original R04 pilot,
with its [known benchmark overlaps](16_PILOT_BENCHMARK_OVERLAP.md); it does not
become benchmark-clean through this experiment. No benchmark selects the recipe.

## Runtime gates and decision rule

Before either long segment, require matching source/data/checkpoint identities,
unchanged k=4 baseline predictions, the recorded R04 CUDA runtime, actual CUDA
interruption/resume equivalence with sampled depth, finite gradients and a measured
k=6 training peak below the A100 40 GB capacity. BF16 uses the original R04
policy; the separate donor GDN BF16 gradient failure is not waived or generalized
away. A failure stops the experiment for inspection.

The planned pair targets one GPU-hour, with a hard 5,400-second queue ceiling.
At the earlier measured approximately two seconds per fixed-k4 step, training
alone would cost about 2,048 seconds across both arms. Variable depth, startup,
validation and checkpoints require fresh measurement. The budget calculator's
memory/speed estimates and design-search output are not measurements of this
warm-start run or of deployment on consumer devices.

At step 512 evaluate both models at k=1,2,4,6,8 on the full, unchanged development
set. The primary seed-0 screening condition is all of:

- Variable-depth k=6 BPB at most 0.995 times its own k=4 BPB, with the upper end
  of a paired-document 95% bootstrap interval for CE(k6)-CE(k4) below zero.
- Variable-depth k=4 BPB no more than 1.01 times the matched fixed-depth control's
  k=4 BPB.
- Variable-depth k=6 improves over the matched fixed-depth control at k=6.

Use 10,000 paired-document bootstrap draws with seed zero. No recipe search or
early stopping based on intermediate development quality is included. A passing
seed-0 result warrants additional training seeds and capability tests, not adoption.
Failure does not justify simply increasing inference loops on the existing model.

## Parameter accounting correction

Historical R04 tables used an older budget estimate of 374,688,512 shared and
920,675,072 unshared parameters. Budget correction `b015121` included GDN gate
biases and output normalization that were already in the models. Exact meta-device
construction now matches the corrected budget at **374,689,648** and **920,679,616**.
No weights or topology change follows; the rounded 59.3% reduction remains the
same. Generated adaptation configurations use the corrected count.

## Driver and operational checks

`scripts/adapt_r04_depth.py` binds the run to the published parent checkpoint,
generated config, plan hash, clean Git revision, driver hash, original CUDA
runtime and corpus fingerprint. It hashes and restricted-loads the exact published
slot, retains the original packing cursor, and verifies k=4 validation before
training. It records actual depth per step in the checkpoint and training log.
A changed arm or contract cannot silently resume an existing output directory.

The preflight mode uses a separate disposable copy, performs three real-shape
optimizer updates at k=6 with full gradients, and checks finite gradients/weights
and peak allocated memory below 90% of the actual GPU capacity. It never modifies
the original checkpoint or any training arm. Its peak excludes non-PyTorch
allocations; the remaining capacity is a reserve, not a measured allocation.

CPU CLI tests use a miniature model and real local text. They verify parent
weights and loader transfer with an empty new optimizer, exact uninterrupted
versus interrupted weights/optimizer/RNG/loader/depth history, changed-arm
rejection, idempotent completion and continuation of an interrupted final depth
evaluation without further training. Both arms also have actual CUDA cases that
must pass without skips before the real preflight. Separate tests reject changed
parent publication/bytes and prove the policy leaves model topology unchanged.
The local targeted result is four passed and two CUDA skips. The actual A100 run
at `4f5c566` subsequently passed all six tests without skips, including both CUDA
restart cases, in 15.70 seconds. These use miniature fixtures, not the 375M model.
The full local suite at that revision passed 807 tests with 18 skips; both CI runs
were green. Passing miniature restart tests does not establish full-size restart
equivalence.

```bash
# Use a fresh output directory for every arm and disposable preflight.
python scripts/adapt_r04_depth.py --parent-run <final-loop-run> \
  --corpus <original-pilot> --arm uniform2to6 --mode preflight --out <preflight>
# Do not start further segments until the real-shape reproducibility issue below is resolved.
# Once all gates, including real-shape restart equivalence, pass:
python scripts/adapt_r04_depth.py --parent-run <final-loop-run> \
  --corpus <original-pilot> --arm uniform2to6 --out <new-training-arm> \
  --max-session-steps 64
# Repeat the same command/contract to continue toward 512 total updates.
```

Periodic and end-of-session checkpoints use the existing two-slot atomic writer.
The driver stops on the first nonfinite update and preserves the last completed
checkpoint. Full final evaluation is written incrementally with `complete=false`
until every planned depth is scored; an interrupted evaluation resumes from the
same weights. Running a completed command again leaves its report unchanged.
The queue, rather than the individual CLI, enforces the paired 5,400-second ceiling
and the ordering of the external CUDA tests/preflights before training.

## First launch stopped for numerical investigation

The [exported evidence](experiments/2026-09-20-r04-depth-adaptation-stop/export-manifest.json)
records both disposable three-update preflights at batch 8, sequence 2,048 and
explicit k=6, with full backpropagation. Both used **13,444,140,032 bytes** of peak
PyTorch allocation (12.52 GiB), below 90% of the A100's 42,405,855,232-byte capacity.
All reported losses, gradients and resulting weights were finite. Per-step times
were approximately 2.74–3.06 seconds, including these preflight updates only.

However, the two processes did not report identical gradients despite the shared
parent, loader cursor, seed and explicit depth. The first loss was exactly
4.108038902282715 in both, but gradient norms were 5.207744598388672 and
5.207338333129883. Subsequent losses also differed. This identifies a missing
full-size reproducibility gate; it does not isolate an operation or demonstrate
that checkpoint restoration is the cause.

The fixed-k4 training process received SIGTERM, completed its current update,
saved step **54** and evaluated k4, then exited with code zero. All six queue
process handles were confirmed terminal. The queue stopped when its expected
step-64 report was absent; neither variable-depth training nor either long segment
started. The retained partial report has 884,736 new input tokens, 54 recorded
k4 depths, no nonfinite skips and CE 3.8533956392204405. This partial CE is not the
planned matched comparison and does not select a recipe. The recorded checkpoint
SHA256 is `a8b86f376632269f6fe51294022619aff737faf8ea7ae267b9fe7fe142742296`;
its metadata was exported, not its 3.23 GB tensor file.

The 124,475-byte evidence ZIP has SHA256
`7f4355f7f0e4cd998e1f8d8bb21449c905946168dddb8459279a7f4b9013aa86`.
The [local verifier](experiments/2026-09-20-r04-depth-adaptation-stop/verify.py)
checks all 22 original file hashes, exact source/protocol identities, all 376
baseline document losses against the frozen original, XML test results, partial
training counters and independently recomputed evaluation aggregates. Large JSON
files are losslessly gzipped. This audit does not reload checkpoint tensors or
repeat GPU computation.

Next require separate-process equality of actual-size inputs, RNG, gradients and
weights, followed by interrupted-versus-continuous actual-size training. A
disposable strict-PyTorch-determinism preflight finished three updates, but a single
run cannot establish reproducibility. The subsequent instrumented two-process
probe failed at launch before producing a comparison; its cause remains to inspect.
Neither observation authorizes resuming the long experiment. Any arithmetic-policy
change must be recorded as a new run contract with a fresh baseline and both arms.
