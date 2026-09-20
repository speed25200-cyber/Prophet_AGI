# 23 — Does training across depths make additional loops useful?

**Status: both matched 512-step adaptations and all final depth evaluations are
complete and independently verified. The primary seed-zero screen fails: variable
depth greatly reduces sensitivity to loop count, but six loops remain worse than
four. All strict numerical, memory and actual-size restart gates pass. The earlier
numerical failure and stopped step-54 run are retained below. No architecture is
adopted and useful additional inference computation remains unproven.**

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

### What the screen can establish

The primary k6-versus-k4 contrast lies **inside** the variable arm's training
support, k2 through k6. A pass would establish a benefit from additional computation
within that support on this language-loss evaluation. It would not establish
extrapolation to unseen depths: k8 is outside the training support and remains a
separately reported secondary result. Neither contrast measures reasoning ability.

The frozen Prophet core uses GDN mixers, pre-normalized residual blocks and
additive input reinjection (`h + injected`). It has no separate normalization at
the core's output. Huginn instead uses attention, sandwich normalization, a
learned concatenation adapter and a normalized core output; its training samples
a heavy-tailed depth distribution with truncated gradients. These differences
are described in [the primary paper, sections 3.2–3.3 and 4.3](https://arxiv.org/html/2502.05171v2).
This experiment changes only the training-depth policy in Prophet. Its outcome
cannot identify normalization, reinjection or mixer choice as the cause of a
remaining failure, and does not reproduce the paper's architecture or scaling
result. Such a causal claim would require another controlled experiment.

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
# The v2 full-size gates below pass with the strict policy and frozen f5a7d71 driver.
# Continue the preselected output, preserving its numerical/training contract:
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

## Strict deterministic policy and separate-process result

The instrumented probe's launch failure was a malformed newline in its generated
source, before any GPU computation. The failed script/log are retained. The
corrected source was syntax-checked before launching two separate processes;
argument abbreviation was also disabled to keep the wrapper's `--model-repo`
separate from the driver's `--mode`.

The [new evidence archive](experiments/2026-09-20-r04-strict-determinism/export-manifest.json)
contains the completed pair. Both processes start from the same 375M parent and
perform three actual batch-8, sequence-2,048, k6 updates, using the original BF16
autocast, FP32 weights, TF32 policy, model, data and optimizer. They enable
`torch.use_deterministic_algorithms(True, warn_only=False)`, disable cuDNN
benchmarking, enable cuDNN determinism and retain
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. PyTorch documents these controls separately
from RNG seeding; custom external kernels still need empirical checks.
[PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

All **106 gradient tensor hashes match at every update**, and all **107 final
state tensor hashes match**. Input hashes, CPU/CUDA RNG before forward and after
backward, initial weights, losses and gradient norms match too. The first gradient
norm is 5.208903789520264 in both processes. Peak allocated memory remains
13,444,140,032 bytes. The pair completed in 178.28 seconds including verification,
loading, evaluation and hashing. Its approximately eight-second instrumented
updates include CPU copies/hashing and are not a training-throughput benchmark.
The separate uninstrumented strict preflight measured roughly 3.03–3.44 seconds
per update; this short test is not a sustained speed measurement.

The downloaded 137,722-byte ZIP has SHA256
`3746241e1fa9771e3d61326a035956fdfd7553ed986e699443c5dc4113cb16d1`.
The [independent verifier](experiments/2026-09-20-r04-strict-determinism/verify.py)
checks all 24 original file hashes, executed wrapper/driver identity, parent/data/
runtime, unchanged full-validation baseline rows, and every reported tensor/RNG/
input hash. It does not execute the GPU computations again or isolate a specific
kernel as the cause of the legacy-policy discrepancy.

The adaptation CLI now records **`r04-depth-adaptation-v2`** and the explicit
strict numerical policy in its run identity and checkpoint contract. It requires
the cuBLAS workspace setting before Python starts. Old v1 outputs are retained;
they must not resume under changed arithmetic. Both v2 arms start afresh from the
same original parent under the already-budgeted 512-step schedules. Policy,
model/data/schedule and any selected backend flags cannot silently change on resume.
The original plan artifacts remain immutable; this section records the numerical
amendment and its evidence.

Before any long v2 segment, run the updated miniature CPU/CUDA suite, both new
memory preflights, and actual-size interruption tests for both arms: eight
continuous updates versus one update plus a new process for the remaining seven.
Compare all checkpoint tensors (model and optimizers), RNG, loader, depth history,
contract and counters exactly, plus per-document evaluation. These short tests
are diagnostic runs and cannot be used to select the depth recipe. The same
published step-512 decision rule remains required afterward.

`scripts/audit_r04_restart.py` verifies both published checkpoint byte hashes,
restricted-loads their tensors, rejects nonfinite state, and compares the entire
state recursively without numerical tolerance. Its report also requires identical
per-document evaluations. It is exercised by both miniature CLI restart cases;
a corruption test demonstrates rejection of changed optimizer momentum or RNG.
The updated local targeted suite passes seven tests with two CUDA skips. The
full strict-policy suite including the auditor passed 810 tests with 18 skips
in 182.38 seconds; that result does not substitute for the actual A100 prefix gate.

## Actual-size restart gates passed

The [exported full-size evidence](experiments/2026-09-20-r04-fullsize-restart/export-manifest.json)
records thirteen separate processes, all terminal with exit zero. Nine CPU/CUDA
tests pass without skips, including the two miniature CUDA restart cases. Both
new actual-size k6 memory preflights pass and reproduce the same losses; peak
allocation remains 13,444,140,032 bytes.

For **each** 374,689,648-parameter arm, the auditor compares eight continuous
updates against one update followed by seven in a new process. All **315 tensors,
867,172,946 tensor elements** and all other checkpoint state compare exactly:
weights, optimizer states, CPU/CUDA RNG, loader, contracts, counters and depth
history. Every saved document-level evaluation matches too. Fixed-depth history
is eight fours; the sampled history is **[6, 6, 5, 2, 5, 6, 4, 5]** in both paths.
The whole gate queue takes 897.32 seconds, below its 1,800-second diagnostic limit.

The 349,577-byte ZIP has SHA256
`e1935e51e12858e4d04099b34331501c824508f4f989fc667035fc3c54c02529`.
The [local verifier](experiments/2026-09-20-r04-fullsize-restart/verify.py) checks all
46 original file hashes, executed driver identity, separate terminal process IDs,
test XML, parent baseline rows, checkpoint metadata, training logs and recomputed
evaluation aggregates. Twelve large JSON files are losslessly gzipped. The
multi-gigabyte weights are excluded; local verification does not repeat the
recorded A100 checkpoint tensor comparison.

The continuous prefixes were designated as production runs in the gate launcher
before observing their quality. They continued with the **same frozen
`f5a7d71` driver and numerical policy**, first 56 more updates to step 64 in both
arms, then 448 more to step 512. This queue completed in 2,920.52 seconds under its
5,400-second limit; diagnostic gate time is reported separately. No intermediate
score selected an arm or changed the recipe. Final results follow below.

## Final comparison implementation

Before the long segments, `scripts/summarize_depth_adaptation.py` implements the
unchanged step-512 decision above. It rejects incomplete runs, changed paired
identities, mismatched document identities/denominators, invalid depth histories,
nonfinite losses and inconsistent aggregates. Every score is recomputed from
the document rows. Training depth counts and the realized mean are reported.

The previously specified 10,000 draws and seed zero are made operational with
NumPy PCG64 and linear 2.5/97.5% quantiles, matching the published frozen-depth
analysis. Each draw samples whole documents jointly; its CE contrast divides
the summed k6-minus-k4 losses by the summed token counts. This is uncertainty
conditional on these checkpoints, not across training seeds. All four primary
checks must pass; the program never selects a recipe from intermediate scores.
It does not reload weights or repeat GPU inference.

Fourteen focused tests cover successful and unsuccessful screens, token-weighted
contrasts with an interval crossing zero, and corrupted or unmatched reports.
Together with the adaptation CLI tests, the local targeted run passes 21 tests
with two CUDA skips. The training driver remains frozen at `f5a7d71`; this
analysis addition does not change its numerical or training contract.

```bash
python scripts/summarize_depth_adaptation.py \
  --fixed <fixed4>/evaluation-step-000512.json \
  --variable <uniform2to6>/evaluation-step-000512.json \
  --out <fresh-summary.json>
```

The [local final-evidence verifier](experiments/2026-09-20-r04-depth-adaptation-final/verify.py)
now passes against the exported evidence. It checks source hashes, terminal process
records, checkpoint metadata, continuity with the previously verified eight-step
prefixes, all 512 training records and exact agreement with a locally recomputed
final screen. It does not reload checkpoint tensors or repeat GPU inference.

## Final matched result: robust to depth, no gain from additional loops

Both models finish 512 additional updates, 8,388,608 input tokens, loader cursor
36,864 and zero skipped nonfinite updates. All five depths use all 376 unchanged
development documents, 393,040 scored tokens and 1,750,592 payload bytes.

| Inference loops | Fixed-four CE | Variable-depth CE | Fixed-four BPB | Variable-depth BPB |
|---:|---:|---:|---:|---:|
| 1 | 4.803387449 | 4.083040002 | 1.555872374 | 1.322543561 |
| 2 | 4.187250954 | 3.874768834 | 1.356298685 | 1.255082137 |
| 4 | 3.842789566 | 3.846897464 | 1.244723685 | 1.246054280 |
| 6 | 4.416418312 | 3.863485786 | 1.430528625 | 1.251427428 |
| 8 | 4.476420184 | 3.980072203 | 1.449963921 | 1.289191108 |

CE is nats per token; lower is better for both metrics. The variable arm improves
k6 BPB by **12.5199%** over the matched fixed-depth arm, while its k4 BPB is only
**0.1069%** worse. However, its own k6 BPB is **0.4312% worse** than k4, instead of
the required improvement of at least 0.5%. Its paired CE(k6)-CE(k4) difference is
**+0.0165883**, with 95% document-bootstrap interval **[+0.0157351, +0.0174359]**.
The interval is conditional on these weights; it is not training-seed uncertainty.

| Preregistered check | Result |
|---|---|
| Variable k6 / own k4 BPB ≤ 0.995 | **Fail:** 1.004312130 |
| Upper paired CE(k6)-CE(k4) interval < 0 | **Fail:** +0.017435901 |
| Variable k4 / control k4 BPB ≤ 1.01 | Pass: 1.001068988 |
| Variable k6 better than control k6 | Pass |

The primary screen therefore **fails**. Training across depths reduces the fixed
model's sharp optimum at k4, but does not make six loops useful relative to four.
At the untrained k8, variable BPB remains **3.4619% worse** than its k4 result.
The secondary k2 result is **0.7245% worse** than its own k4, or **0.8322% worse**
than control k4; it suggests a possible quality/computation tradeoff at reduced
depth, not a measured latency advantage or fulfillment of the extra-computation
objective. Neither lower language loss nor robustness establishes reasoning.

Actual variable-depth counts are k2=116, k3=101, k4=91, k5=105, k6=99, giving a
mean **3.94140625** loops against exactly 4 for the control. Expected depth was
matched, not realized FLOPs. Summed recorded update times are 1,111.60 seconds
for fixed depth and 1,098.24 for variable depth; these exclude checkpoint writes,
validation and process startup, and are not an inference-speed benchmark.

All four queue process handles exit zero. The CPU final-state audit then reloads
the published files, validates their hashes/contracts/counters and inspects all
315 tensors / 867,172,946 elements per arm for finiteness. It does not compare
the two differently trained states for equality. The fourteen screen tests pass
in Colab, and the final audit plus screen takes 30.29 seconds.

The [31-file evidence export](experiments/2026-09-20-r04-depth-adaptation-final/export-manifest.json)
has ZIP size 478,876 bytes and SHA256
`7bac6bc695ccbc7f23345604d595df35eb499ada7143bf62893117e991ae20de`.
Every member hash is verified locally; large JSON/JSONL files are losslessly
gzipped. The [local recomputation](experiments/2026-09-20-r04-depth-adaptation-final/verification.json)
exactly reproduces all scores, the bootstrap interval and every decision bit.

## Consequence for the research sequence

Do not adopt this recipe as evidence of useful additional computation or expand
the seed-zero screen into a positive result by changing its endpoint. The planned
success-triggered multi-seed extension is not triggered. Retain both endpoints as
references for a new controlled hypothesis, while preserving the corpus-pass
ceiling: this adaptation already reaches 3.8877 passes of the original corpus.

The [bounded internal-state observations](24_R04_RECURRENCE_OBSERVATION.md) are
also complete. They help constrain the next hypothesis but do not identify a
causal architecture defect. Any change to normalization, input reinjection or
the mixer still requires budget/search/plan checks, a separately fixed controlled
experiment and demonstrated capability gains before adoption.
