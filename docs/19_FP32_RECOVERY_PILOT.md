# FP32 donor-recovery pilot

The converted GDN candidate fails the production BF16 full-model gradient gate.
Keeping only GDN in FP32 reduces but does not remove the discrepancy. Full FP32
passes and now has measured update costs, matched development baselines and a
CUDA restart check. This enables an explicit FP32 recovery experiment; it does
not clear the BF16 failure described in
[the conversion report](18_DONOR_CONVERSION_REHEARSAL.md). After 256 real updates,
the recovered checkpoint also passes the previously failing FP32 cache case,
as detailed below; broader cached-decoding validation remains outstanding.

## Precision and validation

Recovery, its frozen teacher, the update gate and development evaluation accept
an explicit `--precision float32`. Autocast and PyTorch TF32 matmul/convolution are
disabled for that policy; the validated FLA recurrence retains `tf32x3`. BF16
defaults remain available. Precision, teacher tensor identity and evaluation batch
size belong to the resume contract. Evaluation can keep batch one while training
uses batch four, matching the measured baseline exactly in window/batch policy.

On the actual Colab A100, all eleven CUDA unit cases pass in 9.01 seconds,
including a manual FP32 loss reference and a check of actual projection dtypes.
Two additional miniature donor-recovery CLI tests pass in 16.24 seconds: both CE
and KL reproduce uninterrupted weights and evaluation exactly after a CUDA
checkpoint restart, and reject changing precision during resume. The first
attempt stopped during test collection because an installed package shadowed the
local `tests` namespace; adding `tests/__init__.py` fixes that import ambiguity.
The failed attempt is retained alongside the successful retry.

The real 385.5M GDN model also passes the sequential/fused comparison over the full
512-token training window, at k=5, with identical recurrence randomness and
activation checkpointing. The largest tensor relative L2 discrepancy is
0.000918555, below the unchanged 0.002 FP32 bound. This numerical comparison uses
batch one; the subsequent five actual updates use batch four. The shorter
67-token probes and the full-window probe all pass. These checks concern training
execution, not cached generation or sustained model quality.

## Matched FP32 baselines and cost

All baselines use the audited Colab pair and unchanged donor, 372 development
documents, 366,762 targets, 1,722,551 scored bytes, sequence length 512 and batch
one. The converted models use fixed depth k=5.

| Model before recovery | Nats/token | Bits/byte |
|---|---:|---:|
| Unchanged donor | 3.157778885 | 0.969992773 |
| Shared attention | 9.221068973 | 2.832487832 |
| Shared GDN | 11.308395923 | 3.473663839 |

Four discarded update probes use batch four, 512 tokens per sequence, two warmup
and three synchronized measured updates. All model/optimizer tensors remain
finite, no update is skipped, and both KL teachers remain unchanged and without
gradients. Each probe consumes 10,240 input tokens; its weights are discarded.

| Core / objective | Mean measured update | Peak allocated GPU memory |
|---|---:|---:|
| GDN / CE | 1.1156 s | 8.9844 GiB |
| GDN / CE + KL | 1.3428 s | 12.9365 GiB |
| Attention / CE | 0.8784 s | 8.7907 GiB |
| Attention / CE + KL | 1.1118 s | 12.7441 GiB |

The full-window GDN probe measures 1.1146 seconds per update and the same peak
allocation. These are three-sample measurements excluding evaluation/checkpoint
I/O, not end-to-end throughput or an estimate with a confidence interval.

## Frozen pilot budget

The first actual recovery segment is launched from revision `467d7c8`, after all
the above gates pass. The GPU checkout is frozen while it runs; the R04 checkout
and the still-running CPU evaluation checkout are unchanged.

| Setting | Value |
|---|---|
| Arms | GDN CE; attention CE; GDN CE + KL; attention CE + KL |
| Initialization/training seed | 0; the separately audited Colab pair |
| Steps per arm | 2,048 |
| Input tokens per arm | 4,194,304 |
| Training batch / sequence / accumulation | 4 / 512 / 1 |
| Evaluation batch / sequence | 1 / 512 |
| Core depth | k=5, full backpropagation through all loops |
| Peak Muon / AdamW rates | 0.001 / 0.00003 |
| WSD schedule | 100 warmup, 1,580 plateau, 368 decay steps |
| KL coefficient / temperature | 0.5 / 1.0; frozen FP32 donor |
| Auxiliary objective weights | MTP, confidence, z-loss and ponder all zero |
| Periodic local checkpoint | Every 256 steps |
| First bounded segment | 256 steps, then full development evaluation |

The pilot allocates equal student tokens, not equal FLOPs: the two KL arms also
consume teacher computation. Across all four arms there are 16,777,216 student
input tokens and 8,388,608 teacher input tokens. Each arm consumes about 0.227 of
one 18,486,792-token corpus pass, below the four-pass ceiling. The rates are common
exploratory choices rather than per-arm tuned optima. Extrapolating only the short
update measurements gives 2.53 A100 hours for all four arms, before setup, I/O and
evaluation. This is a budget estimate, not measured completion time.

The unchanged donor and initializations are controls. Development loss and paired
document differences will measure recovery; training minibatch loss alone is not
a quality result. The seed-zero pilot cannot establish multi-seed superiority,
untouched benchmark performance, cache reliability, architecture adoption or AGI.
Later R04 seeds, variable-depth training, persistent-memory/agent experiments and
device deployment remain part of the larger project.

## First measured recovery checkpoint: GDN CE, step 256

The first segment completed 256 updates and 524,288 input tokens with no nonfinite
skips. Full development validation uses the same batch-one FP32 protocol:

| Model state | Nats/token | Bits/byte |
|---|---:|---:|
| GDN before recovery | 11.308395923 | 3.473663839 |
| GDN CE after 256 updates | 5.649684912 | 1.735445620 |
| Unchanged donor | 3.157778885 | 0.969992773 |

All 372 documents improve relative to the initialization. The paired CE difference
is -5.658711, with a 10,000-draw document-bootstrap interval of
[-5.717977, -5.600104]. This closes 69.43% of the *initial CE gap* to the donor;
it is not a percentage of recovered capabilities. The donor still has substantially
lower loss, and the other three recovery arms have no matched trained checkpoint
yet. The paired interval excludes training-seed uncertainty.

The exact evaluated local checkpoint is 3,707,786,137 bytes, SHA256
`edd4632ae4482c72ccf69a19b999c78696eb570cb55fbffd0ea15ad46c2d588a`.
A separate CPU audit verifies its checksum, saved CUDA RNG, complete training
contract/configuration and 165 model plus 301 optimizer tensors, all finite.
Every one of the 256 training-log rows has the expected step/token count and
finite loss/gradient norm. No remote durability claim is made for this intermediate
checkpoint; local rotation may replace the original slot as the same run continues.
An independently checksummed local analysis copy preserves these exact step-256
weights for the cache experiment below.

The downloaded 11-file evidence ZIP is 50,082 bytes, SHA256
`7b867068b83b1777648e0aed60515856878644888cf8c1d2eb62d17b1525a653`.
The archive and audit/evaluation/script hashes were independently checked, along
with input identities, all document denominators and training rows. The paired
comparison was computed independently after download.
[Checkpoint evidence and comparison](experiments/2026-09-20-qwen-recovery-step256/paired-versus-initialization.json).
The same frozen run subsequently completed step 2,048 under its original schedule,
without changing its token budget after observing this intermediate checkpoint.

## Recovered checkpoint: the former cache failure now passes

The cache auditor accepts an evaluated recovery run and a required step. Before
loading weights into the model, it verifies the exact evaluated slot, checksum,
training contract, configuration, step/token budget, CUDA RNG and finite model
and optimizer state. A rotated slot is rejected instead of substituting newer
weights. This uses audit revision `25d24cc`; training remains frozen at `467d7c8`.

The exact step-256 checkpoint passes both the chunk64 and sequential GDN scans,
on the same 128-token prefix that failed before recovery. Both runs use CPU FP32,
k=5, two CPU threads and PyTorch 2.11.0+cu128. The prefix/document and donor hashes,
precision, scan and elementwise tolerance match the earlier reports. Neither FP64
oracle is enabled. The unchanged bound is `1e-4 + 1e-4 * abs(reference)`.

| GDN scan / path | Maximum logit error before | After 256 updates | Maximum tolerance ratio after |
|---|---:|---:|---:|
| chunk64 / 64+63+1 prefill/decode | 0.000219822 | 0.000008583 | 0.04570 |
| chunk64 / tokenwise | 0.001132488 | 0.000017166 | 0.09345 |
| sequential / 64+63+1 prefill/decode | 0.000203729 | 0.000010014 | 0.03705 |
| sequential / tokenwise | 0.001148224 | 0.000018120 | 0.10292 |

All four paths remain finite, match all 128 argmax positions and pass; a tolerance
ratio greater than one would fail. The before-recovery ratios range from 1.55 to
8.91. This demonstrates improvement on the previously failing case, not a general
guarantee of cache equivalence, a proven numerical root cause, or a BF16 fix.
More documents, lengths and the final checkpoint still require evaluation.

[Chunk64 report](experiments/2026-09-20-qwen-recovered-cache-step256/cache-chunk64.json)
and [sequential report](experiments/2026-09-20-qwen-recovered-cache-step256/cache-sequential.json)
include the checkpoint audit. The downloaded five-file ZIP is 13,004 bytes, SHA256
`b7f157fdf44ff800f4de2bed0e1726fe89060710c3c931185dfab9b986946f26`.
Archive/report hashes, checkpoint identity, input identity, unchanged thresholds
and all four results were independently verified after download. CI at `25d24cc`
passes 767 CPU cases with 13 CUDA-only skips, including deliberate corruption and
rotated-slot rejection tests for the new recovery checkpoint path.

The remaining three arms were queued behind the first run. The queue keeps
the original 2,048-step budget and frozen training revision, resumes bounded
segments if needed and stops on process failure, no progress or audit failure.
Each final endpoint receives a CPU checkpoint audit and training-row checks before
the next arm proceeds. The first final is scheduled for a checksummed Drive copy;
that copy will still need a flush/remount verification. The other finals currently
have local storage allocated, with remote persistence pending available Drive
space. Queuing is not evidence that those arms have completed.

The final GDN CE endpoint has completed a separate CPU FP32 development cache suite
at analysis revision `ed72c86`. Before model execution, it selects the
first four distinct eligible document hashes in ascending order, requiring at
least 512 payload tokens. Each document is checked at 128 and 512 tokens, with
chunked prefill/decode and tokenwise execution. Corpus/tokenizer identities are
bound to the audited recovery contract; depth remains the trained k=5 and both
tolerances remain 1e-4. All eight cases are retained, including numerical failures.
These are development texts, not untouched benchmark or deployment certification.
All eight cases pass in 547.09 seconds, covering both paths in each case. All
positions have matching argmax tokens and finite logits. Across 128-token cases,
the maximum logit error is 0.000024319 and the largest tolerance ratio is 0.16183;
across 512-token cases, these are 0.000028610 and 0.24302. A ratio above one would
fail. This covers four selected development documents, CPU FP32 and fixed k=5;
CUDA/BF16 generation, longer contexts, variable depth and wider coverage remain
separate requirements. Thirty focused local tests pass
with two CUDA-only skips; an injected cache error beyond a shorter prefix makes
the longer case fail while preserving both results.
[Full eight-case report](experiments/2026-09-20-qwen-recovery-hybrid-ce-final/cache/chunk64.json).

```bash
python scripts/audit_recovery_cache_suite.py --run <recovery-run> --step 2048 \
  --source <donor-source> --validation <recovery-validation.jsonl> \
  --documents 4 --lengths 128 512 --out <new-report.json>
```

## First final endpoint: GDN CE, step 2,048

The first arm completed its frozen budget of **2,048 updates and 4,194,304 input
tokens**, with zero skipped updates. Full development evaluation gives:

| Model state | Nats/token | Bits/byte |
|---|---:|---:|
| GDN before recovery | 11.308395923 | 3.473663839 |
| GDN CE, step 256 | 5.649684912 | 1.735445620 |
| GDN CE, step 2,048 | **4.571112242** | **1.404134362** |
| Unchanged donor | 3.157778885 | 0.969992773 |

All 372 documents improve both against initialization and against step 256.
The final-minus-initial CE difference is -6.737284, with paired document-bootstrap
95% interval [-6.804932, -6.671657]. Final-minus-step-256 is -1.078573, with interval
[-1.111891, -1.045251]. The endpoint closes 82.66% of the initial *CE gap* to the
donor; this is not a percentage of recovered capabilities. The donor still has
lower loss. These intervals condition on the measured models and do not cover
training-seed uncertainty or establish a winning architecture.

The exact evaluated checkpoint is 3,707,786,649 bytes, SHA256
`9d511123800d71cac0ed4399546ab152e1d70f5b31b4655aa6661ec92da242f8`.
Its CPU audit passes checksum, complete contract/configuration, saved CUDA RNG and
165 model plus 301 optimizer tensor checks. All 2,048 training rows have the exact
expected step/token count and finite losses, rates, durations and auxiliary values.
Evaluation identity is unchanged from step 256; all document identities,
denominators and aggregates are checked independently after download.

The final-evidence ZIP is 166,039 bytes, SHA256
`b811a665c59fec1d06906f2abd59d919798b6f0f3bb258e2f15ef1da4a622883`.
Its 13 files include the frozen plan, exact notebook queue sources, training rows,
evaluation, checkpoint metadata and audit. The downloaded hashes match the audit
and queue records. [Independent paired comparisons](experiments/2026-09-20-qwen-recovery-hybrid-ce-final/paired-comparisons.json)
are computed locally from the downloaded per-document losses.

At this first endpoint, the attention CE arm had started with the same frozen
revision, schedule and token budget; both KL arms were queued. Its completed result
is recorded below. The final GDN CE weights have a checksummed
Drive copy, verified again after a successful 2.85-second flush and remount.
The full CPU audit of the remounted weights reproduces the local audit exactly,
including the checkpoint checksum and all 466 model/optimizer tensors; training
rows, evaluation, configuration, frozen plan and snapshot manifest also match.
The earlier mounted-copy and queue records predate this verification and retain
their original pending flags. The expanded cache suite completed independently
on CPU while attention CE used the GPU. CI at `ed72c86` passes 770 CPU cases with
13 CUDA-only skips.

The downloaded seven-file cache/persistence proof archive is 9,684 bytes, SHA256
`b3f19af9d2d6bf4932f7abfacf1bd7cf856426a716092a33bc5fb507c9e0d4c7`.
Its remounted audit is byte-identical to the local audit. Archive/report hashes,
all sixteen cached path results, exact prefix token hashes and the deterministic
four-document selection were independently verified against the local corpus and
pinned tokenizer. [Persistence proof](experiments/2026-09-20-qwen-recovery-hybrid-ce-final/persistence/hybrid-ce-final-persistence.json).

## Second final endpoint: attention CE, step 2,048

The matched attention CE arm completes **2,048 updates / 4,194,304 input tokens**,
with zero skipped updates. Its full development CE is **4.530511577**, BPB
**1.391662827**, on the same 372 documents, 366,762 targets and 1,722,551 bytes.
Every document improves against its own initialization. The attention-minus-initial
CE difference is -4.690557397, with paired document-bootstrap 95% interval
[-4.761395317, -4.620042312]. This closes 77.36% of its initial CE gap to the donor;
the smaller gap-closure fraction than GDN does not reverse its better final CE,
because attention started closer to the donor.

The interim **GDN-minus-attention CE difference is +0.040600665**, interval
[+0.037362236, +0.043829997]. Attention has lower loss on 324 of 372 documents.
The interval conditions on these two models and excludes training-seed uncertainty.
This is a completed CE-only comparison; the four-arm factorial result remains
unavailable until both KL arms complete. Neither converted model matches the
unchanged donor's CE of 3.157778885. GDN has 385,515,905 unique parameters and the
attention control has 360,087,809: the hybrid is not smaller in stored parameters
in this conversion experiment. The R04 parameter-saving result concerns a different
controlled comparison.

The attention checkpoint is 3,503,292,889 bytes, SHA256
`428a6dd7414054b93d4861535b643a19389b40dd4641de2e0dd03f80bd76527d`.
The separate CPU checkpoint audit passes; the downloaded reports reproduce its
evaluation hash, complete configuration/contract, saved CUDA RNG and finite-state
evidence. Both completed endpoints pass the strict four-arm comparator's per-arm
checks before it correctly refuses the absent third endpoint. Their normalized
runtime/input identities and training contracts match. All 2,048 attention training
rows and their auxiliary metrics are finite, with exact step/token counts.
The median of the last 256 logged attention updates is 0.87114 seconds, excluding
setup, evaluation and checkpointing; this is not billed runtime.

The attention CPU cache suite passes all eight prefixes and sixteen execution paths
in 417.00 seconds. Maximum logit error is 2.76566e-5 and the largest tolerance ratio
is 0.16948; all argmax positions match. Corpus/tokenizer hashes and all eight prefix
identities are independently reproduced from the local source files and match the
GDN suite. Both use FP32, depth five, two CPU threads and unchanged 1e-4 absolute
and relative tolerances.

| Prefix length | GDN cache | Attention cache | Measurement scope |
|---|---:|---:|---|
| 128 tokens | 49.875 MiB | 28 MiB | Stored cache tensors, batch one |
| 512 tokens | 73.875 MiB | 112 MiB | Stored cache tensors, batch one |

GDN retains 41.875 MiB of recurrent state at both lengths; its attention component
grows from 8 to 32 MiB. The attention-only cache grows from 28 to 112 MiB. Thus
bounded recurrent state has an initial cost: it saves cache storage at 512 tokens
but costs more at 128. These are actual CPU tensor-storage measurements, excluding
model weights, workspace, allocator overhead and peak memory. They neither prove
lower total inference memory nor certify numerical behavior beyond the trained
512-token window.

The [downloaded evidence and paired/cache comparison](experiments/2026-09-20-qwen-recovery-attention-ce-final/paired-and-cache-comparison.json)
come from a 171,906-byte, twelve-file ZIP, SHA256
`73fb41313b1fc28b2f88adbe3bf84e6854e43211007b410a81bd2848dc8715e6`.
It includes the exact executed notebook sources for the remaining CPU cache and
four-arm analysis queue, and the post-training GPU cache queue. Attention weights
remain on the Colab runtime; this export contains reports, not a durable weight copy.
The GDN KL process was confirmed live after both CE training/audit processes exited
successfully. Its completed endpoint is recorded below; attention KL follows it
under the frozen pilot protocol.

### Attention checkpoint verified outside Colab

The exact attention CE checkpoint has subsequently been downloaded to the local
PC in seven parts and reconstructed without reserialization. All seven sizes and
SHA256 values match the source transfer manifest; the full file reproduces the
3,503,292,889-byte size and checkpoint SHA256 above. The four metadata files are
byte-identical to the archived Colab reports. Weights remain ignored under
`data/recovered-models/attention-ce-seed0/ckpt_slot1.pt`.

A fresh restricted CPU load checks all 149 model and 253 optimizer tensors,
configuration, complete training contract, exact step/token budget, saved CUDA
RNG and evaluation identities. After JSON normalization, its audit matches the
Colab audit in every field except the reported PyTorch version: local 2.14.0+cpu
versus Colab 2.11.0+cu128. This verifies local persistence outside the ephemeral
runtime; it is not a new inference test, CUDA check or Drive durability claim.
[Transfer proof and local audit](experiments/2026-09-20-qwen-recovery-attention-ce-final/persistence/local-transfer-verification.json).

## Third final endpoint: GDN CE+KL, step 2,048

The hybrid guided by the frozen Qwen donor completes **2,048 updates and
4,194,304 student tokens**, plus the same number of teacher input tokens.
Its full development CE is **4.332629851 nats/token**, BPB **1.330878379**, on the
same 372 documents, 366,762 targets and 1,722,551 bytes. Zero nonfinite updates
are skipped. The first bounded session finishes at step 1,678 and a second session
resumes to step 2,048; both process exit codes and the final audit exit code are zero.

Within the hybrid architecture, **CE+KL minus CE is -0.238482391 nats/token**,
with paired document-bootstrap 95% interval [-0.243985113, -0.233098130]. Every
development document improves. The model closes 85.59% of its initial CE gap to
the donor, which still has lower CE, 3.157778885. This is not a percentage of
recovered capabilities. Equal student tokens do not imply equal compute: KL also
runs the frozen teacher. The median of the last 256 logged hybrid-KL updates is
1.34794 seconds, excluding setup, checkpointing and evaluation.

The exact evaluated checkpoint is 3,707,789,145 bytes, SHA256
`d2b8cc68424da4e3d1ee030fed2ffec4a088e1a14ced9e377965092a7c95d118`.
Its separate Colab CPU audit passes with saved CUDA RNG and 165 model plus 301
optimizer tensors. The downloaded audit/evaluation/cache hashes match the live
queue, respectively:

- `70376ee369f770bdbb60250dd6f690c0d6b6012109afb246f7fa7980bc89aded`
- `9d2e2563bb464d8580d02ebaaf97895d881d956c8720d55744b37f4213787a17`
- `6cb07e1281a463bb82611f6c166a26fe1f8fdec46d842c9c9c8f89221e19da6f`

Independent local analysis validates all three completed endpoint reports against
the strict comparator, which correctly refuses the absent fourth endpoint. It also
checks normalized input/runtime identities and training contracts across the three,
the identical hybrid initialization/configuration, the frozen KL teacher settings,
all 2,048 training rows and finite auxiliary metrics. The document intervals
condition on these models and do not cover training-seed uncertainty.

The hybrid-KL CPU cache suite passes eight prefixes / sixteen execution paths in
549.89 seconds, with maximum logit error 2.86102e-5 and tolerance ratio 0.185764.
All argmax positions agree under the unchanged 1e-4 tolerances. Prefix selection
and token hashes are independently reproduced from the local corpus/tokenizer;
cache storage is identical to hybrid CE at both measured lengths. This does not
certify GPU decoding or contexts beyond the trained 512-token window.

[Evidence and reproducible paired/cache analysis](experiments/2026-09-20-qwen-recovery-hybrid-kl-final/paired-and-cache-comparison.json)
come from a twelve-file, 225,462-byte downloaded ZIP, local SHA256
`10df8cf98212574a4897b318e6f32c6f0b8cb211bb42fa1de6456d95de6da334`.
The archive checksum itself was not retrieved from Colab before browser access
became unavailable; the three substantive report hashes above were independently
matched to the source queue. This export contains reports, not a durable copy of
the hybrid-KL weights. Those weights remain on the Colab runtime.

At the last direct process poll, attention KL PID 141999 was live and its log
reached step 512. The final four-arm comparison, CUDA cache suites and ARC-Easy
evaluation remain queued. Subsequent loss of browser access is not evidence that
training stopped and does not justify restarting that process.

## CUDA cache coverage and queued checks

Revision `bfce7c5` adds explicit `--device cuda` to the expanded cache CLI. It keeps
FP32 and the same thresholds, disables PyTorch TF32, requires FLA chunk32 for GDN,
records CUDA/device/FLA identity and peak allocation, and refuses unavailable CUDA
instead of falling back to CPU. The reference-scan option remains CPU-only.
Tiny trained-model cases cover both correct cached decoding and a deliberate error
beyond the shorter prefix. The local focused run passes 41 cases with four CUDA
skips; full GitHub CI passes **781 CPU cases with 15 CUDA-only skips** in 94.27 seconds.
The earlier actual A100 gates cover thirteen CUDA cases; the two new CUDA cache
cases are **queued, not yet executed**.

The GPU queue waits for all four training/audit processes to finish, then runs the
four CPU/CUDA miniature cache cases and each arm's eight real CUDA prefixes in an
isolated pinned worktree. Input identities and evaluated checkpoint hashes must
match the CPU evidence. Numerical failures retain complete reports and do not
relax thresholds; unexpected execution errors stop the queue. The live training
checkout stays frozen at `467d7c8`.

## Full comparison contract

The final four-arm comparison uses `scripts/summarize_recovery_pilot.py`. It
requires all four planned endpoints and their separately executed checkpoint
audits. It rejects mismatched step/token budgets, initializations, corpus/tokenizer
identities, runtime/precision, evaluation batching, model configurations, training
schedules, KL teacher tensor identities/settings, rotated slots and incomplete
training logs. Reading an audit report is not a new tensor or durability audit.

Four contrasts separate the two experimental factors: GDN minus attention under
CE; GDN minus attention under CE+KL; CE+KL minus CE within GDN; and CE+KL minus CE
within attention. Each uses the same paired documents. The bootstrap intervals
are unadjusted and condition on these trained models; they do not measure seed
uncertainty. Ranking by development CE remains descriptive. Equal student tokens
do not imply equal compute, because KL also evaluates a frozen teacher. The report
therefore includes teacher tokens and logged update times, without treating those
times as billed runtime or making an architecture-adoption decision.

```bash
python scripts/summarize_recovery_pilot.py --root <four-arm-run-root> \
  --baselines <matched-fp32-baselines> --out <new-comparison.json>
```

Analytically known miniature reports test all contrast signs and intervals, plus
ten deliberately incomplete or incompatible evidence cases. Sixteen focused
summary/paired tests pass. The real completed GDN endpoint also passes the report
checks; the comparison correctly refuses to publish before the other endpoints
exist.

[Baselines and four update probes](experiments/2026-09-20-qwen-colab-fp32/policy/queue.json)
use revision `31dafc2`.
[CUDA restart and the full-window gate](experiments/2026-09-20-qwen-colab-fp32/pretrain/queue.json)
use `467d7c8`. The intervening changes add explicit evaluation batching, the longer
probe and tests; the model implementation and FP32 policy are unchanged.

The 25-file archive is 119,798 bytes, SHA256
`81ff17494e343daeba7efc5226ec0d5d86bcc9f5435a6b443b431eb694de54af`.
After download, archive/report hashes, JUnit counts, baseline input identities,
all document denominators/loss aggregates and gate precision/finite-state contracts
were independently checked. CI at `3e092ec` passes 759 CPU cases with 13 CUDA-only
skips; the CUDA executions above cover those thirteen cases separately.

At launch, Drive has 5.72 GB free. Additional removal of the older R04 step-1,024
and step-2,048 checkpoint files has been requested to accommodate the new results;
it is not treated as approved. The final R04 step-4,096 pair remains preserved.
