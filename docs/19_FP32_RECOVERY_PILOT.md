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
The same frozen run is continuing from step 256 toward 2,048 under its original
schedule, rather than changing its token budget after observing this checkpoint.

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

The remaining three arms are now queued behind the live first run. The queue keeps
the original 2,048-step budget and frozen training revision, resumes bounded
segments if needed and stops on process failure, no progress or audit failure.
Each final endpoint receives a CPU checkpoint audit and training-row checks before
the next arm proceeds. The first final is scheduled for a checksummed Drive copy;
that copy will still need a flush/remount verification. The other finals currently
have local storage allocated, with remote persistence pending available Drive
space. Queuing is not evidence that those arms have completed.

## Evidence

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
