# FP32 donor-recovery pilot

The converted GDN candidate fails the production BF16 full-model gradient gate.
Keeping only GDN in FP32 reduces but does not remove the discrepancy. Full FP32
passes and now has measured update costs, matched development baselines and a
CUDA restart check. This enables an explicit FP32 recovery experiment; it does
not clear the BF16 or cached-decoding failures described in
[the conversion report](18_DONOR_CONVERSION_REHEARSAL.md).

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
