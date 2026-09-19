# Real donor conversion rehearsal

The first real-weight rehearsal uses public
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
revision `c1899de289a04d12100db370d81485cdf75e47ca`, whose card records Apache-2.0.
No gated access or remote model code was used. The downloaded safetensors file
contains 1,503,300,328 bytes; its SHA256 matches the Hub LFS metadata:
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The donor tokenizer and source files remain outside Git.

## Conversion defects reproduced and corrected

The generated conversion configuration used Prophet's default RMSNorm epsilon,
1e-5, whereas the donor uses 1e-6. The verifier did not check it. A regression test
on small activations failed with a maximum absolute difference of 0.554 before
the fix. The specification, verifier and generated configuration now carry the
donor epsilon, including Q/K norms. Registry entries remain unverified globally;
the rehearsal marks only its locally checked donor instance verified.

Qwen3-0.6B has 16 query heads of width 128 over a residual width of 1,024. Deriving
the GDN head count from residual width discarded half that projection space and
left Q/K seeds at fresh initialization. A test with the same 2:1 relationship
reproduced both mismatches. The conversion candidate now retains the donor query
head count, allowing all four projection seeds to match. This changes the candidate
initialization, not the frozen R04 training configuration or an adopted architecture.

The transfer report also counted a tied head as both copied and fresh because its
copied entry contained an explanatory suffix. Reports now use exact parameter
names, and a missing donor embedding cannot produce a false copied-head entry.

## Real block equivalence

Three actual donor layers, 0, 13 and 27, were copied into isolated full-attention
Prophet blocks and compared with Transformers 5.17.0 on CPU float32, PyTorch 2.14.0.
Each used seeded synthetic inputs of shape 2×17×1024 at position offsets 0 and 257,
with the same causal mask and rotary positions. **All six outputs matched exactly.**
The old epsilon produced maximum absolute errors from 8.95e-5 to 1.95e-3 on these
weights. This checks unchanged attention blocks, not SWA/NoPE/GDN adaptations or
whole-model equivalence. [Raw block audit](experiments/2026-09-19-qwen-block-parity.json).

The unchanged-stack control then streamed **all 28 copied attention blocks** through
Prophet on the four held-out prefixes below. Its loss exactly reproduced the donor's
2.992543936 nats/token. The ordinary, non-recurrent `trunk` mapping had first been
found to assign no source layers; a failing regression reproduced this omission,
and the planner now maps each trunk layer to its corresponding donor layer.
This control keeps donor precision converted to float32, full attention and rotary
positions throughout. It does not isolate the hybrid's sharing, GDN, NoPE or BF16
initialization changes individually.
[Complete copied-stack audit](experiments/2026-09-19-qwen-stack-parity.json).

## Hybrid initialization and negative forward result

The candidate has four prelude, four shared GDN core and four coda blocks, with
five core iterations: 28 executed blocks. It retains the donor vocabulary and the
project's auxiliary-head defaults. Its **385,515,905 unique parameters** comprise:

| Origin | Parameters |
|---|---:|
| Direct copy | 281,431,040 |
| Average of donor layers | 37,756,928 |
| Heuristic attention-to-GDN seed | 50,331,648 |
| Fresh initialization | 15,996,289 |

Direct copies plus averages cover **82.795%** of actual unique parameters. No shape
mismatch remains. All parameters are finite, and the tied embedding/head remains
one parameter. The seed-0 BF16 initialization was saved and reloaded with exact
tensor equality: **771,101,353 bytes**, SHA256
`cfa8f33e3e588a13fb430a1427a5e9f842395d169822f9c55e7bd13a0ca41848`.
The local artifact is `checkpoints/qwen3-0.6b-rehearsal-init.pt`; it is not committed.
[Full conversion audit](experiments/2026-09-19-qwen-conversion.json).

The budget, design search and compute allocation tools were run before proceeding.
The original analytical count was 3,200 parameters below the actual model; the
historical conversion report retains that difference and uses the actual count for
coverage. The omission is now reproduced and corrected: four GDN output RMSNorms
and gate biases contribute 4 × (256 + 32) = 1,152 parameters, and the MTP block's
two RMSNorms contribute 2,048. The attention control lacked only those 2,048 MTP
parameters. Exact tests construct both audited configurations on the meta device
and now match all 385,515,905 hybrid and 360,087,809 attention parameters. A separate
regression also verifies that fractional GDN expansion rounds per head, as the
live mixer does. This changes the accounting, not the saved weights or frozen R04
execution. Embeddings occupy about
40.4% of the candidate, above the allocation guideline. Memory and device figures
from the budget remain estimates; this candidate has not been profiled on A100.

A CPU float32 forward smoke used the first four eligible validation documents,
truncated to 128 next-token targets each. Both arms used identical donor-tokenizer
IDs: **512 scored targets**, all logits finite.

| Model | Nats/token on these four prefixes |
|---|---:|
| Original donor | 2.992544 |
| Hybrid initialization | 12.092936 |

**The conversion does not preserve donor language quality on this smoke.** Parameter
coverage and isolated block equivalence do not establish a usable hybrid model.
This small prefix check is not full validation, a comparison with R04's different
vocabulary, or an estimate of recovery quality. No recovery training has occurred.
[Raw forward smoke](experiments/2026-09-19-qwen-conversion-smoke.json).

## Controlled initialization diagnostics

Seven arms were declared before this diagnostic and evaluated on the same four
prefixes, with five loops and 28 executed blocks. Every original donor tensor is
already BF16. Copied weights therefore incur no additional BF16 rounding; averaged
weights and fresh hybrid parameters retain the rehearsal's BF16 storage. All
forwards use CPU float32. This does not isolate rounding of averaged weights.

| Core / initialization | Recurrence input | Attention positions | Nats/token |
|---|---|---|---:|
| Attention / contiguous average | Serial | RoPE | 14.605840 |
| Attention / contiguous average | Reinjected | RoPE | 8.776271 |
| Attention / stride selection | Serial | RoPE | 10.842782 |
| GDN / contiguous average | Serial | RoPE | 14.012113 |
| GDN / contiguous average | Reinjected | RoPE | 11.676625 |
| GDN / contiguous average | Serial | NoPE at index 1 | 13.904784 |
| GDN / contiguous average | Reinjected | NoPE at index 1 | 12.092936 |

Serial means the initial state is the prelude output with no repeated injection;
reinjected means zero initial state plus the prelude output at every loop. These
produce the same input to the first core pass. GDN arms load exactly the same saved
hybrid tensors; only these recurrence and positional controls change. Attention
arms use the same donor mappings, embeddings, norms and FFNs. SWA in the outer
sections is unchanged; all 128-token inputs fit within its 2,048-token window.
The original hybrid result is reproduced exactly, and all logits remain finite.
[Raw diagnostic](experiments/2026-09-19-qwen-initialization-ablation.json).

Weight sharing with attention already causes severe loss here. Reinjection helps
the averaged-attention and GDN/RoPE controls, while removing NoPE improves the
reinjected GDN arm by about 0.4163 nats/token. The positional effect changes sign
in the serial arm, so it is not an independent universal benefit. None approaches
the unchanged donor's 2.992544. These four prefixes are now a repeatedly inspected
diagnostic set: they cannot select a production architecture or establish recovery,
generalization, significance, or a best initialization strategy. Unseen validation
and equal-budget recovery runs are required. The current initialization remains
an experimental artifact, not an adopted Prophet-main model.

## Recovery rationale and next gate

The conversion literature does not establish quality from weight transfer alone.
Relaxed Recursive Transformers tests layer-specific low-rank residuals initialized
from donor/shared weight differences, followed by uptraining. Its extended Gemma
experiment uses 60B tokens with forward-KL distillation, beyond the 15B-token
setting. Those budgets and architectures cannot be assumed to transfer to this
Qwen/GDN candidate. [Primary paper, sections 2.3 and 3.6](https://arxiv.org/html/2410.20672v3).

Before spending a recovery budget, retain the unchanged donor control and separate
an all-attention shared candidate from the GDN candidate. Compare plain language
modeling against the same initialization and data order with a frozen-donor
forward-KL objective. Keep the donor tokenizer, record teacher inference cost as
well as student tokens, and use a held-out evaluation not selected by these seven
prefix diagnostics. Restore optimizer, data cursor and RNG under an immutable
teacher/checkpoint contract. Additional per-depth adapters are another architectural
change and require their own parameter/memory accounting and ablation. These are
requirements for the next experiment, not evidence that this initialization will
recover within the available budget.

### Recovery training path implemented; real recovery not yet run

`scripts/recover_qwen.py` supports CE-only or
`(1-alpha) CE + alpha T² KL(donor/T || student/T)` from an audited initialization.
The donor is frozen and evaluated without gradients. KL workspaces are token-chunked;
both full logits tensors still exist. Step timings include the donor forward and
reports count donor tokens. No real recovery-quality result or A100 recovery memory
measurement exists yet.

The training contract binds the initialization, actual teacher tensor hash, source
revision/configuration, tokenizer bytes/policy, corpus hashes, objective, schedule
and runtime versions. A changed donor, alpha, corpus or initialization cannot be
silently resumed. Fixed k=5 with full BPTT is the default; MTP/confidence, ponder and
z-loss weights are zero. Initialization tensors and the original train/evaluation
state-initialization policies are preserved. CE and KL arms reset the training RNG
identically after model loading. Recovery is separate from the frozen R04 runner.

End-to-end miniature tests load a real local Transformers Qwen model, train both
objectives, serialize checkpoints, resume, and evaluate. Resumed weights and
evaluation equal uninterrupted execution exactly on CPU. Loss/gradient tests compare
KL against ordinary PyTorch autograd; a nonfinite objective skips the optimizer even
with finite gradients. These validate implementation, not real-size recovery or
the unresolved cached-decoding gate below.

The plain-text tokenizer adapter retains donor vocabulary and explicit EOS,
disables automatic added-token recognition, and counts NFC-normalized UTF-8 payload
bytes. All **376 validation documents** reproduce the reference tokenizer's IDs
(**372,811 tokens before EOS**); Unicode and literal-control-string probes pass.
[Tokenizer audit](experiments/2026-09-19-qwen-tokenizer-audit.json).

`scripts/prepare_recovery_data.py` creates a separate development corpus with
**19,560 training documents** after removing the nine known lexical overlaps, and
**372 validation documents** after removing the four reused donor diagnostics.
Original R04 corpus hashes are checked and its files stay unchanged. This remains
development validation previously used by R04, not an untouched final benchmark;
seven requested benchmark sources and semantic contamination remain unaudited.
[Recovery data manifest](experiments/2026-09-19-qwen-recovery-data.json).

The paired attention initialization is now serialized and audited: **360,087,809
parameters**, 95.6308% copied/averaged, all finite, with exact save/reload. Its recovery
configuration changes only the four core mixers from GDN to full attention; outer
SWA/NoPE, input reinjection and k=5 stay identical. All **111 common backbone tensors**
are bit-identical to the hybrid. This is distinct from the earlier all-RoPE prefix
diagnostic. Auxiliary heads are disabled in recovery and excluded from that equality
claim. [Attention audit](experiments/2026-09-19-qwen-attention-conversion.json),
[pair audit](experiments/2026-09-19-qwen-recovery-initialization-pair.json).

The original artifacts also have serialization-independent identities covering all
tensor names, shapes, dtypes and bytes plus a separate canonical configuration hash:
[hybrid identity](experiments/2026-09-19-qwen-hybrid-identity.json) and
[attention identity](experiments/2026-09-19-qwen-attention-identity.json).
`scripts/audit_initialization_identity.py` checks the conversion audit before hashing
the restricted-loaded state directly, without allocating a second model. This allows
a Colab reconstruction to be checked against the exact local tensors even when
PyTorch archive metadata differs. Cross-host identity is not assumed from the seed.

`scripts/eval_qwen_recovery.py` now scores the unchanged donor and either audited
initialization over the complete prepared development split. It shares the exact
tokenizer, document/window policy and CE/BPB implementation with the recovery
runner. At the planned sequence length of 512, every target after the first token
is scored exactly once, including EOS with zero payload bytes. The source weights,
tokenizer, initialization and development split must match their audits. Reports
record individual document sums, runtime precision and artifact hashes. The CLI is
tested against independent unpadded, window-by-window losses from a real miniature
Qwen donor and Prophet initialization, including Unicode and empty documents.

The Colab CPU evaluation of the unchanged donor has completed on all **372
development documents**, at sequence length 512 / batch one: **3.157755304
nats/token** and **0.969985530 bits/byte**, scoring 366,762 targets and 1,722,551
payload bytes. All document identities, target/byte counts and loss aggregates
were independently checked after download. The run took 3,390.19 seconds, including
tokenization and CE, with two CPU threads, Torch 2.11.0+cu128 and Transformers 5.17.0.
[Donor report and runtime](experiments/2026-09-19-qwen-colab-development/donor.json).
The queue has moved to the attention initialization; its result and the GDN
initialization result remain pending.
These reports must use the same sequence length and precision for a paired
comparison; the previous four-prefix numbers are not their recovery baselines.
The R04 scores use a different tokenizer, document set and window length, so the
donor's CE/BPB here must not be presented as a controlled comparison against R04.

The independent Windows CPU donor run has also completed: **3.157732405 nats/token**
and **0.969978496 bits/byte**, in 5,042.85 seconds under Torch 2.14.0+cpu and
Transformers 5.17.0, two threads. All 372 document identities and denominators match
Colab, as do the source model, configuration and tokenizer hashes. The local-minus-
Colab CE difference is -0.000022899; the maximum absolute per-document CE difference
is 0.000037289. The two runtime results are retained separately, not treated as
bit-identical or pooled. The data-audit file's byte hash differs because the local
Git working copy uses CRLF and Colab uses the committed LF blob; each matches its
own artifact, and the corpus bytes match exactly. Recovery comparisons will use
the Colab donor baseline measured in the students' runtime.
[Local donor report](experiments/2026-09-19-qwen-development-donor.json).

### Colab reconstruction is a separate initialization pair

An isolated checkout and virtual environment now hold the pinned donor and the exact
prepared development corpus. R04's checkout and dependency versions were checked
before/after setup and remained unchanged. Recovery uses Transformers 5.17.0 and
tokenizers 0.23.2 in its own venv; the shared system Torch remains 2.11.0+cu128.
All preparation and queued baseline evaluation processes hide CUDA explicitly.

The attempted seed-zero reconstruction **failed exact cross-host tensor identity**.
The hybrid's embedding, prelude, coda, output norm and LM head match the Windows
artifacts exactly, while core and auxiliary-head group hashes differ. Both config
hashes match their local counterparts. This does not diagnose which runtime/platform
difference caused the mismatch, and the original initialization was not overwritten.
A small transfer bundle of the original fresh tensors was prepared locally, but the
browser file chooser did not complete and the Drive connector was not connected.
No bundle was uploaded; the notebook upload was cancelled.

The Colab-generated artifacts therefore define a **separate pair**, each with a
successful finite-parameter and exact save/reload audit. Their own pair audit again
checks all 111 shared backbone tensors exactly. Their archive SHA-256 values are:

- GDN: `1b440d1ab490dfc3263a3fb37e1e9ccbde9b72db0c87b4efed4a5e68de0574c8`.
- Attention: `013f6845b9e084aaa7fa1f740878ce65dd9b8c2dd40c3368784ff2f0ee89cfa4`.

All three CPU baselines use the same Colab runtime before comparing recovery;
the donor has completed and the two initialized students remain pending.
The local candidate's prefix/cache reports are not measurements of these new weights;
the real-size GPU and cache gates must be run for the actual training initialization.
The new weights are on local disk and now also have a verified Drive snapshot,
including file-by-file checks after flushing and remounting (persistence proof below).
[Conversion, pair, cross-host failure and identity evidence](experiments/2026-09-19-qwen-colab-initialization/colab-pair-manifest.json).

Before budgeted recovery, measure actual A100 forward/backward memory and kernel
agreement, and freeze equal-token
CE/KL comparisons. The command requires explicit learning rates and a fixed total
schedule. Checkpoints must be snapshotted and remotely verified as in R04.

`scripts/gate_qwen_recovery.py` prepares that measurement; it has not yet been run
on the A100. It verifies the source, initialization and prepared training-file
hashes before checking full-model training logits and parameter gradients against
the sequential GDN reference. Both passes use the same random seed, fixed depth,
full backpropagation and activation checkpointing on a 67-token training prefix.
The existing FP32 relative-L2 bound of 0.002 and BF16 bound of 0.03 are retained;
FP32 logits additionally require maximum absolute error <=0.002. The attention-only
arm reports the GDN comparison as inapplicable. CPU harness tests deliberately
corrupt a candidate backward to verify failure detection; they do not exercise FLA.

After those checks, the gate executes two warmup and three measured updates with
the actual recovery Trainer, using a 128-step diagnostic schedule and explicit
learning rates. CUDA synchronization brackets the measured updates. The report
includes timings, peak allocated CUDA memory, processed tokens, finite model and
optimizer state, skipped steps, and the frozen teacher's unchanged tensor hash for
KL. The normal recovery execution policy remains intact, including auxiliary-head
computation despite zero auxiliary-loss weights. Timings include data loading and
teacher execution, exclude checkpointing and evaluation, and discard updated
weights. These short probes establish neither sustained training stability nor
cached-decoding equivalence. A separate CUDA test compares the custom KL backward
with dense PyTorch KL in FP32 and BF16; its execution is still pending alongside
the real-size gate. R04 retains the GPU until its current queue has completed.

A separate Colab checkout at `4490d66` is staged under
`/content/prophet-recovery/gpu-repo` for those checks. The gate, recovery and
evaluation command entry points all load successfully with CUDA hidden. The active
R04 checkout remains at `e5720d0`, and the CPU baseline checkout remains at `8ad19d3`.
This is code preparation only: no GPU gate or recovery update has run yet. Before
comparing CUDA recovery scores, evaluate the unchanged donor and both exact
initializations in that same BF16-autocast evaluation runtime; the CPU FP32 reports
above do not substitute for those matched pre-recovery references.

## Real-size cached decoding: strict gate remains open

The hybrid was checked on the first diagnostic prefix, with 128 positions, fixed
k=5, batch one and CPU float32. Cached logits were compared against a full forward
using the predeclared elementwise tolerance `1e-4 + 1e-4 * abs(reference)`.

| Model and GDN scan | Prefill 64 + 63, then one token: max error | 128 single-token calls: max error | Strict tokenwise gate |
|---|---:|---:|---|
| Hybrid, chunk 64 | 0.000069380 | 0.001464784 | Fail |
| Hybrid, sequential reference | 0.000071049 | 0.001300752 | Fail |
| Unchanged donor control | 0.000021935 | 0.000083447 | Pass |

All six paths have finite logits and identical argmax predictions at all 128
positions. Both hybrid prefill/decode paths pass the strict tolerance, but the
tokenwise paths exceed it by factors of 10.31 and 8.73. The tolerance was not relaxed.
The sequential reference also fails, so replacing only the blockwise GDN scan does
not resolve the discrepancy. This is an unresolved numerical gate for this real
initialization, not evidence of a wrong prediction on these 128 positions or a
diagnosis of the underlying cause.

Forward-hook tracing reproduces the original failure exactly. Maximum relative L2
hidden-state error across tokenwise calls grows from 1.89e-6 after the first prelude
block to 1.96e-5 after the final core pass and 1.16e-4 after the last coda block.
Differences are already present before GDN; the trace is consistent with amplified
floating-point differences but does not prove that this is the sole cause. The
existing small-model CI tests and separate R04 A100 checks do not substitute for
this real-size gate. No production numerical behavior was changed based on it.

The actual hybrid cache holds **52,297,728 bytes** at this setting: 8,388,608 attention
bytes and 43,909,120 recurrent/short-convolution bytes in 28 slots. This excludes
weights, logits and workspaces, uses FP32, and does not measure device latency.
Bounded recurrent state is not necessarily only a few kilobytes.
[Original audit](experiments/2026-09-19-qwen-cache-audit.json),
[sequential reference](experiments/2026-09-19-qwen-cache-reference-audit.json),
[donor control](experiments/2026-09-19-qwen-cache-donor-control.json),
[layer trace](experiments/2026-09-19-qwen-cache-trace.json).

### Repeat on the separate Colab initialization

The actual Colab hybrid (tensor-state SHA256 `4720d2cb...`) was checked afresh on
the same 128-token diagnostic prefix, using CPU float32, PyTorch 2.11.0+cu128 and
Transformers 5.17.0 with CUDA hidden. The separate Colab initialization is not
tensor-identical to the local one; the earlier local numbers cannot certify it.
The input document and token hashes match the original audit, and tolerances are
unchanged.

| Colab model and scan | Prefill 64 + 63 + 1: max error | Tokenwise: max error | Strict gates |
|---|---:|---:|---|
| Hybrid, chunk 64 | 0.000219822 | 0.001132488 | Both fail |
| Hybrid, sequential reference | 0.000203729 | 0.001148224 | Both fail |
| Unchanged donor | 0.000035286 | 0.000111580 | Both pass |

All six paths remain finite and agree on all 128 argmax predictions. The hybrid's
maximum elementwise tolerance ratios are 1.55/8.91 for prefill/tokenwise with the
chunked scan and 1.61/8.56 with the sequential scan; the donor stays below one.
The trace again finds differences before GDN, with relative L2 error rising from
1.78e-6 after the first prelude block to 1.71e-5 after the last core pass and
6.62e-5 after the last coda block. These checks reproduce the unresolved numerical
limitation on the weights that would actually enter Colab recovery. They do not
establish a wrong token on this prefix, a root cause, or general cached-generation
quality. [Reports, runtime identity and logs](experiments/2026-09-19-qwen-colab-cache/summary.json).

The two Colab initializations and their metadata, 1,491,373,352 bytes in total,
were copied to `Prophet_AGI/R04/donor-recovery/colab-initializations-seed0` and
verified file by file against the source SHA256 digests. Drive was subsequently
flushed and remounted alongside the shared R04 step-4,096 snapshot. All nine
initialization files still matched their source sizes and SHA256 digests.
[Post-remount verification](experiments/2026-09-19-r04-step4096/loop-persistence.json).
The earlier copy manifest retains its historical pre-flush scope:
[Mounted-copy manifest](experiments/2026-09-19-qwen-colab-cache/initialization-snapshot.json).

### Attention and reduced-depth controls

The audit now derives the core identity from the loaded configuration and accepts
an explicit fixed depth. Three additional checks use the exact Colab pair, prefix,
CPU FP32 runtime and unchanged elementwise tolerance. The analysis script is pinned
to `5c117d1`; the model checkout remains at `8ad19d3` while its baseline queue runs.

| Initialization and depth | Prefill max error / tolerance ratio | Tokenwise max error / tolerance ratio | Tokenwise argmax matches |
|---|---:|---:|---:|
| Attention, k=5 | 0.000029564 / 0.234 (pass) | 0.000165224 / 1.073 (fail) | 128/128 |
| Attention, k=1 | 0.000058174 / 0.466 (pass) | 0.000646591 / 4.997 (fail) | 128/128 |
| GDN, k=1 | 0.000311971 / 2.044 (fail) | 0.001334071 / 8.885 (fail) | 127/128 |

All logits remain finite. **GDN is not required for a strict tokenwise failure**:
the attention control also fails. Reducing the depth to one does not remove the
failure, and the GDN depth-one diagnostic changes one argmax prediction. That last
observation concerns the depth-one control, not the planned k=5 recovery candidate.
The results reject a GDN-only explanation and a simple reduction of loop count as
a numerical fix. They do not isolate the cause or establish a monotonic relationship
between depth and error. Production execution and tolerance remain unchanged.
[Reports, block traces and execution records](experiments/2026-09-19-qwen-colab-cache-controls/summary.json).

Three CLI regression cases cover both core types, explicit depth and preservation
of a deliberately corrupted cached-output failure report. The focused recovery and
model tests pass (68 cases); CI at `5c117d1` passes 748 cases with 10 CUDA-only skips.
The cache-control and donor evidence ZIP is 39,499 bytes, SHA256
`3c39c3802fc0cdb758c0215422ebcb064e4f08b9508c65e1a26c19b19b6cab2b`.

### FP64 attention oracle isolates a precision-dependent discrepancy

The attention-only pair member was rerun on the same Colab CPU and prefix with
double-precision weights, attention, residual arithmetic and RMS reductions.
Rotary values retain the production FP32 construction. This diagnostic changes
only its private model instance; production code, weights on disk and the FP32
acceptance tolerance remain unchanged. GDN is explicitly rejected by this attention-only mode
because its internal recurrence and gates otherwise retain FP32 arithmetic.

The oracle uses the stricter threshold `1e-8 + 1e-8 * abs(reference)`:

| Depth | Prefill/decode maximum logit error | Tokenwise maximum logit error | Result |
|---|---:|---:|---|
| k=5 | 4.885e-14 | 2.309e-13 | Both pass; 128/128 argmax matches |
| k=1 | 9.059e-14 | 1.382e-12 | Both pass; 128/128 argmax matches |

The tokenwise FP32 errors for these same weights were 1.652e-4 and 6.466e-4.
This is strong evidence of a precision-dependent discrepancy in the attention
candidate on this prefix, rather than an indexing/state mismatch that persists in
double arithmetic. It does not isolate which FP32 operation dominates, validate
other prefixes or GDN, or make double precision a practical deployment solution.
The normal FP32 cache gate remains failed. Full-forward recovery training has its
own pending kernel/gradient and memory gate; any later recovery checkpoint still
needs fresh cached-decoding checks before adoption.

The analysis script is pinned to `4490d66`; model imports remain at `8ad19d3`.
Both processes exited successfully. Download verification matched the weights,
input, runtime and script identities and confirmed doubled attention-cache bytes.
[Oracle reports and block traces](experiments/2026-09-19-qwen-colab-attention-fp64/summary.json).
The evidence ZIP is 11,235 bytes, SHA256
`70b4f63e6989505e4c7552c21df820d82f606c07d19082fd61bc91455a6b53f7`.
The focused suite now passes 71 cases; CI at `4490d66` passes 751 with 10 CUDA skips.
Tests include an independent double RMS formula, exact preservation of initialized
parameter values, rejection of GDN, and deliberate cached-output corruption that
must still fail the stricter oracle.

### Separate FP64 GDN recurrence also passes

Revision `3c141c9` adds an explicit hybrid diagnostic rather than routing GDN through
its production FP32 internals. Projections, short convolution, key normalization,
gates, sequential rank-one state updates and output normalization use FP64. The
production rotary construction stays FP32. A separate test checks the recurrence
against the dense transition-matrix equation with a nonempty recurrent and
convolution cache, including chunk boundaries and the final state. CLI tests also
inject a cached-logit error that must fail and leave its report intact.

On the same actual Colab hybrid and original 128-token prefix:

| Depth | Prefill/decode maximum logit error | Tokenwise maximum logit error | Result |
|---|---:|---:|---|
| k=5 | 1.454e-13 | 3.534e-12 | Both pass; 128/128 argmax matches |
| k=1 | 2.673e-13 | 2.270e-12 | Both pass; 128/128 argmax matches |

The tolerance remains the stricter `1e-8 + 1e-8 * abs(reference)` for this diagnostic.
The k=5 FP32 sequential control previously failed at 0.001148224 tokenwise error,
so the disappearance cannot be attributed merely to replacing the chunked scan.
The FP32 depth-one control had one argmax disagreement; the two FP64 execution
paths now agree at every position. No comparison of FP32 against FP64 argmax
predictions or semantic correctness is implied.

Together with the attention controls, this supports precision-dependent errors in
both initializations on this prefix. It does not isolate a single offending FP32
operation or clear the production FP32 gate. The hybrid oracle takes 244.52 seconds
at k=5 and 78.11 seconds at k=1 on this CPU, including setup and both cache paths;
these are diagnostic runtimes, not a deployment speed measurement.
[Hybrid oracle reports and traces](experiments/2026-09-19-qwen-colab-hybrid-fp64/summary.json).
The downloaded ZIP is 12,102 bytes, SHA256
`758d19bca831a6992ac863e88a778deb0ce13556dbdd77ce25361547bd3f7c45`.
Its script, model and input identities were independently checked, as were the
finite results and doubled attention/recurrent cache bytes. The focused suite
passes 74 tests; CI at `3c141c9` passes 754 with 10 CUDA-only skips.

## Reproduction and stopped attempts

The first two rehearsal-harness attempts stopped before writing a checkpoint. A
strict analytical-count assertion failed; a later identity assertion exposed that
`load_state_dict(assign=True)` created separate Parameter objects for tied weights.
The successful harness uses ordinary copying into the live model and records the
then-unresolved 3,200-parameter estimate difference, reconciled above. The existing production converter
already used ordinary loading; the assignment issue was in this new rehearsal.

```bash
python scripts/audit_qwen_blocks.py --source data/donor-qwen3-0.6b/source \
  --out /tmp/qwen-blocks.json
python scripts/rehearse_qwen_conversion.py --source data/donor-qwen3-0.6b/source \
  --out /tmp/qwen-initialization.pt --report /tmp/qwen-conversion.json
python scripts/smoke_qwen_conversion.py --source data/donor-qwen3-0.6b/source \
  --checkpoint /tmp/qwen-initialization.pt --audit /tmp/qwen-conversion.json \
  --validation data/fineweb-pilot-v1/validation --out /tmp/qwen-smoke.json
python scripts/audit_qwen_stack.py --source data/donor-qwen3-0.6b/source \
  --validation data/fineweb-pilot-v1/validation --reference /tmp/qwen-smoke.json \
  --out /tmp/qwen-stack.json
python scripts/ablate_qwen_initialization.py --source data/donor-qwen3-0.6b/source \
  --checkpoint /tmp/qwen-initialization.pt --audit /tmp/qwen-conversion.json \
  --validation data/fineweb-pilot-v1/validation --reference /tmp/qwen-smoke.json \
  --out /tmp/qwen-initialization-ablation.json
python scripts/audit_qwen_cache.py --source data/donor-qwen3-0.6b/source \
  --checkpoint /tmp/qwen-initialization.pt --audit /tmp/qwen-conversion.json \
  --validation data/fineweb-pilot-v1/validation --reference /tmp/qwen-smoke.json \
  --out /tmp/qwen-cache.json
python scripts/audit_donor_tokenizer.py --source data/donor-qwen3-0.6b/source \
  --validation data/fineweb-pilot-v1/validation --out /tmp/qwen-tokenizer.json
python scripts/prepare_recovery_data.py --corpus data/fineweb-pilot-v1 \
  --overlap docs/experiments/2026-09-19-pilot-benchmark-overlap.json \
  --smoke docs/experiments/2026-09-19-qwen-conversion-smoke.json \
  --out /tmp/qwen-recovery-data
python scripts/recover_qwen.py --help
python scripts/eval_qwen_recovery.py --source data/donor-qwen3-0.6b/source \
  --validation data/qwen-recovery-v1/validation.jsonl \
  --data-audit docs/experiments/2026-09-19-qwen-recovery-data.json \
  --arm donor --device cpu --seq-len 512 --batch-size 1 --out /tmp/qwen-development-donor.json
# For a converted arm, use --arm initialization --initialization <weights.pt> --audit <conversion.json>.
python scripts/rehearse_qwen_conversion.py --source data/donor-qwen3-0.6b/source \
  --core-mixer full_attn --out /tmp/qwen-attention.pt --report /tmp/qwen-attention.json
python scripts/audit_recovery_pair.py --hybrid /tmp/qwen-initialization.pt \
  --attention /tmp/qwen-attention.pt --out /tmp/qwen-pair.json
```

The hybrid cache audit currently writes its failure report and exits nonzero.
Use a new output path with `--reference-scan`, `--donor-control`, or `--trace-blocks`
to reproduce the additional controls.
For the attention initialization, `--attention-fp64-oracle --loop-k 5` (or `1`)
selects the diagnostic double-precision path, not the production acceptance gate.
For the hybrid, use `--hybrid-fp64-oracle` for the separately checked sequential
double-precision recurrence; it cannot be combined with `--reference-scan`.

The local dependency versions were Transformers 5.17.0, safetensors 0.8.0 and
tokenizers 0.23.2. CPU and recovery test results are reported with their execution environment in the PR.
The separate R04 A100 runtime and its training implementation remain unchanged.
