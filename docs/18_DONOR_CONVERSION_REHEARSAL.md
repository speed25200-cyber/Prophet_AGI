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
The analytical count is 3,200 parameters below the actual model; the report retains
that difference and uses the actual count for coverage. Embeddings occupy about
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

Before budgeted recovery, measure actual A100 forward/backward memory and kernel
agreement, and freeze equal-token
CE/KL comparisons. The command requires explicit learning rates and a fixed total
schedule. Checkpoints must be snapshotted and remotely verified as in R04.

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

## Reproduction and stopped attempts

The first two rehearsal-harness attempts stopped before writing a checkpoint. A
strict analytical-count assertion failed; a later identity assertion exposed that
`load_state_dict(assign=True)` created separate Parameter objects for tied weights.
The successful harness uses ordinary copying into the live model and records the
remaining 3,200-parameter estimate difference. The existing production converter
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
python scripts/rehearse_qwen_conversion.py --source data/donor-qwen3-0.6b/source \
  --core-mixer full_attn --out /tmp/qwen-attention.pt --report /tmp/qwen-attention.json
python scripts/audit_recovery_pair.py --hybrid /tmp/qwen-initialization.pt \
  --attention /tmp/qwen-attention.pt --out /tmp/qwen-pair.json
```

The hybrid cache audit currently writes its failure report and exits nonzero.
Use a new output path with `--reference-scan`, `--donor-control`, or `--trace-blocks`
to reproduce the additional controls.

The local dependency versions were Transformers 5.17.0, safetensors 0.8.0 and
tokenizers 0.23.2. CPU and recovery test results are reported with their execution environment in the PR.
The separate R04 A100 runtime and its training implementation remain unchanged.
