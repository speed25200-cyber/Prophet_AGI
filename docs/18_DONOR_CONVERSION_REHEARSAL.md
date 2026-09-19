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

The next conversion experiment must separate weight sharing from replacement by
GDN, with an unchanged donor baseline and recovery measurements on the same held-out
targets. The current initialization is an experimental artifact, not an adopted
Prophet-main model.

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
```

The local dependency versions were Transformers 5.17.0, safetensors 0.8.0 and
tokenizers 0.23.2. The complete CPU suite passes: **693 tests, eight CUDA skips**.
The separate R04 A100 runtime and its training implementation remain unchanged.
