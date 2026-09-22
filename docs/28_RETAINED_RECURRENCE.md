# Retained-donor recurrence: preserve before adapting

Status: experimental implementation, CPU tests and full-donor CPU gate pass;
not an adopted architecture.
The previous negative screens stand. This candidate retains all donor parameters
and tests a different starting condition: the original computation at one loop.
It does not claim compression, useful additional depth, bounded context, memory,
agent capability or a finished assistant.

## Why this candidate

The native Qwen3 sharing, pruning and calibration scouts damaged language quality
before any recovery training. The four recovery runs did not recover the donor's
capability. Removing that initial damage is a concrete next constraint, rather
than allocating another recovery run to an already poor initialization.

The new isolated module `prophet.convert.retained_recurrence` owns a native Qwen2
model. It leaves every original layer, embedding, norm, bias, rotary position and
output head intact. The first pass executes the original layer order without a
bridge. Later passes reuse the middle layers with a shared re-entry correction:

```
p = Prelude(tokens)
h = Core(p)
repeat k - 1 times:
    h = Core(h + StateDelta(h) + InputDelta(p))
logits = Head(Norm(Coda(h)))
```

Both delta matrices start at zero and remain separate optimizer parameters. No
zero scalar gate blocks their first gradient. This is not a claim that repeatedly
applying the unadapted core preserves quality at k > 1.

The default freezes the entire donor and trains only these two matrices. That
mode preserves the one-loop function even after bridge updates. Explicit
`train_core=True` additionally unfreezes the middle layers; it removes the
post-update guarantee and requires retention evaluation. Neither mode changes
the production Prophet model or default configurations.

Shapiro's retained-path study motivates this initialization and separate re-entry
projections, but uses different bridges, supervision and adaptation recipes; its
controlled-task results do not establish general language improvement here
([primary paper](https://arxiv.org/html/2608.11233v1)). McLeish et al.'s curriculum
and healing experiments operate at billions of training tokens, including a
52-billion-token two-phase run, not the scale of our 512-update screens
([primary paper](https://arxiv.org/html/2511.07384v1)). This module is an independent
experimental implementation, not a reproduction or adoption of either result.

## Frozen donor and arithmetic

Source: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
revision `7ae557604adf67be50417f59c2c2f167def9a775`, ungated Apache-2.0. Hub metadata
and every downloaded file's size/hash are checked. Safetensors weights:
988,097,824 bytes, SHA256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
Loading is local-only with `trust_remote_code=False`.

The 24-layer donor has width 896, 14 query heads, two KV heads of dimension 64,
MLP width 4,864 and vocabulary 151,936. The split is 6 prelude / 12 core / 6 coda.
Native meta-device accounting, counting tied parameters once, gives:

| Quantity | Count |
|---|---:|
| Donor parameters | 494,032,768 |
| Two bridge matrices | 1,605,632 |
| Total resident parameters | 495,638,400 |
| Trainable, bridge only | 1,605,632 |
| Trainable, core plus bridge | 180,554,240 |

This stays within the repository's 50–500M ablation range; it does not reduce
donor parameters. FP32 weights occupy 1,982,553,600 bytes. Bridge-only gradients
and two FP32 Adam moment buffers add 19,267,584 bytes, excluding optimizer steps,
activations, outputs, cache, allocator overhead and framework workspace.

Dense projection FLOPs per token, counting multiplication and addition separately,
are 987,922,432 at k1, 1,348,960,256 at k2 and 2,071,035,904 at k4. These include
the tied output projection and the repeated bridge; they exclude attention QK/AV,
normalization, biases, activation functions and backward. They are arithmetic,
not measured throughput or end-to-end cost ratios.

Before this candidate, `prophet.budget configs/prophet_500m_probe.json`,
`scripts/design_search.py --a100-hours 6 --top 2` and
`prophet.plan --a100-hours 6` were run. The original from-scratch search has no
feasible full-training candidate at that illustrative horizon. The budget report
is for an existing Prophet proxy, not this native model; the native counts above
are separate. Six hours and the tool's default 300-hour projection are not claims
of purchased credit. No GPU training allocation is made by this protocol.

## Cache and restart contracts

Each additional core pass has separate KV storage. First-pass prelude/core and
final coda occupy the original slots; repeated core passes use separate native
caches without mutating any attention module's layer index. Fixed-depth cached
decoding must equal full-prefix execution. Depth and batch size cannot change
inside a cache, caches cannot cross model instances, and an interrupted forward
poisons its partially advanced cache. Cached training is refused.

FP32 cache size at batch one and context 1,024 is 25,165,824 bytes at k1,
37,748,736 at k2 and 62,914,560 at k4. This explicitly grows with context and
depth. Bounded-context memory remains unsolved for this candidate. It must not be
confused with a no-cache full-sequence recurrence experiment.

State dictionaries bind the recurrence partition, native numerical/configuration
contract and parameter dtype. Changed contracts fail on loading. Miniature tests
cover exact 4 versus 1+3 optimizer updates; a real-size, separate-process restart
and CUDA gate are still required before GPU training.

## Preselected CPU gate

`scripts/gate_retained_recurrence.py` accepts only the pinned donor and verified
new train artifact. It selects the first four documents with at least 32 donor
tokens, before any scores. No new validation document is used. FP32, native SDPA,
two CPU threads and deterministic algorithms are fixed.

1. All four 32-token forwards must exactly match the native donor logits and argmax
   at k1 before training.
2. At k1/k2/k4, four 16-token sequences must agree between full execution and a
   12-token prefill followed by four one-token calls, at `atol=1e-4`, `rtol=1e-5`,
   with identical argmax and the exact cache-byte formula.
3. One bridge-only AdamW plumbing update on the first eight tokens must have finite,
   nonzero gradients in both matrices. Every original unique parameter hash must
   remain unchanged, and all initial k1 logits must still match exactly.

This is an identity/cache/update gate, not a trained quality comparison. It does
not establish that extra loops help. A future quality protocol must include
matched controls, untouched evaluation, retention and actual compute accounting.

## Completed full-donor gate

The frozen gate at `7e088f59e2db8f7401664d755a05470f0d642181` passes in
25.44 seconds after source/artifact auditing, including loading, forward checks,
cache cases and the single plumbing update. The actual unique counts match the
meta-device arithmetic: 495,638,400 resident and 1,605,632 trainable parameters.
Initial k1 logits are **bitwise identical** on all four preselected 32-token
prefixes. After the bridge-only update, every original unique parameter hash is
unchanged and all initial k1 logits remain bitwise identical.

| Loops | Maximum cached/full logit error | Argmax identical | Actual cache bytes, batch 4 x 16 |
|---:|---:|---|---:|
| 1 | 0.000028610230 | yes | 1,572,864 |
| 2 | 0.000024795532 | yes | 2,359,296 |
| 4 | 0.000020027161 | yes | 3,932,160 |

Both bridge matrices have finite, nonzero gradients in all 802,816 entries. Their
gradient norms are 417.0892 (state) and 414.6850 (input). This verifies an active
learning path, not a useful update or a stable multi-step training recipe.

Eighteen targeted miniature tests pass, covering exact native identity, learned
bridge/cache equivalence, left padding and explicit positions, causality, frozen
base retention, optional core updates, serialization/numerical-contract rejection,
cache ownership/depth/failure handling and exact optimizer interruption. The wider
modeling/conversion selection passed 113 tests before the additional numerical-
contract test was added and the targeted eighteen rerun.

The [full report](experiments/2026-09-20-retained-recurrence-cpu/gate.json) and
[verified donor manifest](experiments/2026-09-20-retained-recurrence-cpu/donor-manifest.json)
retain exact input and source identities without text or weights. The
[archive verification](experiments/2026-09-20-retained-recurrence-cpu/verification.json)
checks source hashes against frozen Git bytes, recorded checks and cache arithmetic;
it does not repeat model inference or reload tensors. The recorded cache checks
precede the bridge update; real-size post-update cache and interrupted restart
checks remain next gates. No CUDA or quality result is implied.
