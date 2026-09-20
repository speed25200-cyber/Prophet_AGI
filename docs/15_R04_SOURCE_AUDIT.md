# R04: scope of the evidence and depth evaluation

Primary sources were re-read on 2026-09-19 before extending the real-text pilot.
The current training configurations and 4,096-step schedule remain frozen.

## Scale is a protocol choice, not a theorem

MoR's tables distinguish the **base model** size from stored non-embedding
parameters and vary routing, sharing, tokens and FLOPs. Table 9's 360M-base
recursive arm has 118M non-embedding parameters and NLL 3.4864 versus vanilla's
3.3785 at 2e18 FLOPs. Table 10's 135M-base Cycle-2 arm has NLL 3.0071 versus
vanilla's 3.0323 at 10B tokens. These settings do not imply a universal
350M-stored-parameter threshold. The current 374.7M arm remains our chosen test
scale; changing it mid-run would confound the experiment.
[Source: MoR v3, Tables 9–10](https://arxiv.org/html/2507.10524v3).

## A depth dial needs its own measurement

Huginn trains over a heavy-tailed log-normal-Poisson distribution of recurrence
counts, using truncated gradients through the last eight iterations. Our paired
pilot trains at fixed k=4 with full gradients. Extrapolation to k=8 therefore tests
this fixed-depth recipe; a failure would not refute variable-depth training.
[Source: Huginn v2, §3.3](https://arxiv.org/html/2502.05171v2).

DeepLoop changes residual scaling in a Post-LN setting and studies alignment of
repeated parameter visits. Its paper explicitly limits the worst-case alignment
claim. Applying its exponent to Prophet's current normalization without a separate
ablation would change the hypothesis under test. It remains a candidate follow-up.
[Source: DeepLoop v2, §3](https://arxiv.org/html/2607.13491v2).

## Measurement contract

After a completed shared checkpoint, score the full held-out corpus at k=4 first,
then k=1, 2 and 8 with identical weights, vocabulary and context. Verify the k=4
result against the checkpoint's saved validation. Record each depth's CE/BPB,
document scores, evaluation time and peak allocated GPU memory. A shallower pass
changes executed depth to 2+4k+2 blocks. Timings include evaluation I/O and any
compilation; they do not measure autoregressive decoding latency.

This is a language-loss sensitivity probe. Reasoning accuracy, an adaptive stopping
policy, and quality on the target consumer devices each require separate evidence.
No model configuration is adopted by this source audit or by the short pilot.

## Checkpoint identity at periodic boundaries

A periodic checkpoint followed by a session-end checkpoint can create two entries
with the same step. The previous loader preferred the earlier save on a tie; the
loader now prefers the last save, retaining checksum fallback and highest-step
priority. A regression test reproduced the defect before the fix. The existing
128/160-step reports are unaffected.

The frozen R04 training checkout remains unchanged. The depth driver therefore
selects the exact slot and SHA recorded by the published validation, verifies its
manifest membership, size and checksum, and uses restricted tensor loading. It
also checks the original source, runtime and corpus. Standalone evaluation memory
excludes training optimizer state and gradients.
