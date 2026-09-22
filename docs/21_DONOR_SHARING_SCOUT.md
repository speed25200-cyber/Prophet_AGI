# 21 — Diagnose donor damage before another recovery run

The four completed recovery arms improve language loss but retain only
32.5–35.1% raw ARC-Easy accuracy, against 60.9% for the unchanged donor
([complete capability evaluation](20_RECOVERY_CAPABILITY_EVAL.md)). This motivates
isolating conversion damage before paying for a larger recovery run. None of the
interventions below is adopted into the Prophet model or configuration.

## Frozen inputs and scope

These are **untrained initialization diagnostics on development text**, not a new
capability benchmark or evidence of assistant quality. They use the same pinned
Qwen3-0.6B donor (`c1899de…`, weights SHA256 `f47f7117…`), raw tokenizer and frozen
recovery development corpus as the earlier experiment. Full identities and exact
executed scripts are in the [evidence directory](experiments/2026-09-20-qwen-sharing-scout/archive-verification.json).

Both forward probes select the first 16 distinct eligible document SHA256 values
in ascending order, take their first 512 tokens without EOS, and score all 511
next-token targets per document: **8,176 targets**. This choice precedes scoring.
Every variant has identical document/token hashes and target counts. Both separate
donor evaluations return exactly **3.126083612908589 nats/token**.

All model forwards use the original native Qwen implementation, CPU FP32, SDPA,
two CPU threads, no cache and no training. Original embeddings, tied output head,
normalization, positions and outer eight layers remain intact, except that the
pruning control removes eight middle layers. Thus these probes isolate weight
sharing and cannot assess GDN, reinjection, useful variable depth or Prophet's
auxiliary heads. **Do not compare these prefix losses to the full 372-document
recovery losses.**

## Depth-specific residuals

For each of the seven linear projections, four shared bases average donor layers
`4+slot, 8+slot, 12+slot, 16+slot, 20+slot`. Each of the twenty middle depths keeps
its own original norms and a rank-limited SVD approximation to its difference from
the shared base. This explores the depth-specific residual idea discussed in
[Looped Transformers](https://arxiv.org/html/2410.20672v3); it does not reproduce
that paper's uptraining protocol.

Before the forwards, an exact singular-value diagnostic covers all 140 residual
matrices. Rank 64 retains 22.05% of their combined squared Frobenius energy, rank
128 retains 36.61%, and rank 256 retains 57.74%. Cyclic means have less weight
reconstruction error than the previous contiguous-group means at the same serial
positions: the latter's total error is **1.302434 times** the former's. Neither
weight-space statistic proves a language-quality gain.

The ranks `[0, 64, 128, 256]` were fixed before any score:

| Native serial model | Active parameter accounting | Prefix nats/token, lower is better |
|---|---:|---:|
| Unchanged donor | 596,049,920 | 3.126084 |
| Cyclic shared means, rank 0 | 344,391,680 | 13.610855 |
| Depth residual rank 64 | 373,227,520 | 15.794966 |
| Depth residual rank 128 | 402,063,360 | 15.292327 |
| Depth residual rank 256 | 459,735,040 | 14.006550 |

All residual factors remain allocated at rank 256 during the sweep. Smaller-rank
counts are active-factor arithmetic, **not measured resident-memory savings**.
The actual registered resident parameter count is 459,735,040 for this sweep.
Original layer order and norms do not rescue this initialization. Reducing a
matrix approximation error does not guarantee a monotonic improvement in language
loss; all three nonzero residual ranks score worse than the shared-mean control.
This rejects an immediate expensive recovery run using these initializations,
not every trained residual or activation-aware compression method.

Analytic tests check singular directions, shared-base gradient accumulation and
rectangular full-rank reconstruction. An additional native 28-layer miniature Qwen
test applies the actual replacement function at full rank and recovers the original
complete forward within `atol=rtol=2e-5`; setting its residuals to rank zero changes
the logits. This supports the wiring and arithmetic, not real-model capability.

## Selective sharing and a pruning control

A second fixed protocol evaluates three interventions, all under 500M registered
parameters. It restores the exact original module objects between variants:

| Intervention | Registered unique parameters | Prefix nats/token |
|---|---:|---:|
| Share only middle MLP projections across the same cyclic groups | 445,054,976 | 8.862508 |
| Share only middle attention projections across the same cyclic groups | 495,386,624 | 7.688381 |
| Remove original layers 12–19, keep 0–11 and 20–27 | 470,202,368 | **5.188328** |

The donor is 3.126084 on these same inputs. Restricting which projections are
shared reduces damage relative to complete cyclic sharing, but **even simply
removing eight middle layers damages the donor less on this scout**. That control
does not meet Prophet's adjustable recurrent-depth objective and is not a substitute
for it. It establishes a baseline that a proposed sharing mechanism should beat.
Original donor modules remain alive for restoration, so parameter counts again
do not constitute process-memory or deployment measurements.

Tests verify that sharing leaves the unselected projection family, individual
norms and outer layers untouched, and that both sharing and pruning restore original
objects and attention indices even after an exception. Actual model runs verify
the declared parameter counts and restoration identities between variants.

## Budget and next decision

The spectrum diagnostic executed `prophet.budget`, `scripts/design_search.py` and
`prophet.plan` with a hypothetical six-A100-hour horizon before the forward probes.
Their exact outputs are archived. This is a planning scenario, not a claim about
the user's remaining credit or an allocation of six hours. The original large
from-scratch search has **zero feasible candidates** at this horizon.

The budget output describes the existing 360,087,809-parameter Prophet attention
configuration. Its hypothetical residual extension also counts separate norm
residuals and retains auxiliary heads. The native forward scout omits those heads
and directly retains original per-depth norms; therefore its parameter totals
above are deliberately different. Extension FLOPs/weight/state bytes are explicit
arithmetic, not measured activation, optimizer or device-memory consumption.

The local-sharing follow-up is now complete. Ten neighboring pairs `(4,5)` through
`(22,23)` share their seven linear projections, retaining all original norms and
outer layers. Three initializations were fixed before scoring, each with
**438,763,520** registered parameters and the same 8,176 development targets:

| Adjacent-pair initialization | Prefix nats/token |
|---|---:|
| Mean of the pair | 12.213892 |
| First layer's projections reused | 12.330183 |
| Last layer's projections reused | 10.774335 |

The separately rerun donor is again exactly 3.126084. All three local-sharing
initializations remain worse than the previous pruning control (5.188328), despite
keeping 28 executed layers. The negative result rules out allocating a recovery
run simply because layers are adjacent or because averaging was avoided.
[Exact script and reports](experiments/2026-09-20-qwen-adjacent-and-calibration-plan/archive-verification.json).

The next experiment fits shared weights against **training activations**, with a
fixed local ridge objective and independent development evaluation
([protocol](22_ACTIVATION_WEIGHTED_SHARING.md)). Any selected recurrent candidate
still needs an implemented reversible
configuration, exact cache and resume gates, a matched 50–500M recovery ablation,
capability evaluation, multiple training seeds and actual target-device evidence.
No current candidate has passed those adoption requirements.

## Reproduction

```bash
python scripts/probe_qwen_relaxed.py --source <pinned-source> \
  --validation <frozen-development.jsonl> --out <new-output-directory>
python scripts/probe_qwen_selective.py --source <pinned-source> \
  --validation <frozen-development.jsonl> --out <another-new-output-directory>
```

The archived executed sources are exact copies of the local scripts used. The
archive verifier checks every artifact hash and recomputes prefix totals and input
identity equality. It does not claim a separate repeat of each model forward.
