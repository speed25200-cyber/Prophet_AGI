# R04: controlled learned input reinjection

Status: component and executable protocol prepared; no GPU training result yet.
This is an experimental option, disabled in all existing model configurations.

## Why this experiment

The [completed depth-policy comparison](23_R04_DEPTH_ADAPTATION.md) improves
robustness to loop count but fails its primary criterion: six loops remain
0.4312% worse than the variable-depth model's own four loops. The
[internal observations](24_R04_RECURRENCE_OBSERVATION.md) do not show all token
states collapsing into one direction and do not identify a causal defect.

The next hypothesis is that a learned combination of the current state and
original input can make later loops more useful than a fixed sum. It remains a
hypothesis: changes in normalization, supervision or the mixer might matter more.

The primary paper [Scaling by Thinking in Continuous Space, sections 3.2 and 4.3](https://arxiv.org/html/2502.05171v2)
uses a learned projection of concatenated state and input, together with particular
normalization and initialization choices. Its small-scale alternatives behave
similarly, while its larger experiments favor concatenation. This motivates a
controlled test here; it does not establish that reinjection causes our failure
or that the paper's results transfer to our size, GDN core or data budget.

## Exactly one optional component

Let `h` be the previous recurrent state and `e` the prelude output:

```text
fixed_sum:   core_input = h + e
learned_mix: core_input = h + e + W concat(h, e)
```

`W` is one shared, bias-free `2d -> d` matrix initialized to zero. The explicit
original sum preserves the initial computation. Creating the adapter after the
base model's initialization also preserves the base tensors under the same seed.
Training resets the random seed after model construction and parent loading.
No normalizer, cache layout, auxiliary head or mixer is changed.

The switch is `recurrent.input_adapter = "residual_linear"`; its default is
`"none"`. Validation requires recurrence and input injection. Old configurations
that omit the field keep their topology. Historical depth-plan configurations are
compared after resolving defaults, while checkpoint experiment identities remain
strict and prevent resuming an old run under changed source code.

At dimension 1,792, this adds 6,422,528 parameters: 374,689,648 versus 381,112,176.
The experiment therefore tests the added component, not parameter-matched or
FLOP-matched alternatives. The estimated forward FLOP increase at k4 is 2.66%.
The estimated k6 training memory is 7.69 versus 8.69 GiB; actual prior baseline
allocation was higher, so these estimates never replace the CUDA memory gate.

## Frozen comparison and compute limit

The [generated plan](experiments/2026-09-20-r04-reinjection-plan/protocol.json)
binds the original step-4,096 checkpoint, configurations, recipe and decision.
Its SHA256 is `1bd62f90d8c38f2fe61cc1bd6911b5f35279aca598ce96395b1d4627ee11d114`.
Budget, design-search and compute-plan output accompany the plan and its manifest.

Both arms start from that original checkpoint, not the completed depth adaptation.
They use the same loader cursor, seed 0, new optimizers, schedule and 512 updates
of 8 x 2,048 tokens. Both sample depths uniformly from two through six and propagate
gradients through every visited loop. Their actual sampled histories must match.
Each model sees 8,388,608 additional tokens, or 3.8877 cumulative corpus passes;
the four-pass ceiling remains enforced.

The incremental allowance is at most two A100-hours: 1,800 seconds for gates and
5,400 seconds for the paired continuation. This is a bounded additional R04 test,
not an allocation of the unfunded generic depth-ablation line or the whole
illustrative 300-hour project plan. Resource availability must be checked before
launch. There is no intermediate quality-based recipe selection or early stop.

Before continuation, require all of:

1. Exact initial hidden and full-logit equality at k4 and k6 on the next actual
   8 x 2,048 training batch under CUDA BF16 autocast.
2. Original full-validation k4 reproduced in both arms.
3. Three finite full-size k6 optimizer updates, peak allocated memory below 90%
   of actual device capacity.
4. In both arms, eight continuous updates exactly equal one plus seven updates
   in separate processes: every model/optimizer tensor, RNG, loader, contract,
   depth history and per-document evaluation.

The continuous prefixes are preselected for continuation. Training stops on any
failed gate, changed provenance, nonfinite update or exceeded deadline. The driver
uses the previously validated strict CUDA numerical policy and retains its identity
in all reports and checkpoints. CPU tests cannot establish the CUDA gates.

## Primary decision

All 376 original development documents are scored at k1/2/4/6/8. A pass requires
all four conditions:

- Learned k6 BPB is at least 0.5% better than its own k4.
- The paired 95% interval for learned CE(k6)-CE(k4) is entirely below zero.
- Learned k4 BPB is within 1% of control k4.
- Learned k6 BPB is better than control k6.

The bootstrap uses 10,000 paired whole-document draws, PCG64 seed 0, linear
quantiles and token-weighted CE. The program rejects different experiment
identities, depth histories, document rows or aggregate scores. It reuses the
audited statistical calculation without relabeling old depth-policy runs.

Even a pass only justifies confirmation with new seeds, held-out data and actual
capability tasks. Development data has already guided R&D and known benchmark
overlaps remain. Neither this screen nor prediction loss demonstrates reasoning,
an assistant, AGI or readiness for architectural adoption.

## Commands

The full local suite passes **858 tests with 21 skips**. Component tests cover
unchanged initial predictions/base gradients, an effective adapter update, learned
full-versus-cached decoding, checkpointed backward, old-config compatibility,
budget accounting and both miniature CLI interrupted restarts. The new screen
rejects unequal paired depth histories and mislabeled old experiments. CUDA cases
remain pending; neither CPU restart equivalence nor these test counts establish
full-size GPU determinism or scientific improvement.

```bash
python scripts/gate_r04_input_adapter.py --parent-run PARENT --corpus CORPUS --out equality.json
python scripts/adapt_r04_reinjection.py --parent-run PARENT --corpus CORPUS --out FRESH --arm learned_mix --mode preflight
python scripts/adapt_r04_reinjection.py --parent-run PARENT --corpus CORPUS --out RUN --arm learned_mix --max-session-steps 8
python scripts/audit_r04_restart.py --left CONTINUOUS --right RESUMED --expected-step 8 --out restart.json
python scripts/summarize_reinjection.py --fixed-sum CONTROL_REPORT --learned-mix LEARNED_REPORT --out screen.json
```

Repeat preflight and restart gates for `fixed_sum`. CUDA processes must start with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `TRITON_F32_DEFAULT=tf32x3`, in a clean pinned
checkout with the original runtime. The launch queue must enforce both deadlines;
these individual commands do not themselves enforce the pair-wide time limit.
