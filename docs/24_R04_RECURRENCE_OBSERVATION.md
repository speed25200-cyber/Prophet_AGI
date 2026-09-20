# 24 — Observe recurrent states before proposing another architecture

**Status: both actual-size observations complete, with exact observed/ordinary
hidden-state equality for all 32 document forwards. Export identities and scalar
records are verified locally. No architecture or training change.**

The [paired depth adaptation](23_R04_DEPTH_ADAPTATION.md) tests whether changing
training depth makes additional loops useful. Its final primary screen remains
unchanged. Independently of that screen's outcome, describe the internal state
trajectory of both completed checkpoints before attributing an outcome to a
particular mechanism.

## Fixed scope and budget

Run only after both final 512-step reports and their CPU integrity audit complete.
Use the frozen `f5a7d71` model code, exact published checkpoint files, original
corpus/tokenizer and recorded strict numerical policy. There are no optimizer
updates, new parameters, changed inference rules or resumed training processes.

For each of the two arms, use the **first 16 development documents in their
original order**, the first 2,048 token IDs from each including EOS if reached,
batch one padded to 2,048 and exactly eight loops. Exclude padding from state
statistics. Selection is fixed before reading either final quality comparison or
these diagnostics. It is a small descriptive subset, not a replacement validation
set or an additional primary endpoint.

Each document receives one ordinary forward and one observed forward. Their final
hidden tensors must be exactly equal. The observer copies the output of the last
core block after each iteration and always removes its hook, including on failure.
The model remains in evaluation mode with zero recurrent initialization. This
checks that observation leaves the output unchanged for each measured document;
it does not establish every possible inference path's correctness.

Budget: **600 seconds for the pair**, including loading, compilation and CPU
reductions. This is an inference-only diagnostic budget, separately recorded from
the 5,400-second adaptation queue. A timeout or output mismatch stops it for
inspection, without retraining or choosing a different document subset. The
driver itself does not enforce the paired ceiling; the external process queue
must do so. No diagnostic runs concurrently with the training queue.

## Measurements and interpretation

All reductions use detached CPU float64 states. Each per-document/per-loop row
records:

- State RMS: amplitude of the core output before the coda.
- Token-centered energy divided by total energy: variation between token states
  relative to total state energy.
- Mean cosine over all distinct token-position pairs, excluding self-pairs.
- Mean per-token cosine with the previous loop and total change norm divided by
  the previous state's norm. These last two are undefined at the first loop and
  are recorded as null.

For unit token vectors `u_i`, distinct-pair cosine is computed as
`(||sum_i u_i||² - T) / (T(T-1))`; no quadratic token-pair matrix is allocated.
Zero-norm or nonfinite states fail explicitly instead of manufacturing a cosine.
Text, weights and full activation arrays are not exported; document and input-ID
hashes bind the recorded scalar measurements.

Similar token states or small loop changes can occur without loss of useful
information. Large state norms can coexist with normalized readouts. Consequently
there is **no collapse threshold or architecture acceptance rule** based on these
statistics. Report both arms and all measured loops; any proposed normalization,
reinjection or mixer change still needs its own budgeted controlled experiment.
These observations alone cannot establish reasoning or quality at an unseen depth.

## Verification

Eight focused tests check closed-form identical/orthogonal token geometry, agreement
with explicit distinct-pair cosine, rejection of undefined/nonfinite states, and
exact output/weight/RNG preservation on tiny attention and GDN models. A failure
test confirms hook removal when the model raises an exception. The local result
is **8 passed**, without skips. The same eight tests pass in Colab. Both CI runs
at `2e4d25e` pass; the PR suite reports 832 passed and 18 skipped in 88.08 seconds.

```bash
python scripts/diagnose_r04_recurrence.py \
  --model-repo <frozen-f5a7d71-checkout> --run <completed-adaptation-arm> \
  --corpus <original-pilot> --out <fresh-observation.json>
```

## Observed trajectories

The pair completes in **63.57 seconds**, below its 600-second ceiling. All five
setup/test/observation processes exit zero. Each arm uses the same sixteen
document/input-ID identities, exact final checkpoint and frozen numerical policy.
Every observed forward reproduces the ordinary final hidden tensor exactly.

The following are unweighted means over the sixteen documents. Token cosine is
the within-document distinct-position mean; previous-loop cosine compares each
position with itself one iteration earlier.

| Arm | Loop | State RMS | Token-centered energy fraction | Token cosine | Previous-loop cosine |
|---|---:|---:|---:|---:|---:|
| Fixed 4 | 1 | 1,057.76 | 0.5795 | 0.8867 | — |
| Fixed 4 | 2 | 1,949.59 | 0.5560 | 0.8956 | 0.9829 |
| Fixed 4 | 3 | 2,740.02 | 0.6698 | 0.7841 | 0.9770 |
| Fixed 4 | 4 | 3,501.74 | 0.8736 | 0.3935 | 0.8326 |
| Fixed 4 | 5 | 4,435.99 | 0.8677 | 0.4017 | 0.9277 |
| Fixed 4 | 6 | 5,565.19 | 0.8117 | 0.5889 | 0.9293 |
| Fixed 4 | 7 | 6,417.94 | 0.7961 | 0.6094 | 0.9956 |
| Fixed 4 | 8 | 7,248.36 | 0.7978 | 0.5941 | 0.9982 |
| Variable 2–6 | 1 | 912.76 | 0.5806 | 0.8376 | — |
| Variable 2–6 | 2 | 1,760.06 | 0.6446 | 0.7887 | 0.9723 |
| Variable 2–6 | 3 | 2,564.30 | 0.6850 | 0.7028 | 0.9862 |
| Variable 2–6 | 4 | 3,362.91 | 0.7276 | 0.6125 | 0.9890 |
| Variable 2–6 | 5 | 4,157.45 | 0.7753 | 0.5116 | 0.9862 |
| Variable 2–6 | 6 | 4,955.70 | 0.8116 | 0.4312 | 0.9869 |
| Variable 2–6 | 7 | 5,763.78 | 0.8260 | 0.3951 | 0.9890 |
| Variable 2–6 | 8 | 6,591.52 | 0.8236 | 0.3926 | 0.9867 |

These observations do **not** show all token states becoming identical. Variable
depth reduces mean token similarity as loops increase, although its full-set loss
still worsens beyond k4. The fixed model changes direction most strongly around
its trained depth; late loop directions become close while state amplitude still
grows. Both arms have increasing unnormalized core RMS. At k4 its document range
is approximately 380–17,944 for fixed and 381–17,513 for variable, so the means
hide substantial document variation. The coda and final normalization intervene
before prediction; large internal RMS alone is not proof of numerical instability
or the cause of the loss increase.

This narrows the interpretation of the negative depth screen: a claim of simple
token-state collapse is unsupported by these measurements. Whether reinjection,
normalization, the training objective or the mixer limits useful extra computation
remains an experimental question; the observational contrast does not isolate one.

The [11-file export](experiments/2026-09-20-r04-recurrence-observation/export-manifest.json)
has ZIP size 30,444 bytes and SHA256
`4a9854ae7775cfd051308008803511f5766df4d0ce35d5640a811881146e4bcd`.
The [local verifier](experiments/2026-09-20-r04-recurrence-observation/verify.py)
checks all member hashes, the exact executed driver, terminal processes, checkpoint
and document identities, scalar domains and paired input identity. Its
[summary](experiments/2026-09-20-r04-recurrence-observation/verification.json)
contains every loop's mean/minimum/maximum. Full activations and weights are absent;
their original metric calculations and observer equality are Colab measurements,
not repeated by this local metadata verification.
