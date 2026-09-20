# 24 — Observe recurrent states before proposing another architecture

**Status: diagnostic prepared, eight local CPU tests pass; no actual-size GPU
observations yet. No architecture or training change.**

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
is **8 passed**, without skips. Actual-size CUDA observation equivalence remains
pending until the diagnostic executes.

```bash
python scripts/diagnose_r04_recurrence.py \
  --model-repo <frozen-f5a7d71-checkout> --run <completed-adaptation-arm> \
  --corpus <original-pilot> --out <fresh-observation.json>
```
