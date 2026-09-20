# 22 — Train-only activation calibration for shared projections

**Status: experimental driver and analytic tests implemented; real A100 result
pending. No architecture is adopted.** Neighboring-layer means and direct reuse
both damage the donor heavily ([measured scouts](21_DONOR_SHARING_SCOUT.md)). The
next bounded diagnostic tests whether fitting actual layer behavior is more useful
than averaging weights without their input distribution.

This is motivated by feature-space least-squares fitting in
[PLeaS (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Nasery_PLeaS_-_Merging_Models_with_Permutations_and_Least_Squares_CVPR_2025_paper.html).
That work merges separate vision models and includes permutations. The experiment
here shares different depths of one language model, performs no permutation, and
does **not** assume the paper's results transfer.

## Fixed protocol

- Pinned Qwen3-0.6B, original 28-layer native forward and tokenizer.
- Ten adjacent pairs `(4,5)` to `(22,23)`, seven shared projections per pair.
  All original per-depth norms and outer eight layers stay intact.
- Calibration: first 64 eligible **training** document hashes in ascending order,
  first 512 tokens each, no EOS: 32,768 input tokens. The frozen train and development
  corpus hashes must match and selected documents must not overlap.
- Evaluation: the same 16 development prefixes as the earlier scouts, 8,176 targets.
  No development activation enters fitting. No ARC score is used for selection.
- Variants: unchanged donor, adjacent weight mean, activation-fitted sharing.
  Same device and scoring policy for all three. No gradient recovery training.
- Full FP32 donor forwards; FP64 second-moment accumulation and Cholesky solve;
  returned weights cast to FP32. TF32 off, deterministic algorithms on.
- Ridge fraction fixed at **0.01** before real scores. No parameter sweep.
- Whole Colab queue bounded to **1,800 seconds**, including a CUDA solver oracle
  and a deliberate calibration interruption/resume after the first document.

For the two original weights `W_i` and their native donor inputs `X_i`, the shared
projection minimizes

`sum_i ||(W - W_i) X_i||_F² + lambda ||W - mean(W_i)||_F²`.

It uses `H_i = X_i X_i^T`, `H = sum_i H_i`,
`lambda = 0.01 * trace(H) / input_width`, and solves
`W (H + lambda I) = sum_i W_i H_i + lambda mean(W_i)`.

The collector runs **before any weights are shared**. It observes four distinct
input locations per middle layer; Q/K/V reuse one second moment and gate/up reuse
another. Hooks must run exactly once per location per document. No question text
or raw calibration activation is written to Git.

The local fitted objective must be finite and no worse than the weight-mean
baseline; otherwise the experiment stops. This mathematical bound does not imply
better final logits because later student inputs can differ from donor inputs.
The final development score, including a negative result, determines the next step.

## Size, persistence and checks

All shared candidates have **438,763,520** registered unique parameters. The budget
proxy has 438,740,480 parameters and needs 23,040 additional native norm parameters
to match the scout. It has equal linear storage but a different execution order;
it is a sizing proxy, not an implementation of local recurrence. Budget, design
search and plan outputs are [archived](experiments/2026-09-20-qwen-adjacent-and-calibration-plan/budget/budget-scope.json).
The half-hour horizon is a diagnostic ceiling, not sufficient from-scratch training.

The 80 FP64 second-moment matrices occupy exactly **2,516,582,400 bytes (2.34375 GiB)**.
This excludes donor weights, temporary activations, solver workspaces, fitted
weights and retained original modules. Registered parameter counts do not imply
resident memory. This experiment does not measure deployment speed or memory.

Calibration moments and the next document index are atomically checkpointed every
16 documents and at an explicit session boundary. Restricted reload requires the
same complete source, data, prefix, runtime and code contract. A completed report
cannot be overwritten. A partial run resumes its accumulation; fitting/scoring may
be recomputed after interruption. These artifacts stay outside Git.

Four CPU tests check the solver against an independently augmented-design
`lstsq` oracle, behavior in unseen input directions, invalid moments, and exact
atomic snapshot/restricted-load continuation on integer inputs. The oracle also
has a CUDA case; its actual A100 result is required before calibration starts.

```bash
python scripts/probe_qwen_covariance.py --source <pinned-source> \
  --train <frozen-train.jsonl> --validation <frozen-development.jsonl> \
  --out <new-directory> --device cuda --max-documents-per-session 1
# Same command and output directory, with the session limit raised to 64, resumes.
```
