# Bounded loss workspace

The A100 R04 calibration at batch 8 and sequence length 2048 allocated 16.74 GiB
for the shared model and 20.84 GiB for the unshared control. The old budget estimate
underpredicted both. Vocabulary-wide FP32 loss intermediates are one contributor.

`scripts/train.py --loss-chunk-tokens 512` opts into a first-order autograd operation
that computes cross entropy and squared log-partition in token chunks. It keeps the
original logits, aligned targets and one FP32 log-partition value per token. Backward
recomputes probabilities chunk by chunk and combines the CE and z-loss gradients.
The vocabulary logits and their gradient still exist in full; this is not a fused
linear/CE projection. Non-contiguous logits can additionally require a full copy.

The objective is preserved: ignored targets do not contribute to CE, z-loss includes
all positions, shifts stay inside sequence boundaries, and MTP, ponder and jumped-token
weights retain their position-dependent gradients. BF16 can round the combined
gradient differently from separately accumulated reference branches. Higher-order
gradients are deliberately unsupported. The default remains the original autograd
path (`None`), and checkpoints reject a changed chunk setting before loading state.

CPU tests cover weighted values and gradients, ignored targets, large common offsets,
non-contiguous inputs, MTP and ponder, retained tensor storage, and interrupted/resumed
training. CUDA checks cover FP32/BF16 gradients and the actual checkpointed trainer.
Hardware memory and throughput must be measured before choosing a production setting:

```bash
python scripts/gpu_check.py --config configs/prophet_r04_loop.json \
  --batch-size 8 --seq-len 2048 --steps 5 --tokens 100000000 \
  --loss-chunk-tokens 512 --json-output outputs/r04-loop-loss512.json
```

Compare against the same command without `--loss-chunk-tokens`, in a fresh process,
on the same runtime and revision. Repeat for `prophet_r04_plain.json`. These are
synthetic-token resource calibrations, not language-quality ablations.
