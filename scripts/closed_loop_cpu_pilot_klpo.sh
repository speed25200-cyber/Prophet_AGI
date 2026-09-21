#!/usr/bin/env bash
# The fourth arm of the closed-loop CPU pilot (docs/31 §2, H5): rejection fine-tuning
# plus KLPO updates on every episode of the round (docs/research/A5_klpo.md). Reuses the
# seed model the oracle arm trained in OUT_DIR, so run closed_loop_cpu_pilot.sh first.
#
#   scripts/closed_loop_cpu_pilot_klpo.sh FIRST_RUN_DIR OUT_DIR
set -euo pipefail
WORK="${1:?first run directory}"
OUT="${2:?output directory}"
FAMILY="${FAMILY:-calc}"
ROUNDS="${ROUNDS:-5}"
COMMON=(--work "$WORK" --family "$FAMILY" --tasks-per-round 30 --attempts 2
        --steps-per-round 60 --seed-episodes 100 --seed-steps 200 --replay-fraction 0.5
        --bench-tasks 30 --bpb-docs 200 --seq-len 512 --batch-size 8 --minutes 600
        --klpo-steps 60 --klpo-beta 0.1 --klpo-lr 5e-4 --klpo-draws 8 --klpo-temperature 1.0)
for SEED in 0 1; do
  SEED_DIR="$OUT/oracle-seed$SEED/seed"
  [ -d "$SEED_DIR" ] || { echo "missing $SEED_DIR: run closed_loop_cpu_pilot.sh first"; exit 1; }
  python scripts/closed_loop.py "${COMMON[@]}" --arm closed-klpo --seed "$SEED" --rounds "$ROUNDS" \
      --seed-dir "$SEED_DIR" --out "$OUT/closed-klpo-seed$SEED" 2>&1 | tee -a "$OUT/closed-klpo-seed$SEED.log"
done
echo "PILOT_KLPO_COMPLETE"
