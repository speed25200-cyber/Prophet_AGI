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
SEEDS="${SEEDS:-0 1}"  # seeds whose amorce starts strictly between 0 and 1 (docs/31, amendment 2)
LR_SCALE="${LR_SCALE:-1.0}"  # per-round peak-rate multiplier (docs/31, amendment 5; 1.0 = v1/v2 recipe)
SEED_EPISODES="${SEED_EPISODES:-100}"  # amorce size; a lighter amorce leaves a family its margin (docs/31, amendment 10)
SEED_STEPS="${SEED_STEPS:-200}"
COMMON=(--work "$WORK" --family "$FAMILY" --tasks-per-round 30 --attempts 2
        --steps-per-round 60 --seed-episodes "$SEED_EPISODES" --seed-steps "$SEED_STEPS" --replay-fraction 0.5
        --bench-tasks 30 --bpb-docs 200 --seq-len 512 --batch-size 8 --minutes 600
        --lr-scale "$LR_SCALE"
        --klpo-steps "${KLPO_STEPS:-60}" --klpo-beta "${KLPO_BETA:-0.1}" --klpo-lr 5e-4 --klpo-draws 8 --klpo-temperature 1.0)
for SEED in $SEEDS; do
  SEED_DIR="$OUT/oracle-seed$SEED/seed"
  [ -d "$SEED_DIR" ] || { echo "missing $SEED_DIR: run closed_loop_cpu_pilot.sh first"; exit 1; }
  python scripts/closed_loop.py "${COMMON[@]}" --arm closed-klpo --seed "$SEED" --rounds "$ROUNDS" \
      --seed-dir "$SEED_DIR" --out "$OUT/closed-klpo-seed$SEED" 2>&1 | tee -a "$OUT/closed-klpo-seed$SEED.log"
done
echo "PILOT_KLPO_COMPLETE"
