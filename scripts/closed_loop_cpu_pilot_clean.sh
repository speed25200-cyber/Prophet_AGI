#!/usr/bin/env bash
# The closed-clean arm of the closed-loop CPU pilot (docs/31 amendment 7, H8): as closed,
# but only canonical trajectories are promoted. Reuses the seed model the oracle arm
# trained in OUT_DIR, so run closed_loop_cpu_pilot.sh first.
#
#   scripts/closed_loop_cpu_pilot_clean.sh FIRST_RUN_DIR OUT_DIR
set -euo pipefail
WORK="${1:?first run directory}"
OUT="${2:?output directory}"
FAMILY="${FAMILY:-calc}"
ROUNDS="${ROUNDS:-5}"
SEEDS="${SEEDS:-0 1}"
LR_SCALE="${LR_SCALE:-1.0}"
SEED_EPISODES="${SEED_EPISODES:-100}"  # amorce size; a lighter amorce leaves a family its margin (docs/31, amendment 10)
SEED_STEPS="${SEED_STEPS:-200}"
NO_REPEAT="${NO_REPEAT:-0}"  # 1 = --no-repeat-action (docs/31, amendment 11)
SAMPLE_COPY="${SAMPLE_COPY:-0}"  # 1 = --sample-copy at generation (docs/31, amendment 12)
COMMON=(--work "$WORK" --family "$FAMILY" --tasks-per-round 30 --attempts 2
        --steps-per-round 60 --seed-episodes "$SEED_EPISODES" --seed-steps "$SEED_STEPS" --replay-fraction 0.5
        --bench-tasks 30 --bpb-docs 200 --seq-len 512 --batch-size 8 --minutes 600
        --lr-scale "$LR_SCALE" ${NO_REPEAT:+$([ "$NO_REPEAT" = 1 ] && echo --no-repeat-action || true) $([ "$SAMPLE_COPY" = 1 ] && echo --sample-copy || true)})
for SEED in $SEEDS; do
  SEED_DIR="$OUT/oracle-seed$SEED/seed"
  [ -d "$SEED_DIR" ] || { echo "missing $SEED_DIR: run closed_loop_cpu_pilot.sh first"; exit 1; }
  python scripts/closed_loop.py "${COMMON[@]}" --arm closed-clean --seed "$SEED" --rounds "$ROUNDS" \
      --seed-dir "$SEED_DIR" --out "$OUT/closed-clean-seed$SEED" 2>&1 | tee -a "$OUT/closed-clean-seed$SEED.log"
done
echo "PILOT_CLEAN_COMPLETE"
