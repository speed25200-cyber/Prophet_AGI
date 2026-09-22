#!/usr/bin/env bash
# The closed-loop CPU pilot of docs/31 §3: three arms, two seeds, family calc, on the
# 7M-parameter first-run weight. The oracle arm trains the shared seed model first; the
# closed and frozen arms reuse it. Every run resumes if interrupted; rerun the script.
#
#   scripts/closed_loop_cpu_pilot.sh FIRST_RUN_DIR OUT_DIR
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
        --lr-scale "$LR_SCALE")
mkdir -p "$OUT"
ARMS="${ARMS:-oracle closed frozen}"  # arms per seed, in order; the first must train the shared seed
for SEED in $SEEDS; do
  SEED_DIR="$OUT/oracle-seed$SEED/seed"
  for ARM in $ARMS; do
    N="$ROUNDS"; [ "$ARM" = frozen ] && N=1
    if [ "$ARM" = oracle ] && [ ! -d "$SEED_DIR" ]; then
      python scripts/closed_loop.py "${COMMON[@]}" --arm oracle --seed "$SEED" --rounds "$N" \
          --out "$OUT/oracle-seed$SEED" 2>&1 | tee -a "$OUT/oracle-seed$SEED.log"
    else
      python scripts/closed_loop.py "${COMMON[@]}" --arm "$ARM" --seed "$SEED" --rounds "$N" \
          --seed-dir "$SEED_DIR" --out "$OUT/$ARM-seed$SEED" 2>&1 | tee -a "$OUT/$ARM-seed$SEED.log"
    fi
  done
done
echo "PILOT_COMPLETE"
