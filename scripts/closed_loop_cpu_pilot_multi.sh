#!/usr/bin/env bash
# Pilot v10 (docs/31 amendment 14, H14): two families in one loop. One mixed amorce per
# seed (N perfect trajectories of each family, S steps, calibrated: round 0 alone), then
# the mixed closed-clean arm, one closed-clean arm per family alone, and the mixed oracle,
# every arm benched on every family. The reference recipe (v7) throughout.
#
#   scripts/closed_loop_cpu_pilot_multi.sh FIRST_RUN_DIR OUT_DIR
#
# ARMS lists what to run, in order: "calibrate" trains and benches the amorce
# (OUT/calibration/seedN, no rounds), "closed-clean" and "oracle" loop on every family,
# "mono" runs closed-clean once per family (OUT/closed-clean-<family>-seedN).
set -euo pipefail
WORK="${1:?first run directory}"
OUT="${2:?output directory}"
FAMILIES="${FAMILIES:-lookup files}"
ROUNDS="${ROUNDS:-5}"
SEEDS="${SEEDS:-0}"
LR_SCALE="${LR_SCALE:-0.25}"
SEED_EPISODES="${SEED_EPISODES:-50}"  # per family (docs/31, amendment 14)
SEED_STEPS="${SEED_STEPS:-100}"
NO_REPEAT="${NO_REPEAT:-1}"
ARMS="${ARMS:-calibrate closed-clean mono oracle}"
FAMILY_FLAGS=()
BENCH_FLAGS=()
for f in $FAMILIES; do FAMILY_FLAGS+=(--family "$f"); BENCH_FLAGS+=(--bench-family "$f"); done
COMMON=(--work "$WORK" --tasks-per-round 30 --attempts 2
        --steps-per-round 60 --seed-episodes "$SEED_EPISODES" --seed-steps "$SEED_STEPS" --replay-fraction 0.5
        --bench-tasks 30 --bpb-docs 200 --seq-len 512 --batch-size 8 --minutes 600
        --lr-scale "$LR_SCALE" $([ "$NO_REPEAT" = 1 ] && echo --no-repeat-action || true))
mkdir -p "$OUT/calibration"
for SEED in $SEEDS; do
  SEED_DIR="$OUT/calibration/seed$SEED/seed"
  for ARM in $ARMS; do
    case "$ARM" in
      calibrate)
        python scripts/closed_loop.py "${COMMON[@]}" "${FAMILY_FLAGS[@]}" --arm frozen --seed "$SEED" --rounds 0 \
            --out "$OUT/calibration/seed$SEED" 2>&1 | tee -a "$OUT/calibration/seed$SEED.log"
        ;;
      mono)
        [ -d "$SEED_DIR" ] || { echo "missing $SEED_DIR: run with ARMS=calibrate first"; exit 1; }
        for f in $FAMILIES; do
          python scripts/closed_loop.py "${COMMON[@]}" --family "$f" "${BENCH_FLAGS[@]}" --arm closed-clean \
              --seed "$SEED" --rounds "$ROUNDS" --seed-dir "$SEED_DIR" \
              --out "$OUT/closed-clean-$f-seed$SEED" 2>&1 | tee -a "$OUT/closed-clean-$f-seed$SEED.log"
        done
        ;;
      *)
        [ -d "$SEED_DIR" ] || { echo "missing $SEED_DIR: run with ARMS=calibrate first"; exit 1; }
        python scripts/closed_loop.py "${COMMON[@]}" "${FAMILY_FLAGS[@]}" --arm "$ARM" --seed "$SEED" --rounds "$ROUNDS" \
            --seed-dir "$SEED_DIR" --out "$OUT/$ARM-seed$SEED" 2>&1 | tee -a "$OUT/$ARM-seed$SEED.log"
        ;;
    esac
  done
done
echo "PILOT_MULTI_COMPLETE"
