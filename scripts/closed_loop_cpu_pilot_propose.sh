#!/usr/bin/env bash
# Programme 3 on CPU (docs/33 §4): the model proposes its own lookup tasks. One amorce per
# seed (100 perfect trajectories + P perfect proposals, 200 steps) shared by the three
# arms, every arm measured on the generator's bench and on the out-of-distribution bench.
#
#   scripts/closed_loop_cpu_pilot_propose.sh FIRST_RUN_DIR OUT_DIR
#
# ARMS in order (default "closed-propose closed-clean oracle"); the first arm trains the
# amorce at OUT/amorce-seedN/seed, the others reuse it.
set -euo pipefail
WORK="${1:?first run directory}"
OUT="${2:?output directory}"
SEEDS="${SEEDS:-0}"
ROUNDS="${ROUNDS:-5}"
PROPOSE_N="${PROPOSE_N:-30}"
PROPOSE_AMORCE="${PROPOSE_AMORCE:-50}"
SEED_EPISODES="${SEED_EPISODES:-100}"
SEED_STEPS="${SEED_STEPS:-200}"
LR_SCALE="${LR_SCALE:-0.25}"
ARMS="${ARMS:-closed-propose closed-clean oracle}"
COMMON=(--work "$WORK" --family lookup --tasks-per-round 30 --attempts 3
        --steps-per-round 60 --seed-episodes "$SEED_EPISODES" --seed-steps "$SEED_STEPS" --replay-fraction 0.5
        --bench-tasks 30 --bpb-docs 200 --seq-len 512 --batch-size 8 --minutes 600
        --lr-scale "$LR_SCALE" --no-repeat-action --copy-topk 3 --copy-explore observations
        --propose-amorce "$PROPOSE_AMORCE" --hard-bench)
for SEED in $SEEDS; do
  SEED_DIR="$OUT/amorce-seed$SEED/seed"
  for ARM in $ARMS; do
    echo "arm $ARM seed $SEED $(date -u +%FT%TZ)" | tee -a "$OUT/pilot.log"
    python scripts/closed_loop.py "${COMMON[@]}" --arm "$ARM" --seed "$SEED" --rounds "$ROUNDS" \
        --propose-n "$PROPOSE_N" --seed-dir "$SEED_DIR" --out "$OUT/$ARM-seed$SEED" 2>&1 | tee -a "$OUT/$ARM-seed$SEED.log"
  done
done
echo "PILOT_PROPOSE_COMPLETE"
