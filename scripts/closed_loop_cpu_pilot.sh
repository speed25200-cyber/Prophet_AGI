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
COMMON=(--work "$WORK" --family "$FAMILY" --tasks-per-round 30 --attempts 2
        --steps-per-round 60 --seed-episodes 100 --seed-steps 200 --replay-fraction 0.5
        --bench-tasks 30 --bpb-docs 200 --seq-len 512 --batch-size 8 --minutes 600)
mkdir -p "$OUT"
for SEED in $SEEDS; do
  SEED_DIR="$OUT/oracle-seed$SEED/seed"
  python scripts/closed_loop.py "${COMMON[@]}" --arm oracle --seed "$SEED" --rounds "$ROUNDS" \
      --out "$OUT/oracle-seed$SEED" 2>&1 | tee -a "$OUT/oracle-seed$SEED.log"
  python scripts/closed_loop.py "${COMMON[@]}" --arm closed --seed "$SEED" --rounds "$ROUNDS" \
      --seed-dir "$SEED_DIR" --out "$OUT/closed-seed$SEED" 2>&1 | tee -a "$OUT/closed-seed$SEED.log"
  python scripts/closed_loop.py "${COMMON[@]}" --arm frozen --seed "$SEED" --rounds 1 \
      --seed-dir "$SEED_DIR" --out "$OUT/frozen-seed$SEED" 2>&1 | tee -a "$OUT/frozen-seed$SEED.log"
done
echo "PILOT_COMPLETE"
