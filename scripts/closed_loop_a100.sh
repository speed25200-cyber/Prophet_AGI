#!/usr/bin/env bash
# The closed loop at A100 scale (docs/31 §3 and §7): the reference recipe of the CPU
# pilots (docs/32 §0) applied to a loop-core checkpoint (docs/29), one family at a time
# or several, with the amorce calibrated family by family on the canonical success
# (amendments 16 and 20) before any round is trained.
#
#   scripts/closed_loop_a100.sh CORPUS_DIR LOOP_CORE_RUN_DIR CONFIG_JSON OUT_DIR
#
#   CORPUS_DIR          output of scripts/prepare_loop_core_corpus.py (train/, validation/,
#                       manifest.json) with its tokenizer.json alongside or TOKENIZER set
#   LOOP_CORE_RUN_DIR   output of scripts/run_loop_core.py (checkpoints/manifest.json)
#   CONFIG_JSON         the arm's config (configs/loop_core/lc_*.json); action heads are added by the loop
#   OUT_DIR             where the assembled work directory, calibrations and arms go
#
# Environment: FAMILIES (default "calc lookup files"), SEEDS (default "0"), LADDER
# (amorce N/S rungs per family, default "100/200 50/100 200/400"), ROUNDS (8),
# TASKS (200), ATTEMPTS (3), STEPS (300), LR_SCALE (0.25), BENCH (60), BPB_DOCS (400),
# EXPLORE (1 = the exploring retries of docs/32 §16), ARMS ("closed-clean oracle"),
# DEVICE (cuda), DRY (1 = print the commands, run nothing).
#
# Not yet run on a GPU: the --device path of scripts/closed_loop.py is covered by tests
# on CPU only. The first A100 session is the test; this script records what it does.
set -euo pipefail
CORPUS="${1:?corpus directory}"; RUN="${2:?loop-core run directory}"; CONFIG="${3:?config json}"; OUT="${4:?output directory}"
FAMILIES="${FAMILIES:-calc lookup files}"
SEEDS="${SEEDS:-0}"
LADDER="${LADDER:-100/200 50/100 200/400}"
ROUNDS="${ROUNDS:-8}"; TASKS="${TASKS:-200}"; ATTEMPTS="${ATTEMPTS:-3}"; STEPS="${STEPS:-300}"
LR_SCALE="${LR_SCALE:-0.25}"; BENCH="${BENCH:-60}"; BPB_DOCS="${BPB_DOCS:-400}"
EXPLORE="${EXPLORE:-1}"; ARMS="${ARMS:-closed-clean oracle}"; DEVICE="${DEVICE:-cuda}"; DRY="${DRY:-0}"
TOKENIZER="${TOKENIZER:-$CORPUS/tokenizer.json}"
HELDOUT_DOCS="${HELDOUT_DOCS:-2000}"
cd "$(dirname "$0")/.."
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }

# 1. The work directory the loop expects (docs/09 layout), assembled from loop-core parts.
WORK="$OUT/work"
mkdir -p "$WORK/corpus" "$WORK/benchmarks"
[ -e "$WORK/tokenizer.json" ] || ln -s "$(realpath "$TOKENIZER")" "$WORK/tokenizer.json"
[ -e "$WORK/checkpoints" ] || ln -s "$(realpath "$RUN/checkpoints")" "$WORK/checkpoints"
[ -e "$WORK/corpus/fineweb-edu" ] || ln -s "$(realpath "$CORPUS/train/fineweb-edu")" "$WORK/corpus/fineweb-edu"
[ -e "$WORK/corpus/composition" ] || ln -s "$(realpath "$CORPUS/train/composition")" "$WORK/corpus/composition"
if [ ! -s "$WORK/benchmarks/heldout.jsonl" ]; then
  # Held-out text: the first documents of the loop-core validation split, never trained on.
  run python3 - "$CORPUS/validation/fineweb-edu" "$WORK/benchmarks/heldout.jsonl" "$HELDOUT_DOCS" <<'PY'
import json, sys
from pathlib import Path
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
kept = 0
with dst.open("w", encoding="utf-8") as out:
    for path in sorted(src.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            out.write(json.dumps({"text": json.loads(line)["text"]}) + "\n")
            kept += 1
            if kept >= n:
                break
        if kept >= n:
            break
print("heldout docs", kept)
PY
fi

FAMILY_FLAGS=(); for f in $FAMILIES; do FAMILY_FLAGS+=(--family "$f"); done
EXPLORE_FLAGS=(); [ "$EXPLORE" = 1 ] && EXPLORE_FLAGS=(--copy-topk 3 --copy-explore observations)
COMMON=(--work "$WORK" --config "$CONFIG" --device "$DEVICE" --replay-names fineweb-edu,composition
        --tasks-per-round "$TASKS" --attempts "$ATTEMPTS" --steps-per-round "$STEPS" --replay-fraction 0.5
        --bench-tasks "$BENCH" --bpb-docs "$BPB_DOCS" --seq-len 1024 --batch-size 8 --minutes 600
        --lr-scale "$LR_SCALE" --no-repeat-action --no-repeat-emitted "${EXPLORE_FLAGS[@]}")

# 2. Calibration, family by family, on the canonical success (amendment 16): the first rung
#    of the ladder where every family succeeds with a canonical success strictly positive,
#    and at least one family starts below 0.95, is retained. A family already saturated is
#    judged on retention (amendment 20) and reported as such; a rung where every family is
#    saturated is refused, since its rounds could only measure retention
#    (scripts/closed_loop.py calibration_verdict, tested).
verdict() {
python3 - "$1" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from closed_loop import calibration_verdict
r0 = [json.loads(l) for l in open(sys.argv[1])][0]
ok, saturated = calibration_verdict(r0)
by, can = r0["success_by_family"], r0["canonical_by_family"]
print(("OK" if ok else "LADDER") + (" saturated=" + ",".join(saturated) if saturated else ""), json.dumps({"success": by, "canonical": can}))
PY
}
for SEED in $SEEDS; do
  RETAINED=""
  for RUNG in $LADDER; do
    N="${RUNG%/*}"; S="${RUNG#*/}"
    CAL="$OUT/calibration/seed$SEED-$N-$S"
    echo "calibration seed $SEED amorce $N/$S $(date -u +%FT%TZ)" | tee -a "$OUT/pilot.log"
    run python scripts/closed_loop.py "${COMMON[@]}" "${FAMILY_FLAGS[@]}" --arm frozen --seed "$SEED" --rounds 0 \
        --seed-episodes "$N" --seed-steps "$S" --out "$CAL" 2>&1 | tee -a "$OUT/pilot.log"
    [ "$DRY" = 1 ] && { RETAINED="$RUNG"; break; }
    V=$(verdict "$CAL/rounds.jsonl"); echo "CALIBRATION_VERDICT seed $SEED $N/$S $V" | tee -a "$OUT/pilot.log"
    case "$V" in OK*) RETAINED="$RUNG"; break;; esac
  done
  [ -n "$RETAINED" ] || { echo "LADDER_EXHAUSTED seed $SEED" | tee -a "$OUT/pilot.log"; continue; }
  N="${RETAINED%/*}"; S="${RETAINED#*/}"
  # 3. The arms, from the retained amorce; every round trains on the cumulative pool of
  #    every family (docs/32 §19: a round that omits a learned family forgets it at once).
  for ARM in $ARMS; do
    echo "arm $ARM seed $SEED amorce $N/$S $(date -u +%FT%TZ)" | tee -a "$OUT/pilot.log"
    run python scripts/closed_loop.py "${COMMON[@]}" "${FAMILY_FLAGS[@]}" --arm "$ARM" --seed "$SEED" --rounds "$ROUNDS" \
        --seed-episodes "$N" --seed-steps "$S" --seed-dir "$OUT/calibration/seed$SEED-$N-$S/seed" \
        --out "$OUT/$ARM-seed$SEED" 2>&1 | tee -a "$OUT/pilot.log"
  done
done
[ "$DRY" = 1 ] || run python scripts/summarize_closed_loop.py --root "$OUT" --out "$OUT/summary.json" | tee -a "$OUT/pilot.log"
echo "A100_LOOP_COMPLETE $(date -u +%FT%TZ)" | tee -a "$OUT/pilot.log"
