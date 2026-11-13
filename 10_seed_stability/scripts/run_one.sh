#!/bin/bash
# One (arm, seed) job for 10_seed_stability; called by run_all.sh.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

S=$1
M=${MODEL:-vex}
OUT=${OUT:-$PWD/10_seed_stability/runs_out}
mkdir -p "$OUT/${M}${S}"

EXTRA=""
[ -n "${EPOCHS:-}" ] && EXTRA="--epochs $EPOCHS"
if [ "$M" = "vex" ]; then FLAGS="--save-attn-all --save-emb"; else FLAGS="--save-emb"; fi

OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} python -u "scripts/$M.py" \
  --tag mon --model "$M" --split-file 10_seed_stability/splits/mon_composition.npz \
  --data-base "$DATA" --seeds "$S" --no-early-stop $EXTRA $FLAGS \
  --out "$OUT/${M}${S}" > "$OUT/${M}${S}/log.txt" 2>&1
echo "$M seed $S rc=$?"
