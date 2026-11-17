#!/bin/bash
# Two seeds of v(ex) on the SPE composition split, with the attention stored for
# both training and test mixtures. This is the second block of table S11.
#
#   DATA=/path/to/CheMixHub/datasets bash 10_seed_stability/scripts/run_spe.sh
#
# SPE is large (8944 training rows), so two seeds rather than twenty; they were run
# on one GPU each. Expect ~10 minutes per seed on an RTX 4090, much longer on CPU.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export DATA=${DATA:?set DATA to the directory that holds polymer-electrolyte/}
OUT=${OUT:-$PWD/10_seed_stability/runs_out}
mkdir -p "$OUT"

for s in ${SEEDS:-42 44}; do
  mkdir -p "$OUT/s$s"
  OMP_NUM_THREADS=4 python -u scripts/vex.py \
    --tag spe --model vex \
    --split-file 10_seed_stability/splits/spe_composition.npz \
    --data-base "$DATA" \
    --seeds "$s" --no-early-stop \
    --save-attn-all --save-emb \
    --out "$OUT/s$s" > "$OUT/s$s/log.txt" 2>&1
  echo "seed $s rc=$?"
  cp "$OUT/s$s"/*_attn_*.npz "$OUT"/ 2>/dev/null
done

python 10_seed_stability/scripts/stability_table.py \
  --attn-dir "$OUT" --pairs spe:42:44
