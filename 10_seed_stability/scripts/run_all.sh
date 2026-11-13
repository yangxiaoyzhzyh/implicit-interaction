#!/bin/bash
# 20 seeds x 2 arms on MON (composition split), 10 jobs in parallel.
# The attention is read out for training and test mixtures, and the per-molecule
# encoding is stored, which is what table S11 compares.
#
#   DATA=/path/to/CheMixHub/datasets bash 10_seed_stability/scripts/run_all.sh
#   SEEDS="42 43" JOBS=2 ... bash 10_seed_stability/scripts/run_all.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export DATA=${DATA:?set DATA to the directory that holds MON/processed_data}
export EPOCHS=${EPOCHS:-}
JOBS=${JOBS:-10}
SEEDS=${SEEDS:-$(seq 42 61)}
MODELS=${MODELS:-"vex vim"}

mkdir -p 10_seed_stability/runs_out
echo "seeds=$(echo $SEEDS | wc -w) | jobs=$JOBS | arms=$MODELS"
for M in $MODELS; do
  printf '%s\n' $SEEDS | MODEL=$M xargs -P "$JOBS" -n 1 bash 10_seed_stability/scripts/run_one.sh
done

echo "=== per-seed test R ==="
grep -h "test R" 10_seed_stability/runs_out/*/log.txt 2>/dev/null || true
echo "=== pairwise comparison ==="
python 10_seed_stability/scripts/compare_seeds.py --results 10_seed_stability/runs_out
