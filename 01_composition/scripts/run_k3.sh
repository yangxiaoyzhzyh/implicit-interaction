#!/bin/bash
# SPE κ with the training ceiling relaxed to K≤3 (正文 Table S6 的第二行).
# IlT κ 不在此列：它的最大组分数就是 3，放宽后不再构成外推。
# 3 seeds，先 v(im) 后 v(ex)，无早停（组分数轴不早停）。
#
# 用法（在任意目录执行均可，脚本自己回到 repro_v2/）：
#   DATA=/path/to/CheMixHub/datasets bash 01_composition/scripts/run_k3.sh
#
# 划分：本轴用 K≤3 的 composition 索引，需先生成到单独目录，避免覆盖默认的 K≤2：
#   python scripts/make_splits.py --task spe --split composition \
#       --train-max 3 --data-base "$DATA" --out splits_k3
set -u
cd "$(dirname "$0")/../.." || exit 1          # repro_v2/

PY=${PY:-python3}
DATA=${DATA:?请用 DATA=<CheMixHub datasets 目录> 指定数据根}
SPLIT=${SPLIT:-splits_k3/spe_composition.npz}
OUT=${OUT:-01_composition/runs_k3}
SEEDS=${SEEDS:-42 43 44}

mkdir -p "$OUT" 01_composition/logs
echo "python = $PY | data = $DATA | split = $SPLIT | seeds = $SEEDS | started $(date)"

for m in vim vex; do
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} "$PY" -u "scripts/$m.py" --tag spe \
    --split-file "$SPLIT" --data-base "$DATA" \
    --seeds $SEEDS --no-early-stop --out "$OUT" \
    > "01_composition/logs/k3_spe_${m}.log" 2>&1
  echo "spe $m rc=$? $(date)"
done

echo "=== results ==="
ls -l "$OUT"
