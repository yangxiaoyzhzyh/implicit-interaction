#!/bin/bash
# 放宽训练上限：SPE κ 训练 K≤3 / 测试 K>3（splits/spe_composition_k3.npz）
# IlT κ 不在此列：它的最大组分数就是 3，放宽后不再构成外推
# 3 seed 串行（vim 42/43/44 → vex 42/43/44，共 6 次训练），无早停
# 全部完成后：结果复制到 /autodl-fs（持久盘）→ 关机
cd /root/repro_v2 || exit 1
D=/autodl-fs/data/chemixhub/datasets
PY=/root/miniconda3/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
SEEDS="${SEEDS:-42 43 44}"
OUT=/autodl-fs/data/repro_v2_k3
mkdir -p _k3 logs "$OUT"
echo "python = $PY | seeds = $SEEDS | started $(date)"

for m in vim vex; do
  OMP_NUM_THREADS=4 "$PY" -u "scripts/$m.py" --tag spe \
    --split-file splits/spe_composition_k3.npz --data-base "$D" \
    --seeds $SEEDS --no-early-stop --save-preds --out _k3 \
    > "logs/k3_spe_${m}.log" 2>&1
  echo "spe $m rc=$? $(date)"
done

echo "=== 结果 ==="
cat _k3/spe_*_composition.csv 2>/dev/null

# 落到持久盘，避免关机后取不到
cp -f _k3/*.csv "$OUT"/ 2>/dev/null
cp -f logs/k3_spe_*.log "$OUT"/ 2>/dev/null
date > "$OUT/DONE.txt"
sync
echo "results copied to $OUT ; shutting down"
sleep 5
shutdown -h now || poweroff
