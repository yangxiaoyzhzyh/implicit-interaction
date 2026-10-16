#!/bin/bash
# 实验 7 — NIST η 分子轴压力阶梯的驱动
# 调用 run_nist_ladder.py（内部标签 e1 = v(ex)、e2 = v(im)）
# 产出 results/nist_ladder_k{1,2,5}_e{1,2}.csv；logs/extrapolation/nist_ladder.log
# NIST 分子轴压力阶梯：移出 test 总权重贡献最大的 k 个分子（k = 1/2/5），跑 v(im)/v(ex) 两条路线
cd "$(dirname "$0")" || exit 1
for k in 1 2 5; do
  for r in e2 e1; do
    echo "=========== k=$k route=$r ==========="
    /root/miniconda3/bin/python run_nist_ladder.py \
      --route "$r" --seeds 42 --epochs 200 --drop-top-k "$k" \
      --out "ladder_k${k}_${r}.csv"
  done
done
echo "ALL DONE"
