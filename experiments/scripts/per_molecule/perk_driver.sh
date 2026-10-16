#!/bin/bash
# 实验 3 — per-composition-count profile 的串行驱动
# 调用 run_perk_thermo.py（内部标签 e1 = v(ex)、e2 = v(im)）
# 产出 results/perk_*.csv；logs/extrapolation/perk_*.log
# perk_driver.sh — 在服务器上依次跑 ΔHmix / ΔHvap × e1 / e2 的 per-K profile
cd /autodl-fs/data/hmix_kext || exit 1
P=/root/miniconda3/bin/python
D=/autodl-fs/data/hmix_kext/data
O=/autodl-fs/data/hmix_kext/results
cd "$O" || exit 1
for pair in "Enthalpy of mixing:hmix" "Heat of vaporization:hvap"; do
  prop="${pair%%:*}"
  tag="${pair##*:}"
  for r in e1 e2; do
    echo "=== $tag $r start $(date +%H:%M:%S) ==="
    $P /autodl-fs/data/hmix_kext/run_perk_thermo.py \
        --prop "$prop" --route "$r" --seeds 42 43 44 \
        --data-dir "$D" --out-dir "$O" --out-csv "perk_${tag}_${r}.csv"
    echo "=== $tag $r done $(date +%H:%M:%S) ==="
  done
done
echo ALL_DONE
