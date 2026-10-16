#!/bin/bash
# 实验 3 — per-composition-count profile 的并行驱动
# 调用 run_perk_thermo.py（内部标签 e1 = v(ex)、e2 = v(im)）
# 产出 results/perk_*.csv；logs/extrapolation/perk_*.log
# perk_parallel.sh — 停掉串行驱动，四个组合并行跑（单卡 4080S，小模型，显存无压力）
cd /autodl-fs/data/hmix_kext || exit 1
pkill -f perk_driver.sh 2>/dev/null
pkill -f run_perk_thermo.py 2>/dev/null
sleep 3
P=/root/miniconda3/bin/python
D=/autodl-fs/data/hmix_kext/data
O=/autodl-fs/data/hmix_kext/results
mkdir -p logs
for pair in "Enthalpy of mixing:hmix" "Heat of vaporization:hvap"; do
  prop="${pair%%:*}"
  tag="${pair##*:}"
  for r in e1 e2; do
    nohup "$P" /autodl-fs/data/hmix_kext/run_perk_thermo.py \
      --prop "$prop" --route "$r" --seeds 42 43 44 \
      --data-dir "$D" --out-dir "$O" --out-csv "perk_${tag}_${r}.csv" \
      > "logs/perk_${tag}_${r}.log" 2>&1 &
    echo "started ${tag}_${r} pid=$!"
  done
done
sleep 8
echo "--- running: $(ps -ef | grep -c '[r]un_perk_thermo.py') processes ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
