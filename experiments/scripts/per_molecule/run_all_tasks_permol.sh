#!/bin/bash
# 实验 1 — CheMixHub 8 tasks 官方 5-fold，v(im) 路线的批量驱动
# 调用 train_task_permol.py
# 产出 results/chemixhub_permol_results.csv；logs/chemixhub_5fold_permol/
# per-molecule 路线（31 维化学特征 + 常数占位列，摩尔分数凸组合）：
# CheMixHub 8 任务 × 官方 5-fold 随机 CV，单 seed 42
# epochs 上限 200 + val-R 早停(patience 20, min-epochs 10)
# 模型参数与 run_hmix_kext.py (v(im)) 严格一致：dim256/nhead4/TFM2层ff512/GIN×3(64)/head 256→256→256→1
# 并行 2 个 worker，按样本量配平：
#   A: mon(0.1万) + nist_logv(23.9万) + logv(3.4万) + ionic_cond(11.7万)
#   B: density(9万) + heat_vap(9万) + he(9万) + ionic_visc(11.7万)
cd "$(dirname "$0")" || exit 1
PY=/root/miniconda3/bin/python
ROOT=${CHEMIXHUB_ROOT:?set CHEMIXHUB_ROOT to the datasets dir}
LOGS=${LOG_DIR:-./logs}
mkdir -p $LOGS

run_task() {
    task=$1
    echo "[$(date)] START $task"
    if [ "$task" = "mon" ]; then
        # 服务器上 MON 独立存放，其余任务在 datasets/ 下
        DR=${MON_ROOT:?set MON_ROOT}
        $PY train_task_permol.py --task $task --data-root $DR --folds 0 1 2 3 4 \
            --epochs 200 --patience 20 --min-epochs 10 --seed 42 \
            > $LOGS/permol_$task.log 2>&1
    else
        $PY train_task_permol.py --task $task --root $ROOT --folds 0 1 2 3 4 \
            --epochs 200 --patience 20 --min-epochs 10 --seed 42 \
            > $LOGS/permol_$task.log 2>&1
    fi
    echo "[$(date)] DONE $task (exit $?)"
}

( for t in mon nist_logv logv ionic_cond; do run_task $t; done ) &
( for t in density heat_vap he ionic_visc; do run_task $t; done ) &
wait
echo "[$(date)] ALL DONE"
