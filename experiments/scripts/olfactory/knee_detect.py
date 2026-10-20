# -*- coding: utf-8 -*-
"""knee_detect.py — 只用 val 曲线定位拐点 epoch，据此选 ckpt

实验 12 的辅助脚本（Sisson 早停口径，补材 M2 的 "Knee stop"）
读取 results/sisson/curves/curve_*.csv（只读 val 列，不碰 test）
无独立数据产出

方法（与补材 M2 一致，只有这一种）：
  1. val 是 AUROC，饱和在 1.0，直接求导会被天花板压缩 → 变换 h = -log10(1 - val)，
     饱和段被拉直，斜率衰减点才是真正的"学习收益拐点"。
  2. 因果移动平均平滑（窗口 5），抑制逐 epoch 噪声。
  3. 一阶导 d1 = h[t] - h[t-1]；以早期峰值斜率 d1_max 归一化，第一个连续 2 个 epoch
     满足 d1 < 0.1 * d1_max 的点即拐点，且不早于第 10 个 epoch。
  4. 选 ckpt：取 [0, knee] 内 val 最大的 epoch（等价于"在拐点早停 + 回载 best-val"）。

全程只读 val 列，绝不读取 test 列做选择。
用法：python knee_detect.py curve_s42_ck.csv
"""
import argparse
import csv

import numpy as np

WINDOW = 5         # 因果移动平均窗口
RATIO = 0.10       # 一阶导阈值（相对早期峰值斜率）
STREAK = 2         # 需连续满足的 epoch 数
MIN_EPOCHS = 10    # 拐点不得早于此

ap = argparse.ArgumentParser()
ap.add_argument('curve_csv')
args = ap.parse_args()

rows = [r for r in csv.DictReader(open(args.curve_csv, encoding='utf-8'))
        if r.get('val') not in ('', None)]
ep = np.array([int(r['ep']) for r in rows])
val = np.array([float(r['val']) for r in rows])
print(f'curve: {args.curve_csv} | epochs loaded: {len(ep)} '
      f'(ep{ep[0]}..ep{ep[-1]})')

# 1) headroom 对数变换
h = -np.log10(np.clip(1.0 - val, 1e-6, None))

# 2) 因果移动平均
hs = np.array([h[max(0, t - WINDOW + 1):t + 1].mean() for t in range(len(h))])

# 3) 一阶导与拐点
d1 = np.diff(hs, prepend=hs[0])
warm_end = min(len(hs), max(5, len(hs) // 10))
d1_max = d1[:warm_end].max()
print(f'early peak slope d1_max = {d1_max:.4f} '
      f'(at ep{int(ep[:warm_end][d1[:warm_end].argmax()])})')


def knee():
    bad = 0
    for t in range(1, len(hs)):
        if ep[t] < MIN_EPOCHS:
            continue
        if d1[t] < RATIO * d1_max:
            bad += 1
            if bad >= STREAK:
                return int(ep[t])
        else:
            bad = 0
    return int(ep[-1])


k = knee()
idx = int(np.searchsorted(ep, k)) + 1
best = int(ep[:idx][val[:idx].argmax()])
best_val = float(val[np.searchsorted(ep, best)])

print(f'knee = ep{k} (d1 < {RATIO:.0%} of peak for {STREAK} epochs, '
      f'ep >= {MIN_EPOCHS})')
print(f'-> selected ckpt ep{best} (val={best_val:.4f})')
print('note: only the val column is read; test is never used for selection.')
