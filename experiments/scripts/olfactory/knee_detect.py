# -*- coding: utf-8 -*-
"""knee_detect.py — 只用 val 曲线数学地定位拐点 epoch，据此选 ckpt

实验 12 的辅助脚本（Sisson 早停口径，补材 M2 的 "Knee stop"）
读取 results/sisson/curves/curve_*.csv（只读 val 列，不碰 test）
无独立数据产出

思路：
  1. val 是 AUROC，饱和在 1.0，直接求导会被天花板压缩 → 变换 h = -log10(1 - val)
     （headroom 的对数），饱和段被拉直，斜率衰减点才是真正的"学习收益拐点"。
  2. 因果移动平均平滑（窗口 w），抑制 val 的逐 epoch 噪声。
  3. 一阶导 d1 = h[t] - h[t-1]（>0 = 仍在涨）；取早期峰值斜率 d1_max 做归一化，
     第一个连续 k 个 epoch 满足 d1 < ratio * d1_max 的点 = 拐点（一阶导准则）。
  4. 二阶导 d2 = h[t+1] - 2h[t] + h[t-1]：由正转负（曲率翻转）= 拐点（二阶导准则），
     以及 |d2| 最大点（膝点）作为交叉验证。
  5. 选 ckpt：拐点之后不再训练，取 [0, knee] 内 val 最大的 epoch（等价于
     "在该点早停 + 回载 best-val ckpt"）。

全程只读 val 列，绝不读取 test 列做选择。
用法：python knee_detect.py curve_s42_ck.csv [--window 5] [--ratio 0.1] [--streak 2]
"""
import argparse, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('curve_csv')
ap.add_argument('--window', type=int, default=5, help='因果移动平均窗口')
ap.add_argument('--ratio', type=float, default=0.1, help='一阶导阈值（相对早期峰值斜率）')
ap.add_argument('--streak', type=int, default=2, help='需连续满足的 epoch 数')
ap.add_argument('--eps', type=float, default=1e-6)
args = ap.parse_args()

rows = [r for r in csv.DictReader(open(args.curve_csv, encoding='utf-8'))
        if r.get('val') not in ('', None)]
ep = np.array([int(r['ep']) for r in rows])
val = np.array([float(r['val']) for r in rows])
print(f'curve: {args.curve_csv} | epochs loaded: {len(ep)} (ep{ep[0]}..ep{ep[-1]})')

# ── 1) headroom 对数变换 ──
head = np.clip(1.0 - val, args.eps, None)
h = -np.log10(head)

# ── 2) 因果移动平均平滑 ──
w = args.window
hs = np.array([h[max(0, t - w + 1):t + 1].mean() for t in range(len(h))])

# ── 3) 一阶导（h 递增 = 仍在涨）──
d1 = np.diff(hs, prepend=hs[0])
warm_end = min(len(hs), max(5, len(hs) // 10))       # 早期陡升段
d1_max = d1[:warm_end].max()
print(f'\n早期峰值斜率 d1_max = {d1_max:.4f} '
      f'(在 ep{int(ep[:warm_end][d1[:warm_end].argmax()])})')

d2 = np.zeros_like(hs)
for t in range(1, len(hs) - 1):
    d2[t] = hs[t + 1] - 2 * hs[t] + hs[t - 1]

print('\n   ep |    val |  headroom |   h(smooth) |     d1 |  d1/peak |    d2')
for i in range(0, min(len(ep), 26)):
    print(f'  {ep[i]:4d} | {val[i]:.4f} | {head[i]:.6f} | {hs[i]:11.4f} | '
          f'{d1[i]:6.4f} | {d1[i] / d1_max:8.3f} | {d2[i]:7.4f}')


def knee_by_d1(ratio, streak):
    bad = 0
    for t in range(1, len(hs)):
        if d1[t] < ratio * d1_max:
            bad += 1
            if bad >= streak:
                return int(ep[t])
        else:
            bad = 0
    return int(ep[-1])


def pick_ckpt(k):
    idx = int(np.searchsorted(ep, k)) + 1
    seg = val[:idx]
    b = int(ep[:idx][seg.argmax()])
    return b, float(val[np.searchsorted(ep, b)])


print('\n─── 拐点候选（多种准则／阈值，全部只用 val）───')
for ratio in (0.05, 0.10, 0.20):
    for streak in (1, 2, 3):
        k = knee_by_d1(ratio, streak)
        b, bv = pick_ckpt(k)
        print(f'  d1 准则 ratio={ratio:.2f} streak={streak}: knee=ep{k} '
              f'→ 选 ckpt ep{b} (val={bv:.4f})')

# ── 4) 二阶导 / 曲率准则 ──
inner = np.arange(2, len(hs) - 2)
sign_flip = [int(ep[t]) for t in inner if d2[t] < 0 and d2[t - 1] >= 0]
curv_max = int(ep[inner[np.abs(d2[inner]).argmax()]]) if len(inner) else None
print(f'\n  二阶导符号翻转点（凸→凹）: {sign_flip[:6]}')
print(f'  |d2| 最大点（膝点）      : ep{curv_max}')
if sign_flip:
    b, bv = pick_ckpt(sign_flip[0])
    print(f'  → 二阶导准则: knee=ep{sign_flip[0]} → 选 ckpt ep{b} (val={bv:.4f})')

# ── 5) Kneedle（距首尾弦最远点，headroom 空间）──
x = np.arange(len(hs), dtype=float)
xn = (x - x[0]) / max(1e-12, (x[-1] - x[0]))
yn = (hs - hs[0]) / max(1e-12, (hs[-1] - hs[0]))
kd = int(ep[(yn - xn).argmax()])
b, bv = pick_ckpt(kd)
print(f'\n  Kneedle（首尾弦最远点）  : ep{kd} → 选 ckpt ep{b} (val={bv:.4f})')
print('\n注: 以上全部只用 val 列；test 列不参与任何选择，只在选定 ckpt 后做一次性报告。')
