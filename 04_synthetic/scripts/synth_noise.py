# -*- coding: utf-8 -*-
"""synth_noise.py — noise on a purely additive target (table S12)

把训练标签加高斯噪声（以标签标准差为单位），测试标签保持干净，比较两臂：

    v(im)                 : 摩尔分数加权和（聚合无参数）
    v(ex), linear basis   : alpha ∝ eps + w*softplus(s(h))，斜率与截距都可学

目的：在"基是否含解"已经排除之后（纯可加目标上 softmax 版 R² 0.98→0.81，
线性基 1.000），单独检验第二个来源——逐样本估计的权重会把噪声带进外推。

用法：python 04_synthetic/scripts/synth_noise.py [--out runs/synth_noise.csv]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth_routes import Generator, Vp, r2, TRAIN_KS, TEST_KS  # noqa: E402

torch.set_num_threads(6)
SEED, EPOCHS, NTR, NTE = 42, 150, 5000, 2000


class VwLin(nn.Module):
    """v(ex) with a learnable linear aggregation weight."""

    def __init__(self, d=8, hid=64, eps0=0.1):
        super().__init__()
        self.emb = nn.Linear(d + 1, hid)
        self.score = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.eps_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(eps0)))))
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def eps(self):
        return F.softplus(self.eps_raw)

    def forward(self, X, W, M):
        h = self.emb(torch.cat([X, W.unsqueeze(-1)], dim=-1))
        u = F.softplus(self.score(h)).squeeze(-1)
        num = (self.eps() + W * u) * M
        alpha = num / num.clamp_min(1e-9).sum(1, keepdim=True)
        return self.head((alpha.unsqueeze(-1) * h).sum(1)).squeeze(-1)


def fit(model, ytr_n, Xtr, Wtr, Mtr):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(ytr_n)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            F.mse_loss(model(Xtr[idx], Wtr[idx], Mtr[idx]), ytr_n[idx]).backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noises', type=float, nargs='+', default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = []
    print('%-6s %-14s %s' % ('noise', 'arm', ' '.join('K=%d' % k for k in TEST_KS)))
    for noise in args.noises:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        gen = Generator('add_pair', seed=1000 + SEED)       # 函数本身固定，与 K 无关
        Xtr, Wtr, Mtr, ytr = gen.sample(NTR, TRAIN_KS, 0.0)
        te = {k: gen.sample(NTE, (k,), 0.0) for k in TEST_KS}
        ym, ys = ytr.mean(), ytr.std()
        ytr_n = (ytr - ym) / ys
        if noise > 0:                                        # 噪声只进训练标签
            g = torch.Generator().manual_seed(SEED + 7)
            ytr_n = ytr_n + noise * torch.randn(len(ytr_n), generator=g)

        for name, build in (('v(im)', Vp), ('linear', VwLin)):
            torch.manual_seed(SEED)
            m = fit(build(), ytr_n, Xtr, Wtr, Mtr)
            with torch.no_grad():
                rs = [r2(m(te[k][0], te[k][1], te[k][2]), (te[k][3] - ym) / ys)
                      for k in TEST_KS]
            rows.append((noise, name, rs))
            print('%-6.2f %-14s %s' % (noise, name, ' '.join('%.3f' % v for v in rs)))

    out = args.out or os.path.join(HERE, '..', 'runs', 'synth_noise.csv')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write('noise,arm,' + ','.join('R2_K%d' % k for k in TEST_KS) + '\n')
        for noise, name, rs in rows:
            f.write('%.2f,%s,%s\n' % (noise, name, ','.join('%.3f' % v for v in rs)))
    print('wrote', out)


if __name__ == '__main__':
    main()
