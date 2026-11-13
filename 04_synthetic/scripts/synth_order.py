# -*- coding: utf-8 -*-
"""synth_order.py — 聚合权重的阶数（table S13）

同一 embedding、同一读出，只改聚合权重：

    w      : alpha = w                     （不学，理想混合规则）
    lin    : alpha ∝ eps + w*s(h)          （截距 + 斜率，均可学）
    quad   : alpha ∝ eps + w*s(h) + w²*r(h)（再加对角二阶项，均可学）

在一阶可加目标上加噪声，看"多学一项"是否划算。

用法：python 04_synthetic/scripts/synth_order.py [--noises 0.1 0.2] [--n 2000] [--lam 0]
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
from synth_routes import Generator, r2, TRAIN_KS, TEST_KS  # noqa: E402

torch.set_num_threads(6)
SEED, EPOCHS, NTE = 42, 150, 2000


class Agg(nn.Module):
    """mode: 'w'（给定）| 'lin'（截距+斜率）| 'quad'（再加二阶）"""

    def __init__(self, d=8, hid=64, mode='lin', eps0=0.1):
        super().__init__()
        self.mode = mode
        self.emb = nn.Linear(d + 1, hid)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        if mode in ('lin', 'quad'):
            self.score = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
            self.eps_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(eps0)))))
        if mode == 'quad':
            self.score2 = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, X, W, M):
        h = self.emb(torch.cat([X, W.unsqueeze(-1)], dim=-1))
        if self.mode == 'w':
            num = W * M
        else:
            s = F.softplus(self.score(h)).squeeze(-1)
            num = (F.softplus(self.eps_raw) + W * s) * M
            if self.mode == 'quad':
                rr = F.softplus(self.score2(h)).squeeze(-1)
                num = num + (W.pow(2) * rr) * M
        a = num / num.clamp_min(1e-9).sum(1, keepdim=True)
        return self.head((a.unsqueeze(-1) * h).sum(1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noises', type=float, nargs='+', default=[0.1, 0.2])
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = []
    print('n=%d  lam=%.2f  epochs=%d  seeds=%s'
          % (args.n, args.lam, args.epochs, args.seeds))
    for noise in args.noises:
        torch.manual_seed(args.seeds[0])
        np.random.seed(args.seeds[0])
        gen = Generator('add_pair', seed=1000 + args.seeds[0])
        Xtr, Wtr, Mtr, ytr = gen.sample(args.n, TRAIN_KS, args.lam)
        te = {k: gen.sample(NTE, (k,), args.lam) for k in TEST_KS}
        ym, ys = ytr.mean(), ytr.std()
        ytr_n = (ytr - ym) / ys
        g = torch.Generator().manual_seed(SEED + 7)
        ytr_n = ytr_n + noise * torch.randn(len(ytr_n), generator=g)

        print('\n噪声 σ=%.2f' % noise)
        print('%-8s %s' % ('agg', ' '.join('K=%-14d' % k for k in TEST_KS)))
        for mode in ('w', 'lin', 'quad'):
            per_seed = []
            for sd in args.seeds:
                torch.manual_seed(sd)
                m = Agg(mode=mode)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                for _ in range(args.epochs):
                    m.train()
                    perm = torch.randperm(args.n)
                    for i in range(0, args.n, 256):
                        idx = perm[i:i + 256]
                        opt.zero_grad()
                        F.mse_loss(m(Xtr[idx], Wtr[idx], Mtr[idx]), ytr_n[idx]).backward()
                        opt.step()
                m.eval()
                with torch.no_grad():
                    per_seed.append([r2(m(te[k][0], te[k][1], te[k][2]),
                                        (te[k][3] - ym) / ys) for k in TEST_KS])
            arr = np.array(per_seed)
            mu, sd_ = arr.mean(0), (arr.std(0, ddof=1) if len(arr) > 1
                                    else np.zeros(arr.shape[1]))
            rows.append((noise, mode, mu, sd_))
            print('%-8s %s' % (mode, ' '.join('%.3f±%.3f' % (a, b)
                                              for a, b in zip(mu, sd_))))
        sys.stdout.flush()

    out = args.out or os.path.join(HERE, '..', 'runs', 'synth_order.csv')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write('lam,noise,n,agg,nseed,' +
                ','.join('R2_K%d_mean,R2_K%d_sd' % (k, k) for k in TEST_KS) + '\n')
        for noise, mode, mu, sd_ in rows:
            cells = []
            for a, b in zip(mu, sd_):
                cells += ['%.4f' % a, '%.4f' % b]
            f.write('%.2f,%.2f,%d,%s,%d,%s\n'
                    % (args.lam, noise, args.n, mode, len(args.seeds), ','.join(cells)))
    print('wrote', out)


if __name__ == '__main__':
    main()
