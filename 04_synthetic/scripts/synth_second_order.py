# -*- coding: utf-8 -*-
"""run_agg2.py — 给 v(im) 加一条二阶聚合通道，看能否补上 add_pair 缺的 q 项

实验 13 的诊断脚本
产出 /tmp 下打印，不写结果文件

背景
----
family A（add_pair）的目标可以写成

    y = s + (λ/2)(s² − q),   s = Σ w_i u_i,   q = Σ w_i² u_i²

v(im) 的一阶聚合只给 s。s² 可以由读出层的非线性得到，但 q 需要 **w²** 权重，
一阶聚合无论多宽都拿不到。所以 λ 越大，v(im) 的缺口越大。

本脚本把聚合改成

    z = Σ w_i E₁(x_i) + Σ w_i² E₂(x_i)

即加一条二阶通道，参数与一阶分支对称（E₂ 与 E₁ 同结构）。若缺口真在 q，
这一改应当把 λ=2.0 的分数大幅拉回。

用法
----
    python run_agg2.py --lam 2.0 --train-ks 1 2 --seeds 42 43 --epochs 150
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit('\\', 1)[0])
from synth_routes import (Generator, Vp, Vw, TEST_KS,  # noqa: E402
                              n_params, r2)


class VpNorm2(nn.Module):
    """归一化二阶通道：把"会随 K 塌的尺度"与"不塌的形状"分开喂给读出。

        z1   = Σ w_i  E₁(x_i)                 加权平均，尺度不随 K 变
        z2~  = Σ w_i² E₂(x_i) / Σ w_i²        加权平均，尺度不随 K 变
        σ2   = Σ w_i²                        标量，随 K 塌，但显式给出

    这样 q = σ2 · (与 z2~ 同构的量)，读出层可以自己乘上 σ2，不必去猜。
    """

    def __init__(self, dim=8, hid=120):
        super().__init__()
        self.E1 = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.ReLU())
        self.E2 = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.ReLU())
        self.reg = nn.Sequential(nn.Linear(2 * hid + 1, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, X, W, M):
        w = W.unsqueeze(-1) * M.unsqueeze(-1)
        w2 = w * w
        z1 = (self.E1(X) * w).sum(1)
        s2 = w2.sum(1)                                  # σ2，标量
        z2 = (self.E2(X) * w2).sum(1) / s2.clamp_min(1e-9)
        return self.reg(torch.cat([z1, z2, s2], dim=1)).squeeze(-1)


class Vp2(nn.Module):
    """一阶 + 二阶聚合。

    z = Σ w_i E₁(x_i) + Σ w_i² E₂(x_i)
    读出与 Vp 相同：三层 MLP。
    """

    def __init__(self, dim=8, hid=120):
        super().__init__()
        self.E1 = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.ReLU())
        self.E2 = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.ReLU())
        # 拼接版：读出输入 2*hid，让网络自己分配两条通道的权重
        self.reg = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, X, W, M):
        w = W.unsqueeze(-1) * M.unsqueeze(-1)
        z1 = (self.E1(X) * w).sum(1)
        z2 = (self.E2(X) * w * w).sum(1)      # 权重平方
        return self.reg(torch.cat([z1, z2], dim=1)).squeeze(-1)


def run(build, lam, seed, epochs, n_train, n_test, train_ks, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = Generator('add_pair', seed=1000 + seed)
    Xtr, Wtr, Mtr, ytr = gen.sample(n_train, train_ks, lam)
    te = {k: gen.sample(n_test, (k,), lam) for k in TEST_KS}
    ym, ys = ytr.mean(), ytr.std()
    ytr_n = (ytr - ym) / ys

    torch.manual_seed(seed)
    model = build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = ((model(Xtr[b], Wtr[b], Mtr[b]) - ytr_n[b]) ** 2).mean()
            loss.backward()
            opt.step()

    out = {}
    with torch.no_grad():
        for k in TEST_KS:
            X, W, M, y = te[k]
            p = model(X, W, M) * ys + ym
            out[k] = r2(p, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, default=2.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--n-train', type=int, default=5000)
    ap.add_argument('--n-test', type=int, default=2000)
    ap.add_argument('--train-ks', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--out', default=None,
                    help='结果 CSV 路径；归档时显式指定')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tks = tuple(args.train_ks)
    print(f'device={dev} | λ={args.lam} | train_ks={tks} | seeds={args.seeds}')
    print(f'参数量：v(im)={n_params(Vp())} | +agg2={n_params(Vp2())} | '
          f'+norm2={n_params(VpNorm2())} | v(ex)={n_params(Vw())}')

    acc = {name: {k: [] for k in TEST_KS}
           for name in ('v(im)', 'v(im)+agg2', 'v(im)+norm2', 'v(ex)')}
    t0 = time.time()
    for s in args.seeds:
        for name, build in (('v(im)', Vp), ('v(im)+agg2', Vp2),
                            ('v(im)+norm2', VpNorm2), ('v(ex)', Vw)):
            r = run(build, args.lam, s, args.epochs, args.n_train,
                    args.n_test, tks, dev)
            for k in TEST_KS:
                acc[name][k].append(r[k])

    print(f'\n=== add_pair  λ={args.lam}  train K≤{max(tks)} ===')
    print(f'{"K":>4}{"v(im)":>10}{"+agg2":>10}{"+norm2":>10}{"v(ex)":>10}')
    for k in TEST_KS:
        print(f'{k:>4}' + ''.join('%+10.3f' % np.mean(acc[n][k])
                                  for n in ('v(im)', 'v(im)+agg2',
                                            'v(im)+norm2', 'v(ex)')))
    print(f'\n总用时 {time.time() - t0:.0f}s')

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                    exist_ok=True)
        with open(args.out, 'w', encoding='utf-8', newline='') as f:
            f.write('family,lambda,K,model,R2,n_seeds\n')
            for k in TEST_KS:
                for name in ('v(im)', 'v(im)+agg2', 'v(im)+norm2', 'v(ex)'):
                    f.write('add_pair,%s,%d,%s,%.4f,%d\n'
                            % (args.lam, k, name,
                               float(np.mean(acc[name][k])), len(args.seeds)))
        print('saved', args.out)


if __name__ == '__main__':
    main()
