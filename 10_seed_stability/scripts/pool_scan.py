# -*- coding: utf-8 -*-
"""pool_scan.py — how many appearances per molecule remove the underdetermination?

Exactly additive task on a fixed pool of N molecules:
    y = sum_i w_i (u_i . a),   u_i fixed per molecule, w ~ Dirichlet
train K in {1,2}, test K in {3,5,10,20}.

For each pool size N and each target number of appearances per molecule r we
generate n_train ~= r*N/1.5 mixtures, train three seeds, and measure
  * train R^2          (has the model fitted its data?)
  * test R^2 at K=3/10/20
  * cross-seed agreement of the CLS attention (cosine, arg-max flips)

Usage
    python pool_scan.py --nmol 100 --appear 2 --seeds 42 43 44 --out rows.csv
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 8
TRAIN_KS, TEST_KS = (1, 2), (3, 5, 10, 20)
EPOCHS, NTE = 150, 2000


class Vw(nn.Module):
    """1 层自注意力 + CLS 读出（与 repro 中的 v(ex) 同构）。"""

    def __init__(self, d=DIM, hid=64, nhead=2):
        super().__init__()
        self.emb = nn.Linear(d + 1, hid)
        self.cls = nn.Parameter(torch.zeros(1, 1, hid))
        layer = nn.TransformerEncoderLayer(hid, nhead, dim_feedforward=hid,
                                           dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, X, W, M):
        n = X.shape[0]
        h = self.emb(torch.cat([X, W.unsqueeze(-1)], dim=-1))
        h = torch.cat([self.cls.expand(n, -1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=M.device), ~M], dim=1)
        h = self.enc(h, src_key_padding_mask=pad)
        return self.head(h[:, 0]).squeeze(-1)


def make_pool(nmol, seed=7):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(nmol, DIM, generator=g)
    a = torch.randn(DIM, generator=g)
    return U, a / a.norm()


def sample(U, a, n, ks, seed):
    g = torch.Generator().manual_seed(seed)
    maxk = max(ks)
    nmol = U.shape[0]
    X = torch.zeros(n, maxk, DIM)
    W = torch.zeros(n, maxk)
    M = torch.zeros(n, maxk, dtype=torch.bool)
    y = torch.zeros(n)
    for i in range(n):
        k = int(ks[torch.randint(len(ks), (1,), generator=g)])
        idx = torch.randperm(nmol, generator=g)[:k]
        w = torch.rand(k, generator=g)
        w = w / w.sum()
        x = U[idx]
        X[i, :k], W[i, :k], M[i, :k] = x, w, True
        y[i] = (w * (x @ a)).sum()
    return X, W, M, y


def r2(p, y):
    return float(1.0 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def attn_of(model, X, W, M):
    n, k, _ = X.shape
    h = model.emb(torch.cat([X, W.unsqueeze(-1)], dim=-1))
    h = torch.cat([model.cls.expand(n, -1, -1), h], dim=1)
    mha = model.enc.layers[0].self_attn
    E, nh = h.shape[-1], mha.num_heads
    hd = E // nh
    Wq, Wk, _ = mha.in_proj_weight.chunk(3, dim=0)
    bq, bk, _ = mha.in_proj_bias.chunk(3, dim=0)
    q = (h @ Wq.T + bq).view(n, -1, nh, hd).transpose(1, 2)
    kk = (h @ Wk.T + bk).view(n, -1, nh, hd).transpose(1, 2)
    s = (q @ kk.transpose(-1, -2)) / (hd ** 0.5)
    pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=M.device), ~M], dim=1)
    s = s.masked_fill(pad[:, None, None, :], float('-inf'))
    return s.softmax(-1).mean(1)[:, 0, 1:1 + k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nmol', type=int, required=True)
    ap.add_argument('--appear', type=float, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = max(80, int(round(args.appear * args.nmol / 1.5)))
    U, A = make_pool(args.nmol)
    Xtr, Wtr, Mtr, ytr = sample(U, A, n, TRAIN_KS, 11)
    te = {k: sample(U, A, NTE, (k,), 200 + k) for k in TEST_KS}
    Xtr, Wtr, Mtr = Xtr.to(dev), Wtr.to(dev), Mtr.to(dev)
    ytr_n = ((ytr - ytr.mean()) / ytr.std()).to(dev)
    te = {k: tuple(t.to(dev) for t in v) for k, v in te.items()}
    ym, ys = float(ytr.mean()), float(ytr.std())

    alphas, store, tr_r2, test_r2 = {}, {}, [], []
    for sd in args.seeds:
        torch.manual_seed(sd)
        m = Vw().to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(EPOCHS):
            m.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                F.mse_loss(m(Xtr[idx], Wtr[idx], Mtr[idx]), ytr_n[idx]).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            tr_r2.append(r2(m(Xtr, Wtr, Mtr), ytr_n))
            row = []
            for k in TEST_KS:
                X, W, M, y = te[k]
                row.append(r2(m(X, W, M), (y - ym) / ys))
                alphas[(sd, k)] = attn_of(m, X, W, M).cpu().numpy()
                store[(sd, k)] = (W.cpu().numpy(), M.cpu().numpy())
            test_r2.append(row)

    tot = flips = 0
    cs = []
    bias, cross = [], []
    for i in range(1, len(args.seeds)):
        for k in TEST_KS:
            a1 = alphas[(args.seeds[0], k)]
            a2 = alphas[(args.seeds[i], k)]
            Wt, M = store[(args.seeds[0], k)]
            M = M.astype(bool)
            for j in range(a1.shape[0]):
                x, y_ = a1[j][M[j]], a2[j][M[j]]
                if len(x) < 2:
                    continue
                cs.append(float(x @ y_ / (np.linalg.norm(x) * np.linalg.norm(y_) + 1e-12)))
                gx = np.sort(x)[-1] - np.sort(x)[-2]
                gy = np.sort(y_)[-1] - np.sort(y_)[-2]
                if min(gx, gy) <= 0.2 * float(np.mean(x)):
                    continue
                tot += 1
                flips += int(x.argmax() != y_.argmax())
                xn = x / (x.sum() + 1e-12)
                yn = y_ / (y_.sum() + 1e-12)
                wn = Wt[j][M[j]] / (Wt[j][M[j]].sum() + 1e-12)
                bias.append(float(np.abs(xn - wn).mean() / (wn.mean() + 1e-12)))
                cross.append(float(np.abs(xn - yn).mean() / (xn.mean() + 1e-12)))

    mr = np.mean(test_r2, 0)
    print('nmol=%-5d appear=%-4.1f n_train=%-6d trainR2=%.3f  R2(3/10/20)=%.3f/%.3f/%.3f  '
          'flip(non-tie)=%.1f%%  cos=%.4f  dev=%s'
          % (args.nmol, args.appear, n, float(np.mean(tr_r2)), mr[0], mr[2], mr[3],
             100 * flips / max(1, tot), float(np.median(cs)), dev), flush=True)

    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['nmol', 'appear', 'n_train', 'train_r2', 'r2_k3',
                            'r2_k10', 'r2_k20', 'flip_pct', 'n_nontie', 'cos_median',
                            'bias_med', 'cross_seed_med'])
            w.writerow([args.nmol, args.appear, n, '%.3f' % float(np.mean(tr_r2)),
                        '%.3f' % mr[0], '%.3f' % mr[2], '%.3f' % mr[3],
                        '%.1f' % (100 * flips / max(1, tot)), tot,
                        '%.4f' % float(np.median(cs)),
                        '%.4f' % float(np.median(bias)),
                        '%.4f' % float(np.median(cross))])


if __name__ == '__main__':
    main()
