# -*- coding: utf-8 -*-
"""agg_variants.py —— 小样本下比较「逐组分特征的聚合方式」

同一个特征（RDKit 2D，217 维/分子），四种读出：
    sum       Σ w_i x_i / Σ w_i        （固定、无参数；论文现用）
    mean      Σ x_i / K                （不加权，对照）
    concat    按权重降序取前 K 槽、零填充后拼接（无参数，但维数随 K 变）
    attn      单头注意力池化（有参数：score = MLP([x_i, w_i])）

协议与 mlp_small_data.py 一致：官方 fold 0，1% / 5% 标签，2 次重复，
所有投影（StandardScaler / PCA）只在标签子集上拟合，测试集不参与。

用法：python agg_variants.py [--tasks tile ...] [--out out/agg_variants.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from leave_one_task_out import TASKS, load_task  # noqa: E402
from fp_matrix import fp_matrix  # noqa: E402

RATIOS = [0.01, 0.05]
PCA_DIM = 64
MAXK = 5          # concat / attn 的槽数（MON 有 121 个组分，不参与）


def build_tokens(FP, cid2i, ids, ws, maxk=MAXK):
    """返回 (tokens[n, maxk, d], mask[n, maxk], w[n, maxk])，按权重降序排列"""
    n, d = len(ids), FP.shape[1]
    T = np.zeros((n, maxk, d), np.float32)
    M = np.zeros((n, maxk), np.float32)
    W = np.zeros((n, maxk), np.float32)
    for i, (ci, wi) in enumerate(zip(ids, ws)):
        pairs = [(cid2i[int(c)], float(w)) for c, w in zip(ci, wi)
                 if int(c) in cid2i]
        pairs.sort(key=lambda t: -t[1])
        pairs = pairs[:maxk]
        tot = sum(w for _, w in pairs) or 1.0
        for k, (j, w) in enumerate(pairs):
            T[i, k] = FP[j]
            M[i, k] = 1.0
            W[i, k] = w / tot
    return T, M, W


def pool_fixed(T, M, W, kind):
    """固定池化：sum / mean / concat（无参数）"""
    if kind == 'sum':
        return (T * W[:, :, None]).sum(1)
    if kind == 'mean':
        return T.sum(1) / np.maximum(M.sum(1, keepdims=True), 1)
    if kind == 'concat':
        return T.reshape(len(T), -1)
    raise ValueError(kind)


class MLPHead(nn.Module):
    """与 mlp_small_data.Head 同构，输出已 squeeze"""
    def __init__(self, d, h=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttnPool(nn.Module):
    """单头注意力池化：score_i = MLP([x_i, w_i])，z = Σ a_i x_i"""
    def __init__(self, d, h=32, ph=128, p=0.3):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d + 1, h), nn.Tanh(),
                                   nn.Linear(h, 1))
        self.head = nn.Sequential(
            nn.Linear(d, ph), nn.ReLU(), nn.Dropout(p),
            nn.Linear(ph, ph), nn.ReLU(), nn.Dropout(p),
            nn.Linear(ph, 1))

    def forward(self, x, mask, w):
        s = self.score(torch.cat([x, w[:, :, None]], -1)).squeeze(-1)
        s = s.masked_fill(mask < 0.5, -1e9)
        a = torch.softmax(s, dim=1)
        z = (a[:, :, None] * x).sum(1)
        return self.head(z).squeeze(-1)


def fit_torch(model, tr, va, epochs=300, seed=0, lr=3e-3, wd=1e-2):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(tr['y'])
    bs = min(64, max(8, n // 4))
    best, best_state, bad = 1e18, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad()
            loss = F.mse_loss(model(*(a[j] for a in tr['args'])), tr['y'][j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(F.mse_loss(model(*(a for a in va['args'])), va['y']))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def t_of(arr):
    return torch.tensor(arr, dtype=torch.float32)


def run_fixed(X, y, tr, va, te, ratio, n_rep=2):
    rs = []
    for rep in range(n_rep):
        rng = np.random.RandomState(1000 + rep)
        k = max(4, int(len(tr) * ratio))
        sub = rng.choice(tr, size=min(k, len(tr)), replace=False)
        sc = StandardScaler().fit(X[sub])
        d = min(PCA_DIM, X.shape[1], len(sub) - 1)
        if d < 2:
            continue
        pca = PCA(n_components=d, random_state=0).fit(sc.transform(X[sub]))
        Zs, Zv, Zt = (pca.transform(sc.transform(X[i]))
                      for i in (sub, va, te))
        ymu, ysd = y[sub].mean(), y[sub].std() + 1e-9
        tr_ = {'args': (t_of(Zs),), 'y': t_of((y[sub] - ymu) / ysd)}
        va_ = {'args': (t_of(Zv),), 'y': t_of((y[va] - ymu) / ysd)}
        m = MLPHead(d)
        m = fit_torch(m, tr_, va_, seed=rep)
        with torch.no_grad():
            pred = m(t_of(Zt)).numpy() * ysd + ymu
        rs.append(np.corrcoef(pred, y[te])[0, 1])
    return (float(np.mean(rs)), float(np.std(rs))) if rs else (np.nan, np.nan)


def run_attn(T, M, W, y, tr, va, te, ratio, n_rep=2):
    rs = []
    for rep in range(n_rep):
        rng = np.random.RandomState(1000 + rep)
        k = max(4, int(len(tr) * ratio))
        sub = rng.choice(tr, size=min(k, len(tr)), replace=False)
        # 逐维标准化 + 逐 token PCA：统计量只用标签子集的 token
        flat = T[sub][M[sub] > 0.5]
        sc = StandardScaler().fit(flat)
        d = min(PCA_DIM, T.shape[2], len(sub) - 1)
        pca = PCA(n_components=d, random_state=0).fit(sc.transform(flat))
        Z = pca.transform(sc.transform(T.reshape(-1, T.shape[2]))
                          ).reshape(len(T), T.shape[1], d).astype(np.float32)
        ymu, ysd = y[sub].mean(), y[sub].std() + 1e-9
        tr_ = {'args': (t_of(Z[sub]), t_of(M[sub]), t_of(W[sub])),
               'y': t_of((y[sub] - ymu) / ysd)}
        va_ = {'args': (t_of(Z[va]), t_of(M[va]), t_of(W[va])),
               'y': t_of((y[va] - ymu) / ysd)}
        m = AttnPool(d)
        m = fit_torch(m, tr_, va_, seed=rep)
        with torch.no_grad():
            pred = m(t_of(Z[te]), t_of(M[te]), t_of(W[te])).numpy() * ysd + ymu
        rs.append(np.corrcoef(pred, y[te])[0, 1])
    return (float(np.mean(rs)), float(np.std(rs))) if rs else (np.nan, np.nan)


VARIANTS = ['sum', 'mean', 'concat', 'attn']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+',
                    default=[t for t in TASKS if t != 'mon'])
    ap.add_argument('--out', default='out/agg_variants.json')
    a = ap.parse_args()

    cache = np.load(os.path.join(HERE, 'out', 'encoder_cache.npz'),
                    allow_pickle=False)
    smi_all = [str(s) for s in cache['smiles']]
    sidx = {s: i for i, s in enumerate(smi_all)}
    FP = fp_matrix(smi_all, 'rdkit2d')
    print('RDKit2D %s | 分子 %d' % (FP.shape, len(smi_all)))

    print()
    hdr = '%-11s %6s' % ('task', 'ratio') + ''.join('%9s' % v for v in VARIANTS)
    print(hdr)
    print('-' * len(hdr))

    RES = {}
    for t in a.tasks:
        d = load_task(t)
        cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
        split_dir = os.path.join(d['base'], d['split'])
        from safetensors.numpy import load_file
        p0 = os.path.join(split_dir, 'kfold_split_0.safetensors')
        if not os.path.exists(p0):
            print('%-11s 无划分，跳过' % t)
            continue
        f0 = load_file(p0)
        tr, va, te = f0['train_indices'], f0['val_indices'], \
            f0['test_indices']

        T, M, W = build_tokens(FP, cid2i, d['ids'], d['ws'])
        RES[t] = {}
        for ratio in RATIOS:
            row = {}
            line = '%-11s %5.0f%%' % (t, ratio * 100)
            for v in VARIANTS[:3]:
                X = pool_fixed(T, M, W, v)
                if d['cond'] is not None:
                    X = np.concatenate([X, d['cond']], 1)
                r, sd = run_fixed(X, d['y'], tr, va, te, ratio)
                row[v] = dict(R=r, sd=sd)
                line += ' %9.3f' % r
            r, sd = run_attn(T, M, W, d['y'], tr, va, te, ratio)
            row['attn'] = dict(R=r, sd=sd)
            line += ' %9.3f' % r
            print(line, flush=True)
            RES[t]['%.2f' % ratio] = row
        print()

    json.dump(RES, open(os.path.join(HERE, a.out), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('saved', a.out)


if __name__ == '__main__':
    main()
