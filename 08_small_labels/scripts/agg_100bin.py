# -*- coding: utf-8 -*-
"""agg_100bin.py —— 更严苛的小样本：只用 100 条二元（K=2）混合物训练

每个任务：
    训练池 = 官方 train 中 K==2 的行 → 抽 100 条
    验证池 = 同池中余下的行 → 抽 50 条（不足则取 20%）
    测试   = 官方 test 全部
    四种特征 × 四种聚合（管线与 agg_all.py 相同）

用法：python agg_100bin.py [--tasks ...] [--n 100] [--reps 2]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agg_variants as A  # noqa: E402
from fp_matrix import fp_matrix  # noqa: E402
from leave_one_task_out import TASKS, load_task  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

FEATS = ['rdkit2d', 'ecfp4', 'maccs', 'enc']
AGGS = ['sum', 'mean', 'concat', 'attn']


def t_of(x):
    return torch.tensor(x, dtype=torch.float32)


def fit_model(kind, d_in, tr_args, va_args, te_args, y_tr, y_va, seed,
              epochs=200):
    torch.manual_seed(seed)
    m = A.MLPHead(d_in) if kind == 'mlp' else A.AttnPool(d_in)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
    n = len(y_tr)
    bs = min(64, max(8, n // 4))
    best, best_state, bad = 1e18, None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad()
            F.mse_loss(m(*(t_of(a[j]) for a in tr_args)), y_tr[j]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(F.mse_loss(m(*(t_of(a) for a in va_args)), y_va))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return m(*(t_of(a) for a in te_args)).numpy()


def pick(rng, pool, n_main, n_va_max=50):
    sub = rng.choice(pool, size=min(n_main, len(pool)), replace=False)
    rest = np.setdiff1d(pool, sub)
    if len(rest) == 0:
        return sub, sub
    n_va = min(n_va_max, max(8, len(rest) // 4))
    va = rng.choice(rest, size=min(n_va, len(rest)), replace=False)
    return sub, va


def run_vec(X, y, pool, te, n_main, reps, seed0=2000):
    rs = []
    for rep in range(reps):
        rng = np.random.RandomState(seed0 + rep)
        sub, va = pick(rng, pool, n_main)
        mu, sd = y[sub].mean(), y[sub].std() + 1e-9
        pred = fit_model('mlp', X.shape[1], (X[sub],), (X[va],), (X[te],),
                         t_of((y[sub] - mu) / sd),
                         t_of((y[va] - mu) / sd), rep)
        pred = pred * sd + mu
        rs.append(np.corrcoef(pred, y[te])[0, 1])
    return float(np.mean(rs)), float(np.std(rs))


def run_tok(T, M, W, y, pool, te, n_main, reps, seed0=2000):
    rs = []
    for rep in range(reps):
        rng = np.random.RandomState(seed0 + rep)
        sub, va = pick(rng, pool, n_main)
        flat = T[sub][M[sub] > 0.5]
        sc = StandardScaler().fit(flat)
        d = min(64, T.shape[2], len(sub) - 1)
        pca = PCA(n_components=d, random_state=0).fit(sc.transform(flat))
        Z = pca.transform(sc.transform(T.reshape(-1, T.shape[2]))
                          ).reshape(len(T), T.shape[1], d).astype(np.float32)
        mu, sd = y[sub].mean(), y[sub].std() + 1e-9
        pred = fit_model('attn', d,
                         (Z[sub], M[sub], W[sub]),
                         (Z[va], M[va], W[va]),
                         (Z[te], M[te], W[te]),
                         t_of((y[sub] - mu) / sd),
                         t_of((y[va] - mu) / sd), rep)
        pred = pred * sd + mu
        rs.append(np.corrcoef(pred, y[te])[0, 1])
    return float(np.mean(rs)), float(np.std(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default='out/agg_100bin.json')
    a = ap.parse_args()

    cache = np.load(os.path.join(HERE, 'out', 'encoder_cache.npz'),
                    allow_pickle=False)
    smi_all = [str(s) for s in cache['smiles']]
    sidx = {s: i for i, s in enumerate(smi_all)}
    ENC = {k.replace('_vec', ''): cache[k] for k in cache.files
           if k.endswith('_vec')}
    RAW = {f: fp_matrix(smi_all, f) for f in ('rdkit2d', 'ecfp4', 'maccs')}

    RES = {}
    for t in a.tasks:
        d = load_task(t)
        cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
        p0 = os.path.join(d['base'], d['split'], 'kfold_split_0.safetensors')
        if not os.path.exists(p0):
            continue
        f0 = load_file(p0)
        tr, te = f0['train_indices'], f0['test_indices']
        kk = np.array([len(x) for x in d['ids']])
        pool = np.array([i for i in tr if kk[i] == 2])
        if len(pool) < 20:
            print('%-11s 二元训练行仅 %d，跳过' % (t, len(pool)))
            continue

        rng = np.random.RandomState(999)
        head = rng.choice(pool, size=min(a.n, len(pool)), replace=False)
        comps = sorted({int(c) for i in head for c in d['ids'][i]
                        if int(c) in cid2i})
        rows = [cid2i[c] for c in comps]
        mats = dict(RAW)
        mats['enc'] = np.concatenate([ENC[x] for x in sorted(ENC) if x != t],
                                     axis=1)

        print()
        print('=== %s（二元训练池 %d，取 %d；测试 %d）==='
              % (t, len(pool), min(a.n, len(pool)), len(te)))
        hdr = '%-8s' % 'feat' + ''.join('%10s' % v for v in AGGS)
        print(hdr)
        print('-' * len(hdr))
        RES[t] = {}
        for feat in FEATS:
            V = mats[feat]
            if len(rows) > 2:
                sc = StandardScaler().fit(V[rows])
                d_ = min(64, len(rows) - 1, V.shape[1])
                pca = PCA(n_components=d_, random_state=0)
                pca.fit(sc.transform(V[rows]))
                V = pca.transform(sc.transform(V)).astype(np.float32)
            T, M, W = A.build_tokens(V, cid2i, d['ids'], d['ws'], maxk=3)
            row, line = {}, '%-8s' % feat
            for v in AGGS:
                if v == 'attn':
                    r, sd = run_tok(T, M, W, d['y'], pool, te, a.n, a.reps)
                else:
                    X = A.pool_fixed(T, M, W, v)
                    if d['cond'] is not None:
                        X = np.concatenate([X, d['cond']], 1)
                    r, sd = run_vec(X, d['y'], pool, te, a.n, a.reps)
                row[v] = dict(R=r, sd=sd)
                line += '%10.3f' % r
            print(line, flush=True)
            RES[t][feat] = row

    out = os.path.join(HERE, a.out)
    json.dump(RES, open(out, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
