# -*- coding: utf-8 -*-
"""agg_100bin_mixpca.py —— 修正版：PCA 在**混合物层**做，不在单分子层做

上一版把逐分子向量先 PCA 到 64 维（PCA 拟合于分子间差异），会丢掉「该分子在
混合物里怎么表现」的方向，对编码器尤其不利。本版改为：

    sum / mean : 先池化（原生维度，enc 1792、rdkit 217、ecfp4 2048、maccs 167）
                 → 在 100 条标签的混合物上拟合 StandardScaler + PCA(64)
    concat     : 在 100 条标签的拼接向量（≤5 槽 × d）上拟合 PCA(64)；
                 利用线性性，把投影矩阵切成 5 块 W_k，对任意行只用
                 z = Σ_k W_k (x_k − μ_k)/σ_k 计算，无需物化巨大的拼接矩阵

注意力池化需要逐 token 的表示，无法在混合物层做 PCA，本版不包含（另见上一版）。

用法：
    python agg_100bin_mixpca.py --readout mlp
    python agg_100bin_mixpca.py --readout xgb
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_EXT = os.path.abspath(os.path.join(HERE, '..', '..', '06_transfer', 'scripts'))
if _EXT not in sys.path:
    sys.path.insert(0, _EXT)

import agg_variants as A  # noqa: E402
from fp_matrix import fp_matrix  # noqa: E402
from leave_one_task_out import TASKS, load_task  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

FEATS = ['rdkit2d', 'ecfp4', 'maccs', 'enc']
AGGS = ['sum', 'mean', 'concat']
MAXK = 5
PCA_DIM = 64


def pick(rng, pool, n_main, n_va_max=50):
    sub = rng.choice(pool, size=min(n_main, len(pool)), replace=False)
    rest = np.setdiff1d(pool, sub)
    if len(rest) == 0:
        return sub, sub
    n_va = min(n_va_max, max(8, len(rest) // 4))
    return sub, rng.choice(rest, size=min(n_va, len(rest)), replace=False)


def row_slots(ids_row, ws_row, cid2i, maxk=MAXK):
    """[(compound_row, weight_normalised)] 按权重降序，最多 maxk 个"""
    pairs = [(cid2i[int(c)], float(w)) for c, w in zip(ids_row, ws_row)
             if int(c) in cid2i]
    pairs.sort(key=lambda t: -t[1])
    pairs = pairs[:maxk]
    tot = sum(w for _, w in pairs) or 1.0
    return [(j, w / tot) for j, w in pairs]


def pool_rows(V, d, ids, ws, cid2i, rows, kind):
    """对指定行构造池化向量：sum / mean"""
    out = np.zeros((len(rows), V.shape[1]), np.float32)
    for r, i in enumerate(rows):
        slots = row_slots(ids[i], ws[i], cid2i)
        if not slots:
            continue
        if kind == 'sum':
            v = np.zeros(V.shape[1], np.float32)
            for j, w in slots:
                v += w * V[j]
            out[r] = v
        else:  # mean
            v = np.zeros(V.shape[1], np.float32)
            for j, _ in slots:
                v += V[j]
            out[r] = v / len(slots)
    return out


def concat_rows(V, ids, ws, cid2i, rows):
    """拼接向量（物化；只对少量行使用，例如 100 条标签）"""
    d = V.shape[1]
    out = np.zeros((len(rows), MAXK * d), np.float32)
    for r, i in enumerate(rows):
        for k, (j, _) in enumerate(row_slots(ids[i], ws[i], cid2i)):
            out[r, k * d:(k + 1) * d] = V[j]
    return out


def concat_project(V, ids, ws, cid2i, rows, blocks, mu, sc):
    """用切块投影矩阵对任意行算 z = Σ_k W_k (x_k−μ_k)/σ_k"""
    d = V.shape[1]
    out = np.zeros((len(rows), blocks[0].shape[0]), np.float32)
    for r, i in enumerate(rows):
        for k, (j, _) in enumerate(row_slots(ids[i], ws[i], cid2i)):
            x = (V[j] - mu[k * d:(k + 1) * d]) / sc[k * d:(k + 1) * d]
            out[r] += blocks[k] @ x
    return out


def run_mlp(Xs, Xv, Xt, ys, yv, yt_raw, seed, epochs=200):
    import torch
    torch.manual_seed(seed)
    m = A.MLPHead(Xs.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
    xs, yv_t = (torch.tensor(Xs, dtype=torch.float32),
                torch.tensor(ys, dtype=torch.float32))
    xv = torch.tensor(Xv, dtype=torch.float32)
    yvv = torch.tensor(yv, dtype=torch.float32)
    xt = torch.tensor(Xt, dtype=torch.float32)
    n = len(ys)
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
            ((m(xs[j]) - yv_t[j]) ** 2).mean().backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(((m(xv) - yvv) ** 2).mean())
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
        return m(xt).numpy()


def run_xgb(Xs, Xv, Xt, ys, yv, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.05, max_depth=4,
                         subsample=0.9, colsample_bytree=0.5, reg_lambda=1.0,
                         random_state=seed, n_jobs=4,
                         early_stopping_rounds=50, eval_metric='rmse')
    m.fit(Xs, ys, eval_set=[(Xv, yv)], verbose=False)
    return m.predict(Xt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--readout', choices=['mlp', 'xgb'], default='mlp')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or ('../runs/agg_100bin_mixpca_%s.json' % a.readout)

    cache = np.load(os.path.join(_EXT, 'out', 'encoder_cache.npz'),
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

        mats = dict(RAW)
        mats['enc'] = np.concatenate([ENC[x] for x in sorted(ENC) if x != t],
                                     axis=1)
        print()
        print('=== %s（二元池 %d，取 %d；测试 %d；读出 %s）==='
              % (t, len(pool), min(a.n, len(pool)), len(te), a.readout))
        hdr = '%-8s' % 'feat' + ''.join('%10s' % v for v in AGGS)
        print(hdr)
        print('-' * len(hdr))

        RES[t] = {}
        for feat in FEATS:
            V = mats[feat]
            row, line = {}, '%-8s' % feat
            for kind in AGGS:
                rs = []
                for rep in range(a.reps):
                    rng = np.random.RandomState(2000 + rep)
                    sub, va = pick(rng, pool, a.n)
                    cond = d['cond']
                    if kind == 'concat':
                        C = concat_rows(V, d['ids'], d['ws'], cid2i, sub)
                        sc = StandardScaler().fit(C)
                        dd = min(PCA_DIM, C.shape[1], len(sub) - 1)
                        pca = PCA(n_components=dd, random_state=0)
                        pca.fit(sc.transform(C))
                        W = pca.components_
                        blocks = [W[:, k * V.shape[1]:(k + 1) * V.shape[1]]
                                  for k in range(MAXK)]
                        mu = sc.mean_
                        sd = sc.scale_
                        Zs = pca.transform(sc.transform(C))
                        Zv = concat_project(V, d['ids'], d['ws'], cid2i,
                                            va, blocks, mu, sd)
                        Zt = concat_project(V, d['ids'], d['ws'], cid2i,
                                            te, blocks, mu, sd)
                    else:
                        Xs = pool_rows(V, V.shape[1], d['ids'], d['ws'],
                                       cid2i, sub, kind)
                        Xv = pool_rows(V, V.shape[1], d['ids'], d['ws'],
                                       cid2i, va, kind)
                        Xt = pool_rows(V, V.shape[1], d['ids'], d['ws'],
                                       cid2i, te, kind)
                        if cond is not None:
                            Xs = np.concatenate([Xs, cond[sub]], 1)
                            Xv = np.concatenate([Xv, cond[va]], 1)
                            Xt = np.concatenate([Xt, cond[te]], 1)
                        sc = StandardScaler().fit(Xs)
                        dd = min(PCA_DIM, Xs.shape[1], len(sub) - 1)
                        pca = PCA(n_components=dd, random_state=0)
                        pca.fit(sc.transform(Xs))
                        Zs = pca.transform(sc.transform(Xs))
                        Zv = pca.transform(sc.transform(Xv))
                        Zt = pca.transform(sc.transform(Xt))

                    mu_y, sd_y = d['y'][sub].mean(), d['y'][sub].std() + 1e-9
                    ys = (d['y'][sub] - mu_y) / sd_y
                    yv = (d['y'][va] - mu_y) / sd_y
                    if a.readout == 'mlp':
                        pred = run_mlp(Zs, Zv, Zt, ys, yv, d['y'][te], rep)
                        pred = pred * sd_y + mu_y
                    else:
                        pred = run_xgb(Zs, Zv, Zt, d['y'][sub], d['y'][va],
                                       rep)
                    rs.append(np.corrcoef(pred, d['y'][te])[0, 1])
                row[kind] = dict(R=float(np.mean(rs)), sd=float(np.std(rs)))
                line += '%10.3f' % np.mean(rs)
            print(line, flush=True)
            RES[t][feat] = row

    json.dump(RES, open(os.path.join(HERE, out), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
