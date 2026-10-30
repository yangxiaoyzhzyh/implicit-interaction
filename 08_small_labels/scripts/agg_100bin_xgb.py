# -*- coding: utf-8 -*-
"""agg_100bin_xgb.py —— 100 条二元标签下的 XGBoost 读出

与 agg_100bin.py 完全相同的协议与数据，只把读出换成梯度提升树：
    训练 = 100 条 K=2；验证 = 50 条 K=2（用于 early stopping）；测试 = 官方 test 全部
特征 = {RDKit2D, ECFP4, MACCS, 七编码器} × {加权和, 不加权平均, 拼接}
    注意力池化没有对应的 XGB 版本（它本身是带参数的网络），故不列入。
"""
import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agg_variants as A  # noqa: E402
from fp_matrix import fp_matrix  # noqa: E402
from leave_one_task_out import TASKS, load_task  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

FEATS = ['rdkit2d', 'ecfp4', 'maccs', 'enc']
AGGS = ['sum', 'mean', 'concat']


def pick(rng, pool, n_main, n_va_max=50):
    sub = rng.choice(pool, size=min(n_main, len(pool)), replace=False)
    rest = np.setdiff1d(pool, sub)
    if len(rest) == 0:
        return sub, sub
    n_va = min(n_va_max, max(8, len(rest) // 4))
    return sub, rng.choice(rest, size=min(n_va, len(rest)), replace=False)


def fit_xgb(Xtr, ytr, Xva, yva, Xte, seed):
    m = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.5, reg_lambda=1.0,
        random_state=seed, n_jobs=4, early_stopping_rounds=50,
        eval_metric='rmse')
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default='../runs/agg_100bin_xgb.json')
    a = ap.parse_args()

    cache = np.load(os.path.join(HERE, '..', '..', '06_transfer', 'scripts',
                                 'out', 'encoder_cache.npz'), allow_pickle=False)
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

        rng0 = np.random.RandomState(999)
        head = rng0.choice(pool, size=min(a.n, len(pool)), replace=False)
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
                X = A.pool_fixed(T, M, W, v)
                if d['cond'] is not None:
                    X = np.concatenate([X, d['cond']], 1)
                rs = []
                for rep in range(a.reps):
                    rng = np.random.RandomState(2000 + rep)
                    sub, va = pick(rng, pool, a.n)
                    pred = fit_xgb(X[sub], d['y'][sub], X[va], d['y'][va],
                                   X[te], rep)
                    rs.append(np.corrcoef(pred, d['y'][te])[0, 1])
                row[v] = dict(R=float(np.mean(rs)), sd=float(np.std(rs)))
                line += '%10.3f' % np.mean(rs)
            print(line, flush=True)
            RES[t][feat] = row

    out = os.path.join(HERE, a.out)
    json.dump(RES, open(out, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
