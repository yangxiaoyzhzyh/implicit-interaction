# -*- coding: utf-8 -*-
"""additive_baseline.py —— 纯线性可加基线（任务可加性的直接度量）

模型：
    y = Σ_m c_m · x_m + Σ_j β_j · cond_j + b

    x_m = 分子 m 在该样本中的摩尔分数
    c_m = 分子 m 的「固有贡献」系数（每个分子一个）

这就是 paper_v2 (SM1022 §M3) 里的 Additive baseline：
闭式最小二乘，无超参数，无随机性，不涉及任何分子表示。

意义：
    它给出「如果任务完全可加，最好能做到多少」的上限。
    如果编码器的 others 不能显著超过它，说明编码器只是在数组分。

用法：
    python additive_baseline.py                # 全部 8 任务
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from leave_one_task_out import TASKS, load_task  # noqa: E402


def build_design(ids, ws, cid2i, n_mol):
    """稀疏设计矩阵：每行是该样本的摩尔分数分布（行归一）"""
    rows, cols, vals = [], [], []
    for i, (ci, wi) in enumerate(zip(ids, ws)):
        pairs = [(cid2i[int(c)], w) for c, w in zip(ci, wi)
                 if int(c) in cid2i]
        tot = sum(w for _, w in pairs)
        if tot <= 0:
            continue
        for j, w in pairs:
            rows.append(i)
            cols.append(j)
            vals.append(w / tot)
    return csr_matrix((vals, (rows, cols)), shape=(len(ids), n_mol))


def cv_ridge_sparse(X, y, split_dir, cond=None, n_fold=5, alpha=None):
    rs, r2s = [], []
    for fi in range(n_fold):
        p = os.path.join(split_dir, 'kfold_split_%d.safetensors' % fi)
        if not os.path.exists(p):
            continue
        from safetensors.numpy import load_file
        d = load_file(p)
        tr, te = d['train_indices'], d['test_indices']
        Xtr, Xte = X[tr], X[te]
        if cond is not None:
            from scipy.sparse import hstack
            Ctr = csr_matrix(cond[tr].astype(np.float64))
            Cte = csr_matrix(cond[te].astype(np.float64))
            Xtr = hstack([Xtr, Ctr]).tocsr()
            Xte = hstack([Xte, Cte]).tocsr()
        if alpha is None:
            m = RidgeCV(alphas=np.logspace(-3, 3, 13))
        else:
            m = Ridge(alpha=alpha)
        m.fit(Xtr, y[tr])
        pred = m.predict(Xte)
        rs.append(np.corrcoef(pred, y[te])[0, 1])
        r2s.append(1 - ((y[te] - pred) ** 2).sum()
                   / ((y[te] - y[te].mean()) ** 2).sum())
    if not rs:
        return np.nan, np.nan, np.nan
    return float(np.mean(rs)), float(np.std(rs)), float(np.mean(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--out', default='out/additive_baseline.json')
    args = ap.parse_args()

    print('=== 纯线性可加基线（每分子一个系数）===')
    print('%-12s %8s %8s %8s %7s'
          % ('任务', 'R', 'sd', 'R²', '分子数'))
    print('-' * 50)

    RES = {}
    for t in args.tasks:
        d = load_task(t)
        smi = sorted(set(d['cid2smi'].values()))
        cid2i = {c: i for i, c in enumerate(
            sorted(set(int(c) for c in d['cid2smi'])))}
        # 用 cid 而不是 smiles 做索引（同一 smiles 可能对应多个 cid）
        all_cid = sorted({int(c) for ci in d['ids'] for c in ci
                          if int(c) in d['cid2smi']})
        # 但有些 cid 不在 cid2smi 里，过滤
        idx = {c: i for i, c in enumerate(all_cid)}
        X = build_design(d['ids'], d['ws'], idx, len(all_cid))
        split_dir = os.path.join(d['base'], d['split'])
        r, sd, r2 = cv_ridge_sparse(X, d['y'], split_dir, d['cond'])
        print('%-12s %8.3f %8.3f %8.3f %7d'
              % (t, r, sd, r2, len(all_cid)), flush=True)
        RES[t] = dict(R=r, sd=sd, R2=r2, n_mol=len(all_cid),
                      n_sample=int(len(d['y'])))

    a = np.array([RES[t]['R'] for t in args.tasks])
    print('-' * 50)
    print('%-12s %8.3f' % ('均值', np.nanmean(a)))

    json.dump(RES, open(os.path.join(HERE, args.out), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print()
    print('saved', args.out)


if __name__ == '__main__':
    main()
