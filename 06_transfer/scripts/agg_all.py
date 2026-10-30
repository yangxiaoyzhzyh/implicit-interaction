# -*- coding: utf-8 -*-
"""agg_all.py —— 四种特征 × 四种聚合（1% 标签）

特征：rdkit2d(217) / ecfp4(2048) / maccs(167) / enc(1792，七个迁移编码器)
聚合：sum（摩尔分数加权和）/ mean / concat（前 5 槽）/ attn（单头注意力池化）

统一流程（保证各臂可比）：
    逐分子向量 → 在 1% 子集涉及的分子上拟合无监督 PCA 降到 64 维
    → 按组分构造令牌（最多 5 个）→ 池化 → 标准化/PCA(64) → 同一 MLP
MON 的 1% 只有约 5 个标签，结果仅作参考。
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

import agg_variants as A  # noqa: E402
from fp_matrix import fp_matrix  # noqa: E402
from leave_one_task_out import TASKS, load_task  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

MAXK = 5
EMB = 64
FEATS = ['rdkit2d', 'ecfp4', 'maccs', 'enc']
AGGS = ['sum', 'mean', 'concat', 'attn']


def reduce_per_mol(V, rows, dim=EMB):
    sc = StandardScaler().fit(V[rows])
    n = min(dim, len(rows) - 1, V.shape[1])
    pca = PCA(n_components=n, random_state=0).fit(sc.transform(V[rows]))
    return pca.transform(sc.transform(V)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--ratio', type=float, default=0.01)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default='out/agg_all_1pct.json')
    a = ap.parse_args()

    cache = np.load(os.path.join(HERE, 'out', 'encoder_cache.npz'),
                    allow_pickle=False)
    smi_all = [str(s) for s in cache['smiles']]
    sidx = {s: i for i, s in enumerate(smi_all)}
    ENC = {k.replace('_vec', ''): cache[k] for k in cache.files
           if k.endswith('_vec')}
    RAW = {f: fp_matrix(smi_all, f) for f in ('rdkit2d', 'ecfp4', 'maccs')}
    print('分子 %d | %s' % (len(smi_all),
                           ' '.join('%s%s' % (f, v.shape) for f, v in RAW.items())))

    RES = {}
    for t in a.tasks:
        d = load_task(t)
        cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
        p0 = os.path.join(d['base'], d['split'], 'kfold_split_0.safetensors')
        if not os.path.exists(p0):
            continue
        f0 = load_file(p0)
        tr, va, te = f0['train_indices'], f0['val_indices'], f0['test_indices']

        rng = np.random.RandomState(1000)
        k = max(4, int(len(tr) * a.ratio))
        sub = rng.choice(tr, size=min(k, len(tr)), replace=False)
        comps = sorted({int(c) for i in sub for c in d['ids'][i]
                        if int(c) in cid2i})
        rows = [cid2i[c] for c in comps]

        mats = dict(RAW)
        mats['enc'] = np.concatenate([ENC[x] for x in sorted(ENC) if x != t],
                                     axis=1)

        print()
        print('=== %s（训练 %d，1%% = %d 行，可用组分 %d）==='
              % (t, len(tr), len(sub), len(rows)))
        hdr = '%-8s' % 'feat' + ''.join('%10s' % v for v in AGGS)
        print(hdr)
        print('-' * len(hdr))

        RES[t] = {}
        for feat in FEATS:
            V = mats[feat]
            if len(rows) > 2:
                V = reduce_per_mol(V, rows)
            T, M, W = A.build_tokens(V, cid2i, d['ids'], d['ws'], maxk=MAXK)
            row = {}
            line = '%-8s' % feat
            for v in AGGS:
                if v == 'attn':
                    r, sd = A.run_attn(T, M, W, d['y'], tr, va, te, a.ratio,
                                       n_rep=a.reps)
                else:
                    X = A.pool_fixed(T, M, W, v)
                    if d['cond'] is not None:
                        X = np.concatenate([X, d['cond']], 1)
                    r, sd = A.run_fixed(X, d['y'], tr, va, te, a.ratio,
                                        n_rep=a.reps)
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
