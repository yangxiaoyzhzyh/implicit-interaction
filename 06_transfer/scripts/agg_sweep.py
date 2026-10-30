# -*- coding: utf-8 -*-
"""agg_sweep.py —— 全部任务 × 1%/5% × 两种特征 × 四种聚合

特征：
    rdkit   RDKit 2D 描述符（217 维/分子）
    enc     其余七个任务的塑形编码器拼接（1792 维/分子；只在标签子集涉及的
            分子上做一次无监督 PCA 降到 64 维，避免维数爆炸）
聚合：
    sum     摩尔分数加权和（论文现用）
    mean    不加权平均
    concat  按权重降序取前 5 槽、零填充后拼接
    attn    单头注意力池化（可学习，带权重标量作为输入）

协议同 mlp_small_data.py：官方 fold 0，2 次重复，所有投影只在标签子集上拟合。
MON 有 121 个组分，concat/attn 只保留权重最大的 5 个（表注需说明）。
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


def reduce_per_molecule(V, rows, dim=EMB):
    """在给定分子行上拟合无监督 PCA，把每分子向量降到 dim 维"""
    sc = StandardScaler().fit(V[rows])
    n = min(dim, len(rows) - 1, V.shape[1])
    pca = PCA(n_components=n, random_state=0).fit(sc.transform(V[rows]))
    return pca.transform(sc.transform(V)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--ratios', nargs='+', type=float, default=[0.01, 0.05])
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default='out/agg_sweep.json')
    a = ap.parse_args()

    cache = np.load(os.path.join(HERE, 'out', 'encoder_cache.npz'),
                    allow_pickle=False)
    smi_all = [str(s) for s in cache['smiles']]
    sidx = {s: i for i, s in enumerate(smi_all)}
    ENC = {k.replace('_vec', ''): cache[k] for k in cache.files
           if k.endswith('_vec')}
    FP = fp_matrix(smi_all, 'rdkit2d')
    print('分子 %d｜编码器 %d 个｜RDKit2D %s' % (len(smi_all), len(ENC), FP.shape))

    RES = {}
    for t in a.tasks:
        d = load_task(t)
        cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
        p0 = os.path.join(d['base'], d['split'], 'kfold_split_0.safetensors')
        if not os.path.exists(p0):
            print('%-11s 无划分，跳过' % t)
            continue
        f0 = load_file(p0)
        tr, va, te = f0['train_indices'], f0['val_indices'], f0['test_indices']

        # 七个编码器的拼接（不含本任务）
        V = np.concatenate([ENC[k] for k in sorted(ENC) if k != t], axis=1)
        cond = d['cond']
        base_rd = A.build_tokens(FP, cid2i, d['ids'], d['ws'], maxk=MAXK)

        RES[t] = {}
        print()
        print('=== %s（训练 %d / 测试 %d）==='
              % (t, len(tr), len(te)))
        hdr = '%-6s %-6s %8s %8s %8s %8s' % ('ratio', 'feat', 'sum', 'mean',
                                             'concat', 'attn')
        print(hdr)
        print('-' * len(hdr))

        for ratio in a.ratios:
            # 编码器降维用的分子：该比例子集里出现的组分
            rng = np.random.RandomState(1000)
            k = max(4, int(len(tr) * ratio))
            sub = rng.choice(tr, size=min(k, len(tr)), replace=False)
            comps = sorted({int(c) for i in sub for c in d['ids'][i]
                            if int(c) in cid2i})
            V64 = (reduce_per_molecule(V, [cid2i[c] for c in comps])
                   if len(comps) > 2 else None)
            toks = {'rdkit': base_rd}
            if V64 is not None:
                toks['enc'] = A.build_tokens(V64, cid2i, d['ids'], d['ws'],
                                             maxk=MAXK)
            RES[t]['%.2f' % ratio] = {}
            for feat in ('rdkit', 'enc'):
                if feat not in toks:
                    continue
                T, M, W = toks[feat]
                row = {}
                line = '%-6.0f%% %-6s' % (ratio * 100, feat)
                for v in ('sum', 'mean', 'concat'):
                    X = A.pool_fixed(T, M, W, v)
                    if cond is not None:
                        X = np.concatenate([X, cond], 1)
                    r, sd = A.run_fixed(X, d['y'], tr, va, te, ratio,
                                        n_rep=a.reps)
                    row[v] = dict(R=r, sd=sd, dim=int(X.shape[1]))
                    line += ' %8.3f' % r
                r, sd = A.run_attn(T, M, W, d['y'], tr, va, te, ratio,
                                   n_rep=a.reps)
                row['attn'] = dict(R=r, sd=sd)
                line += ' %8.3f' % r
                print(line, flush=True)
                RES[t]['%.2f' % ratio][feat] = row

    out = os.path.join(HERE, a.out)
    json.dump(RES, open(out, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
