# -*- coding: utf-8 -*-
"""IlT κ、1% 标签：七个迁移编码器（1792 维/分子）的聚合方式对比

为了控制内存，先对「逐分子向量」做一次 PCA（只在 1% 标签涉及的分子上拟合），
降到 64 维，再按组分构造成令牌；随后三种池化：加权和 / 不加权平均 / 拼接。
"""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agg_variants as A  # noqa: E402
from leave_one_task_out import load_task  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

TASK = 'ionic_cond'
RATIO = 0.01
EMB = 64          # 每个分子（组分）降到的维数

cache = np.load(os.path.join(HERE, 'out', 'encoder_cache.npz'),
                allow_pickle=False)
smi_all = [str(s) for s in cache['smiles']]
sidx = {s: i for i, s in enumerate(smi_all)}
ENC = {k.replace('_vec', ''): cache[k] for k in cache.files
       if k.endswith('_vec')}
print('编码器 %d 个，每个 %d 维' % (len(ENC), next(iter(ENC.values())).shape[1]))

d = load_task(TASK)
cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
f0 = load_file(os.path.join(d['base'], d['split'],
                            'kfold_split_0.safetensors'))
tr, va, te = f0['train_indices'], f0['val_indices'], f0['test_indices']
rng = np.random.RandomState(1000)
n_lab = max(4, int(len(tr) * RATIO))
sub = rng.choice(tr, size=n_lab, replace=False)

V = np.concatenate([ENC[k] for k in sorted(ENC) if k != TASK], axis=1)
print('拼接编码器矩阵 %s（不含本任务）' % (V.shape,))

# 只用 1% 子集里出现的分子拟合降维
comps_sub = sorted({int(c) for i in sub for c in d['ids'][i]
                    if int(c) in cid2i})
rows_sub = [cid2i[c] for c in comps_sub]
sc = StandardScaler().fit(V[rows_sub])
pca = PCA(n_components=min(EMB, len(rows_sub) - 1), random_state=0)
pca.fit(sc.transform(V[rows_sub]))
V64 = pca.transform(sc.transform(V)).astype(np.float32)
print('逐分子降维：%d → %d 维（用 %d 个分子的特征拟合，不用标签）'
      % (V.shape[1], V64.shape[1], len(rows_sub)))

T, M, W = A.build_tokens(V64, cid2i, d['ids'], d['ws'], maxk=3)
print('令牌张量 %s（IlT κ 最多 3 组分）' % (T.shape,))

res = {}
for v in ('sum', 'mean', 'concat'):
    X = A.pool_fixed(T, M, W, v)
    if d['cond'] is not None:
        X = np.concatenate([X, d['cond']], 1)
    r, sd = A.run_fixed(X, d['y'], tr, va, te, RATIO, n_rep=3)
    res[v] = dict(R=r, sd=sd, dim=int(X.shape[1]))
    print('  %-7s 输入 %5d 维 → R = %.4f ± %.4f' % (v, X.shape[1], r, sd),
          flush=True)

out = os.path.join(HERE, 'out', 'agg_iltk_enc_1pct.json')
json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved', out)
