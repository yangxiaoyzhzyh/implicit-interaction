# -*- coding: utf-8 -*-
"""IlT κ、1% 标签：RDKit2D 加权和 vs 逐组分拼接，都用同一个 MLP 读出"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agg_variants as A  # noqa: E402
from fp_matrix import fp_matrix  # noqa: E402
from leave_one_task_out import load_task  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

TASK = 'ionic_cond'      # IlT κ（电导率）
RATIO = 0.01

cache = np.load(os.path.join(HERE, 'out', 'encoder_cache.npz'),
                allow_pickle=False)
smi_all = [str(s) for s in cache['smiles']]
sidx = {s: i for i, s in enumerate(smi_all)}
FP = fp_matrix(smi_all, 'rdkit2d')
print('RDKit2D %s | 缓存分子 %d' % (FP.shape, len(smi_all)))

d = load_task(TASK)
cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
print('任务 %s：%d 行，可用组分映射 %d / %d'
      % (TASK, len(d['y']), len(cid2i), len(d['cid2smi'])))

f0 = load_file(os.path.join(d['base'], d['split'],
                            'kfold_split_0.safetensors'))
tr, va, te = f0['train_indices'], f0['val_indices'], f0['test_indices']
n_lab = max(4, int(len(tr) * RATIO))
print('训练 %d / 验证 %d / 测试 %d｜1%% 标签 = %d 行'
      % (len(tr), len(va), len(te), n_lab))

T, M, W = A.build_tokens(FP, cid2i, d['ids'], d['ws'])
print('令牌张量 %s，每行最多 %d 个组分' % (T.shape, A.MAXK))

res = {}
for v in ('sum', 'mean', 'concat'):
    X = A.pool_fixed(T, M, W, v)
    if d['cond'] is not None:
        X = np.concatenate([X, d['cond']], 1)
    r, sd = A.run_fixed(X, d['y'], tr, va, te, RATIO, n_rep=3)
    res[v] = dict(R=r, sd=sd, dim=int(X.shape[1]))
    print('  %-7s 输入 %5d 维 → R = %.4f ± %.4f' % (v, X.shape[1], r, sd),
          flush=True)

out = os.path.join(HERE, 'out', 'agg_iltk_1pct.json')
json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved', out)
