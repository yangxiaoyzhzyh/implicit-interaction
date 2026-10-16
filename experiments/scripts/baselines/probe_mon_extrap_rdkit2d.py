# -*- coding: utf-8 -*-
"""probe_mon_extrap_rdkit2d.py — 共享工具：RDKit 2D 描述符（217 维）计算

实验 9 的公共依赖（本身不产出数据）
被 run_5tasks_rdkit_xgb.py / run_spe_xgb_k3.py / run_xgb_coverage.py / run_sisson_rdkit_linear.py 以 `from probe_mon_extrap_rdkit2d import rdkit2d` 复用
注意：该文件 __main__ 部分引用的 scripts/data/MON 目录已不存在，仅 rdkit2d() 函数在用"""

import csv
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
MON_DATA = os.path.join(HERE, '..', 'data', 'MON')
cache = {}


def rdkit2d(smi):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    if smi in cache:
        return cache[smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        cache[smi] = None
        return None
    try:
        d = Descriptors.CalcMolDescriptors(mol)
        v = np.array(list(d.values()), dtype=np.float64)
        v[~np.isfinite(v)] = 0.0
        v = np.clip(v, -1e10, 1e10)  # Ipc 等描述符在聚合物上会爆炸性取值
        cache[smi] = v
    except Exception:
        cache[smi] = None
    return cache[smi]


def main():
    df = pd.read_csv(os.path.join(MON_DATA, 'processed_MON.csv'))
    cmp_smiles = {}
    with open(os.path.join(MON_DATA, 'compounds.csv')) as f:
        for r in csv.DictReader(f):
            cmp_smiles[int(float(r['compound_id']))] = r['smiles']
    X, y, ncomps = [], [], []
    for _, r in df.iterrows():
        cids = eval(r['cmp_ids'])
        mfs = eval(r['cmp_mole_fractions'])
        acc = None
        ok = True
        for cid, w in zip(cids, mfs):
            smi = cmp_smiles.get(int(float(cid)))
            v = rdkit2d(smi) if smi else None
            if v is None:
                ok = False
                break
            acc = w * v if acc is None else acc + w * v
        if not ok:
            continue
        X.append(acc)
        y.append(float(r['value']))
        ncomps.append(len(cids))
    X = np.stack(X)
    y = np.array(y)
    ncomps = np.array(ncomps)
    print('samples:', len(X), '| descriptor dims:', X.shape[1])
    tr = ncomps < 3
    te = ncomps > 3
    print('train %d (1-2 comp) | test %d (>3 comp)' % (tr.sum(), te.sum()))
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                     subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                     n_jobs=8, verbosity=0)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    r = pearsonr(p, y[te])[0]
    mae = float(np.mean(np.abs(p - y[te])))
    print()
    print('>> RDKit2D + XGBoost extrap: test R = %.4f | MAE = %.4f (n=%d)'
          % (r, mae, te.sum()))
    print('   vs frozen task-shaped encoder + XGB: R = 0.9111 | MAE = 2.7460')
    print('   vs end-to-end extrap:                R = 0.3512 | MAE = 11.3180')
    print()
    print('test breakdown by component count:')
    for nc in sorted(set(ncomps[te])):
        msk = te & (ncomps == nc)
        pp = m.predict(X[msk])
        rr = pearsonr(pp, y[msk])[0] if msk.sum() > 2 else float('nan')
        mm = float(np.mean(np.abs(pp - y[msk])))
        print('  %d comp: n=%3d R=%.4f MAE=%.4f' % (nc, msk.sum(), rr, mm))


if __name__ == '__main__':
    main()
