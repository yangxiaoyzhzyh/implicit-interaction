# -*- coding: utf-8 -*-
"""xgb_rdkit.py —— RDKit 2D + XGBoost 对照臂（组分数轴）

与 frozen.py 完全相同的**聚合**（摩尔分数加权求和）和**特征**（RDKit 2D 217 维 + 条件列），
只把读出从 MLP 换成梯度提升树。用途：区分「塑料编码器的增益」与「非线性读出的增益」。

超参与旧归档一致（paper_v2/.../baselines/run_5tasks_rdkit_xgb.py）：
    n_estimators=800, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0

与 engine.py 对齐的两处：
  1. npz 的 df 行标签 → 记录位置；
  2. val_idx 并回训练集（组分数轴不用验证集）。

    python xgb_rdkit.py --task miscible --split-file ../splits/miscible_composition.npz \
                        --data-base D --out ../01_composition/runs
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr
from xgboost import XGBRegressor

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))

from spec import load_records, spec_of  # noqa: E402

CACHE = os.path.join(HERE, '_desc_cache.json')
NDESC = len(Descriptors.descList)
SEEDS = [42, 43, 44]


def descriptors(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    out = []
    for _, fn in Descriptors.descList:
        try:
            v = float(fn(m))
            out.append(v if np.isfinite(v) else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def build(task, data_base, cache):
    import ast
    tk = spec_of(task)
    df, rows, comps = load_records(task, data_base)
    d = os.path.join(data_base, tk['dir'], 'processed_data')
    comp = pd.read_csv(os.path.join(d, 'compounds.csv'))
    smi = {int(r['compound_id']): r.get('smiles') for _, r in comp.iterrows()}

    need = sorted({c for x in comps for c in x})
    for cid in need:
        if str(cid) not in cache:
            cache[str(cid)] = descriptors(smi.get(cid))
    V = {c: np.array(cache[str(c)], dtype=np.float64)
         for c in need if cache[str(c)] is not None}

    ys = np.array([float(df.loc[r, 'value']) for r in rows])
    ok = np.array([all(c in V for c in x) for x in comps])

    mfs = []
    for i in range(len(rows)):
        row = df.loc[rows[i]]
        k = len(comps[i])
        if 'cmp_mole_fractions' in df.columns:
            w = [float(v) for v in ast.literal_eval(row['cmp_mole_fractions'])]
        else:
            w = [1.0 / k] * k
        w = (w[:k] + [0.0] * max(0, k - len(w)))
        tot = sum(w)
        mfs.append([x / tot for x in w] if tot > 0 else [1.0 / k] * k)

    Z = np.zeros((len(rows), NDESC))
    for i, x in enumerate(comps):
        if not ok[i]:
            continue
        for c, w in zip(x, mfs[i]):
            Z[i] += w * V[c]

    conds = np.array([[float(df.loc[r].get(c, np.nan)) for c in tk['conds']]
                      for r in rows], dtype=np.float64).reshape(len(rows), -1)
    return Z, ys, conds, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--n-estimators', type=int, default=800)
    args = ap.parse_args()

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding='utf-8') as f:
            cache = json.load(f)

    Z, ys, conds, ok = build(args.task, args.data_base, cache)
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    sp = np.load(args.split_file, allow_pickle=True)
    _, rows, _ = load_records(args.task, args.data_base)
    pos = {int(r): i for i, r in enumerate(rows)}

    def to_pos(key):
        if key not in sp:
            return np.array([], dtype=int)
        return np.array([pos[int(x)] for x in sp[key]
                         if int(x) in pos and ok[pos[int(x)]]])

    tr = np.concatenate([to_pos('train_idx'), to_pos('val_idx')])
    te = to_pos('test_idx')
    if len(te) == 0:
        raise SystemExit('测试集为空（该任务在此划分下不可评估）')

    # RDKit 部分描述符（如 Ipc）会溢出为 inf 或超过 float32 范围；XGBoost 内部按
    # float32 处理，遇到 inf 会直接报错。这里把非有限值与超范围值统一视为缺失
    # （XGBoost 原生支持 NaN 缺失），转 float32 后再清一次。
    LIM = 1e30
    def clean(A):
        A = np.where(np.isfinite(A) & (np.abs(A) < LIM), A, np.nan)
        return A.astype(np.float32)

    X = clean(np.hstack([Z, conds])) if conds.shape[1] else clean(Z)
    X[~np.isfinite(X)] = np.nan

    rs = []
    for s in args.seeds:
        m = XGBRegressor(n_estimators=args.n_estimators, learning_rate=0.05,
                         max_depth=6, subsample=0.9, colsample_bytree=0.9,
                         reg_lambda=1.0, random_state=s, n_jobs=8,
                         tree_method='hist')
        m.fit(X[tr], ys[tr])
        p = m.predict(X[te])
        r = float(pearsonr(p, ys[te])[0])
        rs.append(r)
        print('  seed %d: R = %.4f' % (s, r), flush=True)

    mu, sd = float(np.mean(rs)), float(np.std(rs))
    print('%s | %s | xgb | train %d / test %d | R = %.4f ± %.4f'
          % (args.task, os.path.basename(args.split_file), len(tr), len(te),
             mu, sd))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(args.split_file).replace('.npz', '')
        dst = os.path.join(args.out, 'xgb_%s.csv' % name)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('task,split,train_n,test_n,mean,sd,per_seed,n_desc,'
                    'n_estimators\n')
            f.write('%s,%s,%d,%d,%.4f,%.4f,%s,%d,%d\n'
                    % (args.task, sp['split'] if 'split' in sp else '',
                       len(tr), len(te), mu, sd,
                       ';'.join('%.4f' % x for x in rs), NDESC,
                       args.n_estimators))
        print('写出', dst)


if __name__ == '__main__':
    main()
