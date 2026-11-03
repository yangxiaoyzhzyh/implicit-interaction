# -*- coding: utf-8 -*-
"""split_statistics.py —— 正文两处体系统计的可复算来源

计算并打印/落盘正文引用的两组数字（均基于组分数外推所用的同一份划分）：

1. IlT κ（`splits/iltk_composition.npz`）
   - 测试集中"含水"的行占比（正文 94%）
   - 测试集中水摩尔分数 > 0.99 的行占比（正文 80%）
   - 训练集中水 > 0.99 的条数与占比（正文 5,397 / 24,688）
   - 训练集中"含水"的行占比（正文 40%）

2. SPE κ（`splits/spe_composition.npz`）
   - 训练/测试集中"含分子量 > 300 g/mol 的组分"的行占比（正文 92% / 30%）
   - 不含此类组分的行占比（正文 8% / 70%）

水的判定：compounds.csv 中 SMILES 恰为 'O' 的组分。
分子量：由 SMILES 用 RDKit 的 Descriptors.MolWt 计算（数据集未提供 Mn 列）。

用法：
  python split_statistics.py --data-base <chemidata 根> \
      --splits ../splits --out ../runs/split_statistics.csv
"""
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'common')))
from spec import load_records  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def head_smiles(key, data_base):
    """{compound_id: SMILES}"""
    from spec import spec_of
    tk = spec_of(key)
    p = os.path.join(data_base, tk['dir'], 'processed_data', 'compounds.csv')
    c = pd.read_csv(p)
    return {int(r['compound_id']): r['smiles'] for _, r in c.iterrows()}


def mol_weights(smiles):
    out = {}
    for cid, smi in smiles.items():
        m = Chem.MolFromSmiles(str(smi)) if smi else None
        out[cid] = Descriptors.MolWt(m) if m is not None else float('nan')
    return out


def mole_fractions(df, rows, comps):
    """逐行取归一化后的摩尔分数（与 engine.py 的处理一致）。"""
    mfs = []
    for i, r in enumerate(rows):
        row = df.loc[r]
        k = len(comps[i])
        try:
            mf = [float(v) for v in ast.literal_eval(row['cmp_mole_fractions'])]
        except Exception:
            mf = [1.0 / k] * k
        mf = (mf + [0.0] * k)[:k]
        tot = sum(mf)
        mfs.append([w / tot for w in mf] if tot > 0 else [1.0 / k] * k)
    return mfs


def split_index(rows, split_file):
    sp = np.load(split_file, allow_pickle=True)
    pos = {int(r): i for i, r in enumerate(rows)}
    tr = np.array([pos[int(x)] for x in sp['train_idx']], int)
    te = np.array([pos[int(x)] for x in sp['test_idx']], int)
    return tr, te


def iltk_stats(data_base, splits):
    df, rows, comps = load_records('iltk', data_base)
    smi = head_smiles('iltk', data_base)
    water = {cid for cid, s in smi.items() if str(s).strip() == 'O'}
    mfs = mole_fractions(df, rows, comps)
    has_w = np.array([any(c in water for c in comps[i])
                      for i in range(len(rows))])
    w_water = np.array([sum(w for c, w in zip(comps[i], mfs[i]) if c in water)
                        for i in range(len(rows))])
    tr, te = split_index(rows, os.path.join(splits, 'iltk_composition.npz'))

    def pct(mask, idx):
        return 100.0 * mask[idx].mean() if len(idx) else float('nan')

    out = [
        dict(dataset='IlT κ', quantity='rows train', value=len(tr), unit='rows'),
        dict(dataset='IlT κ', quantity='rows test', value=len(te), unit='rows'),
        dict(dataset='IlT κ', quantity='test rows with water',
             value=int(has_w[te].sum()), unit='rows'),
        dict(dataset='IlT κ', quantity='test share aqueous',
             value=round(pct(has_w, te), 1), unit='%'),
        dict(dataset='IlT κ', quantity='test share water > 0.99',
             value=round(pct(w_water > 0.99, te), 1), unit='%'),
        dict(dataset='IlT κ', quantity='train rows water > 0.99',
             value=int((w_water[tr] > 0.99).sum()), unit='rows'),
        dict(dataset='IlT κ', quantity='train share water > 0.99',
             value=round(pct(w_water > 0.99, tr), 1), unit='%'),
        dict(dataset='IlT κ', quantity='train share aqueous',
             value=round(pct(has_w, tr), 1), unit='%'),
    ]
    print('[IlT κ] train %d / test %d' % (len(tr), len(te)))
    for d in out:
        print('   %-32s %10.1f %s' % (d['quantity'], d['value'], d['unit']))
    return out


def spe_stats(data_base, splits):
    df, rows, comps = load_records('spe', data_base)
    smi = head_smiles('spe', data_base)
    mw = mol_weights(smi)
    cp = pd.read_csv(os.path.join(data_base, 'polymer-electrolyte',
                                  'processed_data', 'compounds.csv'))
    poly = {int(r['compound_id']): int(r['polymer']) for _, r in cp.iterrows()}
    # 「大分子组分」= 该组分是聚合物，或其 SMILES 的摩尔质量 > 300 g/mol。
    # 只用 MolWt 会漏掉聚合物（其 SMILES 是低聚物，算出来常 < 300），
    # 两类合起来才复现正文的量级。
    heavy = np.array([any(poly.get(c, 0) == 1 or mw.get(c, 0) > 300
                          for c in comps[i]) for i in range(len(rows))])
    tr, te = split_index(rows, os.path.join(splits, 'spe_composition.npz'))

    def pct(mask, idx):
        return 100.0 * mask[idx].mean() if len(idx) else float('nan')

    out = [
        dict(dataset='SPE κ', quantity='rows train', value=len(tr), unit='rows'),
        dict(dataset='SPE κ', quantity='rows test', value=len(te), unit='rows'),
        dict(dataset='SPE κ', quantity='train share with a large component',
             value=round(pct(heavy, tr), 1), unit='%'),
        dict(dataset='SPE κ', quantity='test share with a large component',
             value=round(pct(heavy, te), 1), unit='%'),
        dict(dataset='SPE κ', quantity='train share without one',
             value=round(100 - pct(heavy, tr), 1), unit='%'),
        dict(dataset='SPE κ', quantity='test share without one',
             value=round(100 - pct(heavy, te), 1), unit='%'),
    ]
    print('[SPE κ] train %d / test %d' % (len(tr), len(te)))
    for d in out:
        print('   %-32s %10.1f %s' % (d['quantity'], d['value'], d['unit']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--splits', default=os.path.join(HERE, '..', '..', 'splits'))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'runs',
                                                  'split_statistics.csv'))
    a = ap.parse_args()

    rowsout = iltk_stats(a.data_base, a.splits) + spe_stats(a.data_base, a.splits)
    df = pd.DataFrame(rowsout)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print('\n写出', os.path.abspath(a.out))


if __name__ == '__main__':
    main()
