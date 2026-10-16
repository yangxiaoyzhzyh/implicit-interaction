# -*- coding: utf-8 -*-
"""run_linear_additive.py — 加性基线（additive baseline）

实验 8 — Additive baseline（论文 Table 1 与全文）
路线 无（外部对照，不属于 v(im) 或 v(ex)）
产出 results/linear_additive_results.csv（无日志）

模型
----
    y_hat = Σ_m c_m · x_m  +  Σ_j β_j · ĉ_j  +  b

  · x_m : 分子 m 在该样本中的摩尔分数；每个**训练集出现的分子**有一个线性系数 c_m；
  · ĉ_j : 条件变量（T / P / Frequency），用**训练集**的均值与标准差标准化；
  · b   : 截距。
  · 全部系数由**训练集上的最小二乘**一次解出（闭式解，无超参、无随机性、无迭代）。

未在训练集中出现过的分子，其系数按 0 处理（即对预测无贡献）——这正是"逐分子线性系数"
这一类模型在分子轴外推时的结构性弱点。

协议
----
与两条路线的外推实验**严格同协议**（同一数据文件、同一 train/test 切分规则）：
  · MON / drug / SPE / IlT κ / IlT η : train = K ≤ 2，test = K ≥ 3
  · NIST η                            : train = K ≤ 1，test = K ≥ 2
  · ΔHmix                             : train = K ≤ 2，test = K = 5

用法
----
    python run_linear_additive.py [--out DIR]
"""
import argparse
import ast
import csv
import os
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER_V2 = _os.path.dirname(_os.path.dirname(_HERE))      # experiments/ (holds results/)
PAPER_ROOT = _os.path.dirname(_os.path.dirname(_PAPER_V2))  # <repo>/<paper_root>
REPO = _os.path.dirname(PAPER_ROOT)                        # <repo>

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

CHEMIX_LOCAL = os.path.join(PAPER_ROOT, 'permol_paper', 'data', 'chemixhub_extrap')

SPECS = [
    dict(tag='MON', csv=os.path.join(PAPER_ROOT, 'repro', 'data', 'MON', 'processed_MON.csv'),
         kind='standard', conds=[], prop=None, tr_max=2, te_min=3),
    dict(tag='dHmix', csv=os.path.join(REPO, 'literature', 'CheMixHub', 'datasets', 'miscible-solvent',
                                       'processed_data', 'processed_MiscibleSolventData.csv'),
         kind='standard', conds=[], prop='Enthalpy of mixing', tr_max=2, te_ks=[5]),
    dict(tag='dHvap', csv=os.path.join(REPO, 'literature', 'CheMixHub', 'datasets', 'miscible-solvent',
                                       'processed_data', 'processed_MiscibleSolventData.csv'),
         kind='standard', conds=[], prop='Heat of vaporization', tr_max=2, te_ks=[5]),
    dict(tag='drug', csv=os.path.join(CHEMIX_LOCAL, 'drug-solubility', 'processed_data',
                                      'processed_DrugSolubilityData.csv'),
         kind='drug', conds=['Temperature, K'], prop=None, tr_max=2, te_min=3),
    dict(tag='SPE', csv=os.path.join(CHEMIX_LOCAL, 'polymer-electrolyte', 'processed_data',
                                     'processed_PolymerElectrolyteData.csv'),
         kind='standard', conds=['Temperature, K'], prop=None, tr_max=2, te_min=3),
    dict(tag='IlT_k', csv=os.path.join(CHEMIX_LOCAL, 'ionic-liquids', 'processed_data',
                                       'processed_IlThermoData.csv'),
         kind='standard', conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
         prop='Electrical conductivity', tr_max=2, te_min=3),
    dict(tag='IlT_eta', csv=os.path.join(CHEMIX_LOCAL, 'ionic-liquids', 'processed_data',
                                         'processed_IlThermoData.csv'),
         kind='standard', conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
         prop='Viscosity', tr_max=2, te_min=3),
    dict(tag='NIST_eta', csv=os.path.join(CHEMIX_LOCAL, 'nist-logV', 'processed_data',
                                          'processed_NISTlogV.csv'),
         kind='standard', conds=['T', 'P'], prop=None, tr_max=1, te_min=2),
]


def parse_comps(row, kind):
    """返回 [(cid, mole_fraction), ...]；标注缺损返回 None。"""
    if kind == 'drug':
        ids = list(ast.literal_eval(row['cmp_ids_solvent'])) + list(ast.literal_eval(row['cmp_ids_drug']))
        mfs = list(ast.literal_eval(row['cmp_mole_fractions_solvent'])) + \
              list(ast.literal_eval(row['cmp_mole_fractions_drug']))
        if len(ids) != len(mfs):
            return None
        tot = sum(float(w) for w in mfs)
        if tot <= 0:
            return None
        return [(int(c), float(w) / tot) for c, w in zip(ids, mfs)]
    ids = list(ast.literal_eval(row['cmp_ids']))
    mfs = list(ast.literal_eval(row['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    return [(int(c), float(w)) for c, w in zip(ids, mfs)]


def load(spec):
    """读表 → 逐样本 (mols, conds, y)。"""
    df = pd.read_csv(spec['csv'])
    if spec['prop'] is not None:
        df = df[df['property'] == spec['prop']]
    rows = []
    for _, r in df.iterrows():
        comps = parse_comps(r, spec['kind'])
        if comps is None:
            continue
        y = r['value']
        if pd.isna(y):
            continue
        c = [r[k] for k in spec['conds']]
        if any(pd.isna(x) for x in c):
            continue
        rows.append(dict(mols=comps, conds=[float(x) for x in c], y=float(y)))
    return rows


def design(rows, mol_index, cond_mu, cond_sd):
    """构造设计矩阵：分子摩尔分数列 + 条件列。未见分子自动贡献 0。"""
    n, nm = len(rows), len(mol_index)
    nc = len(cond_mu)
    X = np.zeros((n, nm + nc), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(rows):
        for cid, w in r['mols']:
            j = mol_index.get(cid)
            if j is not None:
                X[i, j] += w
        for k in range(nc):
            X[i, nm + k] = (r['conds'][k] - cond_mu[k]) / cond_sd[k] if cond_sd[k] > 0 else 0.0
        y[i] = r['y']
    return X, y


def run(spec, verbose=True):
    rows = load(spec)
    if not rows:
        return None
    tr = [r for r in rows if len(r['mols']) <= spec['tr_max']]
    if 'te_ks' in spec:
        te = [r for r in rows if len(r['mols']) in spec['te_ks']]
    else:
        te = [r for r in rows if len(r['mols']) >= spec['te_min']]
    if not tr or not te:
        return None

    # 分子表与条件标准化统计，全部只用 train
    mols = sorted({cid for r in tr for cid, _ in r['mols']})
    mol_index = {cid: j for j, cid in enumerate(mols)}
    nc = len(spec['conds'])
    cond_mu, cond_sd = [], []
    for k in range(nc):
        v = np.array([r['conds'][k] for r in tr], dtype=np.float64)
        cond_mu.append(float(v.mean()))
        cond_sd.append(float(v.std()) or 1.0)

    Xtr, ytr = design(tr, mol_index, cond_mu, cond_sd)
    A = np.hstack([Xtr, np.ones((len(tr), 1))])          # 加截距列
    mu_y, sd_y = float(ytr.mean()), float(ytr.std()) or 1.0
    coef, *_ = np.linalg.lstsq(A, (ytr - mu_y) / sd_y, rcond=None)   # 最小二乘闭式解

    Xte, yte = design(te, mol_index, cond_mu, cond_sd)
    Ate = np.hstack([Xte, np.ones((len(te), 1))])
    pred = (Ate @ coef) * sd_y + mu_y

    R = float(pearsonr(pred, yte)[0]) if pred.std() > 1e-12 and yte.std() > 1e-12 else float('nan')
    MAE = float(np.mean(np.abs(pred - yte)))
    unseen = sum(1 for r in te if any(cid not in mol_index for cid, _ in r['mols']))
    out = dict(tag=spec['tag'], n_train=len(tr), n_test=len(te), n_mols=len(mols),
               unseen_frac=unseen / len(te), R=R, MAE=MAE)
    if verbose:
        print(f"  {spec['tag']:9s} train {len(tr):6d} (K<={spec['tr_max']}) | "
              f"test {len(te):6d} | 含新分子样本 {out['unseen_frac']*100:5.1f}% | "
              f"R {R:.4f} | MAE {MAE:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(_PAPER_V2, 'results'))
    a = ap.parse_args()
    print('=== 线性加和基线（y = Σ c_m·x_m + Σ β·cond + b，train 最小二乘）===')
    res = []
    for spec in SPECS:
        if not os.path.exists(spec['csv']):
            print(f"  {spec['tag']:9s} 数据缺失: {spec['csv']}")
            continue
        r = run(spec)
        if r:
            res.append(r)
    if res:
        out = os.path.join(a.out, 'linear_additive_results.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
