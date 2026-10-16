# -*- coding: utf-8 -*-
"""run_xgb_coverage.py — RDKit2D + XGBoost 在训练覆盖变化下的对照

实验 9 — XGBoost coverage sweep（补材 S2）
路线 无（外部对照）
产出 results/xgb_coverage_results.csv

特征与超参与既有 XGB 臂完全一致:
  特征 = 摩尔分数加权 RDKit2D (217 维) + 条件变量 (原始值)
  XGBoost = n_estimators 800, learning_rate 0.05, max_depth 6,
            subsample/colsample_bytree 0.9, reg_lambda 1.0; 10 seeds;
  报 Pearson R 与 MAE 的 mean ± std (ddof=0)。

每个任务跑三格 (k0 = 主配置的训练上限):
  main        train K <= k0       test K >= k0 + 1    ← 与正文/既有 XGB 臂相同
  fixed-test  train K <= k0       test K >= k0 + 2    ← 与 relaxed 同测试集
  relaxed     train K <= k0 + 1   test K >= k0 + 2
main -> fixed-test 只变测试集, fixed-test -> relaxed 只变训练覆盖,
因此 relaxed vs fixed-test 是干净的“提高覆盖”对照。

miscible-solvent 的组分数上限为 5, 其 test 固定为 K = 5:
  main    train K <= 2   test K = 5
  relaxed train K <= 3   test K = 5
"""
import faulthandler
faulthandler.enable()

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_mon_extrap_rdkit2d import rdkit2d

BASE = os.path.join(HERE, '..', '..', '..', '..', '..', 'literature', 'CheMixHub',
                    'datasets')
NSEED = 10

TASKS = [
    dict(name='dHmix', dir='miscible-solvent',
         csv='processed_MiscibleSolventData.csv', prop='Enthalpy of mixing',
         kind='standard', conds=[], k0=2, fix_test_k=5),
    dict(name='dHvap', dir='miscible-solvent',
         csv='processed_MiscibleSolventData.csv', prop='Heat of vaporization',
         kind='standard', conds=[], k0=2, fix_test_k=5),
    dict(name='drug_lnS', dir='drug-solubility',
         csv='processed_DrugSolubilityData.csv', prop=None,
         kind='drug', conds=['Temperature, K'], k0=2),
    dict(name='SPE_lnk', dir='polymer-electrolyte',
         csv='processed_PolymerElectrolyteData.csv', prop=None,
         kind='standard', conds=['Temperature, K'], k0=2),
    dict(name='IlT_lnk', dir='ionic-liquids',
         csv='processed_IlThermoData.csv', prop='Electrical conductivity',
         kind='standard', conds=['Temperature, K', 'Pressure, kPa',
                                 'Frequency, MHz'], k0=2),
    dict(name='IlT_lneta', dir='ionic-liquids',
         csv='processed_IlThermoData.csv', prop='Viscosity',
         kind='standard', conds=['Temperature, K', 'Pressure, kPa',
                                 'Frequency, MHz'], k0=2),
    dict(name='NIST_lneta', dir='nist-logV',
         csv='processed_NISTlogV.csv', prop=None,
         kind='standard', conds=['T', 'P'], k0=1),
    dict(name='MON', dir='MON', csv='processed_MON.csv', prop=None,
         kind='standard', conds=[], k0=2),
]


def components(r, kind):
    """[(cid, w), ...]; 标注缺损 (ids/mfs 长度不一致) 返回 None。"""
    if kind == 'drug':
        ids = (list(eval(r['cmp_ids_solvent'])) +
               list(eval(r['cmp_ids_drug'])))
        mfs = (list(eval(r['cmp_mole_fractions_solvent'])) +
               list(eval(r['cmp_mole_fractions_drug'])))
    else:
        ids = list(eval(r['cmp_ids']))
        mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs) or len(ids) == 0:
        return None
    comps = [(int(c), float(w)) for c, w in zip(ids, mfs)]
    if kind == 'drug':
        tot = sum(w for _, w in comps)
        if tot <= 0:
            return None
        comps = [(c, w / tot) for c, w in comps]
    return comps


def featurize_all(comps_list, cmp_smiles):
    X = np.zeros((len(comps_list), 217))
    for i, comps in enumerate(comps_list):
        acc = None
        for cid, w in comps:
            smi = cmp_smiles.get(cid)
            v = rdkit2d(smi) if isinstance(smi, str) else None
            if v is None:
                continue
            acc = w * v if acc is None else acc + w * v
        X[i] = np.clip(np.zeros(217) if acc is None else acc, -1e10, 1e10)
        if (i + 1) % 5000 == 0:
            print(f'   featurize {i + 1}/{len(comps_list)}', flush=True)
    return X


def fit_eval(tr_X, tr_y, te_X, te_y):
    rs, maes = [], []
    for seed in range(NSEED):
        m = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         random_state=seed, n_jobs=8, verbosity=0)
        m.fit(tr_X, tr_y)
        p = m.predict(te_X)
        rs.append(pearsonr(p, te_y)[0])
        maes.append(float(np.mean(np.abs(p - te_y))))
    rs, maes = np.array(rs), np.array(maes)
    return rs.mean(), rs.std(), maes.mean(), maes.std()


def run_task(tk):
    droot = os.path.join(BASE, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, tk['csv']))
    if tk['prop'] is not None and 'property' in df.columns:
        df = df[df['property'] == tk['prop']].reset_index(drop=True)
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    cid_col = 'compound_id' if 'compound_id' in comp.columns else comp.columns[0]
    cmp_smiles = {}
    for _, r in comp.iterrows():
        cid = r[cid_col]
        cmp_smiles[int(float(cid))] = r['smiles']

    comps_list, ys, conds, skipped = [], [], [], 0
    for _, r in df.iterrows():
        comps = components(r, tk['kind'])
        if comps is None:
            skipped += 1
            continue
        comps_list.append(comps)
        ys.append(float(r['value']))
        conds.append([pd.to_numeric(r.get(c), errors='coerce')
                      for c in tk['conds']])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64).reshape(len(ys), len(tk['conds']))
    nc = np.array([len(c) for c in comps_list])

    k0 = tk['k0']
    if tk.get('fix_test_k'):
        kt = tk['fix_test_k']
        setups = [('main', nc <= k0, nc == kt),
                  ('relaxed', nc <= k0 + 1, nc == kt)]
    else:
        setups = [('main', nc <= k0, nc >= k0 + 1),
                  ('fixed-test', nc <= k0, nc >= k0 + 2),
                  ('relaxed', nc <= k0 + 1, nc >= k0 + 2)]

    kd = {int(k): int((nc == k).sum()) for k in sorted(set(nc))}
    print(f'--- {tk["name"]}: 有效 {len(ys)} (剔除缺损 {skipped}) | K {kd}',
          flush=True)

    X = featurize_all(comps_list, cmp_smiles)
    rows = []
    for tag, tr_m, te_m in setups:
        if tr_m.sum() < 20 or te_m.sum() < 20:
            print(f'   [{tag}] 样本不足 (train {tr_m.sum()} / test {te_m.sum()})，跳过')
            continue
        c = conds.copy()
        for j in range(c.shape[1]):
            med = np.nanmedian(c[tr_m, j])
            c[:, j] = np.where(np.isnan(c[:, j]), med, c[:, j])
        tr_X = np.hstack([X[tr_m], c[tr_m]])
        te_X = np.hstack([X[te_m], c[te_m]])
        r, rs, mae, maes = fit_eval(tr_X, ys[tr_m], te_X, ys[te_m])
        print(f'   [{tag}] train {tr_m.sum()} / test {te_m.sum()} -> '
              f'R = {r:.4f} ± {rs:.4f} | MAE = {mae:.4f} ± {maes:.4f}',
              flush=True)
        rows.append((tk['name'], tag, int(tr_m.sum()), int(te_m.sum()),
                     f'{r:.4f}', f'{rs:.4f}', f'{mae:.4f}', f'{maes:.4f}'))
    return rows


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    out_rows = []
    for tk in TASKS:
        if only and tk['name'] not in only:
            continue
        try:
            out_rows += run_task(tk)
        except Exception as exc:  # 单个任务失败不阻断其余
            print(f'!! {tk["name"]} 失败: {type(exc).__name__}: {exc}',
                  flush=True)
    out = os.path.join(HERE, '..', '..', 'results', 'xgb_coverage_results.csv')
    header = 'task,setup,train_n,test_n,R_mean,R_std,MAE_mean,MAE_std'
    merged = {}
    if os.path.exists(out):  # 分批跑时保留已有行
        with open(out, encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) == 8:
                    merged[(parts[0], parts[1])] = parts
    for row in out_rows:
        merged[(str(row[0]), str(row[1]))] = [str(x) for x in row]
    with open(out, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for key in sorted(merged):
            f.write(','.join(merged[key]) + '\n')
    print(f'\nsaved {out}')
    for key in sorted(merged):
        print(' | '.join(merged[key]))


if __name__ == '__main__':
    main()
