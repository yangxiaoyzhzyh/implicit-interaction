# -*- coding: utf-8 -*-
"""run_5tasks_rdkit_xgb.py — 五任务 RDKit2D + XGBoost 组分尺寸外推（外部对照）

实验 9 — RDKit2D+XGBoost reference（补材 S2 的 external baseline，表 S6）
路线 无（外部对照，不属于 v(im) 或 v(ex)）
产出 results/tasks5_rdkit_xgb_results.csv；logs/extrapolation/tasks5_xgb.log
特征：摩尔分数加权的 RDKit 2D 描述符（217 维）+ 条件变量；10 seeds

统一协议: train = 低组分子集, test = 高组分余量; 无验证集;
特征 = 摩尔分数加权 RDKit2D (217 维) + 条件变量 (原始值);
XGBoost 默认超参, 10 seeds, 报 R/MAE mean±std。
ids/mfs 长度不一致的标注缺损行整行剔除 (drug 453, SPE 283)。
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

BASE = os.path.join(HERE, '..', '..', '..', '..', '..', 'literature',
                    'CheMixHub', 'datasets')

TASKS = [
    dict(name='drug ln(S)', dir='drug-solubility',
         csv='processed_DrugSolubilityData.csv', prop=None, kind='drug',
         conds=['Temperature, K']),
    dict(name='SPE ln(k)', dir='polymer-electrolyte',
         csv='processed_PolymerElectrolyteData.csv', prop=None, kind='standard',
         conds=['Temperature, K']),
    dict(name='IlT ln(k)', dir='ionic-liquids',
         csv='processed_IlThermoData.csv', prop='Electrical conductivity',
         kind='standard', conds=['Temperature, K', 'Pressure, kPa',
                                 'Frequency, MHz']),
    dict(name='IlT ln(eta)', dir='ionic-liquids',
         csv='processed_IlThermoData.csv', prop='Viscosity',
         kind='standard', conds=['Temperature, K', 'Pressure, kPa',
                                 'Frequency, MHz']),
    dict(name='NIST-full ln(eta)', dir='nist-logV',
         csv='processed_NISTlogV.csv', prop=None, kind='standard',
         conds=['T', 'P'], train_max=1),
]


def components(r, kind):
    """返回 [(cid, w), ...]; 标注缺损 (ids/mfs 长度不一致) 返回 None。"""
    if kind == 'drug':
        ids = (list(eval(r['cmp_ids_solvent'])) +
               list(eval(r['cmp_ids_drug'])))
        mfs = (list(eval(r['cmp_mole_fractions_solvent'])) +
               list(eval(r['cmp_mole_fractions_drug'])))
    else:
        ids = list(eval(r['cmp_ids']))
        mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    comps = list(zip(ids, mfs))
    if kind == 'drug':
        tot = sum(w for _, w in comps)
        return [(int(c), w / tot) for c, w in comps]
    return [(int(c), float(w)) for c, w in comps]


def main():
    out_rows = []
    for tk in TASKS:
        name = tk['name']
        droot = os.path.join(BASE, tk['dir'], 'processed_data')
        df = pd.read_csv(os.path.join(droot, tk['csv']))
        if tk['prop'] is not None and 'property' in df.columns:
            df = df[df['property'] == tk['prop']].reset_index(drop=True)
        comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
        cmp_smiles = {int(r['compound_id']): r['smiles']
                      for _, r in comp.iterrows()}

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
        conds = np.array(conds, dtype=np.float64)
        nc = np.array([len(c) for c in comps_list])
        tr_m = nc <= (2 if tk.get('train_max', 2) == 2 else 1)
        te_m = ~tr_m
        print(f'--- {name}: 有效 {len(comps_list)} (剔除缺损 {skipped}) | '
              f'train {tr_m.sum()} / test {te_m.sum()}', flush=True)

        # 条件变量 NaN -> 训练集中位数
        for j in range(conds.shape[1]):
            med = np.nanmedian(conds[tr_m, j])
            conds[:, j] = np.where(np.isnan(conds[:, j]), med, conds[:, j])

        # 加权 RDKit2D 特征
        def featurize(mask, tag):
            X = np.zeros((int(mask.sum()), 217))
            rr = []
            done = 0
            for i in np.where(mask)[0]:
                acc = None
                for cid, w in comps_list[i]:
                    v = rdkit2d(cmp_smiles.get(cid))
                    if v is None:
                        continue
                    acc = w * v if acc is None else acc + w * v
                if acc is None:
                    acc = np.zeros(217)
                acc = np.clip(acc, -1e10, 1e10)  # 加权和二次截断
                X[len(rr)] = acc
                rr.append(i)
                done += 1
                if done % 2000 == 0:
                    print(f'   [{tag}] featurize {done}/{int(mask.sum())}',
                          flush=True)
            return X, rr

        tr_X, tr_pos = featurize(tr_m, 'train')
        te_X, te_pos = featurize(te_m, 'test')
        tr_X = np.hstack([tr_X, conds[tr_pos]])
        te_X = np.hstack([te_X, conds[te_pos]])
        tr_y, te_y = ys[tr_pos], ys[te_pos]

        rs, maes = [], []
        for seed in range(10):
            m = XGBRegressor(n_estimators=800, learning_rate=0.05,
                             max_depth=6, subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=1.0, random_state=seed,
                             n_jobs=8, verbosity=0)
            m.fit(tr_X, tr_y)
            p = m.predict(te_X)
            rs.append(pearsonr(p, te_y)[0])
            maes.append(float(np.mean(np.abs(p - te_y))))
            print(f'   [xgb] seed {seed}: R = {rs[-1]:.4f}', flush=True)
        rs, maes = np.array(rs), np.array(maes)
        print(f'   >> {name}: R = {rs.mean():.4f} ± {rs.std():.4f} | '
              f'MAE = {maes.mean():.4f} ± {maes.std():.4f}', flush=True)
        out_rows.append((name, f'{rs.mean():.4f}', f'{rs.std():.4f}',
                         f'{maes.mean():.4f}', f'{maes.std():.4f}'))

    print('\n===== 五任务 RDKit2D+XGBoost 外推汇总 =====')
    for name, r, rs, m, ms in out_rows:
        print(f'{name:20s} R = {r} ± {rs} | MAE = {m} ± {ms}')
    out = os.path.join(HERE, 'tasks5_rdkit_xgb_results.csv')
    with open(out, 'w') as f:
        f.write('task,R_mean,R_std,MAE_mean,MAE_std\n')
        for row in out_rows:
            f.write(','.join(row) + '\n')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
