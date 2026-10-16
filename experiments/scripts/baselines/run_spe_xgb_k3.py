# -*- coding: utf-8 -*-
"""run_spe_xgb_k3.py — SPE κ、RDKit2D + XGBoost 外推（train K ≤ 3 → test K ≥ 4）

实验 9 — XGBoost reference, relaxed ceiling（补材 S2）
路线 无（外部对照）
产出 results/spe_xgb_k3_results.csv
与 per_molecule/run_spe_vim_k3.py 同协议，作对照
与 run_spe_vim_k3.py 同协议；特征/超参与原 XGB 臂一致；10 seeds。
"""
import faulthandler
faulthandler.enable()

import os
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER_V2 = _os.path.dirname(_os.path.dirname(_HERE))
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_PAPER_V2)))  # <repo> (three levels above experiments/scripts/<sub>)
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_mon_extrap_rdkit2d import rdkit2d

BASE = os.path.join(REPO, 'literature', 'CheMixHub', 'datasets')
TRAIN_MAX = 3
CONDS = ['Temperature, K']


def components(r):
    ids = list(eval(r['cmp_ids']))
    mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    return [(int(c), float(w)) for c, w in zip(ids, mfs)]


def featurize(mask, comps_list, cmp_smiles, tag):
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
        acc = np.clip(np.zeros(217) if acc is None else acc, -1e10, 1e10)
        X[len(rr)] = acc
        rr.append(i)
        done += 1
        if done % 2000 == 0:
            print(f'   [{tag}] featurize {done}/{int(mask.sum())}', flush=True)
    return X, rr


def main():
    droot = os.path.join(BASE, 'polymer-electrolyte', 'processed_data')
    df = pd.read_csv(os.path.join(droot, 'processed_PolymerElectrolyteData.csv'))
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    cmp_smiles = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}

    comps_list, ys, conds, skipped = [], [], [], 0
    for _, r in df.iterrows():
        comps = components(r)
        if comps is None:
            skipped += 1
            continue
        comps_list.append(comps)
        ys.append(float(r['value']))
        conds.append([pd.to_numeric(r.get(c), errors='coerce') for c in CONDS])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64)
    nc = np.array([len(c) for c in comps_list])
    tr_m = nc <= TRAIN_MAX
    te_m = ~tr_m
    kd = {int(k): int((nc == k).sum()) for k in sorted(set(nc))}
    print(f'--- SPE train<=3 -> test>=4: 有效 {len(comps_list)} (缺损 {skipped}) | '
          f'K {kd} | train {tr_m.sum()} / test {te_m.sum()}', flush=True)

    for j in range(conds.shape[1]):
        med = np.nanmedian(conds[tr_m, j])
        conds[:, j] = np.where(np.isnan(conds[:, j]), med, conds[:, j])

    tr_X, tr_pos = featurize(tr_m, comps_list, cmp_smiles, 'train')
    te_X, te_pos = featurize(te_m, comps_list, cmp_smiles, 'test')
    tr_X = np.hstack([tr_X, conds[tr_pos]])
    te_X = np.hstack([te_X, conds[te_pos]])
    tr_y, te_y = ys[tr_pos], ys[te_pos]

    rs, maes = [], []
    for seed in range(3):
        m = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         random_state=seed, n_jobs=8, verbosity=0)
        m.fit(tr_X, tr_y)
        p = m.predict(te_X)
        rs.append(pearsonr(p, te_y)[0])
        maes.append(float(np.mean(np.abs(p - te_y))))
        print(f'   [xgb] seed {seed}: R = {rs[-1]:.4f}', flush=True)
    rs, maes = np.array(rs), np.array(maes)
    print(f'>> SPE ln(k) train<=3 -> test>=4: R = {rs.mean():.4f} ± {rs.std():.4f} | '
          f'MAE = {maes.mean():.4f} ± {maes.std():.4f}', flush=True)
    out = os.path.join(HERE, 'spe_xgb_k3_results.csv')
    with open(out, 'w') as f:
        f.write('task,R_mean,R_std,MAE_mean,MAE_std\n')
        f.write(f'spe_k3,{rs.mean():.4f},{rs.std():.4f},'
                f'{maes.mean():.4f},{maes.std():.4f}\n')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
