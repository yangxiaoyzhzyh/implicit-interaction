# -*- coding: utf-8 -*-
"""train_5fold_vex_current.py —— 官方 5-fold 随机 CV，v(ex) 用**当前实现**重跑

与同目录 `train_5fold_vex.py` 的区别：
  - 那个脚本是上一篇文章 AtomFormer 的整图模型（GIN×3 + TFM×2 + masked mean）；
  - 本脚本用 `common/engine.py` 里的 **Vex**：逐组分编码（GIN×3 + 分子内 TFM×2
    + mean pooling）→ 组分 token（加摩尔分数投影）→ 跨组分 TFM×2 + CLS → 读出。
    也就是正文 Table 1 里 v(ex) 的那个模型。
  - 协议与 v(im) 对齐：Adam 1e-4、batch 128、最多 200 epochs、patience 20 早停，
    以验证集 Pearson R 选最优权重。

用法（在 repro_v2 上一级目录运行）：
  python 07_indistribution/scripts/train_5fold_vex_current.py \
      --data-root /root/autodl-fs/chemixhub/datasets \
      --task ionic_cond --folds 0 1 2 3 4 --out runs_vex_cur/ionic_cond.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
# common/ 可能在上一级或上两级（脚本可平铺到任意位置运行）
for _rel in ('../common', '../../common', '../../../common'):
    _p = os.path.abspath(os.path.join(HERE, _rel))
    if os.path.isdir(_p):
        COMMON = _p
        break
else:
    raise SystemExit('找不到 common/ 目录（需要 repro_v2/common）')
sys.path.insert(0, COMMON)

from engine import Vex, mol_to_graph  # noqa: E402

RDLogger.DisableLog('rdApp.*')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TASK_CONFIG = {
    'mon': dict(dir='MON', csv='processed_MON.csv', prop=None, conds=[],
                split='motor_octane_number_splits', prop_col='property',
                smiles_col='smiles'),
    'density': dict(dir='miscible-solvent',
                    csv='processed_MiscibleSolventData.csv', prop='Density',
                    conds=[], split='density_splits', prop_col='property',
                    smiles_col='smiles'),
    'heat_vap': dict(dir='miscible-solvent',
                     csv='processed_MiscibleSolventData.csv',
                     prop='Heat of vaporization', conds=[],
                     split='heat_of_vaporization_splits',
                     prop_col='property', smiles_col='smiles'),
    'he': dict(dir='miscible-solvent', csv='processed_MiscibleSolventData.csv',
               prop='Enthalpy of mixing', conds=[],
               split='enthalpy_of_mixing_splits', prop_col='property',
               smiles_col='smiles'),
    'logv': dict(dir='logV', csv='processed_logV.csv', prop=None, conds=['T'],
                 split='log_viscosity_splits', prop_col='property',
                 smiles_col='smiles'),
    'nist_logv': dict(dir='nist-logV', csv='processed_NISTlogV.csv', prop=None,
                      conds=['T', 'P'], split='log_viscosity_splits',
                      prop_col='property', smiles_col='smiles'),
    'ionic_cond': dict(dir='ionic-liquids', csv='processed_IlThermoData.csv',
                       prop='Electrical conductivity',
                       conds=['Temperature, K', 'Pressure, kPa',
                              'Frequency, MHz'],
                       split='electrical_conductivity_splits',
                       prop_col='property', smiles_col='smiles'),
    'ionic_visc': dict(dir='ionic-liquids', csv='processed_IlThermoData.csv',
                       prop='Viscosity',
                       conds=['Temperature, K', 'Pressure, kPa',
                              'Frequency, MHz'],
                       split='viscosity_splits', prop_col='property',
                       smiles_col='smiles'),
}


def load_task(key, root):
    """返回 (items, graphs, ys, conds)；items[i] = [(cid, w), ...]。"""
    cfg = TASK_CONFIG[key]
    droot = os.path.join(root, cfg['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, cfg['csv']))
    if cfg['prop'] is not None:
        df = df[df[cfg['prop_col']] == cfg['prop']].reset_index(drop=True)
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    smi = {int(float(r['compound_id'])): r[cfg['smiles_col']]
           for _, r in comp.iterrows()}

    items, ys, conds = [], [], []
    bad_mf = 0
    for _, r in df.iterrows():
        # 关键：**不能跳过任何一行**，否则官方 fold 索引会整体错位。
        # 与 common/spec.py 一致：以 cmp_ids 为准，摩尔分数截断/补零后归一化。
        try:
            cids = [int(float(c)) for c in eval(r['cmp_ids'])]
        except Exception:
            cids = []
        try:
            mfs = [float(w) for w in eval(r['cmp_mole_fractions'])]
        except Exception:
            mfs = []
        if len(mfs) != len(cids):
            bad_mf += 1
            mfs = (mfs + [0.0] * len(cids))[:len(cids)]
        tot = sum(mfs)
        if cids and tot > 0:
            mfs = [w / tot for w in mfs]
        elif cids:
            mfs = [1.0 / len(cids)] * len(cids)
        try:
            y = float(r['value'])
        except (TypeError, ValueError):
            y = float('nan')
        items.append(list(zip(cids, mfs)))
        ys.append(y)
        conds.append([float(r[c]) for c in cfg['conds']] if cfg['conds'] else [])
    ys = np.array(ys, dtype=np.float64)
    conds = (np.array(conds, dtype=np.float64).reshape(len(items), -1)
             if cfg['conds'] else np.zeros((len(items), 0)))

    graphs = {}
    for it in items:
        for cid, _ in it:
            if cid in graphs:
                continue
            g = mol_to_graph(smi.get(cid))
            if g is None:
                graphs[cid] = None
                continue
            x = torch.tensor(g[0], dtype=torch.float32)
            x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)  # +常数列 → 45
            graphs[cid] = (x, torch.tensor(g[1], dtype=torch.long))
    print('[%s] %s 行 %d（与官方 fold 索引一一对应），摩尔分数长度不符 %d 行，'
          '组分图 %d 个' % (key, cfg['prop'] or 'all', len(df), bad_mf,
                            len(graphs)), flush=True)
    return items, graphs, ys, conds


def make_batch(idx, items, graphs, conds, ys, y_mu, y_sd, c_mu, c_sd):
    xs, eis, mids, ws = [], [], [], []
    for j, i in enumerate(idx):
        for cid, w in items[i]:
            g = graphs.get(cid)
            if g is None:
                continue
            xs.append(g[0])
            eis.append(g[1])
            mids.append(j)
            ws.append(w)
    offs = np.cumsum([0] + [x.size(0) for x in xs[:-1]]).tolist() if xs else []
    X = torch.cat(xs).to(DEVICE)
    EI = torch.cat([ei + o for ei, o in zip(eis, offs)], dim=1).to(DEVICE)
    sizes = [x.size(0) for x in xs]
    B = torch.cat([torch.full((s,), k, dtype=torch.long)
                   for k, s in enumerate(sizes)]).to(DEVICE)
    W = torch.tensor(ws, dtype=torch.float32, device=DEVICE)
    M = torch.tensor(mids, dtype=torch.long, device=DEVICE)
    if conds.shape[1]:
        C = torch.tensor((conds[idx] - c_mu) / c_sd, dtype=torch.float32,
                         device=DEVICE)
    else:
        C = torch.zeros(len(idx), 0, device=DEVICE)   # 无条件：空张量，便于 cat
    Y = torch.tensor((ys[idx] - y_mu) / y_sd, dtype=torch.float32,
                     device=DEVICE)
    return X, EI, B, W, M, C, Y


def predict(idx, model, items, graphs, conds, ys, y_mu, y_sd, c_mu, c_sd,
            bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            part = idx[s:s + bs]
            X, EI, B, W, M, C, _ = make_batch(part, items, graphs, conds, ys,
                                             y_mu, y_sd, c_mu, c_sd)
            out.append(model(model.enc(X, EI, B), W, M, len(part), C)
                       .cpu().numpy() * y_sd + y_mu)
    return np.concatenate(out)


def run_fold(key, fold, items, graphs, ys, conds, data_root, cfg, epochs,
             patience, lr, wd, batch, seed):
    from safetensors.torch import load_file
    sp = load_file(os.path.join(data_root, cfg['dir'], 'processed_data',
                                cfg['split'], 'kfold_split_%d.safetensors' % fold))
    tr = sp['train_indices'].numpy().astype(int)
    va = sp['val_indices'].numpy().astype(int)
    te = sp['test_indices'].numpy().astype(int)

    y_mu, y_sd = ys[tr].mean(), ys[tr].std()
    if conds.shape[1]:
        c_mu = np.nanmean(conds[tr], 0)
        c_sd = np.nanstd(conds[tr], 0)
        c_sd[c_sd == 0] = 1.0
        conds = np.where(np.isnan(conds), c_mu, conds)
    else:
        c_mu = c_sd = np.zeros(0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Vex(dim=256, nhead=4, ncond=conds.shape[1]).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - 3))
    best = (-9.0, -1, None)
    bad = 0
    t0 = time.time()
    for ep in range(epochs):
        if ep < 3:
            for pg in opt.param_groups:
                pg['lr'] = lr * (ep + 1) / 3
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for s in range(0, len(perm), batch):
            part = perm[s:s + batch].tolist()
            X, EI, B, W, M, C, Y = make_batch(part, items, graphs, conds, ys,
                                             y_mu, y_sd, c_mu, c_sd)
            opt.zero_grad()
            loss = ((model(model.enc(X, EI, B), W, M, len(part), C) - Y) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(part)
        if ep >= 3:
            sch.step()
        pv = predict(va, model, items, graphs, conds, ys, y_mu, y_sd, c_mu, c_sd)
        vr = pearsonr(pv, ys[va])[0]
        if vr > best[0]:
            best = (vr, ep, {k: v.detach().clone()
                             for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print('    fold %d ep %3d loss %.4f valR %+.4f (best %+.4f@%d)'
                  % (fold, ep, tot / len(perm), vr, best[0], best[1]), flush=True)
        if bad >= patience:
            break
    if best[2] is not None:
        model.load_state_dict(best[2])
    p = predict(te, model, items, graphs, conds, ys, y_mu, y_sd, c_mu, c_sd)
    r = float(pearsonr(p, ys[te])[0])
    mae = float(np.abs(p - ys[te]).mean())
    print('  >> %s fold %d | test R %+.4f MAE %.4f | best_ep %d | %.0fs | %d pars'
          % (key, fold, r, mae, best[1], time.time() - t0, n_par), flush=True)
    return dict(task=key, fold=fold, seed=seed, test_R=round(r, 4),
                test_MAE=round(mae, 4), best_ep=best[1],
                n_train=len(tr), n_val=len(va), n_test=len(te), n_params=n_par)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--task', required=True, choices=list(TASK_CONFIG))
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    items, graphs, ys, conds = load_task(args.task, args.data_root)
    cfg = TASK_CONFIG[args.task]
    rows = []
    for f in args.folds:
        rows.append(run_fold(args.task, f, items, graphs, ys, conds,
                             args.data_root, cfg, args.epochs, args.patience,
                             args.lr, args.wd, args.batch, args.seed))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    print('\n%s 5-fold: R = %.4f ± %.4f' % (args.task, df.test_R.mean(),
                                            df.test_R.std()))
    if args.out:
        print('写出', args.out)


if __name__ == '__main__':
    main()
