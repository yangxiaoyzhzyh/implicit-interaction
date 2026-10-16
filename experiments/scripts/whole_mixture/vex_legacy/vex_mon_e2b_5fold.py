# -*- coding: utf-8 -*-
"""vex_mon_e2b_5fold.py — MON，组分数外推变体

实验 1 — CheMixHub 官方 5-fold（补材 S1 表 S2 的 in-distribution anchors）
路线 v(ex)（该目录即 v(ex) 臂的 per-task 脚本）
产出 logs/chemixhub_5fold_whole/（v(ex) 5-fold 只落日志，无结果 CSV）
说明：文件名已按本仓库规范统一，脚本内容与原始 repro bundle 一致
v(im) 架构（单分子编码 + 摩尔分数加权和 + MLP 头, 端到端, 无预训练）
在 CheMixHub MON 官方 5-fold 随机 CV 上与官方基线同台对比。

协议与 vex_mon_5fold.py 完全一致:
  kfold_split_{0-4}.safetensors 的 train/val/test 三段索引,
  val 选 best (Pearson R), 报 test R / MAE, 60 epochs, warmup 3 + cosine。

对照（官方 Table, R）: RDKit+XGB 0.913±0.019 | v(ex)(统一图+交互) 0.898
用法: python vex_mon_e2b_5fold.py [--folds 0 1 2 3 4] [--epochs 60]
"""
import argparse
import copy
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.utils import scatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'per_molecule'))
from encoder import Encoder, load_compounds, DEVICE  # noqa: E402

MON_DATA = os.environ.get('MON_DATA_ROOT', os.path.join(HERE, 'data', 'MON'))
from probe_mon_extrap_frozen import load_all
from probe_mon_extrap_capacity import build_mol_graphs


def forward_mixtures(ids, meta, mol_graphs, enc, reg, y_mean, y_std,
                     y_all=None, train=False):
    graphs, mix_ids, weights = [], [], []
    for j, mi in enumerate(ids):
        for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(g)
            mix_ids.append(j)
            weights.append(w)
    sizes = [g[0].size(0) for g in graphs]
    xs = torch.cat([g[0] for g in graphs]).to(DEVICE)
    ei = torch.cat([g[1] + sum(sizes[:k])
                    for k, g in enumerate(graphs)], dim=1).to(DEVICE)
    b = torch.cat([torch.full((s,), k, dtype=torch.long)
                   for k, s in enumerate(sizes)]).to(DEVICE)
    z = enc.pooled(xs, ei, b)
    midx = torch.tensor(mix_ids, device=DEVICE)
    wts = torch.tensor(weights, dtype=torch.float32,
                       device=DEVICE).unsqueeze(1)
    mix = scatter(wts * z, midx, dim=0, dim_size=len(ids))
    out = reg(mix).squeeze(1)
    if not train:
        return out.detach().cpu().numpy() * y_std + y_mean
    yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                      dtype=torch.float32, device=DEVICE)
    return out, yb


def run_fold(fold, meta, y_all, mol_graphs, data_root, epochs, seed=42):
    import safetensors.torch as st
    split = st.load_file(os.path.join(
        data_root, 'motor_octane_number_splits', f'kfold_split_{fold}.safetensors'))
    tr_idx, va_idx, te_idx = (split[k].numpy().tolist() for k in
                              ('train_indices', 'val_indices', 'test_indices'))
    print(f'\n===== Fold {fold}: train {len(tr_idx)} / val {len(va_idx)} '
          f'/ test {len(te_idx)} =====', flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    y_mean, y_std = y_all[tr_idx].mean(), y_all[tr_idx].std()
    enc = Encoder(dim=256, nhead=4, pad_from=32, normalize=False).to(DEVICE)
    reg = nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1)).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(reg.parameters()),
                           lr=1e-4, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - 3)

    best_r, best_state, best_ep = -9, None, -1
    for ep in range(epochs):
        if ep < 3:
            for pg in opt.param_groups:
                pg['lr'] = 1e-4 * (ep + 1) / 3
        enc.train()
        reg.train()
        perm = np.random.RandomState(ep).permutation(len(tr_idx))
        for s in range(0, len(perm), 32):
            ids = [tr_idx[i] for i in perm[s:s + 32]]
            out, yb = forward_mixtures(ids, meta, mol_graphs, enc, reg,
                                       y_mean, y_std, y_all, train=True)
            F.mse_loss(out, yb).backward()
            opt.step()
            opt.zero_grad()
        if ep >= 3:
            sch.step()
        enc.eval()
        reg.eval()
        pv = forward_mixtures(va_idx, meta, mol_graphs, enc, reg,
                              y_mean, y_std)
        vr = pearsonr(pv, y_all[va_idx])[0]
        if vr > best_r:
            best_r, best_ep = vr, ep
            best_state = (copy.deepcopy(enc.state_dict()),
                          copy.deepcopy(reg.state_dict()))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'  ep {ep}: val R {vr:.4f} | best {best_r:.4f}@{best_ep}',
                  flush=True)
    enc.load_state_dict(best_state[0])
    reg.load_state_dict(best_state[1])
    enc.eval()
    reg.eval()
    p = forward_mixtures(te_idx, meta, mol_graphs, enc, reg, y_mean, y_std)
    te_y = y_all[te_idx]
    r = pearsonr(p, te_y)[0]
    mae = float(np.mean(np.abs(p - te_y)))
    print(f'>> Fold {fold} test: R = {r:.4f} | MAE = {mae:.4f} '
          f'(best val {best_r:.4f}@{best_ep})', flush=True)
    return r, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=MON_DATA)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=60)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    records, meta, cmp_smiles = load_all(args.data_root)
    y_all = np.array([d.y.item() for d in records])
    mol_graphs = build_mol_graphs(meta, cmp_smiles)
    n_missing = sum(1 for m in meta
                    if any(mol_graphs.get(c) is None for c in m['cids']))
    print(f'样本 {len(meta)}, 含缺失组分: {n_missing}', flush=True)

    rs, maes = [], []
    for fold in args.folds:
        r, mae = run_fold(fold, meta, y_all, mol_graphs,
                          args.data_root, args.epochs)
        rs.append(r)
        maes.append(mae)
    rs, maes = np.array(rs), np.array(maes)
    print('\n===== v(im) (单分子编码+凸组合+MLP头) MON 官方 5-fold 随机 CV =====')
    for f, r, m in zip(args.folds, rs, maes):
        print(f'  fold {f}: R = {r:.4f} | MAE = {m:.4f}')
    print(f'>> mean R = {rs.mean():.4f} ± {rs.std():.4f} | '
          f'MAE = {maes.mean():.4f} ± {maes.std():.4f}')
    print('   官方对照: RDKit+XGB R 0.913±0.019 | '
          'v(ex)(统一图+交互) R 0.898±0.033')
    out = os.path.join(HERE, 'train_mon_e2b_5fold_results.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'R', 'MAE'])
        for fd, r, m in zip(args.folds, rs, maes):
            w.writerow([fd, f'{r:.4f}', f'{m:.4f}'])
        w.writerow(['mean', f'{rs.mean():.4f}', f'{maes.mean():.4f}'])
        w.writerow(['std', f'{rs.std():.4f}', f'{maes.std():.4f}'])
    print(f'saved {out}')


if __name__ == '__main__':
    main()
