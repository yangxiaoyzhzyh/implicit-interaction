# -*- coding: utf-8 -*-
"""run_5tasks_vex.py — 五任务 v(ex) 组分尺寸外推（服务器 GPU）

实验 5 — Five tasks, both routes（论文 Table 1）
路线 v(ex)（命令行与结果文件沿用内部标签 route e1）
产出 results/tasks5_e1_results.csv；logs/extrapolation/tasks5_e1.log
配套 per_molecule/run_5tasks_vim.py（v(im) 臂，同数据、同划分、同协议）

理论对照臂：与 run_5tasks_vim.py（v(im) 因式分解）同数据、同划分、同协议，
唯一差别是模型结构：
  v(im): 每分子独立图 → 分子内 TFM → 分子向量 → 摩尔分数凸组合 → head
  v(ex): 同一样本全部原子拼一张图（第 32 维 = 真实摩尔分数）→
              跨分子 TFM（全体原子联合编码）→ 全体原子 mean → head
理论预测：v(ex) 为欠定系统，外推（新组合无训练方程）下不优于 v(im)。

用法 (服务器):
  python run_5tasks_vex.py --data-base /autodl-fs/data/mon_extrap/data/chemixhub \
      --tasks drug spe iltk ilteta nist --seeds 42 43 44 --epochs 200
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_dense_batch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_transfer_matrix import GINConv, mol_to_graph, DEVICE
print(f'Device: {DEVICE}', flush=True)

TASKS = {
    'drug': dict(name='drug ln(S)', dir='drug-solubility',
                 csv='processed_DrugSolubilityData.csv', prop=None,
                 kind='drug', conds=['Temperature, K'], train_max=2),
    'spe': dict(name='SPE ln(k)', dir='polymer-electrolyte',
                csv='processed_PolymerElectrolyteData.csv', prop=None,
                kind='standard', conds=['Temperature, K'], train_max=2),
    'iltk': dict(name='IlT ln(k)', dir='ionic-liquids',
                 csv='processed_IlThermoData.csv',
                 prop='Electrical conductivity', kind='standard',
                 conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                 train_max=2),
    'ilteta': dict(name='IlT ln(eta)', dir='ionic-liquids',
                   csv='processed_IlThermoData.csv', prop='Viscosity',
                   kind='standard',
                   conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                   train_max=2),
    'nist': dict(name='NIST-full ln(eta)', dir='nist-logV',
                 csv='processed_NISTlogV.csv', prop=None, kind='standard',
                 conds=['T', 'P'], train_max=1),
}


class WholeEncoder(nn.Module):
    """v(ex) 联合编码器：全体原子一张图，跨分子 TFM，全体 masked mean。"""

    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1, pad_from=32):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - pad_from, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                         dim_feedforward=2 * dim,
                                         batch_first=True, dropout=dropout)
        self.tfm = nn.TransformerEncoder(enc, n_layers)

    def atom_embed(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs:
            x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        return self.tfm(x, src_key_padding_mask=~m)[m]

    def whole(self, x, ei, batch):
        xa = self.atom_embed(x, ei, batch)
        bsz = int(batch.max()) + 1
        cnt = scatter(torch.ones_like(batch, dtype=torch.float), batch,
                      dim=0, dim_size=bsz, reduce='sum').unsqueeze(-1)
        return scatter(xa, batch, dim=0, dim_size=bsz,
                       reduce='sum') / cnt.clamp(min=1)


def components(r, kind):
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


def build_mol_graphs(cids, cmp_smiles):
    """第 32 维占位 1（v(ex) 前向时替换为真实摩尔分数）。"""
    cache = {}
    for cid in sorted(set(cids)):
        if cid in cache:
            continue
        smi = cmp_smiles.get(cid)
        g = mol_to_graph(smi) if smi else None
        if g is None:
            cache[cid] = None
            continue
        x = torch.tensor(g['x']).float()[:, :31]
        cache[cid] = (torch.cat([x, torch.ones(x.size(0), 1)], dim=1),
                      torch.tensor(g['ei']))
    return cache


def batch_e1(items, mol_graphs, enc, reg, cond_t):
    """items: [(cids, mfs), ...] → 样本级拼图 → 跨分子 TFM → mean → head。"""
    xs, eis, sids = [], [], []
    off = 0
    for j, (cids, mfs) in enumerate(items):
        n = 0
        for cid, w in zip(cids, mfs):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            x = g[0].clone()
            x[:, -1] = w                      # 第 32 维 = 真实摩尔分数
            xs.append(x)
            eis.append(g[1] + off)
            off += x.size(0)
            n += x.size(0)
        sids.append(torch.full((n,), j, dtype=torch.long))
    if not xs:
        inp = torch.cat([torch.zeros((len(items), 256), device=DEVICE),
                         cond_t], dim=1)
        return reg(inp).squeeze(1)
    X = torch.cat(xs).to(DEVICE)
    EI = torch.cat(eis, dim=1).to(DEVICE)
    B = torch.cat(sids).to(DEVICE)
    z = enc.whole(X, EI, B)
    return reg(torch.cat([z, cond_t], dim=1)).squeeze(1)


def run_task(key, data_base, seeds, epochs, batch_size, out_writer):
    tk = TASKS[key]
    name = tk['name']
    droot = os.path.join(data_base, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, tk['csv']))
    if tk['prop'] is not None and 'property' in df.columns:
        df = df[df['property'] == tk['prop']].reset_index(drop=True)
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    cmp_smiles = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}

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
    tr_pos = np.where(nc <= tk['train_max'])[0]
    te_pos = np.where(nc > tk['train_max'])[0]
    print(f'--- {name}: 有效 {len(comps_list)} (剔除缺损 {skipped}) | '
          f'train {len(tr_pos)} / test {len(te_pos)}', flush=True)

    c_mu = np.nanmean(conds[tr_pos], axis=0)
    c_sd = np.nanstd(conds[tr_pos], axis=0)
    c_sd[c_sd == 0] = 1.0
    conds = np.where(np.isnan(conds), c_mu, conds)
    conds = (conds - c_mu) / c_sd
    n_cond = conds.shape[1]

    meta = [dict(cids=[c for c, _ in comps], mfs=[w for _, w in comps],
                 cond=conds[i]) for i, comps in enumerate(comps_list)]
    all_cids = sorted({c for m in meta for c in m['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles)
    n_fail = sum(1 for c in all_cids if mol_graphs.get(c) is None)
    print(f'    分子图: {len(all_cids) - n_fail}/{len(all_cids)} 成功',
          flush=True)

    tr_idx, te_idx = tr_pos.tolist(), te_pos.tolist()
    y_mean, y_std = ys[tr_idx].mean(), ys[tr_idx].std()

    rs, maes = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = WholeEncoder(dim=256, nhead=4, n_layers=2, dropout=0.1,
                           pad_from=32).to(DEVICE)
        reg = nn.Sequential(
            nn.Linear(256 + n_cond, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters()) + list(reg.parameters()),
                               lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                         T_max=epochs - 3)
        for ep in range(epochs):
            if ep < 3:
                for pg in opt.param_groups:
                    pg['lr'] = 1e-4 * (ep + 1) / 3
            enc.train(); reg.train()
            perm = np.random.RandomState(ep).permutation(len(tr_idx))
            tot, nb = 0.0, 0
            for s in range(0, len(perm), batch_size):
                idx = [tr_idx[i] for i in perm[s:s + batch_size]]
                items = [(meta[i]['cids'], meta[i]['mfs']) for i in idx]
                cond_t = torch.tensor(np.stack([meta[i]['cond'] for i in idx]),
                                      dtype=torch.float32, device=DEVICE)
                out = batch_e1(items, mol_graphs, enc, reg, cond_t)
                yb = torch.tensor((ys[idx] - y_mean) / y_std,
                                  dtype=torch.float32, device=DEVICE)
                loss = F.mse_loss(out, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
                tot += loss.item()
                nb += 1
            if ep >= 3:
                sch.step()
            if ep % 20 == 0 or ep == epochs - 1:
                print(f'    seed {seed} ep {ep}: loss {tot / max(nb, 1):.4f}',
                      flush=True)
        # test：v(ex) 上下文相关，无分子向量缓存，逐 batch 前向
        ps = []
        with torch.no_grad():
            enc.eval(); reg.eval()
            for s in range(0, len(te_idx), batch_size):
                idx = te_idx[s:s + batch_size]
                items = [(meta[i]['cids'], meta[i]['mfs']) for i in idx]
                cond_t = torch.tensor(np.stack([meta[i]['cond'] for i in idx]),
                                      dtype=torch.float32, device=DEVICE)
                out = batch_e1(items, mol_graphs, enc, reg, cond_t)
                ps.append(out.detach().cpu().numpy())
        p = np.concatenate(ps) * y_std + y_mean
        te_y = ys[te_idx]
        r = pearsonr(p, te_y)[0]
        mae = float(np.mean(np.abs(p - te_y)))
        rs.append(r); maes.append(mae)
        print(f'    >> seed {seed}: R = {r:.4f} | MAE = {mae:.4f}', flush=True)
        out_writer.writerow([key, seed, f'{r:.4f}', f'{mae:.4f}'])

    rs, maes = np.array(rs), np.array(maes)
    print(f'>> {name}: R = {rs.mean():.4f} ± {rs.std():.4f} | '
          f'MAE = {maes.mean():.4f} ± {maes.std():.4f}', flush=True)
    out_writer.writerow([key, 'mean', f'{rs.mean():.4f}', f'{maes.mean():.4f}'])
    out_writer.writerow([key, 'std', f'{rs.std():.4f}', f'{maes.std():.4f}'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base',
                    default='/autodl-fs/data/mon_extrap/data/chemixhub')
    ap.add_argument('--tasks', nargs='+',
                    default=['drug', 'spe', 'iltk', 'ilteta', 'nist'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()

    out = os.path.join(HERE, 'tasks5_e1_results.csv')
    f = open(out, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['task', 'seed', 'R', 'MAE'])
    for key in args.tasks:
        run_task(key, args.data_base, args.seeds, args.epochs,
                 args.batch_size, w)
        f.flush()
    f.close()
    print(f'saved {out}')


if __name__ == '__main__':
    main()
