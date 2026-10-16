# -*- coding: utf-8 -*-
"""run_5tasks_vim.py — 五任务 v(im) 组分尺寸外推（服务器 GPU）

实验 5 — Five tasks, both routes（论文 Table 1）
路线 v(im)（命令行与结果文件沿用内部标签 route e2）
产出 results/tasks5_e2_results.csv；logs/extrapolation/tasks5_e2.log
配套 whole_mixture/run_5tasks_vex.py（v(ex) 臂，同数据、同划分、同协议）

协议与本地 RDKit2D+XGB 完全对齐:
  train = 低组分子集, test = 高组分余量; 无验证集, 固定 epochs;
  条件变量 (T/P/Freq) 标准化后拼进 head 输入 (两臂同权)。
  ids/mfs 长度不一致的标注缺损行整行剔除。
  v(im): 每任务每 seed 从头训练 GIN×3+TFM×2+逐分子pooling 单分子编码器,
      摩尔分数加权和 + [条件变量] -> MLP 头, 端到端。

用法 (服务器):
  python run_5tasks_vim.py --data-base <data-root> \
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
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encoder import Encoder, mol_to_graph, DEVICE

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


def build_mol_graphs(cids, cmp_smiles):
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
        mf = torch.ones(x.size(0), 1)
        cache[cid] = (torch.cat([x, mf], dim=1), torch.tensor(g['ei']))
    return cache


def forward_mixtures(ids, meta, mol_graphs, enc, reg, y_mean, y_std,
                     y_all=None, train=False):
    graphs, mix_ids, weights, cond_rows = [], [], [], []
    for j, mi in enumerate(ids):
        for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(g)
            mix_ids.append(j)
            weights.append(w)
        cond_rows.append(meta[mi]['cond'])
    cond_t = torch.tensor(np.stack(cond_rows), dtype=torch.float32,
                          device=DEVICE)
    if not graphs:
        inp = torch.cat([torch.zeros((len(ids), 256), device=DEVICE),
                         cond_t], dim=1)
        out = reg(inp).squeeze(1)
        if not train:
            return out.detach().cpu().numpy() * y_std + y_mean
        yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                          dtype=torch.float32, device=DEVICE)
        return out, yb
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
    inp = torch.cat([mix, cond_t], dim=1)
    out = reg(inp).squeeze(1)
    if not train:
        return out.detach().cpu().numpy() * y_std + y_mean
    yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                      dtype=torch.float32, device=DEVICE)
    return out, yb


@torch.no_grad()
def encode_unique(enc, cids, mol_graphs, batch=256):
    enc.eval()
    uniq = sorted(set(cids))
    graphs, ids = [], []
    for cid in uniq:
        g = mol_graphs.get(cid)
        if g is None:
            continue
        graphs.append(Data(x=g[0], edge_index=g[1]))
        ids.append(cid)
    vecs = []
    for b in DataLoader(graphs, batch_size=batch, shuffle=False):
        b = b.to(DEVICE)
        vecs.append(enc.pooled(b.x, b.edge_index, b.batch).cpu())
    V = torch.cat(vecs).numpy()
    return {cid: V[i] for i, cid in enumerate(ids)}


def run_task(key, data_base, seeds, epochs, batch_size, out_writer):
    tk = TASKS[key]
    name = tk['name']
    droot = os.path.join(data_base, tk['dir'], 'processed_data')
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

    meta = []
    for i, comps in enumerate(comps_list):
        meta.append({'cids': [c for c, _ in comps],
                     'mfs': [w for _, w in comps],
                     'cond': conds[i]})
    all_cids = sorted({c for m in meta for c in m['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles)
    n_fail = sum(1 for c in all_cids if mol_graphs.get(c) is None)
    print(f'    分子图: {len(all_cids) - n_fail}/{len(all_cids)} 成功',
          flush=True)

    tr_idx = tr_pos.tolist()
    te_idx = te_pos.tolist()
    y_mean, y_std = ys[tr_idx].mean(), ys[tr_idx].std()

    rs, maes = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = Encoder(dim=256, nhead=4, pad_from=32,
                      normalize=False).to(DEVICE)
        reg = nn.Sequential(
            nn.Linear(256 + n_cond, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters()) +
                               list(reg.parameters()),
                               lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                         T_max=epochs - 3)
        for ep in range(epochs):
            if ep < 3:
                for pg in opt.param_groups:
                    pg['lr'] = 1e-4 * (ep + 1) / 3
            enc.train()
            reg.train()
            perm = np.random.RandomState(ep).permutation(len(tr_idx))
            tot, nb = 0.0, 0
            for s in range(0, len(perm), batch_size):
                ids = [tr_idx[i] for i in perm[s:s + batch_size]]
                out, yb = forward_mixtures(ids, meta, mol_graphs,
                                           enc, reg, y_mean, y_std,
                                           ys, train=True)
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
        # test: 单分子向量缓存 + numpy 加权和 + head (大数据测试集秒级)
        comp_vec = encode_unique(enc, all_cids, mol_graphs)
        Z = np.zeros((len(te_idx), 256))
        C = np.zeros((len(te_idx), n_cond))
        for j, mi in enumerate(te_idx):
            acc = None
            for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
                v = comp_vec.get(cid)
                if v is None:
                    continue
                acc = w * v if acc is None else acc + w * v
            Z[j] = acc if acc is not None else np.zeros(256)
            C[j] = meta[mi]['cond']
        inp = np.hstack([Z, C])
        te_y = ys[te_idx]
        with torch.no_grad():
            ps = []
            for s in range(0, len(inp), 4096):
                ps.append(reg(torch.tensor(inp[s:s + 4096],
                                           dtype=torch.float32,
                                           device=DEVICE))
                          .squeeze(1).cpu().numpy())
            p = np.concatenate(ps) * y_std + y_mean
        r = pearsonr(p, te_y)[0]
        mae = float(np.mean(np.abs(p - te_y)))
        rs.append(r)
        maes.append(mae)
        print(f'    >> seed {seed}: R = {r:.4f} | MAE = {mae:.4f}',
              flush=True)
        out_writer.writerow([key, seed, f'{r:.4f}', f'{mae:.4f}'])

    rs, maes = np.array(rs), np.array(maes)
    print(f'>> {name}: R = {rs.mean():.4f} ± {rs.std():.4f} | '
          f'MAE = {maes.mean():.4f} ± {maes.std():.4f}', flush=True)
    out_writer.writerow([key, 'mean', f'{rs.mean():.4f}', f'{maes.mean():.4f}'])
    out_writer.writerow([key, 'std', f'{rs.std():.4f}', f'{maes.std():.4f}'])
    return key, rs.mean(), rs.std(), maes.mean(), maes.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base',
                    required=True, help='CheMixHub data root (containing per-task subdirs)')
    ap.add_argument('--tasks', nargs='+',
                    default=['drug', 'spe', 'iltk', 'ilteta', 'nist'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()

    out = os.path.join(HERE, 'tasks5_e2_results.csv')
    f = open(out, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['task', 'seed', 'R', 'MAE'])
    summary = []
    for key in args.tasks:
        summary.append(run_task(key, args.data_base, args.seeds,
                                args.epochs, args.batch_size, w))
        f.flush()
    f.close()

    print('\n===== 五任务 v(im) 外推汇总 =====')
    for key, rm, rs_, mm, ms in summary:
        print(f'{TASKS[key]["name"]:20s} R = {rm:.4f} ± {rs_:.4f} | '
              f'MAE = {mm:.4f} ± {ms:.4f}')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
