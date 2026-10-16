# -*- coding: utf-8 -*-
"""run_spe_vex_groups.py — SPE κ、v(ex) 路线：按 seen/unseen 分组评估

实验 6 — SPE κ subset decomposition（论文 Table 1、补材 S2/S3）
路线 v(ex)（内部标签 e1）
产出 results/spe_e1_groups_results.csv、results/spe_e1_pred.csv；logs/extrapolation/spe_e1_groups.log、spe_e1_pred.log
配套 per_molecule/run_spe_vim_groups.py（v(im) 臂）

配套 run_spe_vim_groups.py（v(im) 同款分组）。要回答的问题：
SPE 的测试集要求新分子外推时，v(ex) 是否比 v(im) 明显更好（v(im) unseen 仅 0.29）？
若成立，则说明 v(ex) 的读出直接在原子图上进行、可利用与训练分子共享的局部结构，
而 v(im) 强制先压出"分子向量"，新分子缺乏校准 → 分子级数据饥渴。

两个配置与 v(im) 组完全同口径：
  config 2: train(K<=2) -> test(K>=3)   （unseen 样本 39.8%）
  config 3: train(K<=3) -> test(K>=4)   （unseen 样本 7.4%）

协议同 run_spe_e1_k3.py（3 seeds x 200 epochs，无早停，bsz 128，条件标准化）。

用法 (服务器):
  python run_spe_vex_groups.py --configs 2 3 --seeds 42 43 44 --epochs 200
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
from torch_geometric.utils import scatter, to_dense_batch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'per_molecule'))
from encoder import GINConv, mol_to_graph, DEVICE  # noqa: E402

print(f'Device: {DEVICE}', flush=True)

DIR = 'polymer-electrolyte'
CSV = 'processed_PolymerElectrolyteData.csv'
CONDS = ['Temperature, K']
CONFIGS = {2: (2, 3), 3: (3, 4)}   # name -> (train_max_k, test_min_k)
# 若设置，写出逐样本预测（config, seed, idx, K, omega, pred, true）→ 剂量-反应分析
DUMP = None
DUMPF = None


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


def components(r):
    ids = list(eval(r['cmp_ids']))
    mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    return [(int(c), float(w)) for c, w in zip(ids, mfs)]


def build_mol_graphs(cids, cmp_smiles):
    """第 32 维占位 1（前向时替换为真实摩尔分数）。"""
    cache = {}
    for cid in sorted(set(cids)):
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
        for cid, mf in zip(cids, mfs):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            x = g[0].clone()
            x[:, -1] = mf                      # 第 32 维 = 真实摩尔分数
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


def group_stats(p, y, idx):
    if len(idx) < 5:
        return len(idx), float('nan'), float('nan')
    pp, yy = p[idx], y[idx]
    r = float('nan')
    if pp.std() > 1e-9 and yy.std() > 1e-9:
        r = float(pearsonr(pp, yy)[0])
    return len(idx), r, float(np.mean(np.abs(pp - yy)))


def run_config(tag, train_max, test_min, data_base, seeds, epochs,
               batch_size, writer):
    droot = os.path.join(data_base, DIR, 'processed_data')
    df = pd.read_csv(os.path.join(droot, CSV))
    if 'property' in df.columns:
        df = df[df['property'] == 'log Conductivity'].reset_index(drop=True)
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    cmp_smiles = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}

    comps_list, ys, conds, skipped = [], [], [], 0
    for _, r in df.iterrows():
        c = components(r)
        if c is None:
            skipped += 1
            continue
        comps_list.append(c)
        ys.append(float(r['value']))
        conds.append([pd.to_numeric(r[c_], errors='coerce') for c_ in CONDS])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64)
    nc = np.array([len(c) for c in comps_list])
    tr_idx = np.where(nc <= train_max)[0].tolist()
    te_idx = np.where(nc >= test_min)[0].tolist()
    print(f'--- {tag}: 有效 {len(comps_list)} (剔除 {skipped}) | '
          f'K 分布 {dict(sorted((int(k), int((nc == k).sum())) for k in set(nc)))}'
          f' | train(K<={train_max}) {len(tr_idx)} / test(K>={test_min}) '
          f'{len(te_idx)}', flush=True)

    c_mu = np.nanmean(conds[tr_idx], axis=0)
    c_sd = np.nanstd(conds[tr_idx], axis=0)
    c_sd[c_sd == 0] = 1.0
    conds = np.where(np.isnan(conds), c_mu, conds)
    conds = (conds - c_mu) / c_sd
    n_cond = conds.shape[1]

    meta = [dict(cids=[c for c, _ in cs], mfs=[w for _, w in cs],
                 cond=conds[i]) for i, cs in enumerate(comps_list)]
    all_cids = sorted({c for m in meta for c in m['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles)
    n_fail = sum(1 for c in all_cids if mol_graphs.get(c) is None)
    print(f'    分子图 {len(all_cids) - n_fail}/{len(all_cids)} 成功', flush=True)

    train_mols = {c for i in tr_idx for c in meta[i]['cids']}
    grp = np.array(['unseen' if set(meta[i]['cids']) - train_mols else 'seen'
                    for i in te_idx])
    n_new_mol = len({c for i in te_idx for c in meta[i]['cids']} - train_mols)
    print(f'    分子轴: test 新分子 {n_new_mol} | unseen 样本 '
          f'{int((grp == "unseen").sum())}/{len(te_idx)} '
          f'({100.0 * (grp == "unseen").mean():.1f}%)', flush=True)

    new_set = {c for i in te_idx for c in meta[i]['cids']} - train_mols
    te_omega = np.array([sum(m for c, m in zip(meta[i]['cids'], meta[i]['mfs'])
                             if c in new_set) for i in te_idx])

    y_mean, y_std = ys[tr_idx].mean(), ys[tr_idx].std()
    te_y = ys[te_idx]
    te_k = nc[te_idx]

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = WholeEncoder(dim=256, nhead=4, n_layers=2, dropout=0.1,
                           pad_from=32).to(DEVICE)
        reg = nn.Sequential(
            nn.Linear(256 + n_cond, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters()) + list(reg.parameters()),
                               lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - 3)
        for ep in range(epochs):
            if ep < 3:
                for pg in opt.param_groups:
                    pg['lr'] = 1e-4 * (ep + 1) / 3
            enc.train()
            reg.train()
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
            if ep % 50 == 0 or ep == epochs - 1:
                print(f'    seed {seed} ep {ep}: loss {tot / max(nb, 1):.4f}',
                      flush=True)

        ps = []
        with torch.no_grad():
            enc.eval()
            reg.eval()
            for s in range(0, len(te_idx), batch_size):
                idx = te_idx[s:s + batch_size]
                items = [(meta[i]['cids'], meta[i]['mfs']) for i in idx]
                cond_t = torch.tensor(np.stack([meta[i]['cond'] for i in idx]),
                                      dtype=torch.float32, device=DEVICE)
                out = batch_e1(items, mol_graphs, enc, reg, cond_t)
                ps.append(out.detach().cpu().numpy())
        p = np.concatenate(ps) * y_std + y_mean

        if DUMP is not None:
            for j, mi in enumerate(te_idx):
                DUMP.writerow([tag, seed, mi, int(te_k[j]),
                               f'{te_omega[j]:.4f}', f'{p[j]:.6f}',
                               f'{te_y[j]:.6f}'])
            DUMPF.flush()

        rows = [('all', np.arange(len(te_idx))),
                ('seen', np.where(grp == 'seen')[0]),
                ('unseen', np.where(grp == 'unseen')[0])]
        for k in sorted(set(te_k.tolist())):
            for g in ('seen', 'unseen'):
                sel = np.where((te_k == k) & (grp == g))[0]
                if len(sel) >= 5:
                    rows.append((f'K{k}_{g}', sel))
        for name, idx in rows:
            n, r, mae = group_stats(p, te_y, idx)
            writer.writerow([tag, seed, name, n, f'{r:.4f}', f'{mae:.4f}'])
            if name in ('all', 'seen', 'unseen'):
                print(f'    >> seed {seed} {name:6s} n={n:5d} | '
                      f'R {r:.4f} | MAE {mae:.4f}', flush=True)
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base',
                    required=True, help='CheMixHub data root (containing per-task subdirs)')
    ap.add_argument('--configs', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--out',
                    default=os.path.join(HERE, 'spe_e1_groups_results.csv'))
    ap.add_argument('--dump-pred', default=None,
                    help='逐样本预测输出路径（config,seed,idx,K,omega,pred,true）')
    args = ap.parse_args()

    global DUMP, DUMPF
    if args.dump_pred:
        DUMPF = open(args.dump_pred, 'w', newline='')
        DUMP = csv.writer(DUMPF)
        DUMP.writerow(['config', 'seed', 'sample_idx', 'K', 'omega',
                       'pred', 'true'])

    f = open(args.out, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['config', 'seed', 'group', 'n', 'R', 'MAE'])
    for name in args.configs:
        train_max, test_min = CONFIGS[name]
        tag = f'spe_e1_traink{train_max}_testk{test_min}'
        run_config(tag, train_max, test_min, args.data_base, args.seeds,
                   args.epochs, args.batch_size, writer)
        f.flush()
    f.close()
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
