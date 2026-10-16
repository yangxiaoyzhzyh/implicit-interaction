# -*- coding: utf-8 -*-
"""run_spe_vim_groups.py — SPE κ、v(im) 路线：按 seen/unseen 分组评估

实验 6 — SPE κ subset decomposition（论文 Table 1、补材 S2/S3）
路线 v(im)（内部标签 e2）
产出 results/spe_e2_groups_results.csv、results/spe_e2_pred.csv；logs/extrapolation/spe_e2_groups.log、spe_e2_pred.log
配套 whole_mixture/run_spe_vex_groups.py（v(ex) 臂）

动机：tasks5 外推实验（train_max=2）中 SPE 是 v(im) 唯一崩的任务（R=0.467）。
先前判读为"组合覆盖断档"，但统计发现该 test 中 46.9% 的样本含训练未见分子
（38 个），而"补三元"实验（train<=3 -> test>=4）把这一比例降到 7.4%（3 个）。
本脚本对两个配置各跑一遍，并在 test 上按 seen / unseen 分子分组算 R/MAE，
以分离"组合轴（K 增加）"与"分子轴（新分子）"的贡献。

协议与原 v(im) 臂完全一致（3 seeds x 200 epochs，无早停，bsz 128，条件标准化）；
唯一可配项是 train_max_k / test_min_k。

用法 (服务器):
  python run_spe_vim_groups.py --configs 2 3 --seeds 42 43 44 --epochs 200
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
from encoder import Encoder, mol_to_graph, DEVICE  # noqa: E402

DIR = 'polymer-electrolyte'
CSV = 'processed_PolymerElectrolyteData.csv'
CONDS = ['Temperature, K']
CONFIGS = {2: (2, 3), 3: (3, 4)}   # name -> (train_max_k, test_min_k)
# 若设置，写出逐样本预测（config, seed, idx, K, omega, pred, true）→ 剂量-反应分析
DUMP = None
DUMPF = None


def components(r):
    ids = list(eval(r['cmp_ids']))
    mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    return [(int(c), float(w)) for c, w in zip(ids, mfs)]


def build_mol_graphs(cids, cmp_smiles):
    cache = {}
    for cid in sorted(set(cids)):
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
    out = reg(torch.cat([mix, cond_t], dim=1)).squeeze(1)
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


def group_stats(p, y, idx):
    """返回 (n, R, MAE)；组内退化时 R = nan。"""
    if len(idx) < 5:
        return len(idx), float('nan'), float('nan')
    pp, yy = p[idx], y[idx]
    r = float('nan')
    if pp.std() > 1e-9 and yy.std() > 1e-9:
        r = float(pearsonr(pp, yy)[0])
    return len(idx), r, float(np.mean(np.abs(pp - yy)))


def run_config(tag, train_max, test_min, data_base, seeds, epochs,
               batch_size, w):
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
          f'K 分布 {{k: n}} {dict(sorted((int(k), int((nc == k).sum())) for k in set(nc)))}'
          f' | train(K<={train_max}) {len(tr_idx)} / test(K>={test_min}) {len(te_idx)}',
          flush=True)

    c_mu = np.nanmean(conds[tr_idx], axis=0)
    c_sd = np.nanstd(conds[tr_idx], axis=0)
    c_sd[c_sd == 0] = 1.0
    conds = np.where(np.isnan(conds), c_mu, conds)
    conds = (conds - c_mu) / c_sd
    n_cond = conds.shape[1]

    meta = [{'cids': [c for c, _ in cs], 'mfs': [v for _, v in cs],
             'cond': conds[i]} for i, cs in enumerate(comps_list)]
    all_cids = sorted({c for m in meta for c in m['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles)
    n_fail = sum(1 for c in all_cids if mol_graphs.get(c) is None)
    print(f'    分子图 {len(all_cids) - n_fail}/{len(all_cids)} 成功', flush=True)

    # 分子轴分组：test 样本是否含"训练集未出现过的分子"
    train_mols = {c for i in tr_idx for c in meta[i]['cids']}
    grp = np.array(['unseen' if set(meta[i]['cids']) - train_mols else 'seen'
                    for i in te_idx])
    n_new_mol = len({c for i in te_idx for c in meta[i]['cids']} - train_mols)
    print(f'    分子轴: test 新分子 {n_new_mol} | unseen 样本 '
          f'{int((grp == "unseen").sum())}/{len(te_idx)} '
          f'({100.0 * (grp == "unseen").mean():.1f}%)', flush=True)

    # 每个测试样本的新分子总摩尔分数（0 = 分子全见）
    new_set = {c for i in te_idx for c in meta[i]['cids']} - train_mols
    te_omega = np.array([sum(m for c, m in zip(meta[i]['cids'], meta[i]['mfs'])
                             if c in new_set) for i in te_idx])

    y_mean, y_std = ys[tr_idx].mean(), ys[tr_idx].std()
    te_y = ys[te_idx]
    te_k = nc[te_idx]

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = Encoder(dim=256, nhead=4, pad_from=32, normalize=False).to(DEVICE)
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
                ids = [tr_idx[i] for i in perm[s:s + batch_size]]
                out, yb = forward_mixtures(ids, meta, mol_graphs, enc, reg,
                                           y_mean, y_std, ys, train=True)
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

        comp_vec = encode_unique(enc, all_cids, mol_graphs)
        Z = np.zeros((len(te_idx), 256))
        C = np.zeros((len(te_idx), n_cond))
        for j, mi in enumerate(te_idx):
            acc = None
            for cid, wt in zip(meta[mi]['cids'], meta[mi]['mfs']):
                v = comp_vec.get(cid)
                if v is None:
                    continue
                acc = wt * v if acc is None else acc + wt * v
            Z[j] = acc if acc is not None else np.zeros(256)
            C[j] = meta[mi]['cond']
        inp = np.hstack([Z, C])
        with torch.no_grad():
            ps = []
            for s in range(0, len(inp), 4096):
                ps.append(reg(torch.tensor(inp[s:s + 4096],
                                           dtype=torch.float32, device=DEVICE))
                          .squeeze(1).cpu().numpy())
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
            w.writerow([tag, seed, name, n, f'{r:.4f}', f'{mae:.4f}'])
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
    ap.add_argument('--out', default=os.path.join(HERE,
                                                  'spe_e2_groups_results.csv'))
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
    w = csv.writer(f)
    w.writerow(['config', 'seed', 'group', 'n', 'R', 'MAE'])
    for name in args.configs:
        train_max, test_min = CONFIGS[name]
        tag = f'spe_e2_traink{train_max}_testk{test_min}'
        run_config(tag, train_max, test_min, args.data_base, args.seeds,
                   args.epochs, args.batch_size, w)
        f.flush()
    f.close()
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
