# -*- coding: utf-8 -*-
"""run_nist_ladder.py — NIST ln(η) 分子轴压力阶梯（按贡献移出 k 个分子后重训）

实验 7 — NIST η molecular-axis ladder（补材 S3）
路线 两条（命令行 --route e2 / e1，即 v(im) / v(ex)）
产出 results/nist_ladder_k{1,2,5}_e{1,2}.csv；logs/extrapolation/nist_ladder.log

用途：为 §5.7 的"剂量–反应"分析提供 NIST 这一 v(im) 占优的对照曲线。
NIST 的特殊性：train K≤1（纯物质）、test K=2（恒定二元）→ **组合侧完全固定**，
因此按 ω（新分子权重）分箱得到的是**纯分子轴**剂量–反应，可与 SPE（K=3）对照。

协议与 run_spe_e{1,2}_groups.py 一致：无 val、200 epochs、warmup 3 + cosine、
bsz 128、条件变量 (T, P) 用训练集统计标准化。

用法 (服务器):
  python run_nist_routes.py --route e2 --seeds 42 --epochs 200 \
      --dump-pred nist_e2_pred.csv
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
from encoder import Encoder, GINConv, mol_to_graph, DEVICE  # noqa: E402

print(f'Device: {DEVICE}', flush=True)

DIR = 'nist-logV'
CSV = 'processed_NISTlogV.csv'
CONDS = ['T', 'P']
TRAIN_MAX = 1
BSZ = 128
DUMP = None
DUMPF = None


class WholeEncoder(nn.Module):
    """v(ex)：全体原子一张图 → 跨分子 TFM → 全体 masked mean。"""

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


def load_data(data_base, limit_test=0):
    droot = os.path.join(data_base, DIR, 'processed_data')
    df = pd.read_csv(os.path.join(droot, CSV))
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    cmp_smiles = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}
    meta, ys, conds, skipped = [], [], [], 0
    for _, r in df.iterrows():
        ids = [int(float(c)) for c in eval(r['cmp_ids'])]
        mfs = [float(w) for w in eval(r['cmp_mole_fractions'])]
        if len(ids) != len(mfs):
            skipped += 1
            continue
        meta.append(dict(cids=ids, mfs=mfs, n_comp=len(ids)))
        ys.append(float(r['value']))
        conds.append([pd.to_numeric(r.get(c), errors='coerce') for c in CONDS])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64)
    nc = np.array([m['n_comp'] for m in meta])
    tr_idx = np.where(nc <= TRAIN_MAX)[0].tolist()
    te_idx = np.where(nc > TRAIN_MAX)[0].tolist()
    if limit_test:
        te_idx = te_idx[:limit_test]
    print(f'--- NIST: 有效 {len(meta)} (剔除 {skipped}) | '
          f'K 分布 {dict(sorted((int(k), int((nc == k).sum())) for k in set(nc)))}'
          f' | train(K<={TRAIN_MAX}) {len(tr_idx)} / test {len(te_idx)}', flush=True)
    return meta, ys, conds, cmp_smiles, tr_idx, te_idx


def build_mol_graphs(meta, cmp_smiles):
    cache = {}
    for m in meta:
        for cid in m['cids']:
            if cid in cache:
                continue
            g = mol_to_graph(cmp_smiles.get(cid)) if cmp_smiles.get(cid) else None
            if g is None:
                cache[cid] = None
                continue
            x = torch.tensor(g['x']).float()[:, :31]
            cache[cid] = (torch.cat([x, torch.ones(x.size(0), 1)], dim=1),
                          torch.tensor(g['ei']))
    return cache


def batch_e1(items, mol_graphs, enc, reg, cond_t):
    """v(ex)：样本级拼图（第 32 维 = 真实摩尔分数）。"""
    xs, eis, sids = [], [], []
    off = 0
    for j, (cids, mfs) in enumerate(items):
        n = 0
        for cid, mf in zip(cids, mfs):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            x = g[0].clone()
            x[:, -1] = mf
            xs.append(x)
            eis.append(g[1] + off)
            off += x.size(0)
            n += x.size(0)
        sids.append(torch.full((n,), j, dtype=torch.long))
    if not xs:
        return reg(torch.cat([torch.zeros((len(items), 256), device=DEVICE),
                              cond_t], dim=1)).squeeze(1)
    X = torch.cat(xs).to(DEVICE)
    EI = torch.cat(eis, dim=1).to(DEVICE)
    B = torch.cat(sids).to(DEVICE)
    z = enc.whole(X, EI, B)
    return reg(torch.cat([z, cond_t], dim=1)).squeeze(1)


def batch_e2(ids, meta, mol_graphs, enc, reg, cond_t):
    """v(im)：分子内 TFM → 分子向量 → 摩尔分数凸组合。"""
    graphs, mix_ids, weights = [], [], []
    for j, mi in enumerate(ids):
        for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(g)
            mix_ids.append(j)
            weights.append(w)
    if not graphs:
        return reg(torch.cat([torch.zeros((len(ids), 256), device=DEVICE),
                              cond_t], dim=1)).squeeze(1)
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
    return reg(torch.cat([mix, cond_t], dim=1)).squeeze(1)


def group_stats(p, y, idx):
    if len(idx) < 5:
        return len(idx), float('nan'), float('nan')
    pp, yy = p[idx], y[idx]
    r = float('nan')
    if pp.std() > 1e-9 and yy.std() > 1e-9:
        r = float(pearsonr(pp, yy)[0])
    return len(idx), r, float(np.mean(np.abs(pp - yy)))


def run_route(route, meta, ys, conds, mol_graphs, tr_idx, te_idx, seeds,
              epochs, writer):
    c_mu = np.nanmean(conds[tr_idx], axis=0)
    c_sd = np.nanstd(conds[tr_idx], axis=0)
    c_sd[c_sd == 0] = 1.0
    conds = np.where(np.isnan(conds), c_mu, conds)
    conds = (conds - c_mu) / c_sd
    n_cond = conds.shape[1]
    y_mean, y_std = ys[tr_idx].mean(), ys[tr_idx].std()

    train_mols = {c for i in tr_idx for c in meta[i]['cids']}
    new_set = {c for i in te_idx for c in meta[i]['cids']} - train_mols
    te_omega = np.array([sum(m for c, m in zip(meta[i]['cids'], meta[i]['mfs'])
                             if c in new_set) for i in te_idx])
    grp = np.array(['unseen' if w > 0 else 'seen' for w in te_omega])
    te_y = ys[te_idx]
    print(f'--- route {route}: train {len(tr_idx)} / test {len(te_idx)} | '
          f'test 新分子 {len(new_set)} | unseen 样本 '
          f'{int((grp == "unseen").sum())}/{len(te_idx)}', flush=True)

    def fwd(ids, enc, reg, train=True):
        cond_t = torch.tensor(np.stack([conds[i] for i in ids]),
                              dtype=torch.float32, device=DEVICE)
        if route == 'e1':
            out = batch_e1([(meta[i]['cids'], meta[i]['mfs']) for i in ids],
                           mol_graphs, enc, reg, cond_t)
        else:
            out = batch_e2(ids, meta, mol_graphs, enc, reg, cond_t)
        if train:
            yb = torch.tensor((ys[ids] - y_mean) / y_std,
                              dtype=torch.float32, device=DEVICE)
            return out, yb
        return out.detach().cpu().numpy() * y_std + y_mean

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = (WholeEncoder(dim=256, nhead=4, pad_from=32).to(DEVICE)
               if route == 'e1' else
               Encoder(dim=256, nhead=4, pad_from=32,
                       normalize=False).to(DEVICE))
        reg = nn.Sequential(
            nn.Linear(256 + n_cond, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)).to(DEVICE)
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
            for s in range(0, len(perm), BSZ):
                ids = [tr_idx[i] for i in perm[s:s + BSZ]]
                out, yb = fwd(ids, enc, reg, train=True)
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

        enc.eval()
        reg.eval()
        ps = []
        with torch.no_grad():
            for s in range(0, len(te_idx), BSZ):
                ps.append(fwd(te_idx[s:s + BSZ], enc, reg, train=False))
        p = np.concatenate(ps)

        if DUMP is not None:
            for j, mi in enumerate(te_idx):
                DUMP.writerow([route, seed, mi, int(meta[mi]['n_comp']),
                               f'{te_omega[j]:.4f}', f'{p[j]:.6f}',
                               f'{te_y[j]:.6f}'])
            DUMPF.flush()

        rows = [('all', np.arange(len(te_idx))),
                ('seen', np.where(grp == 'seen')[0]),
                ('unseen', np.where(grp == 'unseen')[0])]
        for name, idx in rows:
            n, r, mae = group_stats(p, te_y, idx)
            writer.writerow([route, seed, name, n, f'{r:.4f}', f'{mae:.4f}'])
            print(f'    >> route {route} seed {seed} {name:6s} n={n:6d} | '
                  f'R {r:.4f} | MAE {mae:.4f}', flush=True)
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base',
                    required=True, help='CheMixHub data root (containing per-task subdirs)')
    ap.add_argument('--route', default='e2', choices=['e1', 'e2'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--limit-test', type=int, default=0,
                    help='仅取前 N 条 test（冒烟用）')
    ap.add_argument('--drop-top-k', type=int, default=0,
                    help='移出 test 总权重贡献最大的 k 个分子的训练样本')
    ap.add_argument('--out', default=os.path.join(HERE,
                                                  'nist_routes_results.csv'))
    ap.add_argument('--dump-pred', default=None)
    args = ap.parse_args()

    global DUMP, DUMPF
    if args.dump_pred:
        DUMPF = open(args.dump_pred, 'w', newline='')
        DUMP = csv.writer(DUMPF)
        DUMP.writerow(['route', 'seed', 'sample_idx', 'K', 'omega',
                       'pred', 'true'])

    meta, ys, conds, cmp_smiles, tr_idx, te_idx = load_data(
        args.data_base, args.limit_test)

    if args.drop_top_k:
        contrib = {}
        for i in te_idx:
            for c, w in zip(meta[i]['cids'], meta[i]['mfs']):
                contrib[c] = contrib.get(c, 0.0) + w
        top = set(sorted(contrib, key=lambda c: -contrib[c])[:args.drop_top_k])
        before = len(tr_idx)
        tr_idx = [i for i in tr_idx
                  if not any(c in top for c in meta[i]['cids'])]
        om = [sum(w for c, w in zip(meta[i]['cids'], meta[i]['mfs'])
                  if c in top) for i in te_idx
              if any(c in top for c in meta[i]['cids'])]
        pct = len(om) / max(len(te_idx), 1)
        print(f'[drop-top-{args.drop_top_k}] 移出 {sorted(top)} | '
              f'train {before} → {len(tr_idx)} | unseen 占比 {pct:.3f} | '
              f'ω {float(np.median(om)):.3f} | '
              f'压力 {pct * float(np.median(om)):.3f}', flush=True)

    mol_graphs = build_mol_graphs(meta, cmp_smiles)
    n_fail = sum(1 for c in mol_graphs if mol_graphs[c] is None)
    print(f'分子图 {len(mol_graphs) - n_fail}/{len(mol_graphs)} 成功', flush=True)

    f = open(args.out, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['route', 'seed', 'group', 'n', 'R', 'MAE'])
    run_route(args.route, meta, ys, conds, mol_graphs, tr_idx, te_idx,
              args.seeds, args.epochs, writer)
    f.close()
    if DUMPF:
        DUMPF.close()
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
