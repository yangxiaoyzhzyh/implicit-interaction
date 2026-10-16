# -*- coding: utf-8 -*-
"""run_mon_routes.py — MON 组分外推：v(im) 与 v(ex) 在同一协议下严格对照

实验 4 — MON, both routes（论文 Table 1、补材 S2 与表 S8）
路线 两条（命令行 --routes e2 e1，即 v(im) 与 v(ex)）
产出 results/mon_routes_final_results.csv、results/mon_routes_pred.csv；logs/extrapolation/mon_routes_final.log
表 S8 的 per-K 拆分读自 mon_routes_pred.csv

protocol=final（默认，与 MON 外推历史结果 0.9531 完全同口径）：
  复刻外部归档脚本 probe_mon_extrap_final.py（permol_paper bundle）的 v(im) 终版协议：
    - train K<=2 (386) -> test K>=3 (298 = 162×3组分 + 136×>=4组分)
    - 无验证集；固定 200 epochs（无早停、无 best 回载）
    - bsz 32；Adam lr 1e-4 / wd 1e-6；warmup 3 + CosineAnnealingLR(T_max=ep-3)
    - y_mean/y_std 用 tr_idx（386）统计；训练 train() / 测试前 eval()

protocol=e2b（另一份旧协议，仅供对照）：
    - train K<=2 (386) -> test K>=4 (136)
    - train 内部随机 15% 作 val（fit_val_split seed=0），best-val 回载，
      不切 eval；bsz 32；无 scheduler；patience > 40；上限 150 epochs

两条路线唯一差别是结构：
  v(im)：分子内 TFM → 分子向量 → 摩尔分数凸组合 → head
  v(ex)：同一样本全部原子拼一张图 → 跨分子 TFM → 全体原子 mean → head

另报告分子轴分组（seen = test 样本的组分全部出现在 train；unseen = 含未见分子）。

用法 (服务器):
  python run_mon_routes.py --protocol final --routes e2 e1 --seeds 42 43 44 \
      --data-root <data-root>/MON --epochs 200 \
      --dump-pred mon_routes_pred.csv
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
from torch_geometric.utils import scatter, to_dense_batch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encoder import Encoder, GINConv, mol_to_graph, DEVICE  # noqa: E402

print(f'Device: {DEVICE}', flush=True)

BSZ = 32
PATIENCE = 40
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


def load_mon(droot):
    smi = {}
    with open(os.path.join(droot, 'compounds.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            smi[int(float(r['compound_id']))] = r['smiles']
    meta, ys = [], []
    with open(os.path.join(droot, 'processed_MON.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('property') and r['property'] != 'Motor octane number':
                continue
            ids = [int(float(c)) for c in eval(r['cmp_ids'])]
            mfs = [float(w) for w in eval(r['cmp_mole_fractions'])]
            if len(ids) != len(mfs):
                continue
            meta.append(dict(cids=ids, mfs=mfs, n_comp=len(ids)))
            ys.append(float(r['value']))
    return meta, np.array(ys), smi


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


def fit_val_split(n, seed=0, frac=0.15):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_va = max(10, int(n * frac))
    return idx[n_va:], idx[:n_va]


def batch_e1(items, mol_graphs, enc, reg):
    """v(ex)：样本级拼图（第 32 维 = 真实摩尔分数）→ 跨分子 TFM → mean。"""
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
        return torch.zeros(len(items), device=DEVICE)
    X = torch.cat(xs).to(DEVICE)
    EI = torch.cat(eis, dim=1).to(DEVICE)
    B = torch.cat(sids).to(DEVICE)
    z = enc.whole(X, EI, B)
    return reg(z).squeeze(1)


def batch_e2(ids, meta, mol_graphs, enc, reg):
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
        return torch.zeros(len(ids), device=DEVICE)
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
    return reg(mix).squeeze(1)


def group_stats(p, y, idx):
    if len(idx) < 5:
        return len(idx), float('nan'), float('nan')
    pp, yy = p[idx], y[idx]
    r = float('nan')
    if pp.std() > 1e-9 and yy.std() > 1e-9:
        r = float(pearsonr(pp, yy)[0])
    return len(idx), r, float(np.mean(np.abs(pp - yy)))


def run_route(route, meta, ys, mol_graphs, tr_idx, te_idx, seeds, epochs,
              writer, protocol='final'):
    if protocol == 'final':
        fit_g = list(tr_idx)
        va_g = None
        y_mean, y_std = ys[tr_idx].mean(), ys[tr_idx].std()
    else:                                   # e2b：train 内部 15% 作 val
        fit_local, va_local = fit_val_split(len(tr_idx))
        fit_g = [tr_idx[i] for i in fit_local]
        va_g = [tr_idx[i] for i in va_local]
        y_mean, y_std = ys[fit_g].mean(), ys[fit_g].std()

    train_mols = {c for i in tr_idx for c in meta[i]['cids']}
    grp = np.array(['unseen' if set(meta[i]['cids']) - train_mols else 'seen'
                    for i in te_idx])
    te_omega = np.array([sum(w for c, w in zip(meta[i]['cids'], meta[i]['mfs'])
                           if c not in train_mols) for i in te_idx])
    n_new = len({c for i in te_idx for c in meta[i]['cids']} - train_mols)
    print(f'--- route {route} [{protocol}]: train {len(tr_idx)} '
          f'(fit {len(fit_g)}'
          + (f' / val {len(va_g)}' if va_g is not None else '') +
          f') | test {len(te_idx)} | test 新分子 {n_new} | unseen 样本 '
          f'{int((grp == "unseen").sum())}/{len(te_idx)}', flush=True)

    def fwd(ids, enc, reg, train=True):
        if route == 'e1':
            out = batch_e1([(meta[i]['cids'], meta[i]['mfs']) for i in ids],
                           mol_graphs, enc, reg)
        else:
            out = batch_e2(ids, meta, mol_graphs, enc, reg)
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
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters()) + list(reg.parameters()),
                               lr=1e-4, weight_decay=1e-6)
        sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - 3)
               if protocol == 'final' else None)

        best_r, best_state, patience, ep_done = -9, None, 0, epochs - 1
        for ep in range(epochs):
            ep_done = ep
            if protocol == 'final' and ep < 3:      # warmup 3
                for pg in opt.param_groups:
                    pg['lr'] = 1e-4 * (ep + 1) / 3
            enc.train()
            reg.train()
            perm = np.random.RandomState(ep).permutation(len(fit_g))
            for s in range(0, len(perm), BSZ):
                ids = [fit_g[i] for i in perm[s:s + BSZ]]
                out, yb = fwd(ids, enc, reg, train=True)
                loss = F.mse_loss(out, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
            if sch is not None and ep >= 3:
                sch.step()
            if va_g is not None:
                # e2b 原实现：val 整批、不切 eval
                with torch.no_grad():
                    pv = fwd(va_g, enc, reg, train=False)
                rv = pearsonr(pv, ys[va_g])[0]
                if rv > best_r:
                    best_r, patience = rv, 0
                    best_state = (copy.deepcopy(enc.state_dict()),
                                  copy.deepcopy(reg.state_dict()))
                else:
                    patience += 1
                    if patience > PATIENCE:
                        break
        if best_state is not None:
            enc.load_state_dict(best_state[0])
            reg.load_state_dict(best_state[1])
        enc.eval()
        reg.eval()
        with torch.no_grad():
            p = np.concatenate([fwd(te_idx[s:s + BSZ], enc, reg, train=False)
                                for s in range(0, len(te_idx), BSZ)])
        te_y = ys[te_idx]
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
            writer.writerow([route, protocol, seed, name, n,
                             f'{r:.4f}', f'{mae:.4f}'])
            if name == 'all':
                note = (f'best val R {best_r:.3f} @ep{ep_done}'
                        if va_g is not None else f'{epochs} ep')
                print(f'    >> route {route} seed {seed}: {note} | '
                      f'test all R {r:.4f} MAE {mae:.4f}', flush=True)
            else:
                print(f'       {name:6s} n={n:4d} R {r:.4f} MAE {mae:.4f}',
                      flush=True)
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='MON data root')
    ap.add_argument('--protocol', default='final', choices=['final', 'e2b'])
    ap.add_argument('--routes', nargs='+', default=['e2', 'e1'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(HERE,
                                                  'mon_routes_results.csv'))
    ap.add_argument('--dump-pred', default=None)
    args = ap.parse_args()

    global DUMP, DUMPF
    if args.dump_pred:
        DUMPF = open(args.dump_pred, 'w', newline='')
        DUMP = csv.writer(DUMPF)
        DUMP.writerow(['route', 'seed', 'sample_idx', 'K', 'omega',
                       'pred', 'true'])

    meta, ys, cmp_smiles = load_mon(args.data_root)
    test_min = 3 if args.protocol == 'final' else 4
    tr_idx = [i for i, m in enumerate(meta) if m['n_comp'] <= 2]
    te_idx = [i for i, m in enumerate(meta) if m['n_comp'] >= test_min]
    print(f'MON 样本 {len(meta)} | protocol {args.protocol} | '
          f'train(K<=2) {len(tr_idx)} | test(K>={test_min}) {len(te_idx)}',
          flush=True)
    mol_graphs = build_mol_graphs(meta, cmp_smiles)
    n_fail = sum(1 for c in mol_graphs if mol_graphs[c] is None)
    print(f'分子图 {len(mol_graphs) - n_fail}/{len(mol_graphs)} 成功', flush=True)

    f = open(args.out, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['route', 'protocol', 'seed', 'group', 'n', 'R', 'MAE'])
    for route in args.routes:
        run_route(route, meta, ys, mol_graphs, tr_idx, te_idx, args.seeds,
                  args.epochs, writer, protocol=args.protocol)
        f.flush()
    f.close()
    if DUMPF:
        DUMPF.close()
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
