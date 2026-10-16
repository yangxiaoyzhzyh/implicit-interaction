# -*- coding: utf-8 -*-
"""run_hmix_kext.py — ΔHmix / ΔHvap 组分数外推（两条路线）

实验 2 — ΔHmix / ΔHvap four-cell grids（论文 Table 1、补材 S2）
路线 两条（命令行 --route e2 / e1，即 v(im) / v(ex)）
产出 results/hmix_{e1,e2}_traink*.csv 与 hvap_{e1,e2}_traink*.csv（含 _meta.json）；logs/extrapolation/hmix_*.log、hvap_*.log
per-composition-count 拆分由 analysis/summarize_perk.py 汇总（实验 3）
划分: train K<=3 (14,300) | val K=4 (6,122) | test K=5 (9,639)

与五任务 v(im) 臂（per_molecule/run_5tasks_vim.py，同源脚本）
同源同参，差别仅在: ①Hmix 无条件变量(n_cond=0) ②本划分带 K=4 验证集。
  - 原子特征: 31 维化学特征(元素8+手性2+度6+氢5+杂化5+芳香1+电荷3+成环1)
    + 第 32 维（默认常数 1，与 v(im) 一致：摩尔分数只作凸组合权重；
      --mf-as-feature 可改为真实摩尔分数，作对照）
  - Encoder: GIN×3(64) → Linear(64,256) → TFM×2(256,nhead=4) → 逐分子 mean pooling
  - route e2 (per-molecule): Σ_i w_i·v_i → MLP head(256→256→256→1)   [尺寸不变]
  - route e1 (whole-mixture): 全样本原子拼一图 → 跨分子 TFM → 全体原子 mean → head
  - 归一化: train(K<=3) 的 y mean/std；指标还原到 kJ/mol 报 MAE 与 Pearson R

用法（服务器 GPU）:
  python run_hmix_kext.py --route e2 --seeds 42 43 44 --epochs 200
本地冒烟（CPU）:
  python run_hmix_kext.py --route e2 --limit 600 --epochs 3
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROP = 'Enthalpy of mixing'

# ══════════════ 模型 ══════════════


class GINConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
                                 nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))


class Encoder(nn.Module):
    """分子级编码器（与 encoder.py 的 Encoder 同构）。

    pooled(): batch 为**分子**索引 → 每分子一个向量（逐分子 mean pooling）
    whole():  batch 为**样本**索引 → 全体原子跨分子 TFM + 全体 mean pooling
    """

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

    def pooled(self, x, ei, batch):
        xa = self.atom_embed(x, ei, batch)
        return scatter(xa, batch, dim=0, dim_size=int(batch.max()) + 1,
                       reduce='mean')

    def whole(self, x, ei, batch):
        xa = self.atom_embed(x, ei, batch)
        m = torch.zeros(xa.size(0), dtype=torch.bool, device=xa.device)
        # 用样本 batch 做 masked mean
        bsz = int(batch.max()) + 1
        cnt = scatter(torch.ones_like(batch, dtype=torch.float), batch,
                      dim=0, dim_size=bsz, reduce='sum').unsqueeze(-1)
        return scatter(xa, batch, dim=0, dim_size=bsz, reduce='sum') / cnt.clamp(min=1)


# ══════════════ 分子图（31 维特征）══════════════

TARGET_ELEMENTS = [6, 7, 8, 9, 16, 17, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
_gc = {}


def mol_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
               Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
               Chem.HybridizationType.SP3D2]
    nf = []
    for a in mol.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in TARGET_ELEMENTS]
        f += [1 if a.GetChiralTag() == x else 0 for x in CHIRAL_TAGS]
        deg = min(a.GetDegree(), 5)
        f += [1.0 if deg == i else 0.0 for i in range(6)]
        h = min(a.GetTotalNumHs(), 4)
        f += [1.0 if h == i else 0.0 for i in range(5)]
        f += [1.0 if a.GetHybridization() == hyb else 0.0 for hyb in HYBRIDS]
        f += [1.0 if a.GetIsAromatic() else 0.0]
        fc = int(a.GetFormalCharge())
        f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0,
              1.0 if fc > 0 else 0.0]
        f += [1.0 if a.IsInRing() else 0.0]
        nf.append(f[:31])
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
    if not ei:
        ei = [[0, 0]]
    return {'x': np.array(nf, dtype=np.float32),
            'ei': np.array(ei, dtype=np.int64).T}


def build_mol_graphs(cids, cmp_smiles, mf_as_feature=False):
    """每个化合物一张图；返回 {cid: (x[ N,32 ], ei)}。"""
    cache = {}
    for cid in sorted(set(cids)):
        smi = cmp_smiles.get(cid)
        g = mol_to_graph(smi) if smi else None
        if g is None:
            cache[cid] = None
            continue
        x = torch.tensor(g['x']).float()[:, :31]
        mf = torch.zeros(x.size(0), 1) if mf_as_feature else torch.ones(x.size(0), 1)
        cache[cid] = (torch.cat([x, mf], dim=1), torch.tensor(g['ei']))
    return cache


# ══════════════ 数据 ══════════════


def load_records(csv_path, compounds_path):
    df = pd.read_csv(csv_path)
    if 'property' in df.columns:
        df = df[df['property'] == PROP].reset_index(drop=True)
    cmp_smiles = {}
    for r in csv.DictReader(open(compounds_path, encoding='utf-8')):
        cmp_smiles[int(float(r['compound_id']))] = r['smiles']
    recs, skipped = [], 0
    for _, r in df.iterrows():
        ids = list(eval(r['cmp_ids']))
        mfs = list(eval(r['cmp_mole_fractions']))
        if len(ids) != len(mfs):
            skipped += 1
            continue
        recs.append(dict(cids=[int(float(c)) for c in ids],
                         mfs=[float(w) for w in mfs],
                         y=float(r['value'])))
    print(f'[data] {len(recs)} 条有效（剔除缺损 {skipped}）', flush=True)
    return recs, cmp_smiles


# ══════════════ 前向（v(im) / v(ex)）═════════════


def batch_e2(items, mol_graphs, enc, reg):
    """items: [(cids, mfs), ...] → 分子图 batch → 逐分子向量 → 凸组合 → head。"""
    graphs, mix_ids, weights = [], [], []
    for j, (cids, mfs) in enumerate(items):
        for cid, w in zip(cids, mfs):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(g)
            mix_ids.append(j)
            weights.append(w)
    if not graphs:
        z = torch.zeros((len(items), reg[0].in_features), device=DEVICE)
        return reg(z).squeeze(1)
    sizes = [g[0].size(0) for g in graphs]
    xs = torch.cat([g[0] for g in graphs]).to(DEVICE)
    ei = torch.cat([g[1] + sum(sizes[:k]) for k, g in enumerate(graphs)],
                   dim=1).to(DEVICE)
    b = torch.cat([torch.full((s,), k, dtype=torch.long)
                   for k, s in enumerate(sizes)]).to(DEVICE)
    z = enc.pooled(xs, ei, b)
    midx = torch.tensor(mix_ids, device=DEVICE)
    wts = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    mix = scatter(wts * z, midx, dim=0, dim_size=len(items))
    return reg(mix).squeeze(1)


def batch_e1(items, mol_graphs, enc, reg):
    """items → 同一样本所有原子一张图（batch=样本）→ 跨分子 TFM → 全体 mean → head。"""
    xs, eis, sids = [], [], []
    off = 0
    for j, (cids, mfs) in enumerate(items):
        n = 0
        for cid, w in zip(cids, mfs):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            x = g[0].clone()
            x[:, -1] = w                      # v(ex) 把真实摩尔分数放进第 32 维
            xs.append(x)
            eis.append(g[1] + off)
            off += x.size(0)
            n += x.size(0)
        sids.append(torch.full((n,), j, dtype=torch.long))
    if not xs:
        z = torch.zeros((len(items), reg[0].in_features), device=DEVICE)
        return reg(z).squeeze(1)
    X = torch.cat(xs).to(DEVICE)
    EI = torch.cat(eis, dim=1).to(DEVICE)
    B = torch.cat(sids).to(DEVICE)
    return reg(enc.whole(X, EI, B)).squeeze(1)


@torch.no_grad()
def predict(recs, mol_graphs, enc, reg, route, y_mean, y_std, batch_size=128):
    """整段评估；v(im) 走分子向量缓存（化合物仅 81 个，秒级）。"""
    enc.eval()
    reg.eval()
    if route == 'e2':
        cids = sorted({c for r in recs for c in r['cids']})
        graphs, gids = [], []
        for cid in cids:
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(Data(x=g[0], edge_index=g[1]))
            gids.append(cid)
        V = {}
        for b in DataLoader(graphs, batch_size=256, shuffle=False):
            b = b.to(DEVICE)
            v = enc.pooled(b.x, b.edge_index, b.batch).cpu()
            for t, cid in zip(range(v.size(0)), gids[len(V):len(V) + v.size(0)]):
                V[cid] = v[t]
        Z = np.zeros((len(recs), next(iter(V.values())).shape[0]), dtype=np.float32)
        for j, r in enumerate(recs):
            acc = None
            for cid, w in zip(r['cids'], r['mfs']):
                v = V.get(cid)
                if v is None:
                    continue
                acc = w * v.numpy() if acc is None else acc + w * v.numpy()
            if acc is not None:
                Z[j] = acc
        ps = []
        for s in range(0, len(Z), 4096):
            ps.append(reg(torch.tensor(Z[s:s + 4096], device=DEVICE))
                      .squeeze(1).cpu().numpy())
        p = np.concatenate(ps) * y_std + y_mean
    else:
        ps = []
        for s in range(0, len(recs), batch_size):
            items = [(r['cids'], r['mfs']) for r in recs[s:s + batch_size]]
            ps.append(batch_e1(items, mol_graphs, enc, reg).cpu().numpy())
        p = np.concatenate(ps) * y_std + y_mean
    t = np.array([r['y'] for r in recs])
    return pearsonr(p, t)[0], float(np.mean(np.abs(p - t)))


# ══════════════ 早停：val(K=4) 的 Pearson R 连续 patience 个 epoch 不提升即停 ══════════════
# 本划分自带 K=4 验证集，直接用 val patience + best-val 回载即可（无需拐点法）。


# ══════════════ 主流程 ══════════════


def main():
    global PROP
    ap = argparse.ArgumentParser()
    ap.add_argument('--prop', default='Enthalpy of mixing')
    ap.add_argument('--data-dir',
                    default=os.path.abspath(os.path.join(
                        HERE, '..', '..', '..', '..', '..', 'literature', 'CheMixHub',
                        'datasets', 'miscible-solvent', 'processed_data')),
                    help='含 processed_MiscibleSolventData.csv 与 compounds.csv；'
                         '数据根目录（含各任务子目录）'
                         'miscible-solvent/processed_data')
    ap.add_argument('--csv', default='processed_MiscibleSolventData.csv')
    ap.add_argument('--route', default='e2', choices=['e2', 'e1'])
    ap.add_argument('--train-max-k', type=int, default=3)
    ap.add_argument('--val-k', type=int, default=4)
    ap.add_argument('--test-k', type=int, default=5)
    ap.add_argument('--mf-as-feature', action='store_true',
                    help='第32维用真实摩尔分数（默认常数1，浓度只作凸组合权重）')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--early-stop', default='val', choices=['none', 'val'],
                    help='val: val(K=4) 的 Pearson R 连续 patience 个 epoch 不提升即停，'
                         '回载 best-val ckpt')
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--min-epochs', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='冒烟：每集合最多取 N 条')
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results'))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    PROP = args.prop

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.data_dir, args.csv)
    recs, cmp_smiles = load_records(csv_path, os.path.join(args.data_dir,
                                                           'compounds.csv'))
    byk = collections.defaultdict(list)
    for r in recs:
        byk[len(r['cids'])].append(r)
    print('[data] K 分布:', dict(sorted({k: len(v) for k, v in byk.items()}.items())),
          flush=True)

    tr = [r for k, v in byk.items() if k <= args.train_max_k for r in v]
    va = byk.get(args.val_k, [])
    te = byk.get(args.test_k, [])
    if args.limit > 0:
        tr, va, te = tr[:args.limit], va[:max(1, args.limit // 4)], te[:args.limit]
    print(f'[split] train(K<={args.train_max_k}) {len(tr)} | '
          f'val(K={args.val_k}) {len(va)} | test(K={args.test_k}) {len(te)}',
          flush=True)

    trc = {c for r in tr for c in r['cids']}
    for nm, ss in [('val', va), ('test', te)]:
        c = {x for r in ss for x in r['cids']}
        print(f'[split] {nm} 化合物 ⊆ train ? {c <= trc} (独有 {len(c - trc)})',
              flush=True)

    y_tr = np.array([r['y'] for r in tr])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    print(f'[y] train mean={y_mean:.4f} std={y_std:.4f}', flush=True)
    for nm, ss in [('val', va), ('test', te)]:
        y = np.array([r['y'] for r in ss])
        base = float(np.mean(np.abs(y - y_mean)))
        print(f'[baseline] 常数(预测 train 均值) 在 {nm}: MAE={base:.4f} '
              f'| y mean={y.mean():.4f} std={y.std():.4f}', flush=True)

    all_cids = sorted({c for r in tr + va + te for c in r['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles, args.mf_as_feature)
    n_fail = sum(1 for c in all_cids if mol_graphs.get(c) is None)
    print(f'[graph] 化合物 {len(all_cids) - n_fail}/{len(all_cids)} 成功', flush=True)

    tag = args.tag or ("hvap_" if "vapor" in PROP else "hmix_") + f"{args.route}_k{args.train_max_k}to{args.test_k}"
    out_csv = os.path.join(args.out_dir, f'{tag}.csv')
    f = open(out_csv, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['route', 'seed', 'epochs_run', 'stop_ep', 'best_ep',
                'val_R', 'val_MAE', 'test_R', 'test_MAE', 'seconds'])

    rs, maes = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        enc = Encoder(dim=args.dim, nhead=args.nhead, n_layers=args.layers).to(DEVICE)
        reg = nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(args.dim, args.dim), nn.ReLU(),
                            nn.Linear(args.dim, 1)).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters()) + list(reg.parameters()),
                               lr=args.lr, weight_decay=args.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - args.warmup))
        fw = batch_e2 if args.route == 'e2' else batch_e1

        best_r, best_sd, best_ep, stop_ep = -9, None, -1, None
        bad = 0
        t0 = time.time()
        for ep in range(args.epochs):
            if ep < args.warmup:
                for pg in opt.param_groups:
                    pg['lr'] = args.lr * (ep + 1) / args.warmup
            enc.train()
            reg.train()
            perm = np.random.RandomState(seed * 1000 + ep).permutation(len(tr))
            tot, nb = 0.0, 0
            for s in range(0, len(perm), args.batch_size):
                idx = perm[s:s + args.batch_size]
                items = [(tr[i]['cids'], tr[i]['mfs']) for i in idx]
                out = fw(items, mol_graphs, enc, reg)
                yb = torch.tensor([(tr[i]['y'] - y_mean) / y_std for i in idx],
                                  dtype=torch.float32, device=DEVICE)
                loss = F.mse_loss(out, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if ep >= args.warmup:
                sch.step()
            vr, vmae = predict(va, mol_graphs, enc, reg, args.route, y_mean, y_std)
            if vr > best_r:
                best_r, best_ep = vr, ep
                best_sd = {k: v.detach().cpu().clone()
                           for k, v in list(enc.state_dict().items()) +
                           [('reg.' + k, v) for k, v in reg.state_dict().items()]}
                bad = 0
            else:
                bad += 1
            if (args.early_stop == 'val' and ep >= args.min_epochs
                    and bad >= args.patience):
                stop_ep = ep
                print(f'  >> val 连续 {bad} epoch 未提升 → 停于 ep{ep}，'
                      f'回载 best-val ep{best_ep}', flush=True)
                break
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f'  [{tag}] seed {seed} ep {ep:3d} | loss {tot / max(nb, 1):.4f} '
                      f'| val R {vr:.4f} MAE {vmae:.4f} | best {best_r:.4f}@{best_ep}'
                      f' | no-improve {bad} | {time.time() - t0:.0f}s', flush=True)

        if best_sd is not None:
            enc.load_state_dict({k: v for k, v in best_sd.items()
                                 if not k.startswith('reg.')})
            reg.load_state_dict({k[4:]: v for k, v in best_sd.items()
                                 if k.startswith('reg.')})
        vr, vmae = predict(va, mol_graphs, enc, reg, args.route, y_mean, y_std)
        tr_r, tmae = predict(te, mol_graphs, enc, reg, args.route, y_mean, y_std)
        el = time.time() - t0
        rs.append(tr_r)
        maes.append(tmae)
        print(f'>> [{tag}] seed {seed}: best ep{best_ep} '
              f'(stop@{stop_ep if stop_ep is not None else args.epochs - 1}) | '
              f'val(K={args.val_k}) R {vr:.4f} MAE {vmae:.4f} | '
              f'TEST(K={args.test_k}) R {tr_r:.4f} MAE {tmae:.4f} | {el:.0f}s',
              flush=True)
        w.writerow([args.route, seed, stop_ep if stop_ep is not None else args.epochs,
                    stop_ep if stop_ep is not None else '', best_ep,
                    f'{vr:.4f}', f'{vmae:.4f}', f'{tr_r:.4f}', f'{tmae:.4f}',
                    round(el)])
        f.flush()

    f.close()
    rs, maes = np.array(rs), np.array(maes)
    print(f'\n===== {tag} =====')
    print(f'TEST(K={args.test_k}) R = {rs.mean():.4f} ± {rs.std():.4f} | '
          f'MAE = {maes.mean():.4f} ± {maes.std():.4f} (n={len(rs)})', flush=True)
    print(f'saved {out_csv}', flush=True)

    meta = dict(tag=tag, route=args.route, train_max_k=args.train_max_k,
                val_k=args.val_k, test_k=args.test_k,
                mf_as_feature=args.mf_as_feature, epochs=args.epochs,
                early_stop=args.early_stop, seeds=args.seeds,
                n_train=len(tr), n_val=len(va), n_test=len(te),
                y_mean=y_mean, y_std=y_std,
                test_R_mean=float(rs.mean()), test_R_std=float(rs.std()),
                test_MAE_mean=float(maes.mean()), test_MAE_std=float(maes.std()),
                data_csv=os.path.abspath(csv_path))
    with open(os.path.join(args.out_dir, f'{tag}_meta.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
