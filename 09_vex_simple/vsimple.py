# -*- coding: utf-8 -*-
"""v(ex)-simple：整图路线的复现脚本（独立目录，不依赖 common/ 引擎）

模型：一个混合物的全部原子进入同一张图（只在分子内连边，分子间不加边），
每个原子除 44 维化学特征外再带一个该组分的摩尔分数（第 45 维）；GIN×3 之后
接一个跨分子注意力 transformer，最后对全体原子做 masked mean 得到一个向量，
交给 MLP 读出。没有逐组分编码器，也没有 CLS token。

参数量 1,228,420，与 v(im) 相同；v(ex) 为 2,217,604。

本脚本产出补充材料 Table S1 的 "v(ex)-simple" 一列（九个组分轴任务 × 3 seeds）。
原子特征用与正文其余实验相同的 44 维词表（18 元素、度上限 8）。
协议与正文一致：训练取低组分数一侧、测试取高组分数一侧，200 epochs、无早停、
Adam 1e-4、batch 128；划分直接读 01_composition 的同一份 npz（--split-file），
其中 val_idx 会并回训练，与 engine.py 在组分轴的行为一致。

用法：
  python vsimple.py --data-base /path/to/chemidata \
      --tasks drug spe iltk ilteta nist --seeds 42 43 44

输出：results/vsimple_composition.csv（task, seed, test_R, test_R2, MAE,
n_train, n_test），以及逐 seed 的 stdout 摘要。
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr
from torch_geometric.utils import scatter, to_dense_batch

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------- 任务表
TASKS = {
    'drug': dict(dir='drug-solubility', csv='processed_DrugSolubilityData.csv',
                 prop='Log solubility', kind='drug',
                 conds=['Temperature, K'], train_max=2),
    'spe': dict(dir='polymer-electrolyte',
                csv='processed_PolymerElectrolyteData.csv',
                prop='log Conductivity', kind='standard',
                conds=['Temperature, K'], train_max=2),
    'iltk': dict(dir='ionic-liquids', csv='processed_IlThermoData.csv',
                 prop='Electrical conductivity', kind='standard',
                 conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                 train_max=2),
    'ilteta': dict(dir='ionic-liquids', csv='processed_IlThermoData.csv',
                   prop='Viscosity', kind='standard',
                   conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                   train_max=2),
    'nist': dict(dir='nist-logV', csv='processed_NISTlogV.csv', prop=None,
                 kind='standard', conds=['T', 'P'], train_max=1),
    # 组分数轴的其余四个任务（补齐 Table S1）
    'miscible': dict(dir='miscible-solvent',
                     csv='processed_MiscibleSolventData.csv',
                     prop='Enthalpy of mixing', kind='standard',
                     conds=[], train_max=2),
    'miscible_density': dict(dir='miscible-solvent',
                             csv='processed_MiscibleSolventData.csv',
                             prop='Density', kind='standard',
                             conds=[], train_max=2),
    'miscible_hvap': dict(dir='miscible-solvent',
                          csv='processed_MiscibleSolventData.csv',
                          prop='Heat of vaporization', kind='standard',
                          conds=[], train_max=2),
    'mon': dict(dir='MON', csv='processed_MON.csv', prop=None,
                kind='standard', conds=[], train_max=2),
}

# ---------------------------------------------------------------- 特征
TARGET_ELEMENTS = [3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 30, 33, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
           Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
           Chem.HybridizationType.SP3D2]
N_DEG, N_H = 9, 5
# 18 元素 + 手性 2 + 度 9 + 氢 5 + 杂化 5 + 芳香 1 + 电荷 3 + 成环 1 = 44
FEAT = len(TARGET_ELEMENTS) + len(CHIRAL_TAGS) + N_DEG + N_H + len(HYBRIDS) + 1 + 3 + 1
WIDTH = FEAT + 1   # 最后 1 维 = 该组分的摩尔分数

_gc = {}


def mol_to_graph(smi):
    """44 维原子特征 + 边索引；按 SMILES 缓存。"""
    if smi in _gc:
        return _gc[smi]
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        _gc[smi] = None
        return None
    nf = []
    for a in mol.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in TARGET_ELEMENTS]
        f += [1 if a.GetChiralTag() == x else 0 for x in CHIRAL_TAGS]
        deg = min(a.GetDegree(), N_DEG - 1)
        f += [1.0 if deg == i else 0.0 for i in range(N_DEG)]
        h = min(a.GetTotalNumHs(), 4)
        f += [1.0 if h == i else 0.0 for i in range(5)]
        f += [1.0 if a.GetHybridization() == hyb else 0.0 for hyb in HYBRIDS]
        f += [1.0 if a.GetIsAromatic() else 0.0]
        fc = int(a.GetFormalCharge())
        f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0,
              1.0 if fc > 0 else 0.0]
        f += [1.0 if a.IsInRing() else 0.0]
        nf.append(f[:FEAT])
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
    if not ei:
        ei = [[0, 0]]
    out = (torch.tensor(np.array(nf, dtype=np.float32)),
           torch.tensor(np.array(ei, dtype=np.int64).T))
    _gc[smi] = out
    return out


# ---------------------------------------------------------------- 模型
class GIN(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
                                 nn.Linear(d, d), nn.BatchNorm1d(d))
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))


class WholeMixture(nn.Module):
    """GIN×3 → Linear(64→256) → TFM×2 → 全体原子 masked mean → MLP。"""

    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1, ncond=0):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - WIDTH, 0, 0))
        self.convs = nn.ModuleList([GIN(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                           dim_feedforward=2 * dim,
                                           batch_first=True, dropout=dropout)
        self.tfm = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(dim + ncond, dim), nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(dim, dim), nn.ReLU(),
                                  nn.Linear(dim, 1))

    def forward(self, x, ei, batch, cond=None):
        x = self.pad(x)
        for c in self.convs:
            x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        z = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        if cond is not None:
            z = torch.cat([z, cond], dim=-1)
        return self.head(z).squeeze(1)


# ---------------------------------------------------------------- 数据
def components(row, kind):
    if kind == 'drug':
        ids = list(eval(row['cmp_ids_solvent'])) + list(eval(row['cmp_ids_drug']))
        mfs = (list(eval(row['cmp_mole_fractions_solvent'])) +
               list(eval(row['cmp_mole_fractions_drug'])))
    else:
        ids = list(eval(row['cmp_ids']))
        mfs = list(eval(row['cmp_mole_fractions']))
    # 与 common/spec.py 一致：一行都不跳过，摩尔分数按 cmp_ids 截断/补零后归一化
    try:
        ids = [int(float(c)) for c in ids]
    except Exception:
        ids = []
    try:
        mfs = [float(w) for w in mfs]
    except Exception:
        mfs = []
    if len(mfs) != len(ids):
        mfs = (mfs + [0.0] * len(ids))[:len(ids)]
    tot = sum(mfs)
    if ids and tot > 0:
        mfs = [w / tot for w in mfs]
    elif ids:
        mfs = [1.0 / len(ids)] * len(ids)
    return list(zip(ids, mfs))


def load_task(key, data_base):
    tk = TASKS[key]
    droot = os.path.join(data_base, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, tk['csv']))
    if tk['prop'] is not None and 'property' in df.columns:
        df = df[df['property'] == tk['prop']].reset_index(drop=True)
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    smi = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}

    items, ys, conds = [], [], []
    for _, r in df.iterrows():
        items.append(components(r, tk['kind']))
        ys.append(float(r['value']))
        conds.append([pd.to_numeric(r.get(k), errors='coerce')
                      for k in tk['conds']])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64).reshape(len(items), -1)
    return items, ys, conds, smi


def build_batch(idx, items, conds, ys, graphs, y_mu, y_sd, c_mu, c_sd):
    xs, eis, sids = [], [], []
    off = 0
    for j, i in enumerate(idx):
        n = 0
        for cid, w in items[i]:
            g = graphs.get(cid)
            if g is None:
                continue
            x = g[0].clone()
            x[:, -1] = w                      # 最后 1 维 = 摩尔分数
            xs.append(x)
            eis.append(g[1] + off)
            off += x.size(0)
            n += x.size(0)
        sids.append(torch.full((n,), j, dtype=torch.long))
    X = torch.cat(xs).to(DEVICE)
    EI = torch.cat(eis, dim=1).to(DEVICE)
    B = torch.cat(sids).to(DEVICE)
    C = None
    if conds.shape[1]:
        C = torch.tensor((conds[idx] - c_mu) / c_sd, dtype=torch.float32,
                         device=DEVICE)
    Y = torch.tensor((ys[idx] - y_mu) / y_sd, dtype=torch.float32,
                     device=DEVICE)
    return X, EI, B, C, Y


# ---------------------------------------------------------------- 训练
def run_task(key, data_base, seeds, epochs, batch, split_file=None):
    t0 = time.time()
    items, ys, conds, smi = load_task(key, data_base)
    tk = TASKS[key]
    if split_file:
        sp = np.load(split_file, allow_pickle=True)
        tr = sp['train_idx'].tolist()
        # 组分轴不早停：把 npz 里的 val 并回训练（与 engine.py 的行为一致）
        if 'val_idx' in sp:
            tr = tr + sp['val_idx'].tolist()
        te = sp['test_idx'].tolist()
        print('--- %s: 读划分 %s (train %d / test %d)'
              % (key, os.path.basename(split_file), len(tr), len(te)), flush=True)
    else:
        K = np.array([len(c) for c in items])
        tr = [i for i in range(len(items)) if K[i] <= tk['train_max']]
        te = [i for i in range(len(items)) if K[i] > tk['train_max']]
        print('--- %s: train %d (K<=%d) / test %d' %
              (key, len(tr), tk['train_max'], len(te)), flush=True)

    graphs = {cid: mol_to_graph(smi.get(cid))
              for cids in items for cid, _ in cids}
    graphs = {c: (None if g is None else
                  (torch.cat([g[0], torch.ones(g[0].size(0), 1)], 1), g[1]))
              for c, g in graphs.items()}

    y_mu, y_sd = ys[tr].mean(), ys[tr].std()
    if conds.shape[1]:
        c_mu = np.nanmean(conds[tr], 0)
        c_sd = np.nanstd(conds[tr], 0)
        c_sd[c_sd == 0] = 1.0
        conds = np.where(np.isnan(conds), c_mu, conds)
    else:
        c_mu = c_sd = np.zeros(0)
    ncond = conds.shape[1]

    rowsout = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WholeMixture(ncond=ncond).to(DEVICE)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs - 3))
        for ep in range(epochs):
            if ep < 3:
                for pg in opt.param_groups:
                    pg['lr'] = 1e-4 * (ep + 1) / 3
            model.train()
            perm = np.random.permutation(tr)
            tot = 0.0
            for s in range(0, len(perm), batch):
                idx = perm[s:s + batch].tolist()
                X, EI, B, C, Y = build_batch(idx, items, conds, ys, graphs,
                                             y_mu, y_sd, c_mu, c_sd)
                opt.zero_grad()
                loss = ((model(X, EI, B, C) - Y) ** 2).mean()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            if ep >= 3:
                sch.step()
            if ep % 20 == 0 or ep == epochs - 1:
                print('    seed %d ep %3d loss %.4f' % (seed, ep, tot / len(perm)),
                      flush=True)

        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(te), 256):
                idx = te[s:s + 256]
                X, EI, B, C, _ = build_batch(idx, items, conds, ys, graphs,
                                             y_mu, y_sd, c_mu, c_sd)
                preds.append(model(X, EI, B, C).cpu().numpy() * y_sd + y_mu)
        p = np.concatenate(preds)
        t = ys[te]
        r = float(pearsonr(p, t)[0])
        r2 = float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())
        mae = float(np.abs(p - t).mean())
        print('  >> seed %d | test R %+.4f R2 %+.4f MAE %.4f | %d params'
              % (seed, r, r2, mae, n_par), flush=True)
        rowsout.append(dict(task=key, seed=seed, test_R=round(r, 4),
                            test_R2=round(r2, 4), MAE=round(mae, 4),
                            n_train=len(tr), n_test=len(te)))
    print('  %s 总用时 %.0fs' % (key, time.time() - t0), flush=True)
    return rowsout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base', required=True,
                    help='CheMixHub 数据根目录（含 drug-solubility/ 等子目录）')
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results'))
    ap.add_argument('--task', default=None,
                    help='只跑一个任务（配合 --split-file 使用）')
    ap.add_argument('--split-file', default=None,
                    help='npz 划分文件（train_idx / test_idx），与 01_composition 同一份')
    ap.add_argument('--out', default=None, help='单个任务的输出 CSV')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.task:
        rows = run_task(args.task, args.data_base, args.seeds, args.epochs,
                        args.batch, args.split_file)
        out = args.out or os.path.join(args.out_dir, '%s_vsimple.csv' % args.task)
        pd.DataFrame(rows).to_csv(out, index=False)
        df = pd.DataFrame(rows)
        print('\n%s: R = %.4f ± %.4f（写出 %s）'
              % (args.task, df.test_R.mean(), df.test_R.std(), out))
        return
    out = os.path.join(args.out_dir, 'vsimple_composition.csv')
    rows = []
    for key in args.tasks:
        rows += run_task(key, args.data_base, args.seeds, args.epochs,
                         args.batch, args.split_file)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('\n写出 %s（%d 行）' % (out, len(rows)))


if __name__ == '__main__':
    main()
    sys.exit(0)
