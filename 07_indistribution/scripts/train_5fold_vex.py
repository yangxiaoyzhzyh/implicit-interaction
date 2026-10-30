# -*- coding: utf-8 -*-
"""train_5fold_vex.py — 官方 5-fold 随机 CV，v(ex) 路线（AtomFormer）

来自 paper_v2/experiments/scripts/whole_mixture/train_task.py。
本目录 runs/vex_5fold.csv 的数值由这批运行产生，但**本次未重跑**：
数值取自上一篇文章（AtomFormer）已发表结果，见 README。
"""

实验 1 — CheMixHub 8 tasks, official 5-fold（论文/补材的 in-distribution anchors，补材 S1 表 S2）
路线 v(ex)
产出 logs/chemixhub_5fold_whole/（逐折 R 与均值；该臂只落日志，无结果 CSV）
配套 per_molecule/train_task_permol.py（v(im) 臂）
v(ex)（GIN×3 + TFM×2 + MLP）在 CheMixHub 多个混合物理化性质任务上的统一训练/评估。

从 v(ex) 臂的 MON 脚本泛化而来，改动：
- 多任务配置 TASK_CONFIG：CSV 文件名、property 筛选、条件列、split 目录。
- 条件标量（温度/压力/频率）在 masked-mean pooling 之后 concat 进 regressor
  （图级上下文，非 per-component），regressor 输入维自适应。
- split indices 语义：先按 property 筛选子集，再对子集用 indices 划分（官方 load_data 逻辑）。

用法:
    python train_task.py --task mon
    python train_task.py --task logv --folds 0 1 2 3 4
    python train_task.py --task ionic_visc
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOTAL_EPOCHS = 60
WARMUP_EPOCHS = 3
BATCH_SIZE = 256
LR = 1e-4

CHEMIX_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'literature',
    'CheMixHub', 'datasets'))

TARGET_ELEMENTS = [6, 7, 8, 9, 16, 17, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]

# ── 各任务配置 ──
# dir: 数据集目录名（task key 与目录可能不同）
# csv: 相对 processed_data 的文件名
# prop: 若非 None，按该 property 筛选子集；None 表示全量
# conds: 作为图级条件注入的列名（pooling 后 concat）
# split: split 目录名
TASK_CONFIG = {
    'mon': dict(
        dir='MON', csv='processed_MON.csv', prop=None, conds=[],
        split='motor_octane_number_splits',
        prop_col='property', smiles_col='smiles'),
    'density': dict(
        dir='miscible-solvent', csv='processed_MiscibleSolventData.csv',
        prop='Density', conds=[], split='density_splits',
        prop_col='property', smiles_col='smiles'),
    'heat_vap': dict(
        dir='miscible-solvent', csv='processed_MiscibleSolventData.csv',
        prop='Heat of vaporization', conds=[],
        split='heat_of_vaporization_splits',
        prop_col='property', smiles_col='smiles'),
    'he': dict(
        dir='miscible-solvent', csv='processed_MiscibleSolventData.csv',
        prop='Enthalpy of mixing', conds=[],
        split='enthalpy_of_mixing_splits',
        prop_col='property', smiles_col='smiles'),
    'logv': dict(
        dir='logV', csv='processed_logV.csv', prop=None, conds=['T'],
        split='log_viscosity_splits', prop_col='property',
        smiles_col='smiles'),
    'nist_logv': dict(
        dir='nist-logV', csv='processed_NISTlogV.csv', prop=None,
        conds=['T', 'P'], split='log_viscosity_splits', prop_col='property',
        smiles_col='smiles'),
    'ionic_cond': dict(
        dir='ionic-liquids', csv='processed_IlThermoData.csv',
        prop='Electrical conductivity',
        conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
        split='electrical_conductivity_splits', prop_col='property',
        smiles_col='smiles'),
    'ionic_visc': dict(
        dir='ionic-liquids', csv='processed_IlThermoData.csv',
        prop='Viscosity',
        conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
        split='viscosity_splits', prop_col='property', smiles_col='smiles'),
}


class GINConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))


class Model(nn.Module):
    """v2 同款：32 维原子特征(31 化学 + 摩尔分数)，pooling 后 concat 条件标量。"""

    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1,
                 variant='full', n_cond=0):
        super().__init__()
        self.variant = variant
        self.n_cond = n_cond
        self.pad = nn.ZeroPad2d((0, 64 - 32, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        if variant != 'gin_only':
            enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                             dim_feedforward=2 * dim,
                                             batch_first=True, dropout=dropout)
            self.tfm = nn.TransformerEncoder(enc, n_layers)
        # 第一层 LazyLinear 自适应 dim + n_cond
        self.regressor = nn.Sequential(
            nn.LazyLinear(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1))

    def forward(self, x, ei, batch, cond=None):
        x = self.pad(x)
        if self.variant != 'tfm_only':
            for c in self.convs:
                x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        if self.variant != 'gin_only':
            x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)  # [B, dim]
        if cond is not None and self.n_cond > 0:
            x = torch.cat([x, cond], dim=-1)                  # [B, dim+n_cond]
        return self.regressor(x).squeeze(1)


class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'y_orig', 'cond'):
            return None
        return super().__cat_dim__(key, value, *a, **kw)


_gc = {}


def mol_to_graph(smi):
    """31 维原子特征：元素8+手性1+度6+氢5+杂化5+芳香1+电荷3+成环1。"""
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
        f += [1.0 if a.GetHybridization() == hyb else 0.0
              for hyb in HYBRIDS]
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


def build_sample(cmp_ids, mol_fracs, cmp_smiles):
    """混合物图：各组分原子 + 摩尔分数第 32 维。"""
    ax, aei = [], []
    off = 0
    for ci, cid in enumerate(cmp_ids):
        smi = cmp_smiles.get(int(float(cid)))
        if smi is None:
            return None
        if smi not in _gc:
            _gc[smi] = mol_to_graph(smi)
        g = _gc[smi]
        if g is None:
            return None
        x = torch.tensor(g['x']).float()[:, :31]
        mf = torch.full((x.size(0), 1), mol_fracs[ci], dtype=torch.float32)
        ax.append(torch.cat([x, mf], dim=1))
        aei.append(torch.tensor(g['ei']) + off)
        off += x.size(0)
    if off == 0:
        return None
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))


def load_compounds(data_root, smiles_col='smiles'):
    import csv
    cmp_smiles = {}
    with open(os.path.join(data_root, 'compounds.csv')) as f:
        for r in csv.DictReader(f):
            try:
                cmp_smiles[int(float(r['compound_id']))] = r[smiles_col]
            except (KeyError, ValueError):
                pass
    return cmp_smiles


def load_dataset(data_root, cfg):
    df = pd.read_csv(os.path.join(data_root, cfg['csv']))
    if cfg['prop'] is not None:
        df = df[df[cfg['prop_col']] == cfg['prop']].reset_index(drop=True)
    print(f"[{cfg['csv']}] {cfg['prop'] or 'all'} 样本: {len(df):,}")
    cmp_smiles = load_compounds(data_root, cfg['smiles_col'])
    cond_cols = cfg['conds']
    records = []
    skipped = 0
    for _, r in df.iterrows():
        cmp_ids = eval(r['cmp_ids'])
        mfs = eval(r['cmp_mole_fractions'])
        d = build_sample(cmp_ids, mfs, cmp_smiles)
        if d is None:
            skipped += 1
            continue
        try:
            d.y = torch.tensor(float(r['value']), dtype=torch.float32)
        except (TypeError, ValueError):
            skipped += 1
            continue
        d.y_orig = d.y.clone()
        if cond_cols:
            cond = [float(r[c]) for c in cond_cols]
            d.cond = torch.tensor(cond, dtype=torch.float32)
        records.append(d)
    print(f'有效样本: {len(records):,} (跳过 {skipped})')
    return records


def evaluate(model, loader, y_mean, y_std, has_cond):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            cond = b.cond if has_cond else None
            z = model(b.x, b.edge_index, b.batch, cond)
            preds.append((z.cpu() * y_std + y_mean).numpy())
            trues.append((b.y.cpu() * y_std + y_mean).numpy())
    p = np.concatenate(preds).flatten()
    t = np.concatenate(trues).flatten()
    return pearsonr(p, t)[0], np.mean(np.abs(p - t))


def run_fold(fold, records, data_root, cfg, epochs, dim=256, nhead=4,
             n_layers=2, dropout=0.1, lr=LR, wd=1e-6, batch=BATCH_SIZE,
             warmup=WARMUP_EPOCHS, seed=42, variant='full',
             split_type='kfold'):
    import safetensors.torch as st
    split = st.load_file(os.path.join(data_root, cfg['split'],
                                      f'{split_type}_split_{fold}.safetensors'))
    tr_idx, va_idx, te_idx = (split[k].numpy() for k in
                              ('train_indices', 'val_indices', 'test_indices'))
    print(f'\n===== Fold {fold}: train {len(tr_idx):,} / val {len(va_idx):,} '
          f'/ test {len(te_idx):,} =====')

    train_data = [records[i] for i in tr_idx]
    val_data = [records[i] for i in va_idx]
    test_data = [records[i] for i in te_idx]
    for d in train_data + val_data + test_data:
        d.y = d.y_orig.clone()

    all_y = torch.stack([d.y for d in train_data])
    y_mean, y_std = all_y.mean(), all_y.std()
    for d in train_data + val_data + test_data:
        d.y = (d.y - y_mean) / y_std

    # 条件列归一化（用训练集 mean/std）
    has_cond = bool(cfg['conds'])
    cond_mean = cond_std = None
    if has_cond:
        C = torch.stack([d.cond for d in train_data])
        cond_mean = C.mean(dim=0)
        cond_std = C.std(dim=0).clamp(min=1e-8)
        for d in train_data + val_data + test_data:
            d.cond = (d.cond - cond_mean) / cond_std

    tr_loader = DataLoader(train_data, batch_size=batch, shuffle=True,
                           num_workers=0)
    va_loader = DataLoader(val_data, batch_size=256, shuffle=False,
                           num_workers=0)
    te_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                           num_workers=0)

    torch.manual_seed(seed + fold)
    random.seed(seed + fold)
    np.random.seed(seed + fold)
    n_cond = len(cfg['conds'])
    m = Model(dim=dim, nhead=nhead, n_layers=n_layers, dropout=dropout,
              variant=variant, n_cond=n_cond).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs - warmup)

    best_r, best_state, best_ep = -float('inf'), None, -1
    for ep in range(epochs):
        if ep < warmup:
            for pg in opt.param_groups:
                pg['lr'] = lr * (ep + 1) / warmup
        m.train()
        tl = 0
        for b in tr_loader:
            b = b.to(DEVICE)
            cond = b.cond if has_cond else None
            z = m(b.x, b.edge_index, b.batch, cond)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        if ep >= warmup:
            sch.step()
        vr, vmae = evaluate(m, va_loader, y_mean, y_std, has_cond)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            best_ep = ep
            torch.save(best_state,
                       f'model_task_{cfg_key}_{split_type}fold{fold}.pt')
        print(f'  ep {ep:2d} | train {tl/len(tr_loader):.4f} | '
              f'val R {vr:.4f} MAE {vmae:.4f} | best {best_r:.4f}@{best_ep}',
              flush=True)

    m.load_state_dict(best_state)
    tr_, tmae = evaluate(m, te_loader, y_mean, y_std, has_cond)
    print(f'>> Fold {fold}: best ep {best_ep} | test R {tr_:.4f} '
          f'MAE {tmae:.4f}')
    return tr_, tmae


cfg_key = None  # 在 main 里设置，供 run_fold 存 checkpoint 用


def main():
    global cfg_key
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=list(TASK_CONFIG))
    p.add_argument('--epochs', type=int, default=TOTAL_EPOCHS)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--root', default=CHEMIX_ROOT,
                   help='数据集根目录，内含 <dir>/processed_data（默认 CHEMIX_ROOT）')
    p.add_argument('--data-root', default=None,
                   help='直接覆盖单个任务的 processed_data 目录')
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--wd', type=float, default=1e-6)
    p.add_argument('--batch', type=int, default=BATCH_SIZE)
    p.add_argument('--warmup', type=int, default=WARMUP_EPOCHS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variant', choices=['full', 'gin_only', 'tfm_only'],
                   default='full')
    p.add_argument('--split-type', choices=['kfold', 'temperature'],
                   default='kfold',
                   help='分割类型：kfold（随机 CV）或 temperature（温度排除分割）')
    args = p.parse_args()

    cfg_key = args.task
    cfg = TASK_CONFIG[args.task]
    data_root = args.data_root or os.path.join(args.root, cfg['dir'],
                                               'processed_data')
    print(f'Device: {DEVICE}, task: {args.task}, conds: {cfg["conds"]}, '
          f'split: {args.split_type}')
    records = load_dataset(data_root, cfg)
    if len(records) == 0:
        print('!! 无有效样本，退出')
        return
    results = []
    for fold in args.folds:
        r, mae = run_fold(fold, records, data_root, cfg, args.epochs,
                          dim=args.dim, nhead=args.nhead, n_layers=args.layers,
                          dropout=args.dropout, lr=args.lr, wd=args.wd,
                          batch=args.batch, warmup=args.warmup,
                          seed=args.seed, variant=args.variant,
                          split_type=args.split_type)
        results.append((r, mae))
    rs = [x[0] for x in results]
    maes = [x[1] for x in results]
    print(f'\n===== {args.task} 5-fold 汇总 =====')
    for i, (r, mae) in enumerate(results):
        print(f'  Fold {i}: R={r:.4f}  MAE={mae:.4f}')
    print(f'  平均: R={np.mean(rs):.4f} ± {np.std(rs):.4f} | '
          f'MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}')


if __name__ == '__main__':
    main()
