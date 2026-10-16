# -*- coding: utf-8 -*-
"""train_task_permol.py — CheMixHub 官方 5-fold 随机 CV，v(im) 路线（31 维原子特征）

实验 1 — CheMixHub 8 tasks, official 5-fold（补材 S1 表 S2 的 in-distribution anchors）
路线 v(im)
产出 results/chemixhub_permol_results.csv；logs/chemixhub_5fold_permol/
配套 whole_mixture/train_task.py（v(ex) 臂）

与 whole-mixture 基线 train_task.py 严格可比：同数据、同 split（kfold_split_*.safetensors）、
同 31 维化学原子特征、同训练协议（bsz 128 / lr 1e-4 / warmup 3 / cosine / wd 1e-6）、
同评估协议（val Pearson R 选 best，报 test R / MAE）。模型超参与服务器在跑的
Hmix 外推 run_hmix_kext.py（v(im) = per-molecule）严格一致：
dim 256 / nhead 4 / TFM 2 层 ff 512 / GIN×3(64) / pad_from 32（31 维 + 常数 1 占位列）/
head Linear256→256→256→1（仅含条件列的任务首层输入多 n_cond 维）。

架构以逐分子路线正身（原 train_tgsc_permol_31dim.py 的 PerMolModel，该脚本已归档）为准，不采用
whole-mixture 的任何结构部件：
  whole-mixture：全体原子一张图 → TFM 跨分子自注意力 → 全体原子 masked-mean → MLP
  per-molecule ：每分子单独成序列 → TFM 仅分子内自注意力 → 分子内 masked-mean
                 → 分子向量 v_i → 按摩尔分数凸组合 z = Σ x_i·v_i → MLP（+图级条件 concat）
摩尔分数不作原子特征（保持 31 维），只作组合权重 —— 逐分子路线的定义。

用法:
    python train_task_permol.py --task he --root $ROOT
    python train_task_permol.py --task all --folds 0 1 2 3 4 --epochs 200
"""
import argparse
import csv
import os
import random

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
TOTAL_EPOCHS = 200
PATIENCE = 20
MIN_EPOCHS = 10
WARMUP_EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-4

CHEMIX_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'literature',
    'CheMixHub', 'datasets'))

TARGET_ELEMENTS = [6, 7, 8, 9, 16, 17, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]

# ── 任务配置（与 train_task.py 完全一致，保证 split 语义相同）──
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
        split='viscosity_splits', prop_col='property',
        smiles_col='smiles'),
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


class PerMolModel(nn.Module):
    """逐分子路线：分子内 GIN(+TFM) → 分子内 masked-mean → 凸组合 → MLP 回归。

    edge_index 仅含分子内共价键；TFM 在 [n_mol, max_atoms, dim] 序列上运行，
    只做分子内自注意力。z = Σ_i x_i · v_i（wmean，摩尔分数凸组合）。
    """

    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1,
                 variant='full', combine='wmean', n_cond=0):
        super().__init__()
        self.variant, self.combine, self.n_cond = variant, combine, n_cond
        # pad_from=32：31 维化学特征 + 常数 1 占位列（与 run_hmix_kext.py 的 Encoder 对齐）
        self.pad = nn.ZeroPad2d((0, 64 - 32, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        if variant != 'gin_only':
            enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                             dim_feedforward=2 * dim,
                                             batch_first=True, dropout=dropout)
            self.tfm = nn.TransformerEncoder(enc, n_layers)
        self.regressor = nn.Sequential(
            nn.Linear(dim + n_cond, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1))

    def _mol_vectors(self, h, batch, mol_id, amf):
        """每个分子单独成序列 → 分子级向量 v 与该分子的摩尔分数 w。"""
        M = int(mol_id.max().item()) + 1
        key = batch * M + mol_id
        order = torch.argsort(key, stable=True)
        h_s, key_s, amf_s = h[order], key[order], amf[order]
        uniq, inv = torch.unique_consecutive(key_s, return_inverse=True)
        dense, mask = to_dense_batch(h_s, inv)          # [n_mol, max_atoms, dim]
        if self.variant != 'gin_only':
            dense = self.tfm(dense, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        v = (dense * m).sum(1) / m.sum(1).clamp(min=1)  # 分子内 masked-mean
        amf_d, _ = to_dense_batch(amf_s, inv)           # [n_mol, max_atoms]
        w = (amf_d * mask.float()).sum(1) / mask.float().sum(1).clamp(min=1)
        return v, uniq // M, w

    def forward(self, x, ei, batch, mol_id, amf, cond=None):
        x = self.pad(x)
        if self.variant != 'tfm_only':
            for c in self.convs:
                x = c(x, ei)
        h = self.proj(x)
        v, sample_of, w_mol = self._mol_vectors(h, batch, mol_id, amf)
        S = int(batch.max().item()) + 1
        if self.combine == 'wmean':                     # 摩尔分数凸组合
            z = scatter(v * w_mol.unsqueeze(-1), sample_of, dim=0,
                        dim_size=S, reduce='sum')
        elif self.combine == 'mean':                    # 等权（对照）
            z = scatter(v, sample_of, dim=0, dim_size=S, reduce='mean')
        else:                                           # sum（对照）
            z = scatter(v, sample_of, dim=0, dim_size=S, reduce='sum')
        if cond is not None and self.n_cond > 0:
            z = torch.cat([z, cond], dim=-1)
        return self.regressor(z).squeeze(1)


class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'y_orig', 'cond'):
            return None
        return super().__cat_dim__(key, value, *a, **kw)


_gc = {}


def mol_to_graph(smi):
    """31 维原子特征：元素8+手性1+度6+氢5+杂化5+芳香1+电荷3+成环1（无摩尔分数）。"""
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
    """混合物 = 各分子独立子图拼接：31 维原子特征 + mol_id + 原子级摩尔分数 amf。

    edge_index 只含分子内共价键（逐分子路线定义）；amf 供组合权重用。
    """
    ax, aei, amid, aamf = [], [], [], []
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
        # 第 32 维常数 1 占位（信息上仍是 31 维化学特征；与 run_hmix_kext 的
        # build_mol_graphs mf_as_feature=False 完全一致）
        x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
        n = x.size(0)
        ax.append(x)
        aei.append(torch.tensor(g['ei']) + off)
        amid += [ci] * n
        aamf += [float(mol_fracs[ci])] * n
        off += n
    if off == 0:
        return None
    return GData(x=torch.cat(ax),
                 edge_index=torch.cat(aei, dim=1),
                 mol_id=torch.tensor(amid, dtype=torch.long),
                 amf=torch.tensor(aamf, dtype=torch.float32))


def load_compounds(data_root, smiles_col='smiles'):
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
            z = model(b.x, b.edge_index, b.batch, b.mol_id, b.amf, cond)
            preds.append((z.cpu() * y_std + y_mean).numpy())
            trues.append((b.y.cpu() * y_std + y_mean).numpy())
    p = np.concatenate(preds).flatten()
    t = np.concatenate(trues).flatten()
    return pearsonr(p, t)[0], np.mean(np.abs(p - t))


def run_fold(fold, records, data_root, cfg, epochs, patience, min_epochs,
             dim=256, nhead=4, n_layers=2, dropout=0.1, lr=LR, wd=1e-6,
             batch=BATCH_SIZE, warmup=WARMUP_EPOCHS, seed=42,
             variant='full', combine='wmean', ckpt_dir=None):
    import safetensors.torch as st
    split = st.load_file(os.path.join(data_root, cfg['split'],
                                      f'kfold_split_{fold}.safetensors'))
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
    m = PerMolModel(dim=dim, nhead=nhead, n_layers=n_layers, dropout=dropout,
                    variant=variant, combine=combine, n_cond=n_cond).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs - warmup)

    best_r, best_state, best_ep, bad = -float('inf'), None, -1, 0
    for ep in range(epochs):
        if ep < warmup:
            for pg in opt.param_groups:
                pg['lr'] = lr * (ep + 1) / warmup
        m.train()
        tl = 0
        for b in tr_loader:
            b = b.to(DEVICE)
            cond = b.cond if has_cond else None
            z = m(b.x, b.edge_index, b.batch, b.mol_id, b.amf, cond)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        if ep >= warmup:
            sch.step()
        vr, vmae = evaluate(m, va_loader, y_mean, y_std, has_cond)
        star = ''
        if vr > best_r:
            best_r, best_ep, bad, star = vr, ep, 0, ' *'
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            if ckpt_dir:
                torch.save(best_state, os.path.join(
                    ckpt_dir, f'permol_{combine}_{variant}_fold{fold}.pt'))
        else:
            bad += 1
        print(f'  ep {ep:3d} | train {tl / len(tr_loader):.4f} | '
              f'val R {vr:.4f} MAE {vmae:.4f} | best {best_r:.4f}@{best_ep}'
              f'{star}', flush=True)
        if ep >= min_epochs and bad >= patience:
            print(f'  early stop @ep{ep} (no val-R improvement for {patience})',
                  flush=True)
            break

    m.load_state_dict(best_state)
    t_r, t_mae = evaluate(m, te_loader, y_mean, y_std, has_cond)
    print(f'>> Fold {fold}: best ep {best_ep} | val R {best_r:.4f} | '
          f'test R {t_r:.4f} MAE {t_mae:.4f}')
    return dict(best_ep=best_ep, val_r=best_r, test_r=t_r, test_mae=t_mae)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True,
                   choices=list(TASK_CONFIG) + ['all'])
    p.add_argument('--epochs', type=int, default=TOTAL_EPOCHS)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--min-epochs', type=int, default=MIN_EPOCHS)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--root', default=CHEMIX_ROOT,
                   help='数据集根目录，内含 <dir>/processed_data')
    p.add_argument('--data-root', default=None,
                   help='覆盖单个任务的 processed_data 目录（--task 非 all 时有效）')
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
    p.add_argument('--combine', choices=['wmean', 'mean', 'sum'],
                   default='wmean')
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.abspath(os.path.join(base, '..', '..', 'checkpoints'))
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.abspath(os.path.join(base, '..', '..',
                                            'chemixhub_permol_results.csv'))

    tasks = list(TASK_CONFIG) if args.task == 'all' else [args.task]
    for task in tasks:
        cfg = TASK_CONFIG[task]
        data_root = (args.data_root if (args.data_root and task == args.task)
                     else os.path.join(args.root, cfg['dir'], 'processed_data'))
        print(f'\n########## task={task} | conds={cfg["conds"]} | '
              f'data={data_root} ##########', flush=True)
        records = load_dataset(data_root, cfg)
        if not records:
            print('!! 无有效样本，跳过')
            continue
        fold_rows = []
        for fold in args.folds:
            res = run_fold(fold, records, data_root, cfg, args.epochs,
                           args.patience, args.min_epochs,
                           dim=args.dim, nhead=args.nhead,
                           n_layers=args.layers, dropout=args.dropout,
                           lr=args.lr, wd=args.wd, batch=args.batch,
                           warmup=args.warmup, seed=args.seed,
                           variant=args.variant, combine=args.combine,
                           ckpt_dir=ckpt_dir)
            fold_rows.append(res)
            row = dict(task=task, fold=fold, seed=args.seed,
                       variant=args.variant, combine=args.combine,
                       dim=args.dim, epochs=args.epochs, patience=args.patience,
                       n_params=sum(q.numel() for q in
                                    PerMolModel(dim=args.dim,
                                                nhead=args.nhead,
                                                n_layers=args.layers,
                                                variant=args.variant,
                                                combine=args.combine,
                                                n_cond=len(cfg['conds'])
                                                ).parameters()),
                       **{k: round(v, 4) for k, v in res.items()})
            write_header = (not os.path.exists(csv_path)
                            or os.path.getsize(csv_path) == 0)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                wr = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    wr.writeheader()
                wr.writerow(row)
        rs = [r['test_r'] for r in fold_rows]
        maes = [r['test_mae'] for r in fold_rows]
        print(f'\n===== {task} 5-fold 汇总（permol/{args.combine}）=====')
        for i, r in enumerate(fold_rows):
            print(f"  Fold {args.folds[i]}: R={r['test_r']:.4f}  "
                  f"MAE={r['test_mae']:.4f}  (best ep {r['best_ep']})")
        print(f'  平均: R={np.mean(rs):.4f} ± {np.std(rs):.4f} | '
              f'MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}')
        print(f'结果已追加: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
