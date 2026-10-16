"""vex_he_5fold.py — ΔHmix，5-fold（also defines the ± convention）

实验 1 — CheMixHub 官方 5-fold（补材 S1 表 S2 的 in-distribution anchors）
路线 v(ex)（该目录即 v(ex) 臂的 per-task 脚本）
产出 logs/chemixhub_5fold_whole/（v(ex) 5-fold 只落日志，无结果 CSV）
说明：文件名已按本仓库规范统一，脚本内容与原始 repro bundle 一致
CheMixHub Miscible Solvents 超额焓 (Enthalpy of mixing) 5-fold 随机 CV 训练。

与归档脚本 _unused/scripts/vw_legacy/vw_he_20seed.py 相同模型/训练配置（GIN×3(64) + TFM(256,2层,nhead4) + regressor，
40 epochs, LR 1e-4, warmup 3, cosine, batch 256），但使用 CheMixHub 官方的
5-fold 随机 CV 划分（kfold_split_*.safetensors），每折按 val Pearson R 选 best
epoch，评估 test Pearson R 与 MAE（kJ/mol），最后输出 5 折 mean ± std。

用法: python vex_he_5fold.py [--folds 0 1 2 3 4] [--data-root ...]
"""
import argparse
import csv
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
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CHEMIX_ROOT = os.path.join(PROJECT_ROOT, '..', 'literature', 'CheMixHub',
                           'datasets', 'miscible-solvent', 'processed_data')
PROCESSED_CSV = os.path.join(CHEMIX_ROOT, 'processed_MiscibleSolventData.csv')
COMPOUNDS_CSV = os.path.join(CHEMIX_ROOT, 'compounds.csv')
SPLITS_DIR = os.path.join(CHEMIX_ROOT, 'enthalpy_of_mixing_splits')

TOTAL_EPOCHS = 40
WARMUP_EPOCHS = 3
BATCH_SIZE = 256
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROPERTY = 'Enthalpy of mixing'
MAE_UNIT = {'Enthalpy of mixing': 'kJ/mol', 'Heat of vaporization': 'kcal/mol',
            'Density': 'g/m^3'}


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
    """v2 模型（与 train_he_seed_v2.py 一致）：原子特征 10 维（含父分子摩尔分数）。"""

    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 32, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 256)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4,
                                         dim_feedforward=512, batch_first=True,
                                         dropout=0.1)
        self.tfm = nn.TransformerEncoder(enc, 2)
        self.regressor = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs:
            x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        return self.regressor(x).squeeze(1)


class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'y_orig'):
            return None
        return super().__cat_dim__(key, value, *a, **kw)


def load_compounds():
    cmp_smiles = {}
    with open(COMPOUNDS_CSV) as f:
        for r in csv.DictReader(f):
            cmp_smiles[int(float(r['compound_id']))] = r['smiles']
    return cmp_smiles


_gc = {}
TARGET_ELEMENTS = [6, 7, 8, 9, 16, 17, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]


def build_graph(cmp_ids, mol_fracs, cmp_smiles):
    """v2 特征：9 维基础原子特征 + 父分子摩尔分数（第 10 维）。"""
    ax, aei = [], []
    off = 0
    for ci, cid in enumerate(cmp_ids):
        smi = cmp_smiles.get(int(float(cid)))
        if smi is None:
            return None
        if smi not in _gc:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            hyb = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
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
                f += [1.0 if a.GetHybridization() == y else 0.0 for y in hyb]
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
            _gc[smi] = {'x': np.array(nf, dtype=np.float32),
                        'ei': np.array(ei, dtype=np.int64).T}
        g = _gc[smi]
        x = torch.tensor(g['x']).float()[:, :31]
        mf_col = torch.full((x.size(0), 1), mol_fracs[ci], dtype=torch.float32)
        x = torch.cat([x, mf_col], dim=1)  # [N, 10]
        ax.append(x)
        aei.append(torch.tensor(g['ei']) + off)
        off += x.size(0)
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))


def load_dataset():
    df = pd.read_csv(PROCESSED_CSV)
    df = df[df['property'] == PROPERTY].reset_index(drop=True)
    print(f'{PROPERTY} 样本: {len(df):,}')
    cmp_smiles = load_compounds()
    records = []
    for _, r in df.iterrows():
        cmp_ids = eval(r['cmp_ids'])
        mol_fracs = eval(r['cmp_mole_fractions'])
        d = build_graph(cmp_ids, mol_fracs, cmp_smiles)
        if d is None:
            continue
        d.y = torch.tensor(r['value'], dtype=torch.float32)
        d.y_orig = d.y.clone()
        records.append(d)
    print(f'有效样本: {len(records):,}')
    return records


def evaluate(model, loader, y_mean, y_std):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            z = model(b.x, b.edge_index, b.batch)
            preds.append((z.cpu() * y_std + y_mean).numpy())
            trues.append((b.y.cpu() * y_std + y_mean).numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    return pearsonr(p, t)[0], np.mean(np.abs(p - t))


def run_fold(fold, records):
    import safetensors.torch as st
    split = st.load_file(
        os.path.join(SPLITS_DIR, f'kfold_split_{fold}.safetensors'))
    tr_idx, va_idx, te_idx = (split[k].numpy() for k in
                              ('train_indices', 'val_indices', 'test_indices'))
    print(f'\n===== Fold {fold}: train {len(tr_idx):,} / val {len(va_idx):,} '
          f'/ test {len(te_idx):,} =====')

    train_data = [records[i] for i in tr_idx]
    val_data = [records[i] for i in va_idx]
    test_data = [records[i] for i in te_idx]

    # 跨 fold 复用对象：恢复原始标签、清理上次遗留的 nraw
    for d in train_data + val_data + test_data:
        d.y = d.y_orig.clone()
        if hasattr(d, 'nraw'):
            del d.nraw

    all_y = torch.stack([d.y for d in train_data])
    y_mean, y_std = all_y.mean(), all_y.std()
    for d in train_data:
        d.y = (d.y - y_mean) / y_std
    for d in val_data:
        d.nraw = d.y.clone()
        d.y = (d.y - y_mean) / y_std
    for d in test_data:
        d.nraw = d.y.clone()
        d.y = (d.y - y_mean) / y_std

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False,
                             num_workers=0)

    torch.manual_seed(42 + fold)
    random.seed(42 + fold)
    np.random.seed(42 + fold)
    m = Model().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS)

    best_val_r, best_state, best_ep = -float('inf'), None, -1
    for ep in range(TOTAL_EPOCHS):
        if ep < WARMUP_EPOCHS:
            for pg in opt.param_groups:
                pg['lr'] = LR * (ep + 1) / WARMUP_EPOCHS
        m.train()
        tl = 0
        for b in train_loader:
            b = b.to(DEVICE)
            z = m(b.x, b.edge_index, b.batch)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        if ep >= WARMUP_EPOCHS:
            sch.step()
        val_r, val_mae = evaluate(m, val_loader, y_mean, y_std)
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            best_ep = ep
        print(f'  ep {ep:2d} | train {tl/len(train_loader):.4f} | '
              f'val R {val_r:.4f} MAE {val_mae:.4f}', flush=True)

    m.load_state_dict(best_state)
    test_r, test_mae = evaluate(m, test_loader, y_mean, y_std)
    unit = MAE_UNIT.get(PROPERTY, '')
    print(f'>> Fold {fold}: best ep {best_ep} (val R {best_val_r:.4f}) | '
          f'test R {test_r:.4f} MAE {test_mae:.4f} {unit}')
    return test_r, test_mae


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--torch-threads', type=int, default=0)
    p.add_argument('--property', default='Enthalpy of mixing',
                   help='Miscible Solvents 任务: Enthalpy of mixing / Heat of vaporization / Density')
    p.add_argument('--data-root', default=None,
                   help='CheMixHub processed_data 根目录（服务器上使用）')
    args = p.parse_args()
    global PROCESSED_CSV, COMPOUNDS_CSV, SPLITS_DIR, PROPERTY
    PROPERTY = args.property
    if args.data_root:
        PROCESSED_CSV = os.path.join(args.data_root,
                                     'processed_MiscibleSolventData.csv')
        COMPOUNDS_CSV = os.path.join(args.data_root, 'compounds.csv')
        split_dir = {'Enthalpy of mixing': 'enthalpy_of_mixing_splits',
                     'Heat of vaporization': 'heat_of_vaporization_splits',
                     'Density': 'density_splits'}[args.property]
        SPLITS_DIR = os.path.join(args.data_root, split_dir)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    print(f'Device: CPU, torch {torch.__version__}')
    records = load_dataset()
    results = []
    for fold in args.folds:
        r, mae = run_fold(fold, records)
        results.append((r, mae))
        print(f'[进度] {len(results)}/{len(args.folds)} 折完成')

    rs = [x[0] for x in results]
    maes = [x[1] for x in results]
    print('\n===== 5-fold CV 汇总 =====')
    for i, (r, mae) in enumerate(results):
        print(f'  Fold {i}: R={r:.4f}  MAE={mae:.4f} kJ/mol')
    print(f'  平均: R={np.mean(rs):.4f} ± {np.std(rs):.4f} | '
          f'MAE={np.mean(maes):.4f} ± {np.std(maes):.4f} kJ/mol')


if __name__ == '__main__':
    main()
