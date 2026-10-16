"""vex_mon_5fold.py — MON，5-fold 及变体（full / gin_only / tfm_only）

实验 1 — CheMixHub 官方 5-fold（补材 S1 表 S2 的 in-distribution anchors）
路线 v(ex)（该目录即 v(ex) 臂的 per-task 脚本）
产出 logs/chemixhub_5fold_whole/（v(ex) 5-fold 只落日志，无结果 CSV）
说明：文件名已按本仓库规范统一，脚本内容与原始 repro bundle 一致
MON（Motor Octane Number）辛烷值预测，CheMixHub Fuel mixtures 任务。
模型 = 超额焓 v2 同款：原子 10 维特征（9 维化学 + 父分子摩尔分数）
       → GIN×3(64) → TFM×2(256, nhead4) → MLP → MON 回归。
评估 = CheMixHub 官方 5-fold 随机 CV（kfold_split_*.safetensors），
       val 选 best（Pearson R），报 test R / MAE（与 CheMixHub Table 2/10 对比）。

用法: python vex_mon_5fold.py [--data-root ...] [--epochs 60] [--folds 0 1 2 3 4]
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

TARGET_ELEMENTS = [6, 7, 8, 9, 16, 17, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]


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
    """v2 同款：10 维原子特征（含父分子摩尔分数）。"""

    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1,
                 variant='full'):
        super().__init__()
        self.variant = variant
        self.pad = nn.ZeroPad2d((0, 64 - 32, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        if variant != 'gin_only':
            enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                             dim_feedforward=2 * dim,
                                             batch_first=True, dropout=dropout)
            self.tfm = nn.TransformerEncoder(enc, n_layers)
        self.regressor = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1))

    def forward(self, x, ei, batch):
        x = self.pad(x)
        if self.variant != 'tfm_only':
            for c in self.convs:
                x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        if self.variant != 'gin_only':
            x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        return self.regressor(x).squeeze(1)


class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'y_orig'):
            return None
        return super().__cat_dim__(key, value, *a, **kw)


_gc = {}


def mol_to_graph(smi):
    """31 维增强原子特征：元素8+手性1+度6+氢5+杂化5+芳香1+电荷3+成环1（+摩尔分数=32）。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
               Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
               Chem.HybridizationType.SP3D2]
    nf = []
    for a in mol.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in TARGET_ELEMENTS]
        f += [1 if a.GetChiralTag() == x else 0 for x in CHIRAL_TAGS]  # 2 位
        deg = min(a.GetDegree(), 5)
        f += [1.0 if deg == i else 0.0 for i in range(6)]              # 度 6 位
        h = min(a.GetTotalNumHs(), 4)
        f += [1.0 if h == i else 0.0 for i in range(5)]                # 氢 5 位
        f += [1.0 if a.GetHybridization() == hyb else 0.0
              for hyb in HYBRIDS]                                      # 杂化 5 位
        f += [1.0 if a.GetIsAromatic() else 0.0]                       # 芳香 1 位
        fc = int(a.GetFormalCharge())
        f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0,
              1.0 if fc > 0 else 0.0]                                  # 电荷 3 位
        f += [1.0 if a.IsInRing() else 0.0]                            # 成环 1 位
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
    """混合物图：各组分原子 + 摩尔分数第 10 维（v2 特征）。"""
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


def load_dataset(data_root):
    df = pd.read_csv(os.path.join(data_root, 'processed_MON.csv'))
    print(f'MON 样本: {len(df):,}')
    cmp_smiles = {}
    with open(os.path.join(data_root, 'compounds.csv')) as f:
        import csv
        for r in csv.DictReader(f):
            cmp_smiles[int(float(r['compound_id']))] = r['smiles']
    records = []
    for _, r in df.iterrows():
        cmp_ids = eval(r['cmp_ids'])
        mfs = eval(r['cmp_mole_fractions'])
        d = build_sample(cmp_ids, mfs, cmp_smiles)
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
    p = np.concatenate(preds).flatten()
    t = np.concatenate(trues).flatten()
    return pearsonr(p, t)[0], np.mean(np.abs(p - t))


def run_fold(fold, records, data_root, epochs, dim=256, nhead=4,
            n_layers=2, dropout=0.1, lr=LR, wd=1e-6, batch=BATCH_SIZE,
            warmup=WARMUP_EPOCHS, seed=42, variant='full'):
    import safetensors.torch as st
    split = st.load_file(os.path.join(data_root, 'motor_octane_number_splits',
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
    for d in train_data:
        d.y = (d.y - y_mean) / y_std
    for d in val_data:
        d.y = (d.y - y_mean) / y_std
    for d in test_data:
        d.y = (d.y - y_mean) / y_std

    tr_loader = DataLoader(train_data, batch_size=batch, shuffle=True,
                           num_workers=0)
    va_loader = DataLoader(val_data, batch_size=256, shuffle=False,
                           num_workers=0)
    te_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                           num_workers=0)

    torch.manual_seed(seed + fold)
    random.seed(seed + fold)
    np.random.seed(seed + fold)
    m = Model(dim=dim, nhead=nhead, n_layers=n_layers, dropout=dropout,
              variant=variant).to(DEVICE)
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
            z = m(b.x, b.edge_index, b.batch)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        if ep >= warmup:
            sch.step()
        vr, vmae = evaluate(m, va_loader, y_mean, y_std)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            best_ep = ep
            torch.save(best_state, f'model_mon_{variant}_fold{fold}.pt')
        print(f'  ep {ep:2d} | train {tl/len(tr_loader):.4f} | '
              f'val R {vr:.4f} MAE {vmae:.4f} | best {best_r:.4f}@{best_ep}',
              flush=True)

    m.load_state_dict(best_state)
    tr_, tmae = evaluate(m, te_loader, y_mean, y_std)
    print(f'>> Fold {fold}: best ep {best_ep} | test R {tr_:.4f} '
          f'MAE {tmae:.4f}')
    return tr_, tmae

def run_extrap(records, data_root, epochs, lo=30, hi=50, batch=BATCH_SIZE,
               lr=LR, warmup=WARMUP_EPOCHS, wd=1e-6, dim=256, nhead=4,
               n_layers=2, dropout=0.1, variant='full'):
    """外推：train 组分 < lo → val [lo, hi] → test > hi。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(data_root, 'processed_MON.csv'))
    n = df['cmp_ids'].map(lambda s: len(eval(s))).values
    tr_idx = [i for i in range(len(records)) if n[i] < lo]
    va_idx = [i for i in range(len(records)) if lo <= n[i] <= hi]
    te_idx = [i for i in range(len(records)) if n[i] > hi]
    rng = random.Random(0)
    rng.shuffle(va_idx)
    print(f'extrap: train({len(tr_idx)}, <{lo} 组分) / val({len(va_idx)}, {lo}-{hi} 组分) / test({len(te_idx)}, >{hi} 组分)')

    train_data = [records[i] for i in tr_idx]
    val_data = [records[i] for i in va_idx]
    test_data = [records[i] for i in te_idx]
    for d in train_data + val_data + test_data:
        d.y = d.y_orig.clone()
    all_y = torch.stack([d.y for d in train_data])
    y_mean, y_std = all_y.mean(), all_y.std()
    for d in train_data + val_data + test_data:
        d.y = (d.y - y_mean) / y_std

    tr_loader = DataLoader(train_data, batch_size=batch, shuffle=True,
                           num_workers=0)
    va_loader = DataLoader(val_data, batch_size=256, shuffle=False,
                           num_workers=0)
    te_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                           num_workers=0)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    m = Model(dim=dim, nhead=nhead, n_layers=n_layers, dropout=dropout,
              variant=variant).to(DEVICE)
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
            z = m(b.x, b.edge_index, b.batch)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        if ep >= warmup:
            sch.step()
        vr, vmae = evaluate(m, va_loader, y_mean, y_std)
        if vr > best_r:
            best_r = vr
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            best_ep = ep
            torch.save(best_state, f'model_mon_{variant}_extrap.pt')
        print(f'  ep {ep:2d} | train {tl/len(tr_loader):.4f} | '
              f'val R {vr:.4f} MAE {vmae:.4f} | best {best_r:.4f}@{best_ep}',
              flush=True)
    m.load_state_dict(best_state)
    tr_, tmae = evaluate(m, te_loader, y_mean, y_std)
    print(f'>> MON extrap: best ep {best_ep} | test R {tr_:.4f} '
          f'MAE {tmae:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--epochs', type=int, default=TOTAL_EPOCHS)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--mode', choices=['cv', 'extrap'], default='cv')
    p.add_argument('--extrap-lo', type=int, default=30,
                   help='外推: train 组分 < lo; val [lo, hi]; test > hi')
    p.add_argument('--extrap-hi', type=int, default=50)
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
    args = p.parse_args()

    print(f'Device: {DEVICE}')
    records = load_dataset(args.data_root)
    if args.mode == 'extrap':
        run_extrap(records, args.data_root, args.epochs,
                   lo=args.extrap_lo, hi=args.extrap_hi, variant=args.variant)
        return
    results = []
    for fold in args.folds:
        r, mae = run_fold(fold, records, args.data_root, args.epochs,
                          dim=args.dim, nhead=args.nhead, n_layers=args.layers,
                          dropout=args.dropout, lr=args.lr, wd=args.wd,
                          batch=args.batch, warmup=args.warmup,
                          seed=args.seed, variant=args.variant)
        results.append((r, mae))
    rs = [x[0] for x in results]
    maes = [x[1] for x in results]
    print('\n===== MON 5-fold 汇总 =====')
    for i, (r, mae) in enumerate(results):
        print(f'  Fold {i}: R={r:.4f}  MAE={mae:.4f}')
    print(f'  平均: R={np.mean(rs):.4f} ± {np.std(rs):.4f} | '
          f'MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}')


if __name__ == '__main__':
    main()
