# -*- coding: utf-8 -*-
"""mlp_small_data.py —— 小样本 MLP 读出：塑形编码器是否更早发挥优势

设计：
    对每个任务，从官方 train 里抽 1% / 5% / 10% / 100% 的样本
    训练一个小 MLP（PCA 64 维 → hidden 128 → 1）
    在官方 test 上评估

    特征对照（**不含本任务自己的编码器**）：
        others   其余 7 个编码器 concat（1792 → PCA 64）
        ECFP4    通用基线（2048 → PCA 64）
        RDKit2D / MACCS  另两个描述符基线

    为什么排除 own：own 编码器是在该任务全量数据（含测试分子）上训练的，
    用 1%/5% 标签去接它会把测试信息带进特征，属于泄漏。论文 S19 的对照列也
    只有 “seven others” 与描述符，没有 own。

判据：
    小样本（1%/5%）下 others 相对 ECFP4 的优势是否比全量时更大
    → 若是，说明塑形编码器的信息「更容易被少量数据激活」
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from leave_one_task_out import (DS, TASKS, load_task,  # noqa: E402
                                mix_vec)
try:  # 归档自包含：fp_matrix.py 与原 eval_encoders 中的实现逐字一致
    from fp_matrix import fp_matrix  # noqa: E402
except ImportError:
    from eval_encoders import fp_matrix  # noqa: E402

CACHE = os.path.join(HERE, 'out', 'encoder_cache.npz')
# 只跑 1%/5%/10%：100% 就是全量训练，分子层面存在泄漏，且结论无增量
RATIOS = [0.01, 0.05]      # 论文只报 1% 与 5%（10% 不进表，省算力）
PCA_DIM = 64


class Head(nn.Module):
    def __init__(self, d, h=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xva, yva, Xte, epochs=300, seed=0, lr=3e-3,
            wd=1e-2):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    x = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    n = len(y)
    bs = min(64, max(8, n // 4))
    best, best_state, bad = 1e18, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            if len(j) < 2:
                continue
            opt.zero_grad()
            loss = F.mse_loss(m(x[j]), y[j])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(F.mse_loss(m(xv), yv))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return m(xe).numpy()


def run_ratio(X, y, tr, va, te, ratio, dim=PCA_DIM, n_rep=2):
    """在一个数据比例下训练 MLP，返回 R 的均值/标准差"""
    rs = []
    for rep in range(n_rep):
        rng = np.random.RandomState(1000 + rep)
        if tr.size:
            k = max(4, int(len(tr) * ratio))
            sub = rng.choice(tr, size=min(k, len(tr)), replace=False)
        else:
            continue
        sc = StandardScaler().fit(X[sub])
        d = min(dim, X.shape[1], len(sub) - 1)
        if d < 2:
            continue
        pca = PCA(n_components=d, random_state=0).fit(sc.transform(X[sub]))
        Zs = pca.transform(sc.transform(X[sub]))
        Zv = pca.transform(sc.transform(X[va]))
        Zt = pca.transform(sc.transform(X[te]))
        ymu, ysd = y[sub].mean(), y[sub].std() + 1e-9
        pred = fit_mlp(Zs, (y[sub] - ymu) / ysd,
                       Zv, (y[va] - ymu) / ysd,
                       Zt, seed=rep)
        pred = pred * ysd + ymu
        rs.append(np.corrcoef(pred, y[te])[0, 1])
    return (float(np.mean(rs)), float(np.std(rs))) if rs else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--out', default='out/mlp_small_data.json')
    ap.add_argument('--with-linear', action='store_true',
                    help='同时算线性基线（用于对比）')
    args = ap.parse_args()

    if not os.path.exists(CACHE):
        print('!! 缺缓存，请先跑 cache_encoders.py')
        return
    cache = np.load(CACHE, allow_pickle=False)
    smi_all = [str(s) for s in cache['smiles']]
    sidx = {s: i for i, s in enumerate(smi_all)}
    ENC = {k.replace('_vec', ''): cache[k] for k in cache.files
           if k.endswith('_vec')}
    print('缓存：分子 %d | 编码器 %d 个' % (len(smi_all), len(ENC)))

    # 通用基线（todo 3 一并做）
    GEN = {}
    for k in ['ecfp4', 'rdkit2d', 'maccs']:
        try:
            GEN[k] = fp_matrix(smi_all, k)
        except Exception as e:
            print('  跳过 %s (%s)' % (k, type(e).__name__))

    print()
    print('=== MLP 小样本（PCA %d，2 次重复）===' % PCA_DIM)
    hdr = '%-11s %5s' % ('目标', '比例')
    for lab in ['others', 'ECFP4', 'RDKit2D', 'MACCS']:
        hdr += ' %9s' % lab
    print(hdr)
    print('-' * len(hdr))

    RES = {}
    for t in args.tasks:
        d = load_task(t)
        cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
        split_dir = os.path.join(d['base'], d['split'])
        splits = []
        for fi in range(5):
            p = os.path.join(split_dir, 'kfold_split_%d.safetensors' % fi)
            if not os.path.exists(p):
                continue
            from safetensors.numpy import load_file
            splits.append(load_file(p))
        if not splits:
            print('%-11s 无划分，跳过' % t)
            continue

        # 用一个折（fold 0）做基准，避免任务间时间爆炸
        d0 = splits[0]
        tr, va, te = d0['train_indices'], d0['val_indices'], \
            d0['test_indices']

        # 只用其余 7 个任务的编码器。本任务自己的编码器不能进特征：它是在该任务
        # 全量数据（含测试分子）上训练的，用 1%/5% 标签去接它等于泄漏测试信息。
        others = [k for k in sorted(ENC) if k != t]
        Voth = np.concatenate(
            [mix_vec(ENC[k], cid2i, d['ids'], d['ws']) for k in others],
            axis=1)
        FX = {'others': Voth}
        for k, lab in [('ecfp4', 'ECFP4'), ('rdkit2d', 'RDKit2D'),
                       ('maccs', 'MACCS')]:
            if k in GEN:
                FX[lab] = mix_vec(GEN[k], cid2i, d['ids'], d['ws'])

        if d['cond'] is not None:
            for k in FX:
                FX[k] = np.concatenate([FX[k], d['cond']], axis=1)

        RES[t] = {}
        for ratio in RATIOS:
            row = {}
            line = '%-11s %5.0f%%' % (t, ratio * 100)
            for lab in ['others', 'ECFP4', 'RDKit2D', 'MACCS']:
                if lab not in FX:
                    line += ' %9s' % '-'
                    continue
                r, sd = run_ratio(FX[lab], d['y'], tr, va, te, ratio)
                row[lab] = dict(R=r, sd=sd)
                line += ' %9.3f' % r
            print(line, flush=True)
            RES[t]['%.2f' % ratio] = row
        print()

    json.dump(RES, open(os.path.join(HERE, args.out), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
