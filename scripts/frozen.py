# -*- coding: utf-8 -*-
"""frozen.py —— 冻结编码器臂（RDKit 2D 217 维描述符）

与 vim.py 完全相同的聚合 z = Σ w_i v_i、相同的 MLP 读出、相同的训练协议
（固定 200 epoch、batch 128、无验证集、不早停），只把 v_i 换成冻结描述符。
它是“增益来自表示还是来自聚合形式”这一对照的第三臂。

与 engine.py 对齐的两处细节：
  1. npz 里的 val_idx 并回训练集（组分数轴不用验证集）；
  2. batch 128、固定 200 epoch、不早停、不回载。

任务与划分都由参数选，读同一份 npz。

    python frozen.py --task spe --split-file ../splits/spe_composition.npz \
                     --data-base D --out ../01_composition/runs
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr

RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))

from spec import load_records, spec_of  # noqa: E402

CACHE = os.path.join(HERE, '_desc_cache.json')
NDESC = len(Descriptors.descList)
SEEDS = [42, 43, 44]


def descriptors(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    out = []
    for _, fn in Descriptors.descList:
        try:
            v = float(fn(m))
            out.append(v if np.isfinite(v) else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def build(task, data_base, cache):
    import ast
    tk = spec_of(task)
    df, rows, comps = load_records(task, data_base)
    d = os.path.join(data_base, tk['dir'], 'processed_data')
    comp = pd.read_csv(os.path.join(d, 'compounds.csv'))
    smi = {int(r['compound_id']): r.get('smiles') for _, r in comp.iterrows()}

    need = sorted({c for x in comps for c in x})
    for cid in need:
        if str(cid) not in cache:
            cache[str(cid)] = descriptors(smi.get(cid))
    V = {c: np.array(cache[str(c)], dtype=np.float32)
         for c in need if cache[str(c)] is not None}

    ys = np.array([float(df.loc[r, 'value']) for r in rows])
    ok = np.array([all(c in V for c in x) for x in comps])

    mfs = []
    for i in range(len(rows)):
        row = df.loc[rows[i]]
        k = len(comps[i])
        if 'cmp_mole_fractions' in df.columns:
            w = [float(v) for v in ast.literal_eval(row['cmp_mole_fractions'])]
        else:
            w = [1.0 / k] * k
        w = (w[:k] + [0.0] * max(0, k - len(w)))
        tot = sum(w)
        mfs.append([x / tot for x in w] if tot > 0 else [1.0 / k] * k)

    Z = np.zeros((len(rows), NDESC), dtype=np.float32)
    for i, x in enumerate(comps):
        if not ok[i]:
            continue
        for c, w in zip(x, mfs[i]):
            Z[i] += w * V[c]

    conds = np.array([[float(df.loc[r].get(c, np.nan)) for c in tk['conds']]
                      for r in rows], dtype=np.float64).reshape(len(rows), -1)
    return Z, ys, conds, ok


def fit(Z, y, tr, te, seed, epochs=200, batch=128):
    """与 engine 同协议：固定 epoch、无验证集、不早停、余弦退火、前 3 epoch warmup。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ym, ysd = float(y[tr].mean()), float(y[tr].std())
    Xtr = torch.tensor(Z[tr])
    Ytr = torch.tensor((y[tr] - ym) / ysd, dtype=torch.float32)
    Xte = torch.tensor(Z[te])

    net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 3))
    lossf = nn.MSELoss()
    n = len(tr)
    for ep in range(epochs):
        if ep < 3:
            for g in opt.param_groups:
                g['lr'] = 1e-4 * (ep + 1) / 3
        net.train()
        perm = np.random.RandomState(ep).permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]).squeeze(1), Ytr[idx])
            loss.backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        p = net(Xte).squeeze(1).numpy() * ysd + ym
    return float(pearsonr(p, y[te])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = ap.parse_args()

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding='utf-8') as f:
            cache = json.load(f)

    Z, ys, conds, ok = build(args.task, args.data_base, cache)
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    sp = np.load(args.split_file, allow_pickle=True)
    # npz 存的是 df 行标签 → 换算成记录位置后再用 ok 过滤
    _, rows, _ = load_records(args.task, args.data_base)
    pos = {int(r): i for i, r in enumerate(rows)}

    def to_pos(key):
        if key not in sp:
            return np.array([], dtype=int)
        return np.array([pos[int(x)] for x in sp[key]
                         if int(x) in pos and ok[pos[int(x)]]])

    # 与 engine.py 在 --no-early-stop 下的行为一致：npz 里的 val 并回训练集。
    # 组分数轴不使用验证集，这些样本同样满足 K<=train_max，属于合法训练数据。
    tr = np.concatenate([to_pos('train_idx'), to_pos('val_idx')])
    te = to_pos('test_idx')
    if len(te) == 0:
        raise SystemExit('测试集为空（该任务在此划分下不可评估）')

    if conds.shape[1]:
        cm = np.nanmean(conds[tr], 0)
        cs = np.nanstd(conds[tr], 0)
        cs[cs == 0] = 1.0
        Z = np.hstack([Z, ((np.where(np.isnan(conds), cm, conds) - cm) / cs)
                       .astype(np.float32)])

    mu, sd = Z[tr].mean(0), Z[tr].std(0)
    sd[sd == 0] = 1.0
    Z = np.nan_to_num((Z - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    rs = [fit(Z, ys, tr, te, s) for s in args.seeds]
    m, s_ = float(np.mean(rs)), float(np.std(rs))
    print('%s | %s | frozen | train %d / test %d | R = %.4f ± %.4f  (%s)'
          % (args.task, os.path.basename(args.split_file), len(tr), len(te),
             m, s_, '; '.join('%.4f' % x for x in rs)))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(args.split_file).replace('.npz', '')
        dst = os.path.join(args.out, 'frozen_%s.csv' % name)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('task,split,train_n,test_n,mean,sd,per_seed,n_desc\n')
            f.write('%s,%s,%d,%d,%.4f,%.4f,%s,%d\n'
                    % (args.task, sp['split'] if 'split' in sp else '',
                       len(tr), len(te), m, s_,
                       ';'.join('%.4f' % x for x in rs), NDESC))
        print('写出', dst)


if __name__ == '__main__':
    main()
