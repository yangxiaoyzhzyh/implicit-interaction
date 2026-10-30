# -*- coding: utf-8 -*-
"""train_all_permol.py —— 每个任务用「全部数据」训练 100 epoch，只为拿编码器

与 train_task_permol.py 的区别：
    - 不划分 train/val/test，全部样本参与训练
    - 无早停，固定 100 epoch
    - 保存**最后一个 epoch** 的权重（不是 val 最优）
    - 单 seed

目的：这些编码器只用于 CKA（表示几何诊断），不需要泛化性能，
      所以用全量数据能最大化训练信号的覆盖面。
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_task_permol import (TASK_CONFIG, load_dataset, PerMolModel,  # noqa: E402
                               LR, BATCH_SIZE, WARMUP_EPOCHS, DEVICE)


def train_full(task, root, data_root, epochs=100, dim=256, nhead=4,
               n_layers=2, dropout=0.1, lr=LR, wd=1e-6,
               batch=BATCH_SIZE, warmup=WARMUP_EPOCHS, seed=42,
               variant='full', combine='wmean', ckpt_dir=None):
    cfg = TASK_CONFIG[task]
    if data_root is None:
        data_root = os.path.join(root, cfg['dir'], 'processed_data')

    print('[%s] 加载 %s' % (task, data_root), flush=True)
    records = load_dataset(data_root, cfg)
    if not records:
        print('!! 无有效样本，跳过')
        return None
    print('[%s] 样本 %d（全部用于训练）' % (task, len(records)), flush=True)

    # 目标标准化（用全部数据的统计量）
    ys = torch.stack([d.y_orig for d in records])
    y_mean, y_std = ys.mean(), ys.std()
    for d in records:
        d.y = (d.y_orig - y_mean) / y_std

    # 条件列标准化
    has_cond = bool(cfg['conds'])
    if has_cond:
        C = torch.stack([d.cond for d in records])
        c_mean, c_std = C.mean(0), C.std(0).clamp(min=1e-8)
        for d in records:
            d.cond = (d.cond - c_mean) / c_std

    loader = DataLoader(records, batch_size=batch, shuffle=True,
                        num_workers=0)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    m = PerMolModel(dim=dim, nhead=nhead, n_layers=n_layers,
                    dropout=dropout, variant=variant, combine=combine,
                    n_cond=len(cfg['conds'])).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - warmup))

    t0 = time.time()
    for ep in range(epochs):
        if ep < warmup:
            for pg in opt.param_groups:
                pg['lr'] = lr * (ep + 1) / warmup
        m.train()
        tl = 0.0
        for b in loader:
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
        if ep % 10 == 0 or ep == epochs - 1:
            print('  ep %3d | train %.4f | %.0fs'
                  % (ep, tl / len(loader), time.time() - t0), flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    p = os.path.join(ckpt_dir, 'permol_%s_full_all.pt' % task)
    torch.save({'model': m.state_dict(),
                'task': task, 'y_mean': float(y_mean),
                'y_std': float(y_std), 'epochs': epochs,
                'variant': variant, 'combine': combine,
                'dim': dim, 'nhead': nhead, 'n_layers': n_layers},
               p)
    print('[%s] saved %s (%.1f MB, %.1f min)'
          % (task, p, os.path.getsize(p) / 1e6, (time.time() - t0) / 60),
          flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASK_CONFIG) + ['all'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--root', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = args.root or os.environ.get('CHEMIX_ROOT')
    if root is None:
        root = '/root/autodl-fs/chemixhub/datasets'
    out = args.out or os.path.join(root, '..', 'checkpoints_full')
    os.makedirs(out, exist_ok=True)

    tasks = list(TASK_CONFIG) if 'all' in args.tasks else args.tasks
    print('任务:', tasks)
    print('输出:', out)

    for t in tasks:
        if t not in TASK_CONFIG:
            print('跳过未知任务', t)
            continue
        print()
        print('=' * 70)
        train_full(t, root, None, epochs=args.epochs, seed=args.seed,
                   ckpt_dir=out)

    print()
    print('全部完成')


if __name__ == '__main__':
    main()
