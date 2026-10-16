# -*- coding: utf-8 -*-
"""run_perk_thermo.py — ΔHmix / ΔHvap 的 per-composition-count profile

实验 3 — Per-composition-count profile（补材 S2，表 S4）
路线 两条（内部标签 e1 = v(ex)、e2 = v(im)）
产出 results/perk_hmix_e{1,2}.csv、perk_hvap_e{1,2}.csv；logs/extrapolation/perk_*.log
表 S4 的汇总由 analysis/summarize_perk.py 生成

复用 run_hmix_kext 的全部组件（模型、数据、前向、评估），只改一件事：
训练完成后对**多个 K** 各评一次，而不是只评 test_k。

协议与 results/h{mix,vap}_e{1,2}_traink2 严格一致：
  train K<=2 | val K=4（val patience 早停 + best-val 回载）
  评估: K=3, K=4, K=5
  （K=4 同时是验证集；K=3、K=5 为纯测试读数）

输出: results/perk_thermo.csv
  列: task, route, seed, k, n, R, MAE

用法:
  python run_perk_thermo.py --prop "Enthalpy of mixing" --route e1 --seeds 42 43 44
  python run_perk_thermo.py --smoke        # 小规模冒烟
"""
import argparse
import collections
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER_V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import run_hmix_kext as R            # noqa: E402
from run_hmix_kext import (Encoder, batch_e1, batch_e2, build_mol_graphs,   # noqa: E402
                          load_records, predict)


def train_once(tr, va, mol_graphs, route, seed, args, y_mean, y_std):
    """与 run_hmix_kext.main 同一训练循环，返回已回载 best-val 的 (enc, reg)。"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    enc = Encoder(dim=args.dim, nhead=args.nhead, n_layers=args.layers).to(R.DEVICE)
    reg = nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(args.dim, args.dim), nn.ReLU(),
                        nn.Linear(args.dim, 1)).to(R.DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(reg.parameters()),
                           lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - args.warmup))
    fw = batch_e2 if route == 'e2' else batch_e1

    best_r, best_sd, best_ep = -9, None, -1
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
                              dtype=torch.float32, device=R.DEVICE)
            loss = F.mse_loss(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if ep >= args.warmup:
            sch.step()
        vr, _ = predict(va, mol_graphs, enc, reg, route, y_mean, y_std)
        if vr > best_r:
            best_r, best_ep = vr, ep
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in list(enc.state_dict().items()) +
                       [('reg.' + k, v) for k, v in reg.state_dict().items()]}
            bad = 0
        else:
            bad += 1
        if args.early_stop == 'val' and ep >= args.min_epochs and bad >= args.patience:
            print(f'  >> val 连续 {bad} epoch 未提升 → 停于 ep{ep}，'
                  f'回载 best-val ep{best_ep}', flush=True)
            break
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f'  [{route} seed {seed}] ep {ep:3d} | loss {tot / max(nb, 1):.4f} '
                  f'| val R {vr:.4f} | best {best_r:.4f}@{best_ep} | no-improve {bad} '
                  f'| {time.time() - t0:.0f}s', flush=True)

    if best_sd is not None:
        enc.load_state_dict({k: v for k, v in best_sd.items()
                             if not k.startswith('reg.')})
        reg.load_state_dict({k[4:]: v for k, v in best_sd.items()
                             if k.startswith('reg.')})
    return enc, reg, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prop', default='Enthalpy of mixing')
    ap.add_argument('--data-dir',
                    default=os.path.abspath(os.path.join(
                        HERE, '..', '..', '..', '..', '..', 'literature', 'CheMixHub',
                        'datasets', 'miscible-solvent', 'processed_data')))
    ap.add_argument('--csv', default='processed_MiscibleSolventData.csv')
    ap.add_argument('--route', default='e2', choices=['e2', 'e1'])
    ap.add_argument('--train-max-k', type=int, default=2)
    ap.add_argument('--val-k', type=int, default=4)
    ap.add_argument('--eval-ks', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--early-stop', default='val', choices=['none', 'val'])
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--min-epochs', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out-dir', default=os.path.join(_PAPER_V2, 'results'))
    ap.add_argument('--out-csv', default='perk_thermo.csv')
    ap.add_argument('--smoke', action='store_true', help='小规模冒烟（等效 --limit 240 --epochs 2）')
    args = ap.parse_args()
    if args.smoke:
        args.limit, args.epochs, args.seeds = 240, 2, [42]

    R.PROP = args.prop
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.data_dir, args.csv)
    recs, cmp_smiles = load_records(csv_path, os.path.join(args.data_dir, 'compounds.csv'))
    byk = collections.defaultdict(list)
    for r in recs:
        byk[len(r['cids'])].append(r)

    tr = [r for k, v in byk.items() if k <= args.train_max_k for r in v]
    va = byk.get(args.val_k, [])
    if args.limit > 0:
        tr = tr[:args.limit]
        va = va[:max(1, args.limit // 4)]
        byk = {k: v[:args.limit] for k, v in byk.items()}

    task = 'dHvap' if 'vapor' in args.prop else 'dHmix'
    print(f'[task] {task} ({args.prop}) | route {args.route} | '
          f'train K<={args.train_max_k} {len(tr)} | val K={args.val_k} {len(va)}',
          flush=True)

    y_tr = np.array([r['y'] for r in tr])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    all_cids = sorted({c for r in tr + va for c in r['cids']} |
                      {c for k in args.eval_ks for r in byk.get(k, []) for c in r['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles, False)

    rows = []
    for seed in args.seeds:
        enc, reg, best_ep = train_once(tr, va, mol_graphs, args.route, seed,
                                       args, y_mean, y_std)
        line = []
        for k in args.eval_ks:
            ss = byk.get(k, [])
            if not ss:
                continue
            r_, mae_ = predict(ss, mol_graphs, enc, reg, args.route, y_mean, y_std)
            rows.append(dict(task=task, route=args.route, seed=seed, k=k,
                             n=len(ss), R=f'{r_:.4f}', MAE=f'{mae_:.4f}'))
            line.append(f'K={k}: R {r_:.4f} MAE {mae_:.4f} (n={len(ss)})')
        print(f'>> [{task}/{args.route}] seed {seed} best ep{best_ep} | ' + ' | '.join(line),
              flush=True)

    out = os.path.join(args.out_dir, args.out_csv)
    write_header = not os.path.exists(out)
    with open(out, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['task', 'route', 'seed', 'k', 'n', 'R', 'MAE'])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f'append {len(rows)} rows -> {out}', flush=True)


if __name__ == '__main__':
    main()
