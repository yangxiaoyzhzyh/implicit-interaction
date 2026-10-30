# -*- coding: utf-8 -*-
"""cache_encoders.py —— 把 8 个编码器在探针分子上的向量缓存到磁盘

缓存内容：
    out/encoder_cache.npz
        smiles      3540 个 SMILES（顺序固定）
        <task>_vec  (3540, 256) 该任务编码器的分子向量
        ecfp4       (3540, 2048) ECFP4 指纹

用途：leave_one_task_out 与后续 MLP 实验都读这个缓存，
      避免每次重新跑 GNN 前向（每次约 2–3 分钟）。

用法：
    python cache_encoders.py            # 生成缓存
    python cache_encoders.py --verify   # 只检查已有缓存
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, 'out', 'encoder_cache.npz')

from leave_one_task_out import (CKPT, load_model, mol_vectors,  # noqa: E402
                                DS, TASKS)
try:  # 归档自包含：fp_matrix.py 与原 eval_encoders 中的实现逐字一致
    from fp_matrix import fp_matrix  # noqa: E402
except ImportError:
    from eval_encoders import fp_matrix  # noqa: E402


def build_probe_smiles():
    """8 个任务的全部分子并集"""
    all_smi = set()
    for name in TASKS:
        tk = TASKS[name]
        p = os.path.join(DS, tk['dir'], 'processed_data', 'compounds.csv')
        if not os.path.exists(p):
            continue
        c = pd.read_csv(p)
        all_smi |= set(c['smiles'].astype(str).dropna())
    return sorted(all_smi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out', default=CACHE)
    args = ap.parse_args()

    if args.verify:
        if not os.path.exists(args.out):
            print('缓存不存在:', args.out)
            return
        d = np.load(args.out, allow_pickle=False)
        print('缓存 %s (%.1f MB)'
              % (args.out, os.path.getsize(args.out) / 1e6))
        for k in d.files:
            print('  %-16s %s' % (k, d[k].shape))
        return

    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    smi = build_probe_smiles()
    print('探针分子 %d' % len(smi))

    files = {os.path.basename(f).replace('permol_', '').replace(
        '_full_all.pt', ''): f
        for f in glob.glob(os.path.join(CKPT, 'permol_*_full_all.pt'))}
    print('编码器 %d 个: %s' % (len(files), ','.join(sorted(files))))

    out = {'smiles': np.array(smi)}
    t0 = time.time()
    for t in sorted(files):
        print('  %-12s' % t, end='', flush=True)
        m = load_model(files[t], dev)
        v = mol_vectors(m, smi, dev).astype(np.float32)
        out['%s_vec' % t] = v
        print(' %s  %.0fs' % (v.shape, time.time() - t0), flush=True)

    print('  %-12s' % 'ecfp4', end='', flush=True)
    out['ecfp4'] = fp_matrix(smi, 'ecfp4').astype(np.float32)
    print(' %s  %.0fs' % (out['ecfp4'].shape, time.time() - t0), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print()
    print('saved %s (%.1f MB)'
          % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
