# -*- coding: utf-8 -*-
"""make_carved_split.py — 分子级切割，构造纯身份轴划分

原理（同 Sisson 的 Graph Carving，推广到混合物）
------------------------------------------------
一条混合物记录只有当它的**全部组分**都在同一侧时才被保留：

    train: 所有组分 ∈ A 侧
    test : 所有组分 ∈ B 侧
    跨侧  : 丢弃

结果两侧的分子集合**完全不重叠**（ω = 1），而 K 分布由同一批分子随机
分割产生，因此两侧接近相同——组合轴不是变量，唯一变化是"分子是否见过"。

搜索策略
--------
在若干 seed 中挑选同时满足
  1) 两侧样本量都不少于 --min-side
  2) 逐 K 的占比差 ≤ --tol
的切分，取 min(train, test) 最大者。

产出
----
  data/carved/{tag}_split.npz
      train_idx / test_idx   （原始表的行号）
      mols_train / mols_test （两侧分子 id）
      seed, tol, k_train, k_test

用法
----
  python make_carved_split.py --data-base <root> \
      --sets spe:spe ilvis:ilteta ilcond:iltk hmix:miscible
"""
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit('\\', 1)[0])
from spec import TASKS, components  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'splits')

# 额外数据集（不在 run_5tasks_vim.TASKS 里的）
EXTRA = {
    'miscible': dict(dir='miscible-solvent',
                     csv='processed_MiscibleSolventData.csv',
                     prop='Enthalpy of mixing', kind='standard',
                     conds=[], train_max=99),
    'miscible_density': dict(dir='miscible-solvent',
                             csv='processed_MiscibleSolventData.csv',
                             prop='Density', kind='standard',
                             conds=[], train_max=99),
    'miscible_hvap': dict(dir='miscible-solvent',
                          csv='processed_MiscibleSolventData.csv',
                          prop='Heat of vaporization', kind='standard',
                          conds=[], train_max=99),
    'logv': dict(dir='logV', csv='processed_logV.csv', prop=None,
                 kind='standard', conds=['T'], train_max=99),
    'mon': dict(dir='MON', csv='processed_MON.csv', prop=None,
                kind='standard', conds=[], train_max=2),
}


def as_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def spec_of(key):
    if key in TASKS:
        return TASKS[key]
    return EXTRA[key]


def load_records(key, data_base):
    tk = spec_of(key)
    droot = os.path.join(data_base, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, tk['csv']))
    if tk.get('prop') is not None and 'property' in df.columns:
        df = df[df['property'] == tk['prop']].reset_index(drop=True)
    rows, comps = [], []
    for i, r in df.iterrows():
        c = components(r, tk['kind'])
        if c is None:
            # 容错：部分数据集（如 logV 的 K=1）cmp_ids 与 cmp_mole_fractions
            # 长度不一致，此时按 cmp_ids 截断而非整条丢弃
            try:
                ids = [int(v) for v in as_list(r['cmp_ids'])]
            except Exception:
                continue
            if not ids:
                continue
            rows.append(i)
            comps.append(ids)
            continue
        rows.append(i)
        comps.append([cid for cid, _ in c])
    return df, rows, comps


def search_split(comps, n_seed, min_side, tol):
    mols = sorted({c for r in comps for c in r})
    mi = {m: i for i, m in enumerate(mols)}
    NM, NR = len(mols), len(comps)
    K = np.array([len(r) for r in comps])
    Kmax = int(K.max())
    M = np.full((NR, Kmax), -1, dtype=np.int32)
    for i, r in enumerate(comps):
        M[i, :len(r)] = [mi[c] for c in r]
    mask = M >= 0
    Msafe = np.where(mask, M, 0)
    Ks = sorted(set(K.tolist()))

    best = None
    n_valid = 0
    for seed in range(n_seed):
        rng = np.random.RandomState(seed)
        tr = np.zeros(NM, dtype=bool)
        tr[rng.permutation(NM)[:NM // 2]] = True
        ok_tr = ((tr[Msafe] & mask) | ~mask).all(1)
        ok_te = (((~tr[Msafe]) & mask) | ~mask).all(1)
        n_tr, n_te = int(ok_tr.sum()), int(ok_te.sum())
        if min(n_tr, n_te) < min_side:
            continue
        n_valid += 1
        dev = 0.0
        for k in Ks:
            m = K == k
            a = int((ok_tr & m).sum())
            b = int((ok_te & m).sum())
            if a + b < 30:
                continue
            dev = max(dev, abs(a / n_tr - b / n_te))
        if dev > tol:
            continue
        if best is None or min(n_tr, n_te) > best[0]:
            best = (min(n_tr, n_te), n_tr, n_te, seed, dev, ok_tr, ok_te, tr)
    return best, mols, K, n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--sets', nargs='+',
                    default=['spe:spe', 'ilvis:ilteta', 'ilcond:iltk',
                             'hmix:miscible'],
                    help='tag:任务键')
    ap.add_argument('--n-seed', type=int, default=400)
    ap.add_argument('--min-side', type=int, default=500)
    ap.add_argument('--tol', type=float, default=0.06)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for pair in args.sets:
        tag, key = pair.split(':')
        df, rows, comps = load_records(key, args.data_base)
        best, mols, K, n_valid = search_split(comps, args.n_seed,
                                              args.min_side, args.tol)
        print('=' * 74)
        print('%s (task=%s) | %d 条, %d 分子, K %d–%d'
              % (tag, key, len(comps), len(mols), K.min(), K.max()))
        if best is None:
            print('  无满足约束的切分（候选 %d）' % n_valid)
            continue
        _, n_tr, n_te, seed, dev, ok_tr, ok_te, tr = best
        tr_rows = [rows[i] for i in range(len(comps)) if ok_tr[i]]
        te_rows = [rows[i] for i in range(len(comps)) if ok_te[i]]
        m_tr = [m for m, f in zip(mols, tr) if f]
        m_te = [m for m, f in zip(mols, tr) if not f]
        assert not (set(m_tr) & set(m_te))
        print('  seed %d | train %d / test %d | K 偏差 %.3f | 保留 %.1f%%'
              % (seed, len(tr_rows), len(te_rows), dev,
                 100 * (len(tr_rows) + len(te_rows)) / len(comps)))
        for k in sorted(set(K.tolist())):
            a = int((ok_tr & (K == k)).sum())
            b = int((ok_te & (K == k)).sum())
            if a + b < 20:
                continue
            print('     K=%-3d train %5d (%.1f%%)  test %5d (%.1f%%)'
                  % (k, a, 100 * a / len(tr_rows), b, 100 * b / len(te_rows)))

        out = os.path.join(OUT, '%s_split.npz' % tag)
        np.savez_compressed(
            out,
            train_idx=np.array(tr_rows, dtype=np.int64),
            test_idx=np.array(te_rows, dtype=np.int64),
            mols_train=np.array(m_tr, dtype=np.int64),
            mols_test=np.array(m_te, dtype=np.int64),
            k_train=K[ok_tr], k_test=K[ok_te],
            seed=seed, tol=dev, task=key)
        print('  saved', os.path.basename(out))
        print()


if __name__ == '__main__':
    main()