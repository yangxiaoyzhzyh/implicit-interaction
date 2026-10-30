# -*- coding: utf-8 -*-
"""additive.py —— 线性可加基线（闭式解，无随机性）

模型: y_hat = sum_i w_i * c_i + cond . b
即每个组分一个系数 c_i，按摩尔分数加权求和；条件列（温度/压力等）线性叠加。
**没有 seed、没有误差棒**——它是一条参照线，不是一条“路线”。

求解用 ridge（默认 alpha=1e-2），而不是普通最小二乘。原因：
摩尔分数恒有 sum_i w_i = 1，组分列与截距本来就完全共线；当数据里出现一个
占 0.9–0.999 的溶剂、其余是 10^-3 量级的痕量组分时（IlT 两个任务正是如此），
最小二乘返回的最小范数解会给痕量组分分配巨大且相互抵消的系数，
样本内正常、样本外发散（R^2 掉到 -10^4 量级）。ridge 在同一划分上把
IlT κ 从 -0.133 拉回 +0.74，其余七个任务几乎不变。

--plain 可复现旧的普通最小二乘结果，用于对照。

任务与划分都由参数选，与其余三个脚本读同一份 npz。

    python additive.py --task spe --split-file ../splits/spe_composition.npz \
                       --data-base D --out ../01_composition/runs
    python additive.py --task spe ... --plain          # 旧口径（无正则）
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))

from spec import load_records, spec_of  # noqa: E402


def pearson(p, t):
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    p = p - p.mean()
    t = t - t.mean()
    d = np.sqrt((p * p).sum() * (t * t).sum())
    return float((p * t).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--out', default=None, help='汇总 csv 的输出目录')
    ap.add_argument('--alpha', type=float, default=1e-2,
                    help='ridge 强度（作用在标准化目标上）；0 表示不惩罚')
    ap.add_argument('--plain', action='store_true',
                    help='不加 ridge，复现旧的普通最小二乘结果')
    args = ap.parse_args()

    tk = spec_of(args.task)
    df, rows, comps = load_records(args.task, args.data_base)

    # 摩尔分数（与 engine 相同的容错策略）
    import ast
    mfs = []
    for i, r in enumerate(rows):
        row = df.loc[r]
        k = len(comps[i])
        if 'cmp_mole_fractions' in df.columns:
            w = [float(v) for v in ast.literal_eval(row['cmp_mole_fractions'])]
        else:
            w = [1.0 / k] * k
        if len(w) > k:
            w = w[:k]
        elif len(w) < k:
            w += [0.0] * (k - len(w))
        tot = sum(w)
        mfs.append([x / tot for x in w] if tot > 0 else [1.0 / k] * k)

    ys = np.array([float(df.loc[r, 'value']) for r in rows])
    conds = np.array([[float(df.loc[r].get(c, np.nan)) for c in tk['conds']]
                      for r in rows], dtype=np.float64).reshape(len(rows), -1)

    sp = np.load(args.split_file, allow_pickle=True)
    # npz 存的是 df 行标签，这里换算成记录位置（与 comps/ys 的数组下标一致）
    pos = {int(r): i for i, r in enumerate(rows)}
    tr = np.array([pos[int(x)] for x in sp['train_idx']], dtype=int)
    te = np.array([pos[int(x)] for x in sp['test_idx']], dtype=int)

    # 设计矩阵：每个组分一列（摩尔分数加权）+ 条件列
    ids = sorted({c for i in range(len(rows)) for c in comps[i]})
    col = {c: j for j, c in enumerate(ids)}
    X = np.zeros((len(rows), len(ids) + conds.shape[1]))
    for i in range(len(rows)):
        for c, w in zip(comps[i], mfs[i]):
            X[i, col[c]] = w
    if conds.shape[1]:
        cm, cs = np.nanmean(conds[tr], 0), np.nanstd(conds[tr], 0)
        cs[cs == 0] = 1.0
        X[:, len(ids):] = (np.where(np.isnan(conds), cm, conds) - cm) / cs

    # 加截距列、目标标准化；ridge（alpha 默认 1e-2）求解，--plain 时退回普通最小二乘
    ym, ysd = float(ys[tr].mean()), float(ys[tr].std()) or 1.0
    Atr = np.hstack([X[tr], np.ones((len(tr), 1))])
    ytr = (ys[tr] - ym) / ysd
    if args.plain or args.alpha <= 0:
        coef, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
    else:
        M = Atr.T @ Atr + args.alpha * np.eye(Atr.shape[1])
        coef = np.linalg.pinv(M) @ (Atr.T @ ytr)
    Ate = np.hstack([X[te], np.ones((len(te), 1))])
    p = (Ate @ coef) * ysd + ym
    r = pearson(p, ys[te])

    print('%s | %s | 组分 %d + 条件 %d | train %d / test %d | R = %.4f'
          % (args.task, os.path.basename(args.split_file), len(ids),
             conds.shape[1], len(tr), len(te), r))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(args.split_file).replace('.npz', '')
        dst = os.path.join(args.out, 'additive_%s.csv' % name)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('task,split,train_n,test_n,R,alpha\n')
            f.write('%s,%s,%d,%d,%.4f,%s\n'
                    % (args.task, sp['split'] if 'split' in sp else '',
                       len(tr), len(te), r,
                       'plain' if (args.plain or args.alpha <= 0)
                       else '%.0e' % args.alpha))
        print('写出', dst)


if __name__ == '__main__':
    main()
