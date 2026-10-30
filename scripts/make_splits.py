# -*- coding: utf-8 -*-
"""make_splits.py —— 唯一的划分生成脚本

三个划分共用同一入口，四个建模脚本一律读它产出的 npz，因此
v(im) / v(ex) / additive / frozen 面对的是**同一份索引**。

    python make_splits.py --task spe   --split composition --data-base D --out ../splits
    python make_splits.py --task spe   --split unseen      --data-base D --out ../splits
    python make_splits.py --task drug  --split target      --data-base D --out ../splits

产出 `<task>_<split>.npz`，键: task / tag / train_idx / val_idx / test_idx / split。

划分定义
--------
composition : 组分数 K <= spec.train_max 训练, K > train_max 测试。
unseen      : 分子级切割，训练与测试不共享任何分子（调用 common/carved_split.py）。
target      : 目标值四分位带训练（0.25–0.75），两端各 10% 测试（<0.10 或 >0.90）。

验证集
------
`val_idx` 一律从**训练侧**留出 1/8（最多 600 条，RandomState(0)），测试侧不受影响。
组分数轴不使用它（不早停），另两轴用它做 knee 早停。
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))

from spec import load_records, spec_of  # noqa: E402

VAL_FRAC = 0.125
VAL_MAX = 600


def holdout_val(n_fit, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(n_fit)
    n_val = min(VAL_MAX, max(64, int(n_fit * VAL_FRAC)))
    return perm[n_val:], perm[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    help='任务键，见 common/spec.py（如 spe / miscible / mon）')
    ap.add_argument('--split', required=True,
                    choices=['composition', 'unseen', 'target'])
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--out', default=os.path.join(ROOT, 'splits'))
    ap.add_argument('--train-max', type=int, default=None,
                    help='composition: 覆盖 spec 里的 train_max')
    ap.add_argument('--y-train', type=float, nargs=2, default=[0.25, 0.75])
    ap.add_argument('--y-test', type=float, nargs=2, default=[0.10, 0.90])
    args = ap.parse_args()

    tk = spec_of(args.task)
    df, rows, comps = load_records(args.task, args.data_base)
    n = len(rows)
    os.makedirs(args.out, exist_ok=True)

    if args.split == 'composition':
        tmax = args.train_max if args.train_max is not None else tk['train_max']
        kk = np.array([len(c) for c in comps])
        tr = np.where(kk <= tmax)[0]
        te = np.where(kk > tmax)[0]
        # 组分数轴不早停，因此不留验证集：全部 K<=tmax 的样本都用于训练。
        val = np.array([], dtype=int)
        desc = 'K<=%d 训练 / K>%d 测试（无验证集）' % (tmax, tmax)

    elif args.split == 'target':
        ys = np.array([float(df.loc[r, 'value']) for r in rows])
        q = np.percentile(ys, [args.y_test[0] * 100, args.y_train[0] * 100,
                               args.y_train[1] * 100, args.y_test[1] * 100])
        lo_te, lo_tr, hi_tr, hi_te = q
        tr = np.array([i for i in range(n) if lo_tr <= ys[i] <= hi_tr])
        te = np.array([i for i in range(n)
                       if ys[i] < lo_te or ys[i] > hi_te])
        fit, val = holdout_val(len(tr))
        tr, val = tr[fit], tr[val]
        desc = 'y 在 [%.3f, %.3f] 训练 / <%.3f 或 >%.3f 测试' % (
            lo_tr, hi_tr, lo_te, hi_te)

    else:  # unseen：分子二分切割；val 从训练侧按行留出，供 knee 早停使用
        tag = args.task
        cmd = [sys.executable, os.path.join(ROOT, 'common', 'carved_split.py'),
               '--data-base', args.data_base, '--sets', '%s:%s' % (tag, tag)]
        print('调用切割生成器:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        # 生成器固定写到 common/../splits，这里按需拷贝到 --out
        src = os.path.join(ROOT, 'splits', '%s_split.npz' % tag)
        if not os.path.exists(src):
            raise SystemExit('切割生成器没有产出 %s' % src)
        sp = np.load(src, allow_pickle=True)
        tr = np.asarray(sp['train_idx'])
        te = np.asarray(sp['test_idx'])
        # 二分而非三分：再切一刀会显著缩小可用数据。val 只用于 knee 早停，
        # 且 knee 规则本身就是为“val 从 train 内部切出”设计的。
        fit, val = holdout_val(len(tr))
        tr, val = tr[fit], tr[val]
        desc = '分子二分切割（train/test 无共享分子，val 取自 train 内部）'

    # 统一语义：npz 里存的是 **df 行标签**（不是位置）。
    # composition/target 分支前面用的是位置，这里换算；unseen 分支来自切割文件，
    # 本来就是标签。
    if args.split in ('composition', 'target'):
        labels = np.asarray(rows)
        tr, val, te = labels[tr], labels[val], labels[te]

    dst = os.path.join(args.out, '%s_%s.npz' % (args.task, args.split))
    np.savez_compressed(dst, task=args.task, tag=args.task,
                        split=args.split,
                        train_idx=np.array(sorted(tr.tolist()), dtype=np.int64),
                        val_idx=np.array(sorted(val.tolist()), dtype=np.int64),
                        test_idx=np.array(sorted(te.tolist()), dtype=np.int64))
    print('%s | %s | train %d / val %d / test %d'
          % (args.task, desc, len(tr), len(val), len(te)))
    print('写出', dst)


if __name__ == '__main__':
    main()
