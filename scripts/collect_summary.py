# -*- coding: utf-8 -*-
"""collect_summary.py —— 把 run 级 CSV 归一成一张 summary.csv

四个脚本写出的原始表列名不同（vim/vex 每 seed 一行、additive 单行无 seed、
frozen 已含 mean/sd），本脚本把它们统一成同一套列：

    task, model, split, n_seeds, R_mean, R_sd, per_seed, source

用法:
    python collect_summary.py --dir ../01_composition
    → 读 <dir>/runs/*.csv，写 <dir>/summary.csv
"""
import argparse
import csv
import glob
import os
import statistics as st

MODEL_ORDER = {'v(im)': 0, 'v(ex)': 1, 'additive': 2, 'frozen': 3}


def read_rows(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def norm(path):
    """返回 [(model, task, split, [R...]), ...]"""
    rows = read_rows(path)
    if not rows:
        return []
    cols = set(rows[0].keys())

    # frozen.py / xgb_rdkit.py：task,split,...,mean,sd,per_seed
    # 两者列结构相同，只能靠文件名区分，否则会混成一个臂（n 翻倍、xgb 消失）。
    if 'mean' in cols and 'per_seed' in cols:
        arm = 'xgb' if os.path.basename(path).startswith('xgb_') else 'frozen'
        out = []
        for r in rows:
            vals = [float(x) for x in r['per_seed'].split(';') if x]
            out.append((arm, r['task'], r['split'], vals))
        return out

    # additive.py：task,split,train_n,test_n,R（单行）
    if 'R' in cols and 'mean' not in cols and 'seed' not in cols:
        out = []
        for r in rows:
            out.append(('additive', r['task'], r.get('split', ''),
                        [float(r['R'])]))
        return out

    # vim/vex：每 seed 一行，含 model / test_R
    groups = {}
    for r in rows:
        model = 'v(im)' if r.get('model') == 'vim' else 'v(ex)'
        key = (model, r['task'], r.get('split', ''))
        groups.setdefault(key, []).append(float(r['test_R']))
    return [(k[0], k[1], k[2], v) for k, v in groups.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='某个外推目录（含 runs/）')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, 'runs', '*.csv')))
    if not files:
        print('runs/ 为空，未生成 summary.csv')
        return

    agg = {}
    for p in files:
        for model, task, split, vals in norm(p):
            if not vals:
                continue
            agg.setdefault((task, model, split), []).extend(vals)

    dst = os.path.join(args.dir, 'summary.csv')
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['task', 'model', 'split', 'n_seeds', 'R_mean', 'R_sd',
                    'per_seed', 'source'])
        for (task, model, split), vals in sorted(
                agg.items(),
                key=lambda kv: (kv[0][0],
                                MODEL_ORDER.get(kv[0][1], 9))):
            mu = st.mean(vals)
            sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            w.writerow([task, model, split, len(vals),
                        '%.4f' % mu, '%.4f' % sd,
                        ';'.join('%.4f' % v for v in vals),
                        'runs/*.csv'])
            print('%-12s %-9s %-12s n=%d  %.4f ± %.4f'
                  % (task, model, split, len(vals), mu, sd))
    print('\n写出', dst)


if __name__ == '__main__':
    main()
