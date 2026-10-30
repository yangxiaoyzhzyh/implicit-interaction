# -*- coding: utf-8 -*-
"""collect_transfer.py —— 把迁移实验的 JSON 归一成 summary.csv

输入（本目录 runs/）:
  leave_one_out.json      8 任务留一线性读出
  mlp_small_data.json     1%/5% 标签下的 MLP 读出
  mlp_ionic.json          离子任务的 1%/5% MLP
  additive_baseline.json  线性可加基线

输出:
  summary.csv  列: task, family, setting, arm, R, sd, n

用法: python collect_transfer.py
"""
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, 'runs')


def load(name):
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


rows = []

# 1) 留一任务线性读出
for task, d in load('leave_one_out.json').items():
    for arm in ('own', 'others', 'ecfp', 'rdkit2d', 'maccs', 'both'):
        if arm in d:
            rows.append((task, 'loo_linear', 'all-labels', arm,
                         num(d[arm]), num(d.get(arm + '_sd')), d.get('n')))

# 2) 1%/5% 小样本（线性 / MLP）
for fam, fn in (('mlp_small', 'mlp_small_data.json'),):
    for task, per in load(fn).items():
        for setting, arms in per.items():
            for arm, v in arms.items():
                if isinstance(v, dict):
                    rows.append((task, fam, setting, arm,
                                 num(v.get('R')), num(v.get('sd')), None))

# 3) 可加基线
for task, d in load('additive_baseline.json').items():
    rows.append((task, 'additive', 'all-labels', 'additive',
                 num(d.get('R')), num(d.get('sd')), d.get('n_sample')))

dst = os.path.join(ROOT, 'summary.csv')
with open(dst, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['task', 'family', 'setting', 'arm', 'R', 'sd', 'n'])
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3],
                    '' if r[4] is None else '%.4f' % r[4],
                    '' if r[5] is None else '%.4f' % r[5],
                    '' if r[6] is None else r[6]])

print('写出 %s （%d 行）' % (dst, len(rows)))
fams = sorted({r[1] for r in rows})
print('family:', fams)
for fam in fams:
    ts = sorted({r[0] for r in rows if r[1] == fam})
    print('  %-16s 任务 %d 个: %s' % (fam, len(ts), ', '.join(ts)))
