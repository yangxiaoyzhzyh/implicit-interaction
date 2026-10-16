# -*- coding: utf-8 -*-
"""summarize_perk.py — 把 results/perk_*.csv 汇总成 per-composition-count 表

实验 3 — Per-composition-count profile（补材 S2，表 S4）
读取 results/perk_{hmix,hvap}_e{1,2}.csv；v(im) 臂取 e2、v(ex) 臂取 e1
输出到终端（3-seed 均值），供表 S4 使用

输出一段 markdown 表，直接贴进补充材料 S2。
用法: python summarize_perk.py [--dir ../results]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results'))
a = ap.parse_args()

d = defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.dir, 'perk_*.csv'))):
    for r in csv.DictReader(open(f, encoding='utf-8')):
        d[(r['task'], r['route'], int(r['k']))].append(r)


def mean(rows, col):
    return sum(float(x[col]) for x in rows) / len(rows)


print('| Task | test K | n | v(im) R | v(ex) R | v(im) MAE | v(ex) MAE |')
print('|:---|---:|---:|---:|---:|---:|---:|')
for task, label in (('dHmix', 'ΔHmix'), ('dHvap', 'ΔHvap')):
    for k in (3, 4, 5):
        im, ex = d[(task, 'e2', k)], d[(task, 'e1', k)]
        print(f'| {label} | {k} | {im[0]["n"]} | {mean(im,"R"):.4f} | {mean(ex,"R"):.4f} '
              f'| {mean(im,"MAE"):.4f} | {mean(ex,"MAE"):.4f} |')

print()
print('--- seed-level ---')
for task in ('dHmix', 'dHvap'):
    for r_ in ('e2', 'e1'):
        for k in (3, 4, 5):
            rows = d[(task, r_, k)]
            print(task, r_, 'K=', k, 'R:', [x['R'] for x in rows], 'MAE:', [x['MAE'] for x in rows])
