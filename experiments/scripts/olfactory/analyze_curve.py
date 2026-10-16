# -*- coding: utf-8 -*-
"""analyze_curve.py — 离线分析 curve_<tag>.csv：打印 val 序列并模拟 patience=3 早停

实验 12 的辅助脚本
读取 results/sisson/curves/curve_*.csv
无独立数据产出
"""
import csv, sys

p = sys.argv[1]
rows = list(csv.DictReader(open(p, encoding='utf-8')))
rows = [r for r in rows if r['val'] not in ('', None)]
print(f'rows: {len(rows)}')
print('ep:val (前 31 个)')
print(' '.join(f"{r['ep']}:{r['val']}" for r in rows[:31]))

for pat in (3,):
    best, bad, stop = -1.0, 0, None
    for r in rows:
        v = float(r['val'])
        if v > best:
            best, bad = v, 0
        else:
            bad += 1
        if bad >= pat and stop is None:
            stop = int(r['ep'])
    print(f'patience={pat} -> 早停 ep {stop} | best_val {best:.4f}')

print('--- test 采样点（诊断用）---')
for r in rows:
    if r.get('test'):
        print(f"  ep {r['ep']:>3} | val {r['val']} | TEST {r['test']}")
