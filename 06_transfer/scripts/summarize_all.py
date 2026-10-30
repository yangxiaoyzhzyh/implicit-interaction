# -*- coding: utf-8 -*-
"""汇总 agg_all：每个任务取「最佳描述符（任意描述符×任意聚合）」vs「最佳编码器」"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'out', 'agg_all_1pct.json')
d = json.load(open(P, encoding='utf-8'))

AGGS = ['sum', 'mean', 'concat', 'attn']
DESC = ['rdkit2d', 'ecfp4', 'maccs']
LAB = {'rdkit2d': 'RDKit2D', 'ecfp4': 'ECFP4', 'maccs': 'MACCS', 'enc': 'enc'}

print('=== 每个特征的最佳聚合（1% 标签） ===')
print('%-11s %-22s %-22s %-22s %-22s'
      % ('task', 'RDKit2D', 'ECFP4', 'MACCS', 'encoders'))
print('-' * 100)
rows = []
for t, r in d.items():
    cells, best = [], {}
    for f in DESC + ['enc']:
        if f not in r:
            cells.append('%-22s' % '-')
            continue
        v, s = max(((a, r[f][a]['R']) for a in AGGS if a in r[f]),
                   key=lambda x: x[1])
        cells.append('%-22s' % ('%s %.3f' % (v, s)))
        best[f] = (v, s)
    print('%-11s %s' % (t, ' '.join(cells)))

    if not best:
        continue
    bd = max(((f, v, s) for f, (v, s) in best.items() if f in DESC),
             key=lambda x: x[2])
    be = best.get('enc')
    if be is None:
        continue
    rows.append(dict(task=t,
                     desc='%s/%s' % (bd[0], bd[1]), desc_R=round(bd[2], 4),
                     enc=be[0], enc_R=round(be[1], 4),
                     diff=round(be[1] - bd[2], 4)))

print()
print('=== 最佳描述符 vs 最佳编码器 ===')
print('%-11s %-22s %-14s %-12s %9s' % ('task', 'best descriptor', 'R',
                                       'best enc', 'diff'))
print('-' * 76)
for x in rows:
    print('%-11s %-22s %-14.3f %-12s %+9.3f'
          % (x['task'], x['desc'], x['desc_R'], x['enc'], x['diff']))

wins = sum(1 for x in rows if x['diff'] > 0.005)
ties = sum(1 for x in rows if abs(x['diff']) <= 0.005)
print()
print('编码器领先 %d，持平 %d，落后 %d（阈值 ±0.005）'
      % (wins, ties, len(rows) - wins - ties))
print('差值：中位 %+.3f，均值 %+.3f'
      % (sorted(x['diff'] for x in rows)[len(rows) // 2],
         sum(x['diff'] for x in rows) / len(rows)))

out = os.path.join(HERE, 'out', 'agg_all_1pct_best.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('saved', out)
