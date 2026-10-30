# -*- coding: utf-8 -*-
"""汇总 agg_100bin：最佳描述符 vs 最佳编码器"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'out', 'agg_100bin.json'),
                   encoding='utf-8'))
AGGS = ['sum', 'mean', 'concat', 'attn']
DESC = ['rdkit2d', 'ecfp4', 'maccs']

print('%-11s %-24s %-24s %-24s %-24s'
      % ('task', 'RDKit2D', 'ECFP4', 'MACCS', 'encoders'))
print('-' * 108)
rows = []
for t, r in d.items():
    cells, best = [], {}
    for f in DESC + ['enc']:
        if f not in r:
            cells.append('%-24s' % '-')
            continue
        v, s = max(((a, r[f][a]['R']) for a in AGGS if a in r[f]),
                   key=lambda x: x[1])
        cells.append('%-24s' % ('%s %.3f' % (v, s)))
        best[f] = (v, s)
    print('%-11s %s' % (t, ' '.join(cells)))
    bd = max(((f, v, s) for f, (v, s) in best.items() if f in DESC),
             key=lambda x: x[2])
    be = best['enc']
    rows.append(dict(task=t, desc='%s/%s' % (bd[0], bd[1]),
                     desc_R=round(bd[2], 4), enc=be[0], enc_R=round(be[1], 4),
                     diff=round(be[1] - bd[2], 4)))

print()
print('%-11s %-20s %8s %-8s %8s %9s'
      % ('task', 'best descriptor', 'R', 'enc', 'R', 'diff'))
print('-' * 70)
for x in rows:
    print('%-11s %-20s %8.3f %-8s %8.3f %+9.3f'
          % (x['task'], x['desc'], x['desc_R'], x['enc'], x['enc_R'],
             x['diff']))

ds = sorted(x['diff'] for x in rows)
print()
print('编码器领先 %d，持平 %d，落后 %d（±0.005）'
      % (sum(x['diff'] > 0.005 for x in rows),
         sum(abs(x['diff']) <= 0.005 for x in rows),
         sum(x['diff'] < -0.005 for x in rows)))
print('差值中位 %+.3f，均值 %+.3f'
      % ((ds[len(ds) // 2] if len(ds) % 2 else
          0.5 * (ds[len(ds) // 2 - 1] + ds[len(ds) // 2])),
         sum(ds) / len(ds)))

out = os.path.join(HERE, 'out', 'agg_100bin_best.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('saved', out)
