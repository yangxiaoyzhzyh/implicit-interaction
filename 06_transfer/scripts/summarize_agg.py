# -*- coding: utf-8 -*-
"""汇总 agg_sweep 结果：每个任务、每种特征取最佳聚合，打印对照并写 csv"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'out', 'agg_sweep_1pct.json')
d = json.load(open(P, encoding='utf-8'))

AGGS = ['sum', 'mean', 'concat', 'attn']
rows = []
print('%-11s %-26s %-26s %8s' % ('task', 'RDKit2D (best agg)', '7 encoders (best agg)',
                                  'enc − rdkit'))
print('-' * 78)
for t, by in d.items():
    r = by.get('0.01', {})
    if 'rdkit' not in r or 'enc' not in r:
        continue
    def best(feat):
        cand = [(v, r[feat][v]['R']) for v in AGGS if v in r[feat]]
        v, s = max(cand, key=lambda x: x[1])
        return v, s
    bv, bs = best('rdkit')
    ev, es = best('enc')
    print('%-11s %-26s %-26s %+8.3f'
          % (t, '%s %.3f' % (bv, bs), '%s %.3f' % (ev, es), es - bs))
    rows.append(dict(task=t, rdkit_agg=bv, rdkit_R=round(bs, 4),
                     enc_agg=ev, enc_R=round(es, 4),
                     diff=round(es - bs, 4)))

print()
print('%-11s %s' % ('task', ' '.join('%-22s' % ('%s' % a) for a in AGGS)))
for t, by in d.items():
    r = by.get('0.01', {})
    if 'rdkit' not in r:
        continue
    cells = []
    for a in AGGS:
        cells.append('%s/%.3f' % ('rd', r['rdkit'][a]['R'])
                     if a in r['rdkit'] else 'rd/-')
        cells.append('%s/%.3f' % ('en', r['enc'][a]['R'])
                     if 'enc' in r and a in r['enc'] else 'en/-')
    print('%-11s %s' % (t, ' '.join('%-9s' % c for c in cells)))

out = os.path.join(HERE, 'out', 'agg_sweep_1pct_summary.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print()
print('saved', out)
