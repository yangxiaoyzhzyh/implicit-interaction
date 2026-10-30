# -*- coding: utf-8 -*-
"""汇总「混合物层 PCA」两种读出：最佳描述符 vs 最佳编码器"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
AGGS = ['sum', 'mean', 'concat']
DESC = ['rdkit2d', 'ecfp4', 'maccs']

d = {}
for readout, f in (('mlp', 'agg_100bin_mixpca_mlp.json'),
                   ('xgb', 'agg_100bin_mixpca_xgb.json')):
    d[readout] = json.load(open(os.path.join(RUNS, f), encoding='utf-8'))

tasks = list(d['mlp'])
rows = []
print('%-11s %-26s %-26s %8s' % ('task', 'best descriptor', 'best encoder',
                                 'enc − desc'))
print('-' * 78)
for t in tasks:
    best_d, best_e = None, None
    for readout in ('mlp', 'xgb'):
        r = d[readout].get(t, {})
        for feat in DESC:
            if feat not in r:
                continue
            for agg in AGGS:
                if agg not in r[feat]:
                    continue
                v = r[feat][agg]['R']
                if best_d is None or v > best_d[-1]:
                    best_d = ('%s/%s/%s' % (feat, agg, readout), v)
        r_enc = r.get('enc')
        if r_enc:
            for agg in AGGS:
                if agg not in r_enc:
                    continue
                v = r_enc[agg]['R']
                if best_e is None or v > best_e[-1]:
                    best_e = ('%s/%s' % (agg, readout), v)
    print('%-11s %-26s %-26s %+8.3f'
          % (t, '%s %.3f' % best_d, '%s %.3f' % best_e,
             best_e[1] - best_d[1]))
    rows.append(dict(task=t, desc=best_d[0], desc_R=round(best_d[1], 4),
                     enc=best_e[0], enc_R=round(best_e[1], 4),
                     diff=round(best_e[1] - best_d[1], 4)))

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

out = os.path.join(RUNS, 'agg_100bin_mixpca_best.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('saved', out)
