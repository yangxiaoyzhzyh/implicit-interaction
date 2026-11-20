# -*- coding: utf-8 -*-
"""regime_slopes.py —— 正文 IlT κ / SPE κ 两段的定量来源（table S12）

两个失败任务各自被切成两个体系，比较同一份数据里"性质对组成的依赖"是否相同：

1. IlT κ（`processed_IlThermoData.csv`, property='Electrical conductivity'）
   稀：某组分摩尔分数 > 0.99；浓：其余
2. SPE κ（`processed_PolymerElectrolyteData.csv`, property='log Conductivity'）
   小分子：组分最大分子量 ≤ 300 g/mol；聚合物：> 300

每组内把性质对 log10(w_salt) 回归，报告斜率（指数）、组均值（水平）与 n。
SPE 另在配比匹配区间 0.02 ≤ w_salt ≤ 0.2 上比较两组的均值。

w_salt 取 compounds.csv 中 salt=1 的组分摩尔分数之和。

用法：
  python regime_slopes.py --data-base /path/to/CheMixHub/datasets \
      --out ../runs/regime_slopes.csv
"""
import argparse
import ast
import csv
import os

import numpy as np
import pandas as pd

ILT = ('ionic-liquids', 'processed_IlThermoData.csv', 'Electrical conductivity')
SPE = ('polymer-electrolyte', 'processed_PolymerElectrolyteData.csv', 'log Conductivity')


def load(data_base, spec):
    sub, csv_name, prop = spec
    d = os.path.join(data_base, sub, 'processed_data')
    df = pd.read_csv(os.path.join(d, csv_name))
    df = df[df['property'] == prop].copy()
    comp = pd.read_csv(os.path.join(d, 'compounds.csv'))
    salt = dict(zip(comp['compound_id'], comp['salt']))
    df['w'] = df['cmp_mole_fractions'].map(ast.literal_eval)
    df['ids'] = df['cmp_ids'].map(ast.literal_eval)
    df['w_salt'] = [sum(wi for i, wi in zip(ids, ws) if salt.get(i, 0) == 1)
                    for ids, ws in zip(df['ids'], df['w'])]
    df['w_max'] = [max(ws) for ws in df['w']]
    return df


def fit(sub):
    """log10(w_salt) → property 的最小二乘斜率、R² 与组均值。"""
    x = np.log10(sub['w_salt'].to_numpy(float))
    y = sub['value'].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    b = np.polyfit(x, y, 1)[0]
    pred = np.polyval(np.polyfit(x, y, 1), x)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2, float(y.mean()), int(len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'regime_slopes.csv'))
    args = ap.parse_args()

    rows = []

    ilt = load(args.data_base, ILT)
    ilt = ilt[ilt['w_salt'] > 0]
    for name, mask in (('dilute', ilt['w_max'] > 0.99),
                       ('concentrated', ilt['w_max'] <= 0.99)):
        b, r2, lvl, n = fit(ilt[mask])
        rows.append(('iltk', name, n, b, r2, lvl, ''))

    spe = load(args.data_base, SPE)
    spe = spe[spe['w_salt'] > 0]
    mn_cols = [c for c in spe.columns if c.startswith('cmp') and c.endswith('mn_or_mw')]
    spe['mn_max'] = spe[mn_cols].max(axis=1)
    gap = ''
    band = spe[spe['w_salt'].between(0.02, 0.2)]
    small = band[band['mn_max'] <= 300]['value']
    poly = band[band['mn_max'] > 300]['value']
    if len(small) > 20 and len(poly) > 20:
        gap = '%.2f' % (small.mean() - poly.mean())
        print('SPE 匹配配比 %.2f–%.2f：小分子 %+.2f (n=%d) vs 聚合物 %+.2f (n=%d)，差 %s'
              % (0.02, 0.2, small.mean(), len(small), poly.mean(), len(poly), gap))
    for name, mask in (('small molecules', spe['mn_max'] <= 300),
                       ('polymers', spe['mn_max'] > 300)):
        b, r2, lvl, n = fit(spe[mask])
        rows.append(('spek', name, n, b, r2, lvl, gap if name == 'small molecules' else ''))

    print()
    print('%-6s %-16s %8s %8s %6s %8s %8s'
          % ('task', 'regime', 'n', 'exponent', 'R2', 'level', 'gap'))
    for r in rows:
        print('%-6s %-16s %8d %8.2f %6.2f %8.2f %8s' % r)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'regime', 'n', 'exponent', 'r2', 'level',
                    'matched_gap_small_minus_polymer'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], '%.3f' % r[3], '%.3f' % r[4],
                        '%.3f' % r[5], r[6]])
    print('wrote', out)


if __name__ == '__main__':
    main()
