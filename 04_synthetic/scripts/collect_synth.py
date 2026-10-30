# -*- coding: utf-8 -*-
"""collect_synth.py —— 把 runs/synth_*.csv 归一成 04_synthetic/summary.csv

兼容两种 schema：
  synth_routes.py        宽表 family,lambda,K,vp_r2,vw_r2,gap（vp/vw = v(im)/v(ex)）
  synth_second_order.py  长表 family,lambda,K,model,R2,n_seeds

    python scripts/collect_synth.py
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'runs', 'synth_*.csv'))):
        name = os.path.basename(p)
        with open(p, encoding='utf-8') as f:
            rd = list(csv.DictReader(f))
        if not rd:
            continue
        cols = set(rd[0].keys())
        if 'vp_r2' in cols:
            for r in rd:
                rows.append((r['family'], r['lambda'], r['K'], 'v(im)',
                             float(r['vp_r2']), name))
                rows.append((r['family'], r['lambda'], r['K'], 'v(ex)',
                             float(r['vw_r2']), name))
        elif 'model' in cols:
            for r in rd:
                rows.append((r['family'], r['lambda'], r['K'], r['model'],
                             float(r['R2']), name))

    dst = os.path.join(ROOT, 'summary.csv')
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['family', 'lambda', 'K', 'model', 'R2', 'source'])
        for x in rows:
            w.writerow([x[0], x[1], x[2], x[3], '%.4f' % x[4], x[5]])
    print('写出 %s：%d 行' % (dst, len(rows)))


if __name__ == '__main__':
    main()
