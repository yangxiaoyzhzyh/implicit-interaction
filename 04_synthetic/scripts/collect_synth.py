# -*- coding: utf-8 -*-
"""collect_synth.py —— 把逐 seed 运行归一成 04_synthetic/summary.csv

只读 `runs/synth_*_s<seed>.csv`（逐 seed 原始表），按 (family, lambda, K, model)
取三 seed 均值；拼接长表与 3seed 汇总不参与，避免重复计数。

兼容两种 schema：
  synth_routes.py        宽表 family,lambda,K,vp_r2,vw_r2,gap（vp/vw = v(im)/v(ex)）
  synth_second_order.py  长表 family,lambda,K,model,R2,n_seeds

    python scripts/collect_synth.py
"""
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED = re.compile(r'_s\d+\.csv$')


def main():
    acc = {}
    srcs = set()
    for p in sorted(glob.glob(os.path.join(ROOT, 'runs', 'synth_*.csv'))):
        name = os.path.basename(p)
        if not SEED.search(name):          # 只接受逐 seed 文件
            continue
        srcs.add(name)
        with open(p, encoding='utf-8') as f:
            rd = list(csv.DictReader(f))
        if not rd:
            continue
        cols = set(rd[0].keys())
        for r in rd:
            if 'vp_r2' in cols:
                pairs = [('v(im)', float(r['vp_r2'])), ('v(ex)', float(r['vw_r2']))]
            elif 'model' in cols:
                pairs = [(r['model'], float(r['R2']))]
            else:
                continue
            for model, val in pairs:
                key = (r['family'], r['lambda'], r['K'], model)
                acc.setdefault(key, []).append(val)

    dst = os.path.join(ROOT, 'summary.csv')
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['family', 'lambda', 'K', 'model', 'R2', 'source'])
        for (fam, lam, K, model), vals in sorted(acc.items(),
                                                 key=lambda x: (x[0][0],
                                                                float(x[0][2]),
                                                                x[0][3])):
            w.writerow([fam, lam, K, model, '%.4f' % (sum(vals) / len(vals)),
                        'runs/synth_*_s<seed>.csv'])
    print('写出 %s，%d 行，来自 %d 个逐 seed 文件'
          % (dst, len(acc), len(srcs)))


if __name__ == '__main__':
    main()
