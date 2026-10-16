# -*- coding: utf-8 -*-
"""spe_coverage_by_trainmax.py — SPE 覆盖对比（train_max = 2 与 3 两种协议）

实验 11 — Dataset audits（补材 S3）
路线 无（数据审计）
产出 results/analysis/spe_coverage_by_trainmax.out

对应大纲 §5 SPE 补覆盖对照的数据审查：
- 训练二元对数量、测试样本的二元子对覆盖率；
- 测试样本中"含训练零出现分子"的比例（新分子混杂度）。
数字出处：results/analysis/spe_coverage_by_trainmax.out
"""
import os
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER_V2 = _os.path.dirname(_os.path.dirname(_HERE))
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_PAPER_V2)))  # <repo> (three levels above experiments/scripts/<sub>)
import itertools
import pandas as pd
import collections

BASE = os.path.join(REPO, 'literature', 'CheMixHub', 'datasets')
CSV = os.path.join(BASE, 'polymer-electrolyte', 'processed_data',
                   'processed_PolymerElectrolyteData.csv')

df = pd.read_csv(CSV)
recs = []
for _, r in df.iterrows():
    try:
        ids = list(eval(r['cmp_ids'])); mfs = list(eval(r['cmp_mole_fractions']))
    except Exception:
        continue
    if len(ids) != len(mfs):
        continue
    recs.append(sorted(set(int(c) for c in ids)))

for tm in [2, 3]:
    tr = [s for s in recs if len(s) <= tm]
    te = [s for s in recs if len(s) >= tm + 1]
    P = set(frozenset(s) for s in tr if len(s) == 2)
    mol_cnt = collections.Counter(c for s in tr for c in s)
    full = partial = zero = 0
    cov_sum = 0.0
    newmol = 0
    byk = collections.defaultdict(lambda: [0, 0.0])
    for s in te:
        pairs = [frozenset(p) for p in itertools.combinations(s, 2)]
        hit = sum(1 for p in pairs if p in P)
        cov = hit / len(pairs)
        cov_sum += cov
        if cov == 1.0: full += 1
        elif cov > 0: partial += 1
        else: zero += 1
        byk[len(s)][0] += 1
        byk[len(s)][1] += cov
        if any(mol_cnt.get(c, 0) == 0 for c in s):
            newmol += 1
    print(f'=== train_max={tm}: train {len(tr)} 条 | 训练二元对 {len(P)} 个')
    print(f'  test {len(te)} 条: 子对覆盖 完全={full} 部分={partial} 零={zero} '
          f'| 平均覆盖={cov_sum/len(te):.3f}')
    for k in sorted(byk):
        n, sc = byk[k]
        print(f'  test K={k}: n={n} 平均子对覆盖={sc/n:.3f}')
    print(f'  含训练零出现分子的测试样本: {newmol}/{len(te)} '
          f'({newmol/len(te)*100:.0f}%)')
    print()
