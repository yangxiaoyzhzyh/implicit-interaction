import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER_V2 = _os.path.dirname(_os.path.dirname(_HERE))
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_PAPER_V2)))  # <repo> (three levels above experiments/scripts/<sub>)
# -*- coding: utf-8 -*-
"""spe_subpair_coverage.py — SPE 测试样本的二元子对在训练集中的覆盖情况

实验 11 — Dataset audits（补材 S3）
路线 无（数据审计）
产出 results/analysis/spe_subpair_coverage.out
v(ex) 可把学过的二元交互直接复用到测试组合；v(im) 依赖单分子在多样搭档中的修正。
"""
import pandas as pd, itertools, collections

CSV = os.path.join(REPO, 'literature', 'CheMixHub', 'datasets', 'polymer-electrolyte', 'processed_data', 'processed_PolymerElectrolyteData.csv')

df = pd.read_csv(CSV)
recs = []
for _, r in df.iterrows():
    try:
        ids = list(eval(r['cmp_ids'])); mfs = list(eval(r['cmp_mole_fractions']))
    except Exception:
        continue
    if len(ids) != len(mfs):
        continue
    recs.append(sorted(set(int(c) for c in ids)))  # 组分去重集合

tr = [s for s in recs if len(s) <= 2]
te = [s for s in recs if len(s) >= 3]
P_train = set()
for s in tr:
    if len(s) == 2:
        P_train.add(frozenset(s))
print(f'train(<=2): {len(tr)} 条，其中二元 {len([s for s in tr if len(s)==2])} 条，'
      f'不同二元对 {len(P_train)} 个')
print(f'test(>=3): {len(te)} 条\n')

# 测试样本的二元子对覆盖率
full_cover = partial = zero = 0
ratio_by_k = collections.defaultdict(list)
for s in te:
    pairs = [frozenset(p) for p in itertools.combinations(s, 2)]
    hit = sum(1 for p in pairs if p in P_train)
    cov = hit / len(pairs)
    ratio_by_k[len(s)].append(cov)
    if cov == 1.0: full_cover += 1
    elif cov > 0: partial += 1
    else: zero += 1
print(f'测试样本二元子对覆盖率（子对在训练二元对中出现的比例）：')
print(f'  完全覆盖(100%): {full_cover} | 部分: {partial} | 零覆盖: {zero}')
for k in sorted(ratio_by_k):
    v = ratio_by_k[k]
    print(f'  K={k}: n={len(v)} 平均子对覆盖率={sum(v)/len(v):.3f} '
          f'完全覆盖样本={sum(1 for x in v if x==1.0)}')

# 分子级：测试化合物的训练出现次数（混合物条数）
mol_cnt = collections.Counter()
for s in tr:
    for c in s:
        mol_cnt[c] += 1
te_mols = collections.Counter()
zero_appear = collections.Counter()
for s in te:
    for c in s:
        te_mols[c] += 1
        if mol_cnt[c] == 0:
            zero_appear[c] += 1
print(f'\n测试集涉及化合物 {len(te_mols)} 个；训练中零出现的: {len(zero_appear)} 个 '
      f'(涉及 {sum(zero_appear.values())} 条测试样本)')
# 分子在训练中的出现次数分布（测试涉及的）
cnts = [mol_cnt[c] for c in te_mols]
import numpy as np
print(f'测试化合物训练出现次数: median={np.median(cnts):.0f} '
      f'p25={np.percentile(cnts,25):.0f} p75={np.percentile(cnts,75):.0f} '
      f'min={min(cnts)} max={max(cnts)}')
