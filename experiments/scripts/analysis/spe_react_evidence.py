# -*- coding: utf-8 -*-
"""spe_react_evidence.py — 检验 SPE 的「局部结构迁移」假设

实验 11 — Dataset audits（补材 S3）
路线 无（数据审计）
产出 results/analysis/spe_react_evidence.out
1) SPE 化合物含盐/离子结构？（化学前提：A+B→A+ + B- 的解离/缔合）
2) 二元数据中非可加偏差多大？（ln κ_mix vs 摩尔分数加权的单组分 ln κ）
3) 测试组合的局部结构（Morgan r=1 位点）在训练二元对中的覆盖率（迁移通道）。
"""
import os
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER_V2 = _os.path.dirname(_os.path.dirname(_HERE))
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_PAPER_V2)))  # <repo> (three levels above experiments/scripts/<sub>)
import pandas as pd
import numpy as np
import collections
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

BASE = os.path.join(REPO, 'literature', 'CheMixHub', 'datasets', 'polymer-electrolyte', 'processed_data')
df = pd.read_csv(os.path.join(BASE, 'processed_PolymerElectrolyteData.csv'))
comp = pd.read_csv(os.path.join(BASE, 'compounds.csv'))
smiles = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}

recs = []
for _, r in df.iterrows():
    try:
        ids = list(eval(r['cmp_ids'])); mfs = list(eval(r['cmp_mole_fractions']))
    except Exception:
        continue
    if len(ids) != len(mfs):
        continue
    recs.append((sorted(set(int(c) for c in ids)),
                 [(int(c), float(w)) for c, w in zip(ids, mfs)],
                 float(r['value'])))

# ── 1) 化合物类型 ──
mols = {}
salt, polymer_like, other = [], [], []
for cid, smi in smiles.items():
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    mols[cid] = m
    has_metal = any(a.GetSymbol() in ('Li', 'Na', 'K', 'Mg') for a in m.GetAtoms())
    n_heavy = m.GetNumHeavyAtoms()
    if has_metal:
        salt.append(cid)
    elif n_heavy >= 20:
        polymer_like.append(cid)
    else:
        other.append(cid)
print(f'化合物总数(可解析): {len(mols)} | 含碱金属(盐): {len(salt)} | '
      f'大分子(>=20重原子,聚合物): {len(polymer_like)} | 其它小分子: {len(other)}')
for cid in salt[:4]:
    print(f'  盐例: {smiles[cid][:70]}')
for cid in polymer_like[:3]:
    print(f'  聚合物例: {smiles[cid][:70]}')

# ── 2) 二元数据的非可加偏差 ──
mono = {}
for s, comps, y in recs:
    if len(s) == 1 and s[0] in mono:
        continue
    if len(s) == 1:
        mono[s[0]] = y
print(f'纯化合物记录: {len(mono)} 个（去重后）')
devs = []
n_ok = 0
for s, comps, y in recs:
    if len(s) != 2:
        continue
    c1, w1 = comps[0]; c2, w2 = comps[1]
    if c1 in mono and c2 in mono:
        ideal = w1 * mono[c1] + w2 * mono[c2]
        devs.append(y - ideal)
        n_ok += 1
devs = np.array(devs)
print(f'二元且两组分均有纯组分参考: {n_ok} 条')
print(f'非可加偏差 (y - Σw·lnκ_i): mean={devs.mean():.4f} std={devs.std():.4f} '
      f'|abs|>0.1 的比例: {(np.abs(devs)>0.1).mean()*100:.0f}% '
      f'|>0.3: {(np.abs(devs)>0.3).mean()*100:.0f}%')

# ── 3) 局部结构（Morgan r=1 位点）覆盖：训练二元 vs 测试 ──
def sites(cid):
    fp = AllChem.GetMorganFingerprint(mols[cid], 1)
    return set(fp.GetNonzeroElements().keys())

train_pairs_sites = collections.defaultdict(set)   # 二元对 id → 位点并集
for s, comps, y in recs:
    if len(s) == 2:
        for c in s:
            if c in mols:
                train_pairs_sites[frozenset(s)] |= sites(c)
all_train_sites = set()
for v in train_pairs_sites.values():
    all_train_sites |= v

tr = [s for s, c, y in recs if len(s) <= 2]
te = [s for s, c, y in recs if len(s) >= 3]
byk = collections.defaultdict(lambda: [0, 0.0])
for s in te:
    ss = set()
    for c in s:
        if c in mols:
            ss |= sites(c)
    cov = len(ss & all_train_sites) / max(1, len(ss))
    byk[len(s)][0] += 1
    byk[len(s)][1] += cov
print(f'训练位点总数(二元对并集): {len(all_train_sites)}')
for k in sorted(byk):
    n, sc = byk[k]
    print(f'  test K={k}: n={n} 平均局部结构覆盖(对训练二元并集)={sc/n:.3f}')
