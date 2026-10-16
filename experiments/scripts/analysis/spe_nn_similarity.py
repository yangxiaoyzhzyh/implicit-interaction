# -*- coding: utf-8 -*-
"""spe_nn_similarity.py — SPE 测试化合物与训练化合物的 Morgan 最近邻相似度

实验 11 — Dataset audits（补材 S3）
路线 无（数据审计）
产出 results/analysis/spe_nn_similarity.out
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
        ids = list(eval(r['cmp_ids']))
    except Exception:
        continue
    recs.append(sorted(set(int(c) for c in ids)))
tr = [s for s in recs if len(s) <= 2]
te = [s for s in recs if len(s) >= 3]

mol, fp = {}, {}
for cid, smi in smiles.items():
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    mol[cid] = m
    fp[cid] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

tr_mols = sorted({c for s in tr for c in s if c in mol})
tr_seen_cnt = collections.Counter(c for s in tr for c in s)
te_mols = sorted({c for s in te for c in s if c in mol})

new_mols = [c for c in te_mols if tr_seen_cnt[c] == 0]
seen_mols = [c for c in te_mols if tr_seen_cnt[c] > 0]
print(f'测试化合物 {len(te_mols)}：已见 {len(seen_mols)}，训练零出现(新) {len(new_mols)}')


def nn_tanimoto(cid):
    best = 0.0
    for t in tr_mols:
        sim = DataStructs.TanimotoSimilarity(fp[cid], fp[t])
        if sim > best:
            best = sim
    return best


for name, group in [('已见分子', seen_mols), ('新分子(零出现)', new_mols)]:
    sims = [nn_tanimoto(c) for c in group]
    s = np.array(sims)
    print(f'{name}: n={len(s)} 最近邻Tanimoto mean={s.mean():.3f} '
          f'median={np.median(s):.3f} | >0.5: {(s>0.5).mean()*100:.0f}% '
          f'| >0.7: {(s>0.7).mean()*100:.0f}% | <0.3: {(s<0.3).mean()*100:.0f}%')

# 各举一个新分子的最近邻
for c in new_mols[:3]:
    best_t, best_s = None, -1
    for t in tr_mols:
        sim = DataStructs.TanimotoSimilarity(fp[c], fp[t])
        if sim > best_s:
            best_s, best_t = sim, t
    print(f'  新分子 {c}: nn_sim={best_s:.3f}\n    新: {smiles[c][:80]}\n    邻: {smiles[best_t][:80]}')
