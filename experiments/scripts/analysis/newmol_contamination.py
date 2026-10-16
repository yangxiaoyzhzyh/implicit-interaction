# -*- coding: utf-8 -*-
"""newmol_contamination.py — 五任务外推测试集的新分子混杂度

实验 11 — Dataset audits（补材 S3）
路线 无（数据审计）
产出 results/analysis/newmol_contamination.out
"""
import os
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PAPER_V2 = _os.path.dirname(_os.path.dirname(_HERE))
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_PAPER_V2)))  # <repo> (three levels above experiments/scripts/<sub>)
import pandas as pd
import collections

BASE = os.path.join(REPO, 'literature', 'CheMixHub', 'datasets')
TASKS = [
    ('drug', 'drug-solubility', 'processed_DrugSolubilityData.csv', None, 'drug', 2),
    ('spe', 'polymer-electrolyte', 'processed_PolymerElectrolyteData.csv', None, 'standard', 2),
    ('iltk', 'ionic-liquids', 'processed_IlThermoData.csv', 'Electrical conductivity', 'standard', 2),
    ('ilteta', 'ionic-liquids', 'processed_IlThermoData.csv', 'Viscosity', 'standard', 2),
    ('nist', 'nist-logV', 'processed_NISTlogV.csv', None, 'standard', 1),
]

def comps(r, kind):
    try:
        if kind == 'drug':
            ids = list(eval(r['cmp_ids_solvent'])) + list(eval(r['cmp_ids_drug']))
        else:
            ids = list(eval(r['cmp_ids']))
        return sorted(set(int(c) for c in ids))
    except Exception:
        return None

for key, d, csvf, prop, kind, tm in TASKS:
    pdir = os.path.join(BASE, d, 'processed_data')
    df = pd.read_csv(os.path.join(pdir, csvf))
    if prop is not None and 'property' in df.columns:
        df = df[df['property'] == prop].reset_index(drop=True)
    recs = []
    for _, r in df.iterrows():
        s = comps(r, kind)
        if s:
            recs.append(s)
    tr = [s for s in recs if len(s) <= tm]
    te = [s for s in recs if len(s) > tm]
    mol = collections.Counter(c for s in tr for c in s)
    seen = sum(1 for s in te if all(mol.get(c, 0) > 0 for c in s))
    byk = collections.Counter(len(s) for s in te)
    new_byk = collections.Counter()
    for s in te:
        if any(mol.get(c, 0) == 0 for c in s):
            new_byk[len(s)] += 1
    detail = ', '.join(f'K={k}: 新分子 {new_byk.get(k,0)}/{byk[k]}' for k in sorted(byk))
    print(f'[{key}] train {len(tr)} | test {len(te)}: 纯已见分子 {seen} ({seen/len(te)*100:.0f}%) | '
          f'含新分子 {len(te)-seen} ({(len(te)-seen)/len(te)*100:.0f}%) | {detail}')
