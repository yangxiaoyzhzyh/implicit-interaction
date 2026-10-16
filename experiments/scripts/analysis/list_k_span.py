# -*- coding: utf-8 -*-
"""list_k_span.py — 打印各化学任务的组分数分布（train 上限 / test 上限）

实验 16 — 附表：任务与组分跨度的核对
路线 无（数据审计，无产出文件）
读取 CheMixHub 各任务的 processed_*.csv

用于 Figure 1a 的组分轴压力 ΔK = max(test K) − max(train K)。
"""
import ast
import collections
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

TASKS = [
    ('drug lnS', os.path.join(REPO, 'atomformer_paper', 'permol_paper', 'data', 'chemixhub_extrap', 'drug-solubility',
                              'processed_data', 'processed_DrugSolubilityData.csv'), None, 2),
    ('SPE κ', os.path.join(REPO, 'atomformer_paper', 'permol_paper', 'data', 'chemixhub_extrap', 'polymer-electrolyte',
                           'processed_data', 'processed_PolymerElectrolyteData.csv'), None, 2),
    ('IlT κ', os.path.join(REPO, 'atomformer_paper', 'permol_paper', 'data', 'chemixhub_extrap', 'ionic-liquids',
                           'processed_data', 'processed_IlThermoData.csv'), 'Electrical conductivity', 2),
    ('IlT η', os.path.join(REPO, 'atomformer_paper', 'permol_paper', 'data', 'chemixhub_extrap', 'ionic-liquids',
                           'processed_data', 'processed_IlThermoData.csv'), 'Viscosity', 2),
    ('NIST η', os.path.join(REPO, 'atomformer_paper', 'permol_paper', 'data', 'chemixhub_extrap', 'nist-logV',
                            'processed_data', 'processed_NISTlogV.csv'), None, 1),
    ('MON', os.path.join(REPO, 'atomformer_paper', 'repro', 'data', 'MON', 'processed_MON.csv'), None, 2),
    ('ΔHmix', os.path.join(REPO, 'literature', 'CheMixHub', 'datasets', 'miscible-solvent',
                           'processed_data', 'processed_MiscibleSolventData.csv'), 'Enthalpy of mixing', 2),
    ('ΔHvap', os.path.join(REPO, 'literature', 'CheMixHub', 'datasets', 'miscible-solvent',
                           'processed_data', 'processed_MiscibleSolventData.csv'), 'Heat of vaporization', 2),
]

print('%-10s %10s %10s %8s %s' % ('task', 'train maxK', 'test maxK', 'dK', 'train K / test K'))
for name, path, prop, tm in TASKS:
    if not os.path.exists(path):
        print('%-10s  MISSING %s' % (name, path))
        continue
    df = pd.read_csv(path)
    if prop is not None:
        df = df[df['property'] == prop]
    if 'cmp_ids_solvent' in df.columns:
        ks = df['cmp_ids_solvent'].apply(
            lambda s: len(ast.literal_eval(s))) + df['cmp_ids_drug'].apply(
            lambda s: len(ast.literal_eval(s)))
    else:
        ks = df['cmp_ids'].apply(lambda s: len(ast.literal_eval(s)))
    tr = ks[ks <= tm]
    te = ks[ks > tm]
    ctr = dict(sorted(collections.Counter(tr).items()))
    cte = dict(sorted(collections.Counter(te).items()))
    print('%-10s %10d %10d %8d %s / %s' % (
        name, max(tr) if len(tr) else 0, max(te) if len(te) else 0,
        (max(te) - max(tr)) if len(tr) and len(te) else 0,
        ctr, {k: cte[k] for k in list(cte)[:6]}))
