# -*- coding: utf-8 -*-
"""spec.py —— 任务注册表与数据载入（全仓库唯一来源）

键 -> CheMixHub 目录 / CSV / property / 条件列 / 组分轴训练上限。
engine.py 只认这里的键，所有外推脚本通过键引用任务，因此同一任务在任何
外推方式下用的都是同一份数据、同一套条件列处理、同一套组分解析。

注意：这里的 `train_max` 只对 `--split-by k` 有意义（组分轴）；新分子轴与
极值轴由划分文件/分位区间决定，不使用该字段。
"""
import ast
import os

import pandas as pd

# ---------------------------------------------------------------- 任务注册表
# 组分轴的五任务（drug / spe / iltk / ilteta / nist）
TASKS = {
    'drug': dict(name='drug ln(S)', dir='drug-solubility',
                 csv='processed_DrugSolubilityData.csv', prop='Log solubility',
                 kind='drug', conds=['Temperature, K'], train_max=2),
    'spe': dict(name='SPE ln(k)', dir='polymer-electrolyte',
                csv='processed_PolymerElectrolyteData.csv',
                prop='log Conductivity', kind='standard',
                conds=['Temperature, K'], train_max=2),
    'iltk': dict(name='IlT ln(k)', dir='ionic-liquids',
                 csv='processed_IlThermoData.csv',
                 prop='Electrical conductivity', kind='standard',
                 conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                 train_max=2),
    'ilteta': dict(name='IlT ln(eta)', dir='ionic-liquids',
                   csv='processed_IlThermoData.csv', prop='Viscosity',
                   kind='standard',
                   conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                   train_max=2),
    'nist': dict(name='NIST-full ln(eta)', dir='nist-logV',
                 csv='processed_NISTlogV.csv', prop=None, kind='standard',
                 conds=['T', 'P'], train_max=1),
}

# 其余任务（同一套 CheMixHub 目录，另一批 property 列或数据集）
EXTRA = {
    'miscible': dict(name=r'$\Delta H_{mix}$', dir='miscible-solvent',
                     csv='processed_MiscibleSolventData.csv',
                     prop='Enthalpy of mixing', kind='standard',
                     conds=[], train_max=2),
    'miscible_density': dict(name='density', dir='miscible-solvent',
                             csv='processed_MiscibleSolventData.csv',
                             prop='Density', kind='standard',
                             conds=[], train_max=2),
    'miscible_hvap': dict(name=r'$\Delta H_{vap}$', dir='miscible-solvent',
                          csv='processed_MiscibleSolventData.csv',
                          prop='Heat of vaporization', kind='standard',
                          conds=[], train_max=2),
    'logv': dict(name='logV', dir='logV', csv='processed_logV.csv',
                 prop='Log viscosity', kind='standard', conds=['T'],
                 train_max=2),
    'mon': dict(name='MON', dir='MON', csv='processed_MON.csv',
                prop='Motor octane number', kind='standard', conds=[],
                train_max=2),
}

ALL = dict(TASKS, **EXTRA)


def as_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def spec_of(key):
    if key in TASKS:
        return TASKS[key]
    return EXTRA[key]


def components(r, kind):
    """返回 [(compound_id, mole_fraction), ...]；长度不符返回 None。

    drug 任务把溶剂段与药物段拼接后归一化；其余任务直接取 cmp_ids。
    """
    if kind == 'drug':
        ids = list(eval(r['cmp_ids_solvent'])) + list(eval(r['cmp_ids_drug']))
        mfs = (list(eval(r['cmp_mole_fractions_solvent'])) +
               list(eval(r['cmp_mole_fractions_drug'])))
    else:
        ids = list(eval(r['cmp_ids']))
        mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    comps = list(zip(ids, mfs))
    if kind == 'drug':
        tot = sum(w for _, w in comps)
        return [(int(c), w / tot) for c, w in comps]
    return [(int(c), float(w)) for c, w in comps]


def load_records(key, data_base):
    """载入某任务的全部记录，返回 (df, rows, comps)。

    comps[i] 是第 i 条的组分 id 列表；摩尔分数在 engine 里另行解析。
    """
    tk = spec_of(key)
    droot = os.path.join(data_base, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, tk['csv']))
    if tk.get('prop') is not None and 'property' in df.columns:
        df = df[df['property'] == tk['prop']].reset_index(drop=True)
    rows, comps = [], []
    for i, r in df.iterrows():
        c = components(r, tk['kind'])
        if c is None:
            # 容错：部分数据集（如 logV 的纯组分行）cmp_ids 与摩尔分数长度
            # 不一致，此时按 cmp_ids 截断而非整条丢弃
            try:
                ids = [int(v) for v in as_list(r['cmp_ids'])]
            except Exception:
                continue
            if not ids:
                continue
            rows.append(i)
            comps.append(ids)
            continue
        rows.append(i)
        comps.append([cid for cid, _ in c])
    return df, rows, comps
