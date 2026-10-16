# -*- coding: utf-8 -*-
"""summarize_fig1.py — Figure 1 / Table 1 的 3-seed 均值与标准差（ddof = 0）

实验 16 — Figure 1 / Table 1 汇总（无数据产出，输出到终端）
读取 results/ 下：hmix|hvap_{e1,e2}_traink2.csv、mon_routes_final_results.csv、tasks5_{e1,e2}_results.csv、spe_{e1,e2}_groups_results.csv
v(im) 臂取 e2 文件、v(ex) 臂取 e1 文件（沿用结果文件里的内部标签）
输出的 literal 供 figures/make_figures.py 的 EIGHT 表使用

从归档结果文件中取出每个任务两条路线的 per-seed R，输出 mean ± std，
供 make_figures.py 的 EIGHT 表与正文 Table 1 直接使用。

用法: python summarize_fig1.py [--results ../results]
"""
import argparse
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=DEFAULT)
a = ap.parse_args()
R = a.results


def rows(name):
    with open(os.path.join(R, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def sd(v):
    return st.pstdev(v)          # ddof = 0


def ms(v):
    return st.mean(v), sd(v)


def from_simple(name, task):
    """hmix/hvap 的 per-seed 结果：test_R 列。"""
    return [float(r['test_R']) for r in rows(name) if r['route'] == task]


def from_tasks5(name, task):
    """tasks5_*_results.csv：取 seed 行的 R。"""
    return [float(r['R']) for r in rows(name)
            if r['task'] == task and r['seed'] not in ('mean', 'std')]


def from_groups(name, task):
    """spe_e*_groups_results.csv：traink2_testk3 / all 组。"""
    return [float(r['R']) for r in rows(name)
            if r['group'] == 'all' and r['config'].startswith(task)]


def from_mon(name, route):
    return [float(r['R']) for r in rows(name)
            if r['route'] == route and r['group'] == 'all']


SPECS = [
    ('ΔHmix',   lambda r: from_simple('hmix_e2_traink2.csv', 'e2'),
                lambda r: from_simple('hmix_e1_traink2.csv', 'e1')),
    ('ΔHvap',   lambda r: from_simple('hvap_e2_traink2.csv', 'e2'),
                lambda r: from_simple('hvap_e1_traink2.csv', 'e1')),
    ('MON',     lambda r: from_mon('mon_routes_final_results.csv', 'e2'),
                lambda r: from_mon('mon_routes_final_results.csv', 'e1')),
    ('IlT η',   lambda r: from_tasks5('tasks5_e2_results.csv', 'ilteta'),
                lambda r: from_tasks5('tasks5_e1_results.csv', 'ilteta')),
    ('IlT κ',   lambda r: from_tasks5('tasks5_e2_results.csv', 'iltk'),
                lambda r: from_tasks5('tasks5_e1_results.csv', 'iltk')),
    ('NIST η',  lambda r: from_tasks5('tasks5_e2_results.csv', 'nist'),
                lambda r: from_tasks5('tasks5_e1_results.csv', 'nist')),
    ('drug lnS', lambda r: from_tasks5('tasks5_e2_results.csv', 'drug'),
                lambda r: from_tasks5('tasks5_e1_results.csv', 'drug')),
    ('SPE κ',   lambda r: from_groups('spe_e2_groups_results.csv', 'spe_e2_traink2_testk3'),
                lambda r: from_groups('spe_e1_groups_results.csv', 'spe_e1_traink2_testk3')),
]

print('%-9s %22s %22s %9s' % ('task', 'v(im) mean ± sd', 'v(ex) mean ± sd', 'gap'))
print('-' * 68)
out = []
for name, fim, fex in SPECS:
    im, ex = fim(None), fex(None)
    mi, si = ms(im)
    mx, sx = ms(ex)
    out.append((name, mi, si, mx, sx))
    print('%-9s %10.4f ± %-10.4f %10.4f ± %-10.4f %+9.4f' %
          (name, mi, si, mx, sx, mi - mx))

print()
print('python literal for make_figures.py EIGHT '
      '(name, dK, omega, additive, v(im), v(ex), sd_im, sd_ex):')
DK = {'ΔHmix': 3, 'ΔHvap': 3, 'MON': 119, 'IlT η': 1, 'IlT κ': 1,
      'NIST η': 1, 'drug lnS': 1, 'SPE κ': 3}
OM = {'ΔHmix': 0.0, 'ΔHvap': 0.0, 'MON': 0.018, 'IlT η': 0.001, 'IlT κ': 0.001,
      'NIST η': 0.016, 'drug lnS': 0.400, 'SPE κ': 0.570}
ADD = {'ΔHmix': 0.830, 'ΔHvap': 0.978, 'MON': 0.905, 'IlT η': 0.007, 'IlT κ': -0.133,
       'NIST η': 0.878, 'drug lnS': 0.624, 'SPE κ': 0.001}
for name, mi, si, mx, sx in out:
    print("    ('%s', %s, %s, %s, %.4f, %.4f, %.4f, %.4f)," %
          (name, DK[name], OM[name], ADD[name], mi, mx, si, sx))
