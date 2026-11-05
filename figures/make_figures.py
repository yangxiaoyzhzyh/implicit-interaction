# -*- coding: utf-8 -*-
"""Four main-text figures for Main1030 / SM1030.

Sources
-------
A  repro_v2/01_composition/summary.csv        (Table 1, ridge additive)
B  repro_v2/04_synthetic/runs/synth_add_pair_3seeds_summary.csv  (Table 2)
C  repro_v2/08_small_labels/summary.csv       (Table 3; mixpca grid only)
D  IlT kappa per-regime scores transcribed from the 12_iltk_dry runs, and
   SPE kappa from Supplementary Table S4 (see DATA_FIG_D).

    python make_figures.py            # writes FigureA-D .pdf and .png
"""
import csv
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPRO = os.path.join(ROOT, 'repro_v2')
OUT = HERE

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.labelsize': 7, 'legend.fontsize': 6.5, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'hatch.linewidth': 0.6,
})

C_IM, C_EX, C_ADD = '#1f6fb4', '#c94a4a', '#8a8a8a'
C_GRID, C_FILL = '#cccccc', '#eef2f7'

LAB = {'miscible_density': 'density', 'miscible_hvap': r'$\Delta H_{\rm vap}$',
       'miscible': r'$\Delta H_{\rm mix}$', 'mon': 'MON',
       'nist': r'NIST $\eta$', 'drug': r'drug ln$S$',
       'ilteta': r'IlT $\eta$', 'iltk': r'IlT $\kappa$', 'spe': r'SPE $\kappa$'}
ORDER = ['miscible_density', 'miscible_hvap', 'mon', 'nist', 'miscible',
         'drug', 'ilteta', 'iltk', 'spe']


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), dpi=400)
    plt.close(fig)
    print('wrote', name)


def load_summary(axis):
    with open(os.path.join(REPRO, axis, 'summary.csv'), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def clean(ax, axis='both'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if axis in ('both', 'y'):
        ax.grid(axis='y', color=C_GRID, lw=0.4, ls=':', zorder=0)
    if axis in ('both', 'x'):
        ax.grid(axis='x', color=C_GRID, lw=0.4, ls=':', zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# A -- composition extrapolation
# --------------------------------------------------------------------------- #
def fig_a():
    R = {}
    for r in load_summary('01_composition'):
        R.setdefault(r['task'], {})[r['model']] = (float(r['R_mean']),
                                                   float(r['R_sd']))
    tasks = sorted(ORDER, key=lambda t: R[t]['additive'][0])
    y = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(3.7, 2.75))
    for i, t in enumerate(tasks):
        add = R[t]['additive'][0]
        im, im_sd = R[t]['v(im)']
        ex, ex_sd = R[t]['v(ex)']
        ax.plot([add, add], [y[i] - 0.16, y[i] + 0.16], color=C_ADD, lw=1.5,
                zorder=2)
        ax.errorbar(im, y[i], xerr=im_sd, fmt='o', ms=3.2, color=C_IM,
                    mfc=C_IM, mec='white', mew=0.4, ecolor=C_IM,
                    elinewidth=0.5, capsize=1.6, capthick=0.5, zorder=3)
        ax.errorbar(ex, y[i], xerr=ex_sd, fmt='o', ms=3.2, mfc='white',
                    mec=C_EX, mew=0.9, ecolor=C_EX, elinewidth=0.5,
                    capsize=1.6, capthick=0.5, zorder=3)

    for i in range(len(tasks) - 1):
        ax.axhline(i + 0.5, color='#dddddd', lw=0.5, ls=(0, (2, 2)),
                   zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([LAB[t] for t in tasks])
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.set_xlabel('Pearson $R$ under composition extrapolation')
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    clean(ax, 'x')

    h = [plt.Line2D([], [], marker='o', ls='none', color=C_IM, ms=3.4),
         plt.Line2D([], [], marker='o', ls='none', mfc='white', mec=C_EX,
                    mew=0.9, ms=3.4),
         plt.Line2D([], [], marker='|', ls='none', color=C_ADD, ms=7,
                    mew=1.6)]
    ax.legend(h, ['v(im)', 'v(ex)', 'additive'], loc='lower right',
              handletextpad=0.5, borderaxespad=0.4, labelspacing=0.3)
    save(fig, 'Figure1')


# --------------------------------------------------------------------------- #
# B -- synthetic additive target
# --------------------------------------------------------------------------- #
def fig_b():
    p = os.path.join(REPRO, '04_synthetic', 'runs',
                     'synth_add_pair_3seeds_summary.csv')
    rows = sorted(csv.DictReader(open(p, encoding='utf-8')),
                  key=lambda r: int(r['K']))
    K = [int(r['K']) for r in rows]
    im = np.array([float(r['vim_mean']) for r in rows])
    im_sd = np.array([float(r['vim_sd']) for r in rows])
    ex = np.array([float(r['vex_mean']) for r in rows])
    ex_sd = np.array([float(r['vex_sd']) for r in rows])

    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    x = np.arange(len(K))
    ax.fill_between(x, ex, im, color=C_FILL, zorder=0, label='_nolegend_')
    ax.errorbar(x, im, yerr=im_sd, color=C_IM, marker='o', ms=3.4, lw=1.2,
                capsize=1.6, elinewidth=0.6, zorder=3, label='v(im)')
    ax.errorbar(x, ex, yerr=ex_sd, color=C_EX, marker='s', ms=3.2, lw=1.2,
                mfc='white', mew=0.9, capsize=1.6, elinewidth=0.6, zorder=3,
                label='v(ex)')
    import matplotlib.patheffects as pe
    for i in range(len(K)):
        ax.annotate('%.3f' % im[i], (x[i], im[i] + 0.012), fontsize=5.6,
                    color=C_IM, ha='center', va='bottom', zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground='white')])
        ax.annotate('%.3f' % ex[i], (x[i], ex[i] - 0.014), fontsize=5.6,
                    color=C_EX, ha='center', va='top', zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground='white')])
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel('components in the test mixture, $K$')
    ax.set_ylabel('$R^2$ (exactly additive target)')
    ax.set_ylim(0.72, 1.07)
    ax.set_xlim(-0.35, len(K) - 0.65)
    clean(ax, 'y')
    ax.legend(loc='lower left', handletextpad=0.5, labelspacing=0.3,
              handlelength=1.6)
    save(fig, 'Figure3')


# --------------------------------------------------------------------------- #
# C -- transfer at 100 labels
# --------------------------------------------------------------------------- #
def fig_c():
    best = {}
    for r in load_summary('08_small_labels'):
        if not r['protocol'].startswith('100bin_mixpca'):
            continue
        key = (r['task'], r['feature'])
        best[key] = max(best.get(key, -9.0), float(r['R']))

    NAME = {'density': 'density', 'heat_vap': r'$\Delta H_{\rm vap}$',
            'logv': 'logV', 'nist_logv': r'NIST $\eta$',
            'ionic_visc': r'IlT $\eta$', 'he': r'$\Delta H_{\rm mix}$',
            'ionic_cond': r'IlT $\kappa$', 'mon': 'MON'}
    data = []
    for t, nm in NAME.items():
        enc = best[(t, 'enc')]
        des = max(best[(t, f)] for f in ('ecfp4', 'maccs', 'rdkit2d'))
        data.append((nm, des, enc))
    data.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, (nm, des, enc) in enumerate(data):
        ax.plot([des, enc], [i, i], color=C_GRID, lw=1.2, zorder=1)
        ax.scatter(des, i, s=20, facecolor='white', edgecolor=C_EX,
                   linewidth=0.9, zorder=3)
        ax.scatter(enc, i, s=20, color=C_IM, edgecolor='white',
                   linewidth=0.4, zorder=4)
        d = enc - des
        ax.annotate('%+.3f' % d, (1.045, i), fontsize=5.8, ha='right',
                    va='center', color=C_IM if d >= 0 else C_EX)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data])
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlim(0.50, 1.05)
    ax.set_xlabel('Pearson $R$ with 100 binary mixtures')
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    clean(ax, 'x')
    h = [plt.Line2D([], [], marker='o', ls='none', mfc='white', mec=C_EX,
                    mew=0.9, ms=3.4),
         plt.Line2D([], [], marker='o', ls='none', color=C_IM, ms=3.4)]
    ax.legend(h, ['best descriptor', 'borrowed encoders'], loc='upper left',
              handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3)
    save(fig, 'Figure4')


# --------------------------------------------------------------------------- #
# Figure 2 -- coverage: SPE kappa ceiling, IlT kappa regime
# --------------------------------------------------------------------------- #
# SPE kappa: original split and the relaxed-ceiling run (Supplementary Table S4)
SPE = {'subsets': ['default', 'raised ceiling'],
       'v(im)': [0.733, 0.911], 'v(im)_sd': [0.034, 0.023],
       'v(ex)': [0.756, 0.868], 'v(ex)_sd': [0.068, 0.022]}

# IlT kappa: component-count split vs official five-fold random split
ILTK = [('v(im)', C_IM, 0.732, 0.016, 0.929, 0.005),
        ('v(ex)', C_EX, 0.829, 0.046, 0.975, 0.001)]


def fig_d():
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35))
    w = 0.30

    # ---- (a) SPE kappa, ceiling raised ------------------------------------
    ax = axes[0]
    x = np.arange(len(SPE['subsets']))
    for j, (mod, col) in enumerate((('v(im)', C_IM), ('v(ex)', C_EX))):
        ax.bar(x + (j - 0.5) * w, SPE[mod], w, color=col, edgecolor='white',
               linewidth=0.5, zorder=3, yerr=SPE[mod + '_sd'],
               ecolor='#333333', error_kw=dict(elinewidth=0.5, capsize=1.2))
        for i in range(len(x)):
            top = SPE[mod][i] + SPE[mod + '_sd'][i] + 0.035
            ax.annotate('%.2f' % SPE[mod][i],
                        (x[i] + (j - 0.5) * w, top),
                        ha='center', fontsize=5.6, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(SPE['subsets'])
    ax.set_xlim(-0.62, len(x) - 0.38)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel(r'SPE $\kappa$', labelpad=3)
    ax.set_ylabel('Pearson $R$')
    clean(ax, 'y')
    ax.set_title('a  raising the training ceiling', loc='left')
    h = [plt.Rectangle((0, 0), 1, 1, color=C_IM),
         plt.Rectangle((0, 0), 1, 1, color=C_EX)]
    ax.legend(h, ['v(im)', 'v(ex)'], loc='upper left', handlelength=1.0,
              handletextpad=0.4, labelspacing=0.25, borderaxespad=0.35)

    # ---- (b) IlT kappa, grouped by route --------------------------------
    ax2 = axes[1]
    x2 = np.arange(len(ILTK))                       # one group per route
    for j, (lab, col, e, es, f, fs) in enumerate(ILTK):
        vals = [e, f]
        sds = [es, fs]
        shades = [dict(facecolor='white', edgecolor=col, hatch='///'),
                  dict(facecolor=col, edgecolor='white')]
        for i, (v, sd, sh) in enumerate(zip(vals, sds, shades)):
            ax2.bar(x2[j] + (i - 0.5) * w, v, w,
                    linewidth=0.5, zorder=3, yerr=sd, ecolor='#333333',
                    error_kw=dict(elinewidth=0.5, capsize=1.2), **sh)
            ax2.annotate('%.2f' % v,
                         (x2[j] + (i - 0.5) * w, v + sd + 0.035),
                         ha='center', fontsize=5.6, color=col)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([r[0] for r in ILTK])
    ax2.set_xlim(-0.62, len(x2) - 0.38)
    ax2.set_ylim(0, 1.16)
    ax2.set_xlabel(r'IlT $\kappa$', labelpad=3)
    clean(ax2, 'y')
    ax2.set_title('b  coverage restored on the same task', loc='left')
    h2 = [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#555555',
                        hatch='///', linewidth=0.5),
          plt.Rectangle((0, 0), 1, 1, facecolor='#555555', edgecolor='none')]
    ax2.legend(h2, ['component split', 'five-fold'], loc='upper center',
               ncol=2, handlelength=1.0, handletextpad=0.35,
               columnspacing=1.2, borderaxespad=0.15)
    save(fig, 'Figure2')


if __name__ == '__main__':
    fig_a()
    fig_b()
    fig_c()
    fig_d()
