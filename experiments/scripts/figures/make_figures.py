"""make_figures.py — 生成论文与补材的数据图

出图脚本（实验 16 的下游）
读取 results/ 下的任务结果与合成结果
产出 figures/Figure*.png 与 .pdf（Figure 1 的 EIGHT 表与 analysis/summarize_fig1.py 的输出对齐）

All numbers are read from the archived result files under ``results/``; the eight-task
summary (Fig. 1) is transcribed from the outline table, which is the authoritative
source for those values.

Usage:
    python scripts/figures/make_figures.py            # all figures
    python scripts/figures/make_figures.py --only 1 2 # selected figures

Outputs PNG (300 dpi) and PDF into ``figures/``.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(PAPER, 'experiments', 'results')
SYNTH = os.path.join(PAPER, 'experiments', 'results',
                     'synth_routes_results.csv')
OUT = os.path.join(PAPER, 'figures')

C_VP = '#1f6fb4'      # per-element route
C_VW = '#d1495b'      # joint route
C_ADD = '#8a8a8a'     # additive baseline
C_LAM = ['#1f6fb4', '#e08a1e', '#7b3f9d']

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

TASK_LABEL = {
    'drug lnS': 'drug lnS',
    'SPE κ': r'SPE $\kappa$',
    'NIST η': r'NIST $\eta$',
    'MON': 'MON',
    'IlT η': r'IlT $\eta$',
    'IlT κ': r'IlT $\kappa$',
    'ΔHmix': r'$\Delta H_{\mathrm{mix}}$',
    'ΔHvap': r'$\Delta H_{\mathrm{vap}}$',
}

# --------------------------------------------------------------------------- #
# Fig. 1 — eight chemical tasks on the two axes, and the two routes' performance
# --------------------------------------------------------------------------- #
# task, dK (composition-axis pressure = max test K - max train K), omega,
# additive-baseline score a, v(im), v(ex), sd(v(im)), sd(v(ex))   [sd = 3-seed, ddof = 0].
# Identity-axis pressure is the composite omega * (1 - a).
EIGHT = [
    ('drug lnS', 1, 0.400, 0.624, 0.7220, 0.6189, 0.0138, 0.0381),
    ('SPE κ', 3, 0.570, 0.001, 0.4565, 0.6497, 0.0363, 0.1061),
    ('NIST η', 1, 0.016, 0.878, 0.8760, 0.7207, 0.0010, 0.0433),
    ('MON', 119, 0.018, 0.905, 0.9499, 0.8339, 0.0083, 0.0372),
    ('IlT η', 1, 0.001, 0.007, 0.8431, 0.8110, 0.0018, 0.0222),
    ('IlT κ', 1, 0.001, -0.133, 0.7289, 0.7443, 0.0142, 0.0675),
    ('ΔHmix', 3, 0.0, 0.830, 0.9036, 0.7268, 0.0014, 0.0140),
    ('ΔHvap', 3, 0.0, 0.978, 0.9965, 0.9799, 0.0008, 0.0028),
]

OM_FLOOR = 5e-4          # log-axis floor for zero identity-axis pressure
# points sharing a ΔK are spread horizontally for legibility (ΔHmix/ΔHvap coincide exactly)
DK_JITTER = {'ΔHvap': 1.30, 'IlT η': 0.58, 'NIST η': 1.65}

# manual label offsets (dx, dy in points, horizontal alignment) to avoid overlap
LABEL_POS = {
    'drug lnS': (9, -3, 'left'),
    'SPE κ': (9, 1, 'left'),
    'NIST η': (9, -2, 'left'),
    'MON': (-4, -13, 'center'),
    'IlT η': (0, -11, 'center'),
    'IlT κ': (-1, 9, 'center'),
    'ΔHmix': (-11, 3, 'right'),
    'ΔHvap': (10, 3, 'left'),
}


def fig1():
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0],
                          height_ratios=[1.0, 1.15], hspace=0.12, wspace=0.32)
    ax1a = fig.add_subplot(gs[0, 0])                 # identity axis, upper band
    ax1b = fig.add_subplot(gs[1, 0], sharex=ax1a)    # identity axis, lower band
    ax2 = fig.add_subplot(gs[:, 1])

    # (A) position on the two axes; colour = which route leads
    for name, dk, om, add, vp, vw, sp, sw in EIGHT:
        lead = C_VP if vp > vw else C_VW
        edge = 'none'
        if abs(vp - vw) < sp + sw:       # within the combined seed spread
            lead, edge = 'white', C_ADD
        xi = dk * DK_JITTER.get(name, 1.0)
        yi = om * (1.0 - add)
        ax = ax1a if yi > 0.01 else ax1b
        ax.scatter([xi], [yi], s=52, facecolor=lead,
                   edgecolor=edge if edge != 'none' else lead,
                   linewidth=1.0, zorder=3)
        dx, dy, ha = LABEL_POS[name]
        ax.annotate(TASK_LABEL[name], (xi, yi),
                    textcoords='offset points', xytext=(dx, dy),
                    ha=ha, fontsize=7)

    for ax in (ax1a, ax1b):
        ax.set_xscale('log')
        ax.set_xlim(0.32, 220)
    ax1a.set_ylim(0, 0.65)
    ax1b.set_ylim(0, 0.0024)
    ax1a.set_yticks([0, 0.2, 0.4, 0.6])
    ax1b.set_yticks([0, 0.001, 0.002])
    ax1a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1a.spines['bottom'].set_visible(False)
    ax1b.spines['top'].set_visible(False)
    # axis-break marks (left edge only)
    d = 0.016
    for a, ys in ((ax1a, (-d, +d)), (ax1b, (1 - d, 1 + d))):
        kw = dict(transform=a.transAxes, color='k', clip_on=False, lw=0.8)
        a.plot((-2 * d, 0), ys, **kw)

    ax1b.set_xlabel(r'composition-axis pressure   ($\Delta K$, components gained)')
    fig.text(0.005, 0.52, r'identity-axis pressure   ($\omega\,(1-a)$)',
             rotation=90, va='center', ha='left', fontsize=8)
    ax1a.set_title('a  Where the eight tasks sit', loc='left')
    handles = [
        plt.Line2D([], [], marker='o', ls='', ms=6, mfc=C_VP, mec=C_VP, label='v(im) leads'),
        plt.Line2D([], [], marker='o', ls='', ms=6, mfc=C_VW, mec=C_VW, label='v(ex) leads'),
        plt.Line2D([], [], marker='o', ls='', ms=6, mfc='white', mec=C_ADD, label='within spread'),
    ]
    ax1a.legend(handles=handles, loc='upper right', frameon=False, handletextpad=0.3)

    # (B) paired performance, grouped horizontal bars
    order = sorted(EIGHT, key=lambda t: t[4] - t[5])
    y = np.arange(len(order))[::-1].astype(float)
    h = 0.34                                           # bar thickness
    for yi, (name, dk, om, add, vp, vw, sp, sw) in zip(y, order):
        first = (yi == y[-1])
        ax2.barh(yi - h / 2, vp, height=h, color=C_VP, zorder=3,
                 label='v(im)' if first else None)
        ax2.barh(yi + h / 2, vw, height=h, color=C_VW, zorder=3,
                 label='v(ex)' if first else None)
        ax2.errorbar([vw], [yi + h / 2], xerr=[sw], fmt='none', ecolor='k',
                     elinewidth=0.35, capsize=1.5, capthick=0.35, zorder=4)
        ax2.errorbar([vp], [yi - h / 2], xerr=[sp], fmt='none', ecolor='k',
                     elinewidth=0.35, capsize=1.5, capthick=0.35, zorder=4)
    ax2.set_yticks(y)
    ax2.set_yticklabels([TASK_LABEL[t[0]] for t in order])
    ax2.set_xlabel('correlation (Pearson $r$)')
    ax2.set_xlim(0, 1.06)
    ax2.set_ylim(-0.78, len(order) - 0.30)
    ax2.grid(axis='x', lw=0.4, color='0.85')
    ax2.set_axisbelow(True)
    ax2.set_title('b  Paired performance, same protocol', loc='left')
    ax2.legend(loc='upper right', frameon=False, handletextpad=0.3)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.185, wspace=0.34)
    save(fig, 'Figure1')


# --------------------------------------------------------------------------- #
# Fig. 2 — synthetic tasks: the two axes isolated
# --------------------------------------------------------------------------- #
def _synth_rows():
    d = defaultdict(dict)
    with open(SYNTH, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d[(r['family'], float(r['lambda']), int(r['K']))] = (
                float(r['vp_r2']), float(r['vw_r2']))  # vp_r2 = v(im), vw_r2 = v(ex)
    return d


def fig2():
    d = _synth_rows()
    fams = [('add_pair', 'a  additive + pairwise interaction'),
            ('add_hidden', 'b  additive, hidden nonlinearity'),
            ('pair_hidden', 'c  purely pairwise (adversarial)')]
    lams = [0.0, 0.5, 2.0]
    Ks = [3, 5, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)

    for ax, (fam, title) in zip(axes, fams):
        ls = lams if fam == 'add_pair' else [None]
        for li, lam in enumerate(ls):
            c = C_LAM[li] if fam == 'add_pair' else C_VP
            key = (fam, lam if lam is not None else 0.0)
            vp = [d[(key[0], key[1], k)][0] for k in Ks]
            vw = [d[(key[0], key[1], k)][1] for k in Ks]
            lab = rf'$\lambda$={lam}' if fam == 'add_pair' else None
            ax.plot(Ks, vp, '-o', ms=3, lw=1.2, color=c,
                    label=(lab + ' , v(im)') if lab else 'v(im)')
            ax.plot(Ks, vw, '--s', ms=3, lw=1.2, color=c,
                    label=(lab + ' , v(ex)') if lab else 'v(ex)')
        ax.axhline(0, color=C_ADD, lw=0.6, ls=':')
        ax.set_title(title, loc='left')
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel('test composition count  K')
        if fam == 'pair_hidden':
            ax.set_ylim(-7.5, 1.2)
        else:
            ax.set_ylim(-0.5, 1.1)
    axes[0].set_ylabel(r'$R^2$')
    axes[0].legend(loc='lower left', frameon=False, handletextpad=0.4,
                   labelspacing=0.25, columnspacing=0.8, ncol=2)
    axes[1].legend(loc='lower left', frameon=False, handletextpad=0.4)
    axes[2].legend(loc='lower left', frameon=False, handletextpad=0.4)

    fig.tight_layout(w_pad=1.0)
    save(fig, 'FigureS4')


# --------------------------------------------------------------------------- #
# Fig. S1 — composition-axis profile on MON
# --------------------------------------------------------------------------- #
# Transcribed from the archived MON analysis (split by test composition count).
# Both columns reproduce exactly from results/mon_routes_pred.csv: the first is Pearson r
# computed within the bin (NOT R2), the second is MAE.
# Verified by _unused/scripts/analysis/check_mon_profile_metric.py (已归档).
# (bin label, n, v(im) r, v(ex) r, v(im) MAE, v(ex) MAE)
MON_PROFILE = [
    ('3', 486, 0.9543, 0.8873, 3.638, 6.311),
    ('4', 195, 0.9704, 0.8318, 3.288, 5.436),
    ('7–10', 24, 0.7868, 0.1391, 1.396, 5.062),
    ('21–40', 147, 0.9005, 0.1627, 2.645, 6.105),
    ('41–121', 42, 0.9087, -0.0022, 2.636, 4.988),
]


def fig_s1():
    labs = [b[0] for b in MON_PROFILE]
    x = np.arange(len(labs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))

    for i, (route, col, mk) in enumerate((('v(im)', C_VP, 'o'), ('v(ex)', C_VW, 's'))):
        r2 = [b[2 + i] for b in MON_PROFILE]
        mae = [b[4 + i] for b in MON_PROFILE]
        ax1.plot(x, r2, '-' + mk, ms=3.4, lw=1.2, color=col, label=route)
        ax2.plot(x, mae, '-' + mk, ms=3.4, lw=1.2, color=col, label=route)

    ax1.axhline(0, color=C_ADD, lw=0.6, ls=':')
    ax1.set_ylabel('Pearson $r$')
    ax1.set_title('a', loc='left')
    ax2.set_ylabel('MAE')
    ax2.set_title('b', loc='left')
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_xlabel('test composition count  K')
        ax.legend(frameon=False, handletextpad=0.3, loc='best')
    fig.tight_layout(w_pad=1.4)
    save(fig, 'FigureS1')


# --------------------------------------------------------------------------- #
# Fig. S2 — dose–response: composition count fixed, omega swept
# --------------------------------------------------------------------------- #
# SPE kappa, trained on <=2 components -> tested at K=3, split by whether every
# component of a test sample was present in training.
# spe_e2_groups_results.csv = v(im); spe_e1_groups_results.csv = v(ex).
SPE_ROUTE = {'spe_e2_groups_results.csv': 'v(im)', 'spe_e1_groups_results.csv': 'v(ex)'}


def _spe_split():
    out = {}
    for fn, route in SPE_ROUTE.items():
        seen, unseen = [], []
        with open(os.path.join(RES, fn), encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if 'traink2_testk3' not in r['config']:
                    continue
                if r['group'] == 'seen':
                    seen.append(float(r['R']))
                elif r['group'] == 'unseen':
                    unseen.append(float(r['R']))
        out[route] = (seen, unseen)
    return out


def fig_s2():
    d = _spe_split()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.5))

    x = np.arange(2)
    w = 0.34
    for i, (route, col) in enumerate((('v(im)', C_VP), ('v(ex)', C_VW))):
        m = [np.mean(d[route][j]) for j in (0, 1)]
        s = [np.std(d[route][j]) for j in (0, 1)]
        ax1.bar(x + (i - 0.5) * w, m, w, yerr=s, color=col, label=route,
                error_kw=dict(lw=0.7, capsize=2))
    ax1.set_xticks(x)
    ax1.set_xticklabels(['seen\n(all components\nknown)', 'unseen\n(contains new\ncomponents)'])
    ax1.set_ylabel(r'$R^2$')
    ax1.set_title(r'a  SPE $\kappa$, split within one test set', loc='left')
    ax1.legend(frameon=False, handletextpad=0.3)

    # (b) the three seeds on the unseen subset
    for i, (route, col, mk) in enumerate((('v(im)', C_VP, 'o'), ('v(ex)', C_VW, 's'))):
        vals = d[route][1]
        ax2.hlines(np.mean(vals), i - 0.22, i + 0.22, color=col, lw=1.4, zorder=2)
        ax2.scatter([i] * len(vals), vals, s=26, color=col, marker=mk, zorder=3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['v(im)', 'v(ex)'])
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylabel(r'$R^2$, unseen subset')
    ax2.set_title('b  Seed-to-seed behaviour', loc='left')

    fig.tight_layout(w_pad=1.6)
    save(fig, 'FigureS2')


# --------------------------------------------------------------------------- #
# Fig. S3 — multi-document reasoning
# --------------------------------------------------------------------------- #
MD_NAME = {'加性基准': 'additive (mean embedding)', '加性基准 ': 'additive (mean embedding)',
           'v(p)': 'v(im)', 'v(p)-small': 'v(im) (small)', 'v(w)': 'v(ex)'}  # 旧标签 → 统一命名（结果文件里仍用旧标签）
MD_ORDER = ['additive (mean embedding)', 'v(im) (small)', 'v(im)', 'v(ex)']
MD_STYLE = {'additive (mean embedding)': (C_ADD, '^', '--'),
            'v(im)': (C_VP, 'o', '-'),
            'v(im) (small)': ('#7fb2de', 'D', '-'),
            'v(ex)': (C_VW, 's', '-')}


def _md(path):
    d = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(RES, path), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            name = MD_NAME.get(r['model'], MD_NAME.get(r['model'].strip(), r['model'].strip()))
            d[name][int(r['k'])].append(float(r['acc']))
    return d


def fig_s4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, path, title in ((ax1, 'multidoc_set_results.csv', 'a  dominant-topic label'),
                            (ax2, 'multidoc_set_homo_results.csv', 'b  all-same-topic label')):
        p = os.path.join(RES, path)
        if not os.path.exists(p):
            continue
        d = _md(path)
        ks = sorted({k for v in d.values() for k in v})
        if title.startswith('b'):
            ks = [k for k in ks if k <= 60]      # S5.2 reports k ≤ 60 for this variant
        x = np.arange(len(ks))
        for name in MD_ORDER:
            if name not in d:
                continue
            col, mk, ls = MD_STYLE[name]
            m = [np.mean(d[name][k]) if d[name].get(k) else np.nan for k in ks]
            ax.plot(x, m, ls + mk, ms=3.4, lw=1.2, color=col, label=name)
        if title.startswith('b'):
            ax.axhline(0.5, color=C_ADD, lw=0.6, ls=':')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel('set size  k')
        ax.set_ylim(0.4, 1.02)
        ax.set_title(title, loc='left')
        ax.legend(frameon=False, handletextpad=0.3, loc='best')
    ax1.set_ylabel('accuracy')
    fig.tight_layout(w_pad=1.4)
    save(fig, 'FigureS3')


# --------------------------------------------------------------------------- #
def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)))
    plt.close(fig)
    print('  saved', name)


FIGS = {'1': fig1, '2': fig2, 's1': fig_s1, 's2': fig_s2, 's3': fig_s4}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', default=list(FIGS))
    args = ap.parse_args()
    for key in args.only:
        k = key.lower()
        if k in FIGS:
            print('drawing figure', k)
            FIGS[k]()
