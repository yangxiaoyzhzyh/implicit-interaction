# -*- coding: utf-8 -*-
"""Compare two seeds of the same arm on the same test set.

v(im): per-molecule encodings  -> cosine (and linear CKA, which is invariant to
       the linear change of basis that a representation is only defined up to)
v(ex): CLS attention over components -> cosine, arg-max flips, mean |dw|

Usage:
    python compare_seeds.py --results results
    python compare_seeds.py --results results --model vex --pair 42 57
"""
import argparse
import glob
import os
import re

import numpy as np

LOGR = re.compile(r'test R\s*([+-]\d+\.\d+)')


def read_r(d):
    p = os.path.join(d, 'log.txt')
    if not os.path.exists(p):
        return None
    m = LOGR.search(open(p, encoding='utf-8', errors='replace').read())
    return float(m.group(1)) if m else None


def load(results):
    """{(model, seed): {'R': r, 'emb': (cids, E), 'attn': [(row, k, w, cids)]}}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(results, '*'))):
        if not os.path.isdir(d):
            continue
        m = re.match(r'^(vex|vim)(\d+)$', os.path.basename(d))
        if not m:
            continue
        model, seed = m.group(1), int(m.group(2))
        rec = {'R': read_r(d)}
        for p in glob.glob(os.path.join(d, '*_emb.npz')):
            z = np.load(p, allow_pickle=True)
            rec['emb'] = (z['cid'].astype(int), z['emb'].astype(np.float64))
        for p in glob.glob(os.path.join(d, '*_attn.npz')):
            z = np.load(p, allow_pickle=True)
            rec['attn'] = list(zip(z['row'].astype(int), z['k'].astype(int),
                                   z['attn'], z['cids']))
        out[(model, seed)] = rec
    return out


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def cka(X, Y):
    """Linear CKA on row-aligned representations."""
    K, L = X @ X.T, Y @ Y.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))


def pair_vim(A, B):
    ca, ea = A['emb']
    cb, eb = B['emb']
    common, ia, ib = np.intersect1d(ca, cb, return_indices=True)
    v = [cos(ea[i], eb[j]) for i, j in zip(ia, ib)]
    return dict(n=len(common), cos_med=float(np.median(v)), cos_mean=float(np.mean(v)),
                cka=cka(ea[ia], eb[ib]), worst=float(np.min(v)))


def concentration(attn):
    """How far the attention is from uniform: top-1 weight vs 1/K, and H/Hmax."""
    top1, ent = [], []
    for _r, k, w, _c in attn:
        w = np.asarray(w, float)
        if len(w) < 2:
            continue
        p = w / w.sum()
        top1.append(p.max() - 1.0 / len(p))
        ent.append(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return dict(top1=float(np.mean(top1)) if top1 else float('nan'),
                ent=float(np.mean(ent)) if ent else float('nan'))


def pair_vex(A, B):
    ra = {r: (w, c) for r, k, w, c in A['attn']}
    rb = {r: (w, c) for r, k, w, c in B['attn']}
    common = sorted(set(ra) & set(rb))
    v, flip = [], 0
    for r in common:
        wa, wb = np.asarray(ra[r][0], float), np.asarray(rb[r][0], float)
        n = min(len(wa), len(wb))
        wa, wb = wa[:n], wb[:n]
        v.append(cos(wa, wb))
        flip += int(wa.argmax() != wb.argmax())
    ca = concentration(A['attn'])
    return dict(n=len(common), cos_med=float(np.median(v)), cos_mean=float(np.mean(v)),
                flip=100.0 * flip / max(1, len(common)), worst=float(np.min(v)),
                top1=ca['top1'], ent=ca['ent'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--model', choices=['vex', 'vim'], default=None)
    ap.add_argument('--pair', type=int, nargs=2, default=None)
    args = ap.parse_args()

    data = load(args.results)
    models = sorted({m for m, _ in data})
    if args.model:
        models = [args.model]

    rows = []
    for model in models:
        seeds = sorted(s for m, s in data if m == model and data[(m, s)]['R'] is not None)
        if len(seeds) < 2:
            print('%s: 只有 %d 个可用 seed' % (model, len(seeds)))
            continue
        pairs = ([tuple(args.pair)] if args.pair
                 else [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]])
        for a, b in pairs:
            A, B = data[(model, a)], data[(model, b)]
            r = pair_vim(A, B) if model == 'vim' else pair_vex(A, B)
            rows.append(dict(model=model, seed_a=a, seed_b=b,
                             dR=abs(A['R'] - B['R']), **r))

    if not rows:
        raise SystemExit('没有可比较的 seed 对')

    print('\n%-5s %-11s %-7s %-7s %-7s %-7s %-8s' %
          ('arm', 'pair', 'Δtest R', 'cos中位', 'cos均值', 'cos最小', '其余'))
    for r in sorted(rows, key=lambda x: -x['dR']):
        if 'cka' in r:
            extra = 'CKA %.3f' % r['cka']
        else:
            extra = '翻转 %.1f%% | 集中度 %+.3f | H/Hmax %.2f' % (
                r['flip'], r['top1'], r['ent'])
        print('%-5s %-11s %-7.4f %-7.3f %-7.3f %-7.3f %-8s' %
              (r['model'], '%d vs %d' % (r['seed_a'], r['seed_b']),
               r['dR'], r['cos_med'], r['cos_mean'], r['worst'], extra))

    out = os.path.join(args.results, 'pair_summary.csv')
    keys = ['model', 'seed_a', 'seed_b', 'dR', 'n', 'cos_med', 'cos_mean', 'worst',
            'cka', 'flip', 'top1', 'ent']
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(keys) + '\n')
        for r in sorted(rows, key=lambda x: -x['dR']):
            f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    print('\n已写 %s（%d 对）' % (out, len(rows)))


if __name__ == '__main__':
    main()
