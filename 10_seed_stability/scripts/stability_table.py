# -*- coding: utf-8 -*-
"""stability_table.py — rebuild table S11 from the stored attention files

For two seeds of the same arm, count how often the leading component of a mixture
changes. Rules (same as the table caption):

  * single-component mixtures are left out (their leading component cannot change);
  * a mixture is left out when either seed's leading component is within 0.05 of
    the runner-up, i.e. when the lead is not clear.

Both the last encoder layer and the rollout over both layers are reported.

Usage
-----
    python 10_seed_stability/scripts/stability_table.py \
        --attn-dir 10_seed_stability/attn \
        --out 10_seed_stability/runs/stability_table.csv

The attention files come from `--save-attn-all`, which writes
`<tag>_<model>_<split_label>_s<seed>_attn_{train,test}.npz`.
"""
import argparse
import csv
import glob
import os

import numpy as np


def load(pattern):
    """Return {row: (last_layer_attention, rollout_attention)}."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit('no file matches %s' % pattern)
    z = np.load(paths[0], allow_pickle=True)
    out = {}
    for r, a0, a1 in zip(z['row'], z['attn_l0'], z['attn_l1']):
        n = len(a1)
        a1 = np.asarray(a1, float)
        A0 = np.zeros((n + 1, n + 1))
        A1 = np.zeros((n + 1, n + 1))
        A0[0, 1:] = np.asarray(a0, float)
        A1[0, 1:] = np.asarray(a1, float)
        M = (np.eye(n + 1) + A0) @ (np.eye(n + 1) + A1)
        v = M[0, 1:]
        out[int(r)] = (a1, v / v.sum())
    return out


def compare(a, b, tol=0.05):
    """Return (n_kept, flips_last, flips_roll) for one pair of seeds."""
    kept = last = roll = 0
    for r in sorted(set(a) & set(b)):
        la, ra = a[r]
        lb, rb = b[r]
        n = min(len(la), len(lb))
        if n < 2:
            continue
        la, lb = la[:n], lb[:n]
        ra, rb = ra[:n], rb[:n]
        ga = np.sort(la)[-1] - np.sort(la)[-2]
        gb = np.sort(lb)[-1] - np.sort(lb)[-2]
        if ga <= tol or gb <= tol:
            continue
        kept += 1
        last += int(la.argmax() != lb.argmax())
        roll += int(ra.argmax() != rb.argmax())
    return kept, last, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attn-dir', default='10_seed_stability/attn')
    ap.add_argument('--pairs', nargs='+', default=['mon:42:61', 'spe:42:44'],
                    help='task:seedA:seedB')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = []
    print('%-5s %-6s %-6s %8s %8s %9s' %
          ('task', 'split', 'seeds', 'kept', 'last', 'rollout'))
    for spec in args.pairs:
        task, sa, sb = spec.split(':')
        for split in ('train', 'test'):
            pat = os.path.join(args.attn_dir,
                               '%s_vex_composition_s%%s_attn_%s.npz' % (task, split))
            a = load(pat % sa)
            b = load(pat % sb)
            kept, last, roll = compare(a, b)
            rows.append((task, split, '%s/%s' % (sa, sb), kept,
                         100.0 * last / kept, 100.0 * roll / kept))
            print('%-5s %-6s %-6s %8d %7.1f%% %8.1f%%' %
                  (task, split, '%s/%s' % (sa, sb), kept,
                   100.0 * last / kept, 100.0 * roll / kept))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task', 'split', 'seeds', 'n_clear_lead',
                        'flip_last_pct', 'flip_rollout_pct'])
            for r in rows:
                w.writerow([r[0], r[1], r[2], r[3], '%.1f' % r[4], '%.1f' % r[5]])
        print('wrote', args.out)


if __name__ == '__main__':
    main()
