# -*- coding: utf-8 -*-
"""run_composition_xgb.py —— 组分数轴 RDKit+XGBoost 对照臂批量运行

三路并行；每任务写 01_composition/runs/xgb_<task>_composition.csv，
日志写 logs/xgb_<task>.log。协议与 vim/vex/frozen 对齐（同一 npz、val 并回训练、3 seed）。

    python scripts/run_composition_xgb.py --data-base <DATA> [--jobs 3]
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS = ['drug', 'spe', 'iltk', 'ilteta', 'nist',
         'miscible', 'miscible_hvap', 'miscible_density', 'mon']


def run(task, data_base):
    t0 = time.time()
    log = os.path.join(ROOT, 'logs', 'xgb_%s.log' % task)
    sf = os.path.join(ROOT, 'splits', '%s_composition.npz' % task)
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    if not os.path.exists(sf):
        return task, time.time() - t0, 1, 'MISSING %s' % sf
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, 'xgb_rdkit.py'),
             '--task', task, '--split-file', sf, '--data-base', data_base,
             '--out', os.path.join(ROOT, '01_composition', 'runs')],
            capture_output=True, text=True, encoding='utf-8')
        f.write((r.stdout or '') + '\n' + (r.stderr or ''))
    line = [l for l in (r.stdout or '').split('\n') if '| xgb |' in l]
    return task, time.time() - t0, r.returncode, line[0] if line else \
        (r.stderr or '')[-160:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--tasks', nargs='+', default=TASKS)
    args = ap.parse_args()

    t_all = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run, t, args.data_base): t for t in args.tasks}
        for f in as_completed(futs):
            task, dt, rc, line = f.result()
            print('[%5.0fs] %-16s rc=%d  %s' % (dt, task, rc, line.strip()),
                  flush=True)
    print('全部完成 %.1f 分钟' % ((time.time() - t_all) / 60))


if __name__ == '__main__':
    main()
