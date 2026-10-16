# -*- coding: utf-8 -*-
"""make_molaxis_stats.py — 分子（身份）轴统计：ω 与最近邻相似度

实验 10 — Two-axis statistics（论文 Table 1、补材 S3 表 S9）
路线 无（数据统计）
产出 results/molaxis_stats.csv（--out 指定；默认只打印）
ω 的定义与计算在此文件内

ω 的口径（与正文、补充材料一致）
------------------------------------------------------------------
对每个**测试样本**：ω_i = 该样本中"训练集未出现过的组分"的摩尔分数之和。
对每个**任务**  ：ω   = median over the test samples that contain at least one
                        such component（即只在 ω_i > 0 的样本上取中位数）。

⚠ 不要用"全部测试样本"取中位数：SPE κ 约 60% 的测试样本不含新组分，
   那样得到的是 0，会把高 ω 任务误判成零压力任务。

划分协议
------------------------------------------------------------------
train = 组分数 ≤ train_max 的样本；test = 其余样本。
train_max 按原协议：drug / SPE / IlT κ / IlT η / MON 取 2，NIST logV 取 1。
组分数按**原始组分个数**计（不去重）——与两个实验脚本的划分一致。

分子（身份）轴列
------------------------------------------------------------------
n_train_mols : 训练集（train 侧）出现过的独特分子数
new_mols     : 测试集中训练零出现的独特分子数
nn_*         : 这些新分子到最近训练分子的 Morgan(r=2, 2048 bit) Tanimoto——
               nn_med 为中位数，nn_gt05_pct / nn_lt03_pct 为 >0.5 / <0.3 的占比。

数据来源：CheMixHub（`literature/CheMixHub/datasets/`）与 `atomformer_paper/repro/data/MON/`。

用法
------------------------------------------------------------------
    python make_molaxis_stats.py                     # 打印，并与归档 csv 对比
    python make_molaxis_stats.py --out <path.csv>    # 写出 csv
"""
import argparse
import collections
import csv
import os
import statistics as st

REPO = os.environ.get(
    'REPO_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))))
BASE = os.path.join(REPO, 'literature', 'CheMixHub', 'datasets')

# key, 目录, 文件, property 过滤, train_max
TASKS = [
    ('drug', 'drug-solubility', 'processed_DrugSolubilityData.csv', None, 2),
    ('SPE κ', 'polymer-electrolyte', 'processed_PolymerElectrolyteData.csv', None, 2),
    ('IlT κ', 'ionic-liquids', 'processed_IlThermoData.csv', 'Electrical conductivity', 2),
    ('IlT η', 'ionic-liquids', 'processed_IlThermoData.csv', 'Viscosity', 2),
    ('NIST η', 'nist-logV', 'processed_NISTlogV.csv', None, 1),
]
MON_DIR = os.path.join(REPO, 'atomformer_paper', 'repro', 'data', 'MON')

# 基准值（本脚本口径下的期望输出），用于自检
REFERENCE = {'drug': 0.400, 'SPE κ': 0.570, 'IlT κ': 0.001,
             'IlT η': 0.001, 'NIST η': 0.016}


def read_rows(path, prop):
    """返回 [(k_ids, w_ids, w_mfs), ...]。

    k_ids  — 决定组分数的分子（用于 train/test 划分）；
    w_ids / w_mfs — 用于计算 ω 的分子与其摩尔分数。

    drug 特例：组分数按"溶剂 + 药物"计（否则 K 恒为 1、没有测试样本），
    但药物的 `cmp_mole_fractions_drug` 恒为占位值 1.0，不参与归一化，
    因此 ω 只按溶剂计算。
    """
    rows = []
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if prop is not None and r.get('property') not in (None, prop):
                continue
            try:
                if r.get('cmp_ids_solvent') is not None:
                    sol = [int(float(x)) for x in eval(r['cmp_ids_solvent'])]
                    solw = [float(x) for x in eval(r['cmp_mole_fractions_solvent'])]
                    drug = [int(float(x)) for x in eval(r['cmp_ids_drug'])]
                    k_ids, w_ids, w_mfs = sol + drug, sol, solw
                else:
                    ids = [int(float(x)) for x in eval(r['cmp_ids'])]
                    mfs = [float(x) for x in eval(r['cmp_mole_fractions'])]
                    k_ids, w_ids, w_mfs = ids, ids, mfs
            except Exception:
                continue
            if k_ids and len(w_ids) == len(w_mfs):
                rows.append((k_ids, w_ids, w_mfs))
    return rows


def read_smiles(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            try:
                out[int(float(r['compound_id']))] = r['smiles']
            except Exception:
                continue
    return out


def make_fps(smiles_map, cids):
    """Morgan(r=2, 2048 bit) 指纹；rdkit 缺失或分子不可解析时返回空 dict。"""
    try:
        from rdkit import Chem, RDLogger, DataStructs      # noqa: F401
        from rdkit.Chem import AllChem
    except Exception:
        return {}, None
    RDLogger.DisableLog('rdApp.*')
    fps = {}
    for cid in cids:
        smi = smiles_map.get(cid)
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is not None:
            fps[cid] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    return fps, DataStructs


def nn_stats(new_mols, tr_mols, fps, DataStructs):
    """新分子到最近训练分子的最大 Tanimoto。"""
    sims = []
    for c in new_mols:
        if c not in fps:
            continue
        best = 0.0
        for t in tr_mols:
            if t not in fps:
                continue
            s = DataStructs.TanimotoSimilarity(fps[c], fps[t])
            if s > best:
                best = s
        sims.append(best)
    if not sims:
        return None, None, None
    return (round(st.median(sims), 3),
            round(100.0 * sum(1 for s in sims if s > 0.5) / len(sims), 1),
            round(100.0 * sum(1 for s in sims if s < 0.3) / len(sims), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='写出 csv 的路径（默认只打印）')
    args = ap.parse_args()

    specs = [(k, os.path.join(BASE, d, 'processed_data', f), p, tm)
             for k, d, f, p, tm in TASKS]
    specs.append(('MON', os.path.join(MON_DIR, 'processed_MON.csv'), None, 2))

    out_rows = []
    print('%-8s %8s %8s %8s %10s %10s %8s' %
          ('task', 'n_train', 'n_test', 'new_mol', 'unseen%', 'omega', 'ref'))
    for key, path, prop, tm in specs:
        rows = read_rows(path, prop)
        tr = [r for r in rows if len(r[0]) <= tm]
        te = [r for r in rows if len(r[0]) > tm]
        if not te:
            print('%-8s  (no test rows)' % key)
            continue

        seen = {c for _, w_ids, _ in tr for c in w_ids}
        omega = [sum(w for c, w in zip(w_ids, w_mfs) if c not in seen)
                 for _, w_ids, w_mfs in te]
        pos = [x for x in omega if x > 1e-9]

        n_new = len({c for _, w_ids, _ in te for c in w_ids} - seen)
        unseen_pct = 100.0 * len(pos) / len(te)
        omega_med = st.median(pos) if pos else 0.0

        n_train_mols = len({c for _, w_ids, _ in tr for c in w_ids})

        smiles_map = read_smiles(os.path.join(os.path.dirname(path), 'compounds.csv'))
        all_ids = sorted({c for _, w_ids, _ in tr + te for c in w_ids})
        fps, DataStructs = make_fps(smiles_map, all_ids)
        tr_mols = sorted({c for _, w_ids, _ in tr for c in w_ids})
        new_list = sorted({c for _, w_ids, _ in te for c in w_ids} - seen)
        nn_med, nn_gt, nn_lt = (nn_stats(new_list, tr_mols, fps, DataStructs)
                                if DataStructs is not None else (None, None, None))

        ref = REFERENCE.get(key)
        flag = '' if ref is None else (
            ' OK' if abs(omega_med - ref) < max(0.001, 0.05 * max(ref, 1e-9)) else ' DIFF')
        print('%-8s %8d %8d %8d %9.1f%% %10.4f %8.3f%s' %
              (key, len(tr), len(te), n_new, unseen_pct, omega_med, ref or 0, flag))

        out_rows.append({
            'task': key, 'n_train_mols': n_train_mols, 'n_test': len(te),
            'unseen_pct': round(unseen_pct, 1), 'omega_med': round(omega_med, 4),
            'new_mols': n_new, 'nn_med': nn_med,
            'nn_gt05_pct': nn_gt, 'nn_lt03_pct': nn_lt,
        })

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print('\nwritten: %s' % args.out)


if __name__ == '__main__':
    main()
