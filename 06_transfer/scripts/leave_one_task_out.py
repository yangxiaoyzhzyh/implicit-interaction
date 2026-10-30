# -*- coding: utf-8 -*-
"""leave_one_task_out.py —— 8 个任务轮流留一验证

对每个任务 T：
    特征 own      = 编码器_T 的混合物向量（用来做上限参照）
    特征 others   = 其余 7 个编码器的 concat（零数据迁移）
    特征 ecfp     = ECFP4 平均向量（通用基线）
    特征 others+ecfp = 二者拼接

    读出：岭回归（线性，5 折，官方划分）
    指标：Pearson R（越大越好）

判据：
    others 接近 own          → 组合能力可迁移（核心命题）
    others > ecfp            → 塑形编码器优于通用基线
"""
import argparse
import ast
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# PerMolModel / encoder 的位置：原位置在 paper_v2/.../per_molecule/，
# 归档后与脚本同目录（若同目录能找到则优先用同目录）
PM = os.path.abspath(os.path.join(
    HERE, '..', 'atomformer_paper', 'paper_v2', 'experiments',
    'scripts', 'per_molecule'))
if not os.path.exists(os.path.join(PM, 'train_task_permol.py')):
    PM = HERE
sys.path.insert(0, PM)

from train_task_permol import PerMolModel  # noqa: E402
from encoder import mol_to_graph  # noqa: E402
try:  # 归档自包含：fp_matrix.py 与原 eval_encoders 中的实现逐字一致
    from fp_matrix import fp_matrix  # noqa: E402
except ImportError:
    from eval_encoders import fp_matrix  # noqa: E402

def _find_ds():
    """CheMixHub 数据集根目录：环境变量 > 原位置 > 向上逐级搜索"""
    env = os.environ.get('CHEMIXHUB_ROOT')
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    cands = [os.path.join(HERE, '..', 'literature', 'CheMixHub', 'datasets')]
    for up in range(1, 7):
        cands.append(os.path.join(HERE, *(['..'] * up),
                                  'literature', 'CheMixHub', 'datasets'))
    for c in cands:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, 'MON')):
            return c
    return os.path.abspath(cands[0])


DS = _find_ds()


def _find_ckpt():
    """编码器权重目录：归档位置 > 原位置；也接受环境变量覆写"""
    env = os.environ.get('TRANSFER_CKPT')
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    cands = [
        os.path.join(HERE, '..', 'ckpt'),                  # 归档位置 06_transfer/ckpt
        os.path.join(HERE, '..', '..', 'task_encoders',
                     'checkpoints'),                       # 兼容旧布局
        os.path.join(HERE, '..', 'atomformer_paper', 'paper_v3', 'repro',
                     'task_encoders', 'checkpoints'),      # 原位置
    ]
    for c in cands:
        c = os.path.abspath(c)
        if os.path.isdir(c) and glob.glob(os.path.join(
                c, 'permol_*_full_all.pt')):
            return c
    return os.path.abspath(cands[0])


CKPT = _find_ckpt()

TASKS = {
    'mon': dict(dir='MON', csv='processed_MON.csv', prop=None,
                split='motor_octane_number_splits', conds=[]),
    'density': dict(dir='miscible-solvent',
                    csv='processed_MiscibleSolventData.csv',
                    prop='Density', split='density_splits', conds=[]),
    'heat_vap': dict(dir='miscible-solvent',
                     csv='processed_MiscibleSolventData.csv',
                     prop='Heat of vaporization',
                     split='heat_of_vaporization_splits', conds=[]),
    'he': dict(dir='miscible-solvent',
               csv='processed_MiscibleSolventData.csv',
               prop='Enthalpy of mixing',
               split='enthalpy_of_mixing_splits', conds=[]),
    'logv': dict(dir='logV', csv='processed_logV.csv', prop=None,
                 split='log_viscosity_splits', conds=['T']),
    'nist_logv': dict(dir='nist-logV', csv='processed_NISTlogV.csv',
                      prop=None, split='log_viscosity_splits',
                      conds=['T', 'P']),
    'ionic_cond': dict(dir='ionic-liquids',
                       csv='processed_IlThermoData.csv',
                       prop='Electrical conductivity',
                       split='electrical_conductivity_splits',
                       conds=['Temperature, K']),
    'ionic_visc': dict(dir='ionic-liquids',
                       csv='processed_IlThermoData.csv',
                       prop='Viscosity', split='viscosity_splits',
                       conds=['Temperature, K']),
}


def parse_ids(s):
    if isinstance(s, list):
        return [int(x) for x in s]
    return [int(float(x)) for x in ast.literal_eval(str(s))]


def parse_w(s):
    if isinstance(s, list):
        return [float(x) for x in s]
    return [float(x) for x in ast.literal_eval(str(s))]


def load_task(name):
    tk = TASKS[name]
    base = os.path.join(DS, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(base, tk['csv']))
    if tk['prop'] is not None:
        df = df[df['property'] == tk['prop']]
    df = df[df['cmp_ids'].notna()].reset_index(drop=True)

    cmp = pd.read_csv(os.path.join(base, 'compounds.csv'))
    cid2smi = {int(r['compound_id']): str(r['smiles'])
               for _, r in cmp.iterrows()}

    ids = [parse_ids(x) for x in df['cmp_ids']]
    ws = [parse_w(x) for x in df['cmp_mole_fractions']]
    y = df['value'].to_numpy(dtype=float)

    # 条件列
    cond = None
    if tk['conds']:
        cols = []
        for c in tk['conds']:
            if c in df.columns:
                cols.append(df[c].to_numpy(dtype=float))
        if cols:
            cond = np.column_stack(cols)

    return dict(df=df, ids=ids, ws=ws, y=y, cond=cond,
                cid2smi=cid2smi, base=base, split=tk['split'])


@torch.no_grad()
def mol_vectors(model, smiles, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(smiles), bs):
        chunk = smiles[i:i + bs]
        xs, eis, bas, mids, amfs, ok = [], [], [], [], [], []
        off = 0
        for j, s in enumerate(chunk):
            g = mol_to_graph(s)
            if g is None:
                continue
            x = torch.tensor(g['x'], dtype=torch.float32)
            ei = torch.tensor(g['ei'], dtype=torch.long)
            x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
            n = x.shape[0]
            xs.append(x)
            eis.append(ei + off)
            bas.append(torch.full((n,), len(ok), dtype=torch.long))
            mids.append(torch.zeros(n, dtype=torch.long))
            amfs.append(torch.ones(n, dtype=torch.float32))
            off += n
            ok.append(j)
        if not xs:
            out.append(np.zeros((len(chunk), 256), np.float32))
            continue
        x = torch.cat(xs).to(dev)
        ei = torch.cat(eis, 1).to(dev)
        ba = torch.cat(bas).to(dev)
        mi = torch.cat(mids).to(dev)
        am = torch.cat(amfs).to(dev)
        h = model.pad(x)
        if model.variant != 'tfm_only':
            for c in model.convs:
                h = c(h, ei)
        h = model.proj(h)
        v, _, _ = model._mol_vectors(h, ba, mi, am)
        v = v.cpu().numpy()
        full = np.zeros((len(chunk), v.shape[1]), np.float32)
        for k, j in enumerate(ok):
            full[j] = v[k]
        out.append(full)
    return np.concatenate(out, 0)


def load_model(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    meta = ck if isinstance(ck, dict) else {}
    dim = meta.get('dim', 256)
    n_cond = max(0, sd['regressor.0.weight'].shape[1] - dim)
    m = PerMolModel(dim=dim, nhead=meta.get('nhead', 4),
                    n_layers=meta.get('n_layers', 2),
                    variant=meta.get('variant', 'full'),
                    combine=meta.get('combine', 'wmean'), n_cond=n_cond)
    m.load_state_dict(sd, strict=False)
    return m.to(dev).eval()


def mix_vec(V, cid2i, ids, ws):
    """混合物向量 = Σ w_i v_i / Σ w_i"""
    out = np.zeros((len(ids), V.shape[1]), np.float32)
    for i, (ci, wi) in enumerate(zip(ids, ws)):
        v = np.zeros(V.shape[1], np.float32)
        tot = 0.0
        for c, w in zip(ci, wi):
            j = cid2i.get(int(c))
            if j is None:
                continue
            v += w * V[j]
            tot += w
        if tot > 0:
            out[i] = v / tot
    return out


def cv_ridge(X, y, split_dir, n_fold=5, pca_dim=128):
    """岭回归 5 折。高维特征先 PCA 降维（无监督，不引入标签）。

    1792 维 × 9 万样本时，直接 RidgeCV(20 alphas) 要算很久；
    PCA 到 128 维后快两个数量级，同时抑制过拟合。
    """
    from sklearn.decomposition import PCA
    rs = []
    for fi in range(n_fold):
        p = os.path.join(split_dir, 'kfold_split_%d.safetensors' % fi)
        if not os.path.exists(p):
            continue
        from safetensors.numpy import load_file
        d = load_file(p)
        tr = d['train_indices']
        te = d['test_indices']
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        if X.shape[1] > pca_dim and len(tr) > pca_dim * 4:
            k = min(pca_dim, X.shape[1], len(tr) - 1)
            pca = PCA(n_components=k, random_state=0).fit(Xtr)
            Xtr = pca.transform(Xtr)
            Xte = pca.transform(Xte)
        m = RidgeCV(alphas=np.logspace(-1, 4, 8))
        m.fit(Xtr, y[tr])
        pred = m.predict(Xte)
        rs.append(np.corrcoef(pred, y[te])[0, 1])
    return (float(np.mean(rs)), float(np.std(rs))) if rs else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--out', default='out/leave_one_out.json')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = {os.path.basename(f).replace('permol_', '').replace(
        '_full_all.pt', ''): f
        for f in glob.glob(os.path.join(CKPT, 'permol_*_full_all.pt'))}
    print('编码器 %d 个: %s' % (len(files), ','.join(sorted(files))))

    # 每个编码器的分子向量（按需提取，全部任务的分子并集）
    task_data = {}
    all_smi = set()
    for t in args.tasks:
        d = load_task(t)
        task_data[t] = d
        all_smi |= set(d['cid2smi'].values())
    all_smi = sorted(all_smi)
    sidx = {s: i for i, s in enumerate(all_smi)}
    print('探针分子并集 %d' % len(all_smi))

    print()
    # 优先读缓存（cache_encoders.py 生成）
    cache_p = os.path.join(HERE, 'out', 'encoder_cache.npz')
    ENC = {}
    ecfp = None
    if os.path.exists(cache_p):
        ck = np.load(cache_p, allow_pickle=False)
        cached_smi = [str(s) for s in ck['smiles']]
        if cached_smi == all_smi:
            print('=== 读缓存 %s ===' % os.path.basename(cache_p))
            for t in sorted(files):
                key = '%s_vec' % t
                if key in ck.files:
                    ENC[t] = ck[key]
            ecfp = ck['ecfp4'] if 'ecfp4' in ck.files else None
            print('  编码器 %s' % ','.join(sorted(ENC)))
        else:
            print('  缓存分子顺序不一致（%d vs %d），重新提取'
                  % (len(cached_smi), len(all_smi)))

    if not ENC:
        print('=== 提取编码器向量 ===')
        for t in sorted(files):
            print('  %-12s' % t, end='', flush=True)
            m = load_model(files[t], dev)
            ENC[t] = mol_vectors(m, all_smi, dev)
            print(' ok %s' % (ENC[t].shape,), flush=True)
        ecfp = None

    if ecfp is None:
        ecfp = fp_matrix(all_smi, 'ecfp4')

    # 额外通用基线（RDKit2D 在 MON 上远强于 ECFP4，必须一起比）
    GEN = {}
    for k in ['rdkit2d', 'maccs']:
        try:
            GEN[k] = fp_matrix(all_smi, k)
            print('  通用基线 %s %s' % (k, GEN[k].shape))
        except Exception as e:
            print('  跳过 %s (%s)' % (k, type(e).__name__))

    print()
    print('=== Leave-one-task-out ===')
    print('%-11s %6s %6s %6s %6s %6s %6s %7s'
          % ('目标', 'own', 'others', 'ECFP4', 'RDKit', 'MACCS', 'o+e',
             '样本'))
    print('-' * 68)
    RES = {}
    for t in args.tasks:
        d = task_data[t]
        cid2i = {c: sidx[s] for c, s in d['cid2smi'].items() if s in sidx}
        split_dir = os.path.join(d['base'], d['split'])

        # 目标任务的混合物向量
        Vown = mix_vec(ENC[t], cid2i, d['ids'], d['ws'])
        # 其余 7 个
        others = [k for k in sorted(ENC) if k != t]
        Voth = np.concatenate(
            [mix_vec(ENC[k], cid2i, d['ids'], d['ws']) for k in others],
            axis=1)
        Vecfp = mix_vec(ecfp, cid2i, d['ids'], d['ws'])
        Vrdk = (mix_vec(GEN['rdkit2d'], cid2i, d['ids'], d['ws'])
                if 'rdkit2d' in GEN else None)
        Vmcs = (mix_vec(GEN['maccs'], cid2i, d['ids'], d['ws'])
                if 'maccs' in GEN else None)

        base_x = d['cond'] if d['cond'] is not None else None

        def run(X):
            if X is None:
                return np.nan, np.nan
            if base_x is not None:
                X = np.concatenate([X, base_x], axis=1)
            return cv_ridge(X, d['y'], split_dir)

        r_own, _ = run(Vown)
        r_oth, s_oth = run(Voth)
        r_fp, _ = run(Vecfp)
        r_rdk, _ = run(Vrdk)
        r_mcs, _ = run(Vmcs)
        r_both, _ = run(np.concatenate([Voth, Vecfp], axis=1))

        print('%-11s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %7d'
              % (t, r_own, r_oth, r_fp, r_rdk, r_mcs, r_both,
                 len(d['y'])), flush=True)
        RES[t] = dict(own=r_own, others=r_oth, others_sd=s_oth,
                      ecfp=r_fp, rdkit2d=r_rdk, maccs=r_mcs,
                      both=r_both, n=int(len(d['y'])))

    print()
    print('=== 汇总（均值）===')
    for k in ['own', 'others', 'ecfp', 'rdkit2d', 'maccs', 'both']:
        v = np.array([RES[t][k] for t in args.tasks])
        print('  %-9s %.3f' % (k, v.mean()))
    own = np.array([RES[t]['own'] for t in args.tasks])
    oth = np.array([RES[t]['others'] for t in args.tasks])
    print('  own - others = %.3f' % (own - oth).mean())

    json.dump(RES, open(os.path.join(HERE, args.out), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print()
    print('saved', args.out)


if __name__ == '__main__':
    main()
