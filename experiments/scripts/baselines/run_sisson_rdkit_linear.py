"""run_sisson_rdkit_linear.py — Sisson 74 类上的描述符-线性基线（外部对照）

实验 12 — Sisson olfactory task（补材 S3，表 S12 的 0.7511 一行）
路线 无（外部对照：RDKit 2D 描述符 + 摩尔分数加权和 + 线性 logistic）
产出 results/sisson/（与 olfactory/train_sisson_permol_cls.py 同目录）；logs/sisson_rdkit_linear2.log、.err

问题：Sisson 的测试分子与训练集交集为 0（50:50 molecule-level carving），
所以「分子身份线性系数」基线（Σ c_m·x_m）在它上面无定义（系数恒为 0）。
这里换一种线性：结构 → 贡献 由 RDKit2D 描述符的线性函数给出，对未见分子有定义。

特征  = Σ_i w_i · rdkit2d(M_i)     （Sisson 为等摩尔 1:1 对，即两分子描述符的平均）
模型  = Ridge（多输出回归）与 LogisticRegression（OvR 多标签）
指标  = macro-AUROC（与 v(im)/v(ex) 的 74 类口径一致）

用法：python run_sisson_rdkit_linear.py
"""
import os
import sys
import types
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))          # paper_v2/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PAPER, 'scripts', 'baselines'))

from probe_mon_extrap_rdkit2d import rdkit2d           # noqa: E402

ROOT = r'C:\Users\yangx\v3-mixture-token\sisson_repro'
DSET = os.path.join(ROOT, 'dataset')

# ---- PairData 注册（unpickle 需要） ----
try:
    import pairing.data  # noqa: F401
except Exception as e:
    import torch_geometric.data as gd

    class PairData(gd.Data):
        pass

    _pkg = types.ModuleType('pairing')
    _sub = types.ModuleType('pairing.data')
    _sub.PairData = PairData
    _pkg.data = _sub
    sys.modules['pairing'] = _pkg
    sys.modules['pairing.data'] = _sub
    print(f'pairing.data stub ({type(e).__name__})', flush=True)


def load(fn):
    d, _ = torch.load(os.path.join(DSET, fn), weights_only=False)
    return d


def macro_auc(Y, Z):
    """逐类 AUC 后取平均，跳过测试集中只含单一类别的列（照 train_sisson_permol_cls.py）。"""
    aucs = []
    for k in range(Y.shape[1]):
        y, z = Y[:, k], Z[:, k]
        pos, neg = int(y.sum()), int((1 - y).sum())
        if pos == 0 or neg == 0:
            continue
        order = np.argsort(z, kind='mergesort')
        ranks = np.empty(len(z), dtype=np.float64)
        ranks[order] = np.arange(1, len(z) + 1)
        zs = z[order]
        i = 0
        while i < len(zs):                       # 并列取平均秩
            j = i
            while j + 1 < len(zs) and zs[j + 1] == zs[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = np.arange(i + 1, j + 2).mean()
            i = j + 1
        pos_ranks = ranks[y == 1].sum()
        aucs.append((pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))
    return float(np.mean(aucs)), len(aucs)


def main():
    t0 = time.time()
    tr, te = load('trainpairs74.pt'), load('testpairs74.pt')
    print(f'train pairs {tr.y.size(0)} | test pairs {te.y.size(0)}', flush=True)

    # 分子级描述符缓存（两集分子不重叠，仍一次性算完）
    mols = sorted(set(tr.smiles_s) | set(tr.smiles_t) |
                  set(te.smiles_s) | set(te.smiles_t))
    print(f'唯一分子 {len(mols)}，开始算 RDKit2D 描述符 ...', flush=True)
    cache, dim = {}, None
    bad = 0
    for i, smi in enumerate(mols):
        v = rdkit2d(smi)
        if v is None:
            bad += 1
            continue
        v = np.nan_to_num(np.asarray(v, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        dim = len(v)
        cache[smi] = v
        if (i + 1) % 500 == 0:
            print(f'  desc {i + 1}/{len(mols)}', flush=True)
    print(f'描述符 {dim} 维，失败分子 {bad}，用时 {time.time() - t0:.0f}s', flush=True)

    def featurize(d):
        n = d.y.size(0)
        X = np.zeros((n, dim), dtype=np.float32)
        miss = 0
        for i in range(n):
            acc, cnt = None, 0
            for smi in (d.smiles_s[i], d.smiles_t[i]):
                v = cache.get(smi)
                if v is None:
                    continue
                acc = v.copy() if acc is None else acc + v
                cnt += 1
            if acc is None:
                miss += 1
                continue
            X[i] = acc / cnt                     # 等摩尔加权平均
        if miss:
            print(f'  [warn] {miss} 对无可用描述符')
        return X, np.asarray(d.y, dtype=np.float64)

    Xtr, Ytr = featurize(tr)
    Xte, Yte = featurize(te)
    print(f'feature shape train {Xtr.shape} / test {Xte.shape}', flush=True)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    # ---- 1) Ridge（多输出线性回归） ----
    from sklearn.linear_model import Ridge
    print('\n--- Ridge（多输出线性回归）---', flush=True)
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha, solver='svd').fit(Xtr_s, Ytr)
        Z = m.predict(Xte_s)
        auc, nk = macro_auc(Yte, Z)
        print(f'  alpha={alpha:<6} macro-AUROC = {auc:.4f}  ({nk} classes)', flush=True)

    # ---- 2) Logistic（OvR 多标签） ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    print('\n--- LogisticRegression（显式 OvR 多标签）---', flush=True)
    for C in [0.1, 1.0]:
        m = OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=1000, solver='liblinear'))
        m.fit(Xtr_s, Ytr)
        Z = m.predict_proba(Xte_s)
        auc, nk = macro_auc(Yte, Z)
        print(f'  C={C:<6} macro-AUROC = {auc:.4f}  ({nk} classes)', flush=True)

    print(f'\n参考：v(im) 0.7542 | v(ex) 0.7635 | Sisson MPNN-GNN 文献 0.7627',
          flush=True)
    print(f'总用时 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
