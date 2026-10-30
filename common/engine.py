# -*- coding: utf-8 -*-
"""engine.py —— 唯一训练/评估引擎（四条路线共用）

来源: paper_v2/experiments/scripts/per_molecule/run_carved.py
改动: (1) 任务注册表改为 common/spec.py；
      (2) 新增 --split-file，直接读 make_splits.py 产出的 npz，
          保证 v(im)/v(ex)/additive/frozen 四臂用的是同一份索引。

--split-file 优先于 --split-by；两者都没给时会报错提示。
早停: 默认开启（knee）；--no-early-stop 关闭。
"""
"""run_carved.py — 在"分子完全不重叠"的划分上比较 v(im) 与 v(ex)

划分来自 make_carved_split.py：
    train: 全部组分在 A 侧
    test : 全部组分在 B 侧      （两侧分子集合不相交，ω = 1）
两侧 K 分布接近，所以组合轴不是变量，差异只来自"分子是否见过"。

用法
----
  python run_carved.py --data-base <root> --tag ilvis --model vim --seeds 42
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.utils import scatter, to_dense_batch

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, __file__.rsplit('\\', 1)[0])
from spec import spec_of, load_records  # noqa: E402
from encoder import Encoder  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data', 'carved')

ELEMENTS = [3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 30, 33, 35, 53]
CHIRAL = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
          Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
           Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
           Chem.HybridizationType.SP3D2]
N_DEG, N_H = 9, 5
FEAT = len(ELEMENTS) + len(CHIRAL) + N_DEG + N_H + len(HYBRIDS) + 1 + 3 + 1


def mol_to_graph(smi):
    m = Chem.MolFromSmiles(str(smi)) if smi else None
    if m is None:
        return None
    nf = []
    for a in m.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in ELEMENTS]
        f += [1 if a.GetChiralTag() == x else 0 for x in CHIRAL]
        d = min(a.GetDegree(), N_DEG - 1)
        f += [1.0 if d == i else 0.0 for i in range(N_DEG)]
        h = min(a.GetTotalNumHs(), N_H - 1)
        f += [1.0 if h == i else 0.0 for i in range(N_H)]
        f += [1.0 if a.GetHybridization() == x else 0.0 for x in HYBRIDS]
        f += [1.0 if a.GetIsAromatic() else 0.0]
        fc = int(a.GetFormalCharge())
        f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0,
              1.0 if fc > 0 else 0.0]
        f += [1.0 if a.IsInRing() else 0.0]
        nf.append(f[:FEAT])
    ei = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
    if not ei:
        ei = [[0, 0]]
    return np.array(nf, np.float32), np.array(ei, np.int64).T


class Enc(nn.Module):
    """直接用 encoder.py 的 Encoder（GIN×3 + TFM×2 + mean pooling）。"""

    def __init__(self, dim=256, nhead=4):
        super().__init__()
        self.inner = Encoder(dim=dim, nhead=nhead, pad_from=FEAT + 1,
                             normalize=False)

    def forward(self, x, ei, batch):
        return self.inner.pooled(x, ei, batch)


class Vim(nn.Module):
    def __init__(self, dim=256, nhead=4, ncond=0):
        super().__init__()
        self.enc = Enc(dim, nhead)
        self.reg = nn.Sequential(
            nn.Linear(dim + ncond, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, z, ws, mix_ids, n_mix, cond):
        mix = scatter(ws.unsqueeze(1) * z, mix_ids, dim=0, dim_size=n_mix)
        return self.reg(torch.cat([mix, cond], 1)).squeeze(1)


class Vex(nn.Module):
    def __init__(self, dim=256, nhead=4, ncond=0):
        super().__init__()
        self.enc = Enc(dim, nhead)
        self.w_proj = nn.Linear(1, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                           dim_feedforward=2 * dim,
                                           batch_first=True, dropout=0.1)
        self.mix_tfm = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(dim + ncond, dim), nn.ReLU(),
                                  nn.Linear(dim, 1))

    def forward(self, z, ws, mix_ids, n_mix, cond):
        B = n_mix
        D = z.size(1)
        counts = torch.bincount(mix_ids, minlength=B)
        K = int(counts.max())
        tok = z + self.w_proj(ws.unsqueeze(1))

        # 组分按所属混合物排序，组内保持原顺序（stable）
        order = torch.argsort(mix_ids, stable=True)
        s_mix = mix_ids[order]
        # 组内序号 = 组内累计计数 − 该组起始偏移
        csum = torch.cumsum(torch.ones_like(s_mix), 0) - 1
        starts = torch.cumsum(counts, 0) - counts
        pos = csum - starts[s_mix]
        # 展平写入位置（第 0 列留给 CLS）
        flat = s_mix * (K + 1) + (pos + 1)

        pad = torch.zeros(B * (K + 1), D, device=z.device)
        pad[flat] = tok[order]
        pad = pad.view(B, K + 1, D)
        pad[:, 0] = self.cls

        msk = torch.zeros(B * (K + 1), dtype=torch.bool, device=z.device)
        msk[flat] = True
        msk = msk.view(B, K + 1)
        msk[:, 0] = True

        out = self.mix_tfm(pad, src_key_padding_mask=~msk)
        return self.head(torch.cat([out[:, 0], cond], 1)).squeeze(-1)


def pearson(p, t):
    p = np.asarray(p, float) - np.mean(p)
    t = np.asarray(t, float) - np.mean(t)
    den = np.sqrt((p * p).sum() * (t * t).sum())
    return float((p * t).sum() / den) if den > 0 else 0.0


def r2(p, t):
    return 1 - float(((t - p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())


class Knee:
    def __init__(self, w=5, ratio=0.1, streak=2, min_ep=10):
        self.w, self.ratio, self.streak, self.min_ep = w, ratio, streak, min_ep
        self.h, self.sm, self.peak, self.bad, self.knee = [], [], 0.0, 0, None

    def update(self, ep, s):
        self.h.append(-np.log10(max(1e-6, 1.0 - min(s, 1 - 1e-6))))
        if len(self.h) < self.w:
            return False
        self.sm.append(float(np.mean(self.h[-self.w:])))
        if len(self.sm) < 2:
            return False
        d = self.sm[-1] - self.sm[-2]
        self.peak = max(self.peak, d)
        self.bad = self.bad + 1 if (ep >= self.min_ep and self.peak > 0
                                    and d < self.ratio * self.peak) else 0
        if self.bad >= self.streak:
            self.knee = ep
            return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base', required=True)
    ap.add_argument('--split-dir', default=DATA,
                    help='划分文件所在目录（默认 experiments/data/carved）')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split-file', default=None,
                    help='直接读 make_splits.py 产出的 npz（含 train_idx/'
                         'test_idx，可选 val_idx）；给了它就不再按 --split-by 现算')
    ap.add_argument('--out', default=None,
                    help='汇总 csv 的输出目录；写入 <tag>_<model>_<split>.csv，'
                         '每个 seed 一行（追加）')
    ap.add_argument('--split-by', choices=['carved', 'carved3', 'k', 'y'],
                    default='carved',
                    help='carved=分子不重叠；carved3=三向分子切割；'
                         'k=按组分数；y=按目标值分位')
    ap.add_argument('--train-max', type=int, default=2,
                    help='--split-by k 时，K<=train_max 为训练')
    ap.add_argument('--y-train', type=float, nargs=2, default=[0.25, 0.75],
                    help='--split-by y 时，训练侧的目标值分位区间')
    ap.add_argument('--y-test', type=float, nargs=2, default=[0.10, 0.90],
                    help='--split-by y 时，测试侧取两端各自的这个分位之外')
    ap.add_argument('--model', choices=['vim', 'vex'], default='vim')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--no-early-stop', action='store_true')
    ap.add_argument('--save-ckpt', default=None,
                    help='权重目录；每个 seed 存一个 .pt（权重 + 配置 + 逐行预测）')
    ap.add_argument('--save-preds', action='store_true',
                    help='把逐行预测写进 <tag>_<model>_<split>_preds.csv；'
                         '之后任何分组（K 分档、seen/unseen）都能离线重算，无需重训')
    args = ap.parse_args()

    va_rows = None
    if args.split_file:
        sp = np.load(args.split_file, allow_pickle=True)
        task = str(sp['task']) if 'task' in sp else args.tag
        tr_rows = sp['train_idx'].tolist()
        te_rows = sp['test_idx'].tolist()
        va_rows = (sp['val_idx'].tolist() if 'val_idx' in sp else None)
        print('  读划分文件 %s: train %d / val %d / test %d'
              % (os.path.basename(args.split_file), len(tr_rows),
                 0 if va_rows is None else len(va_rows), len(te_rows)),
              flush=True)
    elif args.split_by == 'carved':
        sp = np.load(os.path.join(args.split_dir,
                                  '%s_split.npz' % args.tag),
                     allow_pickle=True)
        task = str(sp['task'])
        tr_rows = sp['train_idx'].tolist()
        te_rows = sp['test_idx'].tolist()
    elif args.split_by == 'carved3':
        sp = np.load(os.path.join(args.split_dir,
                                  '%s_split3.npz' % args.tag),
                     allow_pickle=True)
        task = str(sp['task'])
        tr_rows = sp['train_idx'].tolist()
        va_rows = sp['val_idx'].tolist()
        te_rows = sp['test_idx'].tolist()
    else:
        task = args.tag

    df, rows, comps = load_records(task, args.data_base)
    if args.split_by == 'k':
        kk = np.array([len(c) for c in comps])
        tr_rows = [rows[i] for i in range(len(rows))
                   if kk[i] <= args.train_max]
        te_rows = [rows[i] for i in range(len(rows))
                   if kk[i] > args.train_max]
        print('  按 K 划分: train K<=%d (%d 条) / test K>%d (%d 条)'
              % (args.train_max, len(tr_rows), args.train_max,
                 len(te_rows)), flush=True)
    elif args.split_by == 'y':
        yv = np.array([float(df.loc[r, 'value']) for r in rows])
        q = np.percentile(yv, [args.y_test[0] * 100,
                               args.y_train[0] * 100,
                               args.y_train[1] * 100,
                               args.y_test[1] * 100])
        lo_te, lo_tr, hi_tr, hi_te = q
        tr_rows = [rows[i] for i in range(len(rows))
                   if lo_tr <= yv[i] <= hi_tr]
        te_rows = [rows[i] for i in range(len(rows))
                   if yv[i] < lo_te or yv[i] > hi_te]
        print('  按 y 分位划分: 训练 [%.3f, %.3f] (%d 条) / '
              '测试 <%.3f 或 >%.3f (%d 条)'
              % (lo_tr, hi_tr, len(tr_rows), lo_te, hi_te, len(te_rows)),
              flush=True)
    tk = spec_of(task)
    d = os.path.join(args.data_base, tk['dir'], 'processed_data')
    comp = pd.read_csv(os.path.join(d, 'compounds.csv'))
    smi = {int(r['compound_id']): r['smiles'] for _, r in comp.iterrows()}

    pos = {r: i for i, r in enumerate(rows)}
    ys, conds = [], []
    for r in rows:
        row = df.loc[r]
        ys.append(float(row['value']))
        conds.append([pd.to_numeric(row.get(c), errors='coerce')
                      for c in tk['conds']])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64).reshape(len(rows), -1)

    meta = [{'cids': comps[pos[r]], 'mfs': None, 'cond': None}
            for r in rows]
    # 摩尔分数（长度不符时按组分数截断/补齐并归一化）
    import ast as _ast
    for i, r in enumerate(rows):
        row = df.loc[r]
        k = len(comps[pos[r]])
        if 'cmp_mole_fractions' in df.columns:
            mf = [float(v) for v in _ast.literal_eval(
                row['cmp_mole_fractions'])]
        else:
            mf = [1.0 / k] * k
        if len(mf) > k:
            mf = mf[:k]
        elif len(mf) < k:
            mf = mf + [0.0] * (k - len(mf))
        tot = sum(mf)
        meta[i]['mfs'] = ([w / tot for w in mf] if tot > 0
                          else [1.0 / k] * k)

    itr = [pos[r] for r in tr_rows]
    ite = [pos[r] for r in te_rows]
    if conds.shape[1] > 0:
        cm, cs = np.nanmean(conds[itr], 0), np.nanstd(conds[itr], 0)
        cs[cs == 0] = 1.0
        conds = (np.where(np.isnan(conds), cm, conds) - cm) / cs
    for i in range(len(meta)):
        meta[i]['cond'] = conds[i]
    ncond = conds.shape[1]

    all_cids = {c for m in meta for c in m['cids']}
    graphs = {}
    for cid in all_cids:
        g = mol_to_graph(smi.get(cid))
        if g is None:
            graphs[cid] = None
            continue
        x = torch.tensor(g[0])
        mf = torch.ones(x.size(0), 1)
        graphs[cid] = (torch.cat([x, mf], 1), torch.tensor(g[1]))

    print('tag=%s task=%s model=%s | train %d / test %d | feat %d | %s'
          % (args.tag, task, args.model, len(itr), len(ite), FEAT, DEVICE),
          flush=True)

    # 验证集：开启早停时才需要。
    #  - 关闭早停（组分数轴）：不使用验证集；切分文件若带 val_idx，把它并回训练
    #    （这些样本同样满足 K<=train_max，本来就是合法训练数据）。
    #  - 开启早停：三分时用切分文件给的 val；否则从训练侧内部切一小块。
    if args.no_early_stop:
        va_idx = []
        fit_idx = list(itr)
        if va_rows is not None:
            fit_idx = fit_idx + [pos[r] for r in va_rows]
    elif va_rows is not None:
        va_idx = [pos[r] for r in va_rows]
        fit_idx = list(itr)
    else:
        rng0 = np.random.RandomState(0)
        perm = rng0.permutation(len(itr))
        n_val = min(600, max(64, len(itr) // 8))
        va_idx = [itr[i] for i in perm[:n_val]]
        fit_idx = [itr[i] for i in perm[n_val:]]
    print('  fit %d / val %d' % (len(fit_idx), len(va_idx)), flush=True)

    ym, ysd = float(ys[itr].mean()), float(ys[itr].std())

    def fwd(idx, model, train=False):
        gs, mids, ws, cs_ = [], [], [], []
        for j, mi in enumerate(idx):
            for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
                g = graphs.get(cid)
                if g is None:
                    continue
                gs.append(g)
                mids.append(j)
                ws.append(w)
            cs_.append(meta[mi]['cond'])
        cond_t = torch.tensor(np.stack(cs_), dtype=torch.float32,
                              device=DEVICE)
        sizes = [g[0].size(0) for g in gs]
        xs = torch.cat([g[0] for g in gs]).to(DEVICE)
        offs = np.cumsum([0] + sizes[:-1]).tolist()
        ei = torch.cat([g[1] + o for g, o in zip(gs, offs)], 1).to(DEVICE)
        b = torch.cat([torch.full((s,), k, dtype=torch.long)
                       for k, s in enumerate(sizes)]).to(DEVICE)
        z = model.enc(xs, ei, b)
        midx = torch.tensor(mids, device=DEVICE)
        wts = torch.tensor(ws, dtype=torch.float32, device=DEVICE)
        out = model(z, wts, midx, len(idx), cond_t)
        if not train:
            return out.detach().cpu().numpy() * ysd + ym
        yb = torch.tensor((ys[idx] - ym) / ysd, dtype=torch.float32,
                          device=DEVICE)
        return out, yb

    t0 = time.time()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cls_ = Vim if args.model == 'vim' else Vex
        model = cls_(dim=256, nhead=4, ncond=ncond).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - 3))
        st = None if args.no_early_stop else Knee()
        best = {'r': -9, 'ep': -1, 'state': None}
        knee = None
        for ep in range(args.epochs):
            if ep < 3:
                for pg in opt.param_groups:
                    pg['lr'] = args.lr * (ep + 1) / 3
            model.train()
            pm = np.random.permutation(fit_idx)
            tot = 0.0
            for i in range(0, len(pm), args.batch):
                bidx = pm[i:i + args.batch].tolist()
                opt.zero_grad()
                out, yb = fwd(bidx, model, train=True)
                loss = ((out - yb) ** 2).mean()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(bidx)
            if ep >= 3:
                sch.step()
            if st is not None:
                model.eval()
                with torch.no_grad():
                    pv = np.concatenate([fwd(va_idx[i:i + 256], model)
                                         for i in range(0, len(va_idx), 256)])
                vr = pearson(pv, ys[va_idx])
                if vr > best['r']:
                    best = {'r': vr, 'ep': ep,
                            'state': {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}}
            else:
                vr = float('nan')
            if ep % 10 == 0 or ep == args.epochs - 1:
                print('  seed %d ep %3d loss %.4f val R %s'
                      % (seed, ep, tot / len(pm),
                         ('%+.4f' % vr) if st is not None else '(no-val)'),
                      flush=True)
            if st is not None and st.update(ep, vr):
                knee = ep
                break
        # 只有开启早停时才按验证集回载；关闭时用最后一轮权重
        if st is not None and best['state'] is not None:
            model.load_state_dict(best['state'])
        model.eval()
        with torch.no_grad():
            p = np.concatenate([fwd(ite[i:i + 256], model)
                                for i in range(0, len(ite), 256)])
        yt = ys[ite]
        kk = np.array([len(meta[i]['cids']) for i in ite])
        print('  >> seed %d knee %s best ep %s | val R %s | test R %+.4f '
              'R2 %+.4f MAE %.4f'
              % (seed, knee,
                 best['ep'] if st is not None else '(final)',
                 ('%+.4f' % best['r']) if st is not None else '(no-val)',
                 pearson(p, yt), r2(p, yt),
                 float(np.abs(p - yt).mean())), flush=True)
        print('     pred mean %+.3f vs true %+.3f' % (p.mean(), yt.mean()),
              flush=True)

        # ---- 分组评测：逐 K，以及 seen / unseen（测试样本的组分是否都在训练侧出现）----
        seen_set = set()
        for j in fit_idx:
            seen_set.update(meta[j]['cids'])
        sub = np.array([all(c in seen_set for c in meta[i]['cids'])
                        for i in ite])
        groups = []
        for kv in sorted(set(kk.tolist())):
            m = kk == kv
            if m.sum() >= 10:
                groups.append(('K', 'K=%d' % kv, m))
        # 大 K 样本稀疏，单点会少于 10 条；额外给几档区间，供逐 K 剖面使用
        for lo, hi, lab in ((5, 10, 'K=5-10'), (11, 40, 'K=11-40'),
                            (41, 10 ** 9, 'K>=41')):
            m = (kk >= lo) & (kk <= hi)
            if m.sum() >= 1:
                groups.append(('Kbin', lab, m))
        for name, m in (('seen', sub), ('unseen', ~sub)):
            if m.sum() >= 10:
                groups.append(('subset', name, m))
        for gtype, gname, m in groups:
            print('       %-9s n=%5d  R %+.4f  MAE %.4f'
                  % (gname, m.sum(), pearson(p[m], yt[m]),
                     float(np.abs(p[m] - yt[m]).mean())), flush=True)
        if args.out:
            import csv as _csv
            os.makedirs(args.out, exist_ok=True)
            stag = (os.path.splitext(os.path.basename(args.split_file))[0]
                    if args.split_file else 'split_%s' % args.split_by)
            stag = stag.replace(args.tag + '_', '', 1) if stag.startswith(
                args.tag + '_') else stag
            dst = os.path.join(args.out, '%s_%s_%s.csv'
                               % (args.tag, args.model, stag))

            # 分组表（逐 K 与 seen/unseen）：必须在 dst 之后写
            bdst = os.path.splitext(dst)[0] + '_breaks.csv'
            bnew = not os.path.exists(bdst)
            with open(bdst, 'a', newline='', encoding='utf-8') as _bf:
                _bw = _csv.writer(_bf)
                if bnew:
                    _bw.writerow(['task', 'model', 'split', 'seed',
                                  'group_type', 'group', 'n', 'R', 'MAE'])
                for gtype, gname, m in groups:
                    _bw.writerow([args.tag, args.model, stag, seed, gtype,
                                  gname, int(m.sum()),
                                  '%.4f' % pearson(p[m], yt[m]),
                                  '%.4f' % float(np.abs(p[m] - yt[m]).mean())])

            new = not os.path.exists(dst)
            with open(dst, 'a', newline='', encoding='utf-8') as _f:
                _w = _csv.writer(_f)
                if new:
                    _w.writerow(['task', 'model', 'split', 'seed', 'knee',
                                 'best_ep', 'val_R', 'test_R', 'test_R2',
                                 'MAE', 'n_train', 'n_test'])
                _w.writerow([args.tag, args.model, stag, seed,
                             '' if knee is None else knee,
                             best['ep'] if st is not None else 'final',
                             ('%.4f' % best['r']) if st is not None else '',
                             '%.4f' % pearson(p, yt), '%.4f' % r2(p, yt),
                             '%.4f' % float(np.abs(p - yt).mean()),
                             len(fit_idx), len(ite)])
            print('     wrote', os.path.basename(dst), flush=True)

            # ---- 逐行预测：任何后续分组都可离线重算 ----
            if args.save_preds:
                pdst = os.path.splitext(dst)[0] + '_preds.csv'
                pnew = not os.path.exists(pdst)
                with open(pdst, 'a', newline='', encoding='utf-8') as _pf:
                    _pw = _csv.writer(_pf)
                    if pnew:
                        _pw.writerow(['task', 'model', 'split', 'seed', 'row',
                                      'k', 'all_seen', 'pred', 'true'])
                    for i, r in enumerate(ite):
                        _pw.writerow([args.tag, args.model, stag, seed,
                                      int(rows[r]), int(kk[i]), int(sub[i]),
                                      '%.6f' % p[i], '%.6f' % yt[i]])
                print('     wrote', os.path.basename(pdst), flush=True)

            # ---- 权重：后续可做注意力/表示分析，不必重训 ----
            if args.save_ckpt:
                os.makedirs(args.save_ckpt, exist_ok=True)
                cdst = os.path.join(args.save_ckpt, '%s_%s_%s_s%d.pt'
                                    % (args.tag, args.model, stag, seed))
                torch.save({
                    'state_dict': {k: v.detach().cpu()
                                   for k, v in model.state_dict().items()},
                    'config': {'tag': args.tag, 'model': args.model,
                               'split': stag, 'seed': seed,
                               'epochs': args.epochs, 'batch': args.batch,
                               'lr': args.lr,
                               'no_early_stop': bool(args.no_early_stop),
                               'n_train': len(fit_idx), 'n_test': len(ite)},
                    'row_test': [int(rows[i]) for i in ite],
                    'row_fit': [int(rows[i]) for i in fit_idx],
                    'k_test': kk.astype(int).tolist(),
                    'all_seen': sub.astype(int).tolist(),
                    'pred': p, 'true': yt,
                }, cdst)
                print('     wrote', os.path.basename(cdst), flush=True)
        for k in sorted(set(kk.tolist())):
            m = kk == k
            if m.sum() < 10:
                continue
            print('       K=%d (n=%4d) R %+.4f' % (k, m.sum(),
                                                  pearson(p[m], yt[m])),
                  flush=True)
    print('总用时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()