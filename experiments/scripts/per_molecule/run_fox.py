# -*- coding: utf-8 -*-
"""run_fox.py — 把混合定律写进聚合，参数由编码器学：粘度上的可行性检验

实验 13 的展望验证（未进论文），结果写入 results/fox_results.csv

动机
----
工程上常见"公式已知、参数未知"的情形：粘度的 Fox 方程给出

    ln η_mix = Σ_i w_i ln η_i(T)

而纯组分粘度 η_i 通常是 VFT 形式 ln η_i = a_i + b_i/(T − c_i)。若把
(a_i, b_i, c_i) 交给编码器从分子结构预测，就得到

    ln η = Σ_i w_i [ a_i + b_i / (T − c_i) ]

即"聚合按公式、参数由模型学"。本脚本把它与数据驱动的 v(im) 在同一数据、
同一划分、同一训练预算下对比。

两个模型（--model）
------------------
  std   每分子一个 256 维向量，加权和 + [条件] -> MLP（即 run_5tasks_vim.py）
  fox   每分子 (a, b, c) 三维，按上式聚合；无 MLP 头

读法
----
  fox 超过 std   → 单分子参数的公式确实更强，值得作为 v(im) 的聚合先验
  fox 接近 std   → 该任务不缺温度形状，公式先验增益有限
  fox 明显落后   → 公式形式与数据不符（如该体系的粘度不服从 Fox）

用法 (服务器):
  python run_fox.py --data-base <data-root> --tasks ilteta nist \
      --model fox --seeds 42 43 44 --epochs 200
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encoder import Encoder, mol_to_graph, DEVICE

TASKS = {
    'drug': dict(name='drug ln(S)', dir='drug-solubility',
                 csv='processed_DrugSolubilityData.csv', prop=None,
                 kind='drug', conds=['Temperature, K'], train_max=2),
    'spe': dict(name='SPE ln(k)', dir='polymer-electrolyte',
                csv='processed_PolymerElectrolyteData.csv', prop=None,
                kind='standard', conds=['Temperature, K'], train_max=2),
    'iltk': dict(name='IlT ln(k)', dir='ionic-liquids',
                 csv='processed_IlThermoData.csv',
                 prop='Electrical conductivity', kind='standard',
                 conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                 train_max=2),
    'ilteta': dict(name='IlT ln(eta)', dir='ionic-liquids',
                   csv='processed_IlThermoData.csv', prop='Viscosity',
                   kind='standard',
                   conds=['Temperature, K', 'Pressure, kPa', 'Frequency, MHz'],
                   train_max=2),
    'nist': dict(name='NIST-full ln(eta)', dir='nist-logV',
                 csv='processed_NISTlogV.csv', prop=None, kind='standard',
                 conds=['T', 'P'], train_max=1),
}


def components(r, kind):
    """返回 [(cid, w), ...]; 标注缺损 (ids/mfs 长度不一致) 返回 None。"""
    if kind == 'drug':
        ids = (list(eval(r['cmp_ids_solvent'])) +
               list(eval(r['cmp_ids_drug'])))
        mfs = (list(eval(r['cmp_mole_fractions_solvent'])) +
               list(eval(r['cmp_mole_fractions_drug'])))
    else:
        ids = list(eval(r['cmp_ids']))
        mfs = list(eval(r['cmp_mole_fractions']))
    if len(ids) != len(mfs):
        return None
    comps = list(zip(ids, mfs))
    if kind == 'drug':
        tot = sum(w for _, w in comps)
        return [(int(c), w / tot) for c, w in comps]
    return [(int(c), float(w)) for c, w in comps]


def build_mol_graphs(cids, cmp_smiles):
    cache = {}
    for cid in sorted(set(cids)):
        if cid in cache:
            continue
        smi = cmp_smiles.get(cid)
        g = mol_to_graph(smi) if smi else None
        if g is None:
            cache[cid] = None
            continue
        x = torch.tensor(g['x']).float()[:, :31]
        mf = torch.ones(x.size(0), 1)
        cache[cid] = (torch.cat([x, mf], dim=1), torch.tensor(g['ei']))
    return cache


def forward_mixtures(ids, meta, mol_graphs, enc, reg, y_mean, y_std,
                     y_all=None, train=False):
    graphs, mix_ids, weights, cond_rows = [], [], [], []
    for j, mi in enumerate(ids):
        for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(g)
            mix_ids.append(j)
            weights.append(w)
        cond_rows.append(meta[mi]['cond'])
    cond_t = torch.tensor(np.stack(cond_rows), dtype=torch.float32,
                          device=DEVICE)
    if not graphs:
        inp = torch.cat([torch.zeros((len(ids), 256), device=DEVICE),
                         cond_t], dim=1)
        out = reg(inp).squeeze(1)
        if not train:
            return out.detach().cpu().numpy() * y_std + y_mean
        yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                          dtype=torch.float32, device=DEVICE)
        return out, yb
    sizes = [g[0].size(0) for g in graphs]
    xs = torch.cat([g[0] for g in graphs]).to(DEVICE)
    ei = torch.cat([g[1] + sum(sizes[:k])
                    for k, g in enumerate(graphs)], dim=1).to(DEVICE)
    b = torch.cat([torch.full((s,), k, dtype=torch.long)
                   for k, s in enumerate(sizes)]).to(DEVICE)
    z = enc.pooled(xs, ei, b)
    midx = torch.tensor(mix_ids, device=DEVICE)
    wts = torch.tensor(weights, dtype=torch.float32,
                       device=DEVICE).unsqueeze(1)
    mix = scatter(wts * z, midx, dim=0, dim_size=len(ids))
    inp = torch.cat([mix, cond_t], dim=1)
    out = reg(inp).squeeze(1)
    if not train:
        return out.detach().cpu().numpy() * y_std + y_mean
    yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                      dtype=torch.float32, device=DEVICE)
    return out, yb


GN_DIM = 16      # fox_gn 交互项向量 u 的维数默认值（可被 --gn-dim 覆盖）
ENC_DIM = 256    # 编码器宽度默认值（可被 --enc-dim 覆盖）


def forward_fox(ids, meta, mol_graphs, enc, proj, y_mean, y_std,
                y_all=None, train=False, t_mean=0.0, t_std=1.0,
                with_interaction=False):
    """ln η = Σ_i w_i [a_i + b_i / (T − c_i)] (+ 交互项)。

    前 3 维是 (a, b, c)，其后若 with_interaction 则取 G 维作为 u_i，
    交互项写作 Σ_{i<j} w_i w_j (u_i·u_j)，用恒等式化为
    ½[(Σ w u)² − Σ w² u²]，只需一阶聚合与一个 w² 项。

    温度取自 meta 的 cond 第 0 列（标准化过的），用 t_mean/t_std 还原到 K。
    c 用 sigmoid 映射到 (0, 200)，保证 T − c > 0 且分母不爆。
    """
    graphs, mix_ids, weights, temps = [], [], [], []
    for j, mi in enumerate(ids):
        T = meta[mi]['cond'][0] * t_std + t_mean
        for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
            g = mol_graphs.get(cid)
            if g is None:
                continue
            graphs.append(g)
            mix_ids.append(j)
            weights.append(w)
            temps.append(T)

    if not graphs:
        out = torch.zeros(len(ids), device=DEVICE)
        if not train:
            return out.detach().cpu().numpy() * y_std + y_mean
        yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                          dtype=torch.float32, device=DEVICE)
        return out, yb

    sizes = [g[0].size(0) for g in graphs]
    xs = torch.cat([g[0] for g in graphs]).to(DEVICE)
    ei = torch.cat([g[1] + sum(sizes[:k])
                    for k, g in enumerate(graphs)], dim=1).to(DEVICE)
    b = torch.cat([torch.full((s,), k, dtype=torch.long)
                   for k, s in enumerate(sizes)]).to(DEVICE)

    p = proj(enc.pooled(xs, ei, b))                 # 256 -> 3 或 3+G
    a, bcoef, craw = p[:, 0], p[:, 1], p[:, 2]
    T = torch.tensor(temps, dtype=torch.float32, device=DEVICE)
    c = 200.0 * torch.sigmoid(craw)
    ln_eta = a + bcoef / torch.clamp(T - c, min=1.0)

    midx = torch.tensor(mix_ids, device=DEVICE)
    wts = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    out = scatter(wts * ln_eta, midx, dim=0, dim_size=len(ids))

    if with_interaction:
        u = p[:, 3:]                                # [n_mol, G]
        s = scatter(wts.unsqueeze(-1) * u, midx, dim=0, dim_size=len(ids))
        q = scatter((wts ** 2).unsqueeze(-1) * u * u, midx, dim=0,
                    dim_size=len(ids))
        # Σ_{i<j} w_i w_j (u_i·u_j) = ½[(Σ w u)² − Σ w² u²]
        out = out + 0.5 * ((s * s).sum(1) - q.sum(1))

    if not train:
        return out.detach().cpu().numpy() * y_std + y_mean
    yb = torch.tensor((y_all[ids] - y_mean) / y_std,
                      dtype=torch.float32, device=DEVICE)
    return out, yb


@torch.no_grad()
def encode_unique(enc, cids, mol_graphs, batch=256):
    enc.eval()
    uniq = sorted(set(cids))
    graphs, ids = [], []
    for cid in uniq:
        g = mol_graphs.get(cid)
        if g is None:
            continue
        graphs.append(Data(x=g[0], edge_index=g[1]))
        ids.append(cid)
    vecs = []
    for b in DataLoader(graphs, batch_size=batch, shuffle=False):
        b = b.to(DEVICE)
        vecs.append(enc.pooled(b.x, b.edge_index, b.batch).cpu())
    V = torch.cat(vecs).numpy()
    return {cid: V[i] for i, cid in enumerate(ids)}



class KneeStop:
    """拐点早停：验证曲线饱和时停止。

    h = -log10(1 - score) 把趋近 1 的曲线拉成可用斜率判断的形状；
    窗 5 平滑后取一阶差分，连续两次低于峰值斜率的 ratio 倍即判为拐点。
    """

    def __init__(self, window=5, ratio=0.1, streak=2, min_epochs=10):
        self.w = window
        self.ratio = ratio
        self.streak = streak
        self.min_epochs = min_epochs
        self.h = []
        self.sm = []
        self.d = []
        self.peak = 0.0
        self.bad = 0
        self.knee = None

    def update(self, ep, score):
        self.h.append(-np.log10(max(1e-6, 1.0 - min(score, 1 - 1e-6))))
        if len(self.h) < self.w:
            return False
        self.sm.append(float(np.mean(self.h[-self.w:])))
        if len(self.sm) < 2:
            return False
        d = self.sm[-1] - self.sm[-2]
        self.d.append(d)
        if d > self.peak:
            self.peak = d
        if ep >= self.min_epochs and self.peak > 0 and d < self.ratio * self.peak:
            self.bad += 1
        else:
            self.bad = 0
        if self.bad >= self.streak:
            self.knee = ep
            return True
        return False


@torch.no_grad()
def score_batch(idx, meta, mol_graphs, enc, proj, reg, model,
                y_mean, y_std, t_mean, t_std, bs=512):
    """对一组样本算预测值（原尺度）。"""
    enc.eval()
    if reg is not None:
        reg.eval()
    ps = []
    for s in range(0, len(idx), bs):
        chunk = idx[s:s + bs]
        if model in ('fox', 'fox_gn'):
            ps.append(forward_fox(chunk, meta, mol_graphs, enc, proj,
                                  y_mean, y_std, t_mean=t_mean, t_std=t_std,
                                  with_interaction=(model == 'fox_gn')))
        else:
            ps.append(forward_mixtures(chunk, meta, mol_graphs, enc, reg,
                                       y_mean, y_std, train=False))
    return np.concatenate(ps)


def snapshot(*mods):
    out = {}
    for i, m in enumerate(mods):
        if m is not None:
            out[i] = {k: v.detach().clone() for k, v in m.state_dict().items()}
    return out


def restore(snap, *mods):
    for i, m in enumerate(mods):
        if m is not None and i in snap:
            m.load_state_dict(snap[i])

def run_task(key, data_base, seeds, epochs, batch_size, out_writer,
             model='std', enc_dim=256, gn_dim=16, early_stop=True):
    tk = TASKS[key]
    name = tk['name']
    droot = os.path.join(data_base, tk['dir'], 'processed_data')
    df = pd.read_csv(os.path.join(droot, tk['csv']))
    if tk['prop'] is not None and 'property' in df.columns:
        df = df[df['property'] == tk['prop']].reset_index(drop=True)
    comp = pd.read_csv(os.path.join(droot, 'compounds.csv'))
    cmp_smiles = {int(r['compound_id']): r['smiles']
                  for _, r in comp.iterrows()}

    comps_list, ys, conds, skipped = [], [], [], 0
    for _, r in df.iterrows():
        comps = components(r, tk['kind'])
        if comps is None:
            skipped += 1
            continue
        comps_list.append(comps)
        ys.append(float(r['value']))
        conds.append([pd.to_numeric(r.get(c), errors='coerce')
                      for c in tk['conds']])
    ys = np.array(ys)
    conds = np.array(conds, dtype=np.float64)
    nc = np.array([len(c) for c in comps_list])
    tr_pos = np.where(nc <= tk['train_max'])[0]
    te_pos = np.where(nc > tk['train_max'])[0]
    print(f'--- {name}: 有效 {len(comps_list)} (剔除缺损 {skipped}) | '
          f'train {len(tr_pos)} / test {len(te_pos)}', flush=True)

    c_mu = np.nanmean(conds[tr_pos], axis=0)
    c_sd = np.nanstd(conds[tr_pos], axis=0)
    c_sd[c_sd == 0] = 1.0
    conds = np.where(np.isnan(conds), c_mu, conds)
    conds = (conds - c_mu) / c_sd
    n_cond = conds.shape[1]

    meta = []
    for i, comps in enumerate(comps_list):
        meta.append({'cids': [c for c, _ in comps],
                     'mfs': [w for _, w in comps],
                     'cond': conds[i]})
    all_cids = sorted({c for m in meta for c in m['cids']})
    mol_graphs = build_mol_graphs(all_cids, cmp_smiles)
    n_fail = sum(1 for c in all_cids if mol_graphs.get(c) is None)
    print(f'    分子图: {len(all_cids) - n_fail}/{len(all_cids)} 成功',
          flush=True)

    tr_idx = tr_pos.tolist()
    te_idx = te_pos.tolist()
    # 固定 1000 个训练样本做验证：Knee stop 需要一条验证曲线，
    # 而全量验证会让每 epoch 的额外前向过重（CPU 侧是瓶颈）
    _rng = np.random.RandomState(0)
    _perm = _rng.permutation(len(tr_idx))
    n_val = min(1000, len(tr_idx) // 10)
    val_idx = [tr_idx[i] for i in _perm[:n_val]]
    fit_idx = [tr_idx[i] for i in _perm[n_val:]]
    y_mean, y_std = ys[fit_idx].mean(), ys[fit_idx].std()

    t_mean = float(c_mu[0]) if len(c_mu) else 0.0
    t_std = float(c_sd[0]) if len(c_sd) else 1.0

    rs, maes = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # 三档共用同一个编码器：差别只在投影层与公式。
        # nhead 取 4，enc_dim 需能被 4 整除。
        nh = 4 if enc_dim % 4 == 0 else 1
        enc = Encoder(dim=enc_dim, nhead=nh, pad_from=32,
                      normalize=False).to(DEVICE)
        if model in ('fox', 'fox_gn'):
            dim_out = 3 + (gn_dim if model == 'fox_gn' else 0)
            proj = nn.Linear(enc_dim, dim_out).to(DEVICE)
            reg = None
        else:
            proj = None
            reg = nn.Sequential(
                nn.Linear(enc_dim + n_cond, enc_dim), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(enc_dim, enc_dim), nn.ReLU(),
                nn.Linear(enc_dim, 1)).to(DEVICE)
        params = list(enc.parameters())
        if proj is not None:
            params += list(proj.parameters())
        if reg is not None:
            params += list(reg.parameters())
        opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                         T_max=epochs - 3)
        stopper = KneeStop() if early_stop else None
        best = {'r': -9.0, 'ep': -1, 'snap': None}
        for ep in range(epochs):
            if ep < 3:
                for pg in opt.param_groups:
                    pg['lr'] = 1e-4 * (ep + 1) / 3
            enc.train()
            if reg is not None:
                reg.train()
            perm = np.random.RandomState(ep).permutation(len(fit_idx))
            tot, nb = 0.0, 0
            for s in range(0, len(perm), batch_size):
                ids = [fit_idx[i] for i in perm[s:s + batch_size]]
                if model in ('fox', 'fox_gn'):
                    out, yb = forward_fox(ids, meta, mol_graphs, enc, proj,
                                          y_mean, y_std, ys, train=True,
                                          t_mean=t_mean, t_std=t_std,
                                          with_interaction=(model == 'fox_gn'))
                else:
                    out, yb = forward_mixtures(ids, meta, mol_graphs,
                                               enc, reg, y_mean, y_std,
                                               ys, train=True)
                loss = F.mse_loss(out, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
                tot += loss.item()
                nb += 1
            if ep >= 3:
                sch.step()
            if ep % 20 == 0 or ep == epochs - 1:
                print(f'    seed {seed} ep {ep}: loss {tot / max(nb, 1):.4f}',
                      flush=True)
            if stopper is not None:
                vp = score_batch(val_idx, meta, mol_graphs, enc, proj, reg,
                                 model, y_mean, y_std, t_mean, t_std)
                vr = float(pearsonr(vp, ys[val_idx])[0])
                if vr > best['r']:
                    best['r'] = vr
                    best['ep'] = ep
                    best['snap'] = snapshot(enc, proj, reg)
                if stopper.update(ep, vr):
                    print(f'    seed {seed} knee at ep {ep} '
                          f'(best ep {best["ep"]}, val R {best["r"]:.4f})',
                          flush=True)
                    break
        if early_stop and best['snap'] is not None:
            restore(best['snap'], enc, proj, reg)

        # test
        te_y = ys[te_idx]
        if model in ('fox', 'fox_gn'):
            # Fox 的温度项逐样本不同，不能预聚合，按批前向
            enc.eval()
            with torch.no_grad():
                ps = []
                for s in range(0, len(te_idx), 512):
                    ps.append(forward_fox(te_idx[s:s + 512], meta, mol_graphs,
                                          enc, proj, y_mean, y_std,
                                          t_mean=t_mean, t_std=t_std,
                                          with_interaction=(model == 'fox_gn')))
                p = np.concatenate(ps)
        else:
            # 单分子向量缓存 + numpy 加权和 + head (大数据测试集秒级)
            comp_vec = encode_unique(enc, all_cids, mol_graphs)
            Z = np.zeros((len(te_idx), 256))
            C = np.zeros((len(te_idx), n_cond))
            for j, mi in enumerate(te_idx):
                acc = None
                for cid, w in zip(meta[mi]['cids'], meta[mi]['mfs']):
                    v = comp_vec.get(cid)
                    if v is None:
                        continue
                    acc = w * v if acc is None else acc + w * v
                Z[j] = acc if acc is not None else np.zeros(256)
                C[j] = meta[mi]['cond']
            inp = np.hstack([Z, C])
            with torch.no_grad():
                ps = []
                for s in range(0, len(inp), 4096):
                    ps.append(reg(torch.tensor(inp[s:s + 4096],
                                               dtype=torch.float32,
                                               device=DEVICE))
                              .squeeze(1).cpu().numpy())
                p = np.concatenate(ps) * y_std + y_mean
        r = pearsonr(p, te_y)[0]
        mae = float(np.mean(np.abs(p - te_y)))
        rs.append(r)
        maes.append(mae)
        print(f'    >> seed {seed}: R = {r:.4f} | MAE = {mae:.4f}',
              flush=True)
        out_writer.writerow([key, seed, f'{r:.4f}', f'{mae:.4f}'])

    rs, maes = np.array(rs), np.array(maes)
    print(f'>> {name}: R = {rs.mean():.4f} ± {rs.std():.4f} | '
          f'MAE = {maes.mean():.4f} ± {maes.std():.4f}', flush=True)
    out_writer.writerow([key, 'mean', f'{rs.mean():.4f}', f'{maes.mean():.4f}'])
    out_writer.writerow([key, 'std', f'{rs.std():.4f}', f'{maes.std():.4f}'])
    return key, rs.mean(), rs.std(), maes.mean(), maes.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-base',
                    required=True, help='CheMixHub data root (containing per-task subdirs)')
    ap.add_argument('--tasks', nargs='+',
                    default=['drug', 'spe', 'iltk', 'ilteta', 'nist'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--enc-dim', type=int, default=256,
                    help='encoder width; must be divisible by 4 for 4 heads')
    ap.add_argument('--no-early-stop', action='store_true',
                    help='disable Knee stop and train the full epoch budget')
    ap.add_argument('--gn-dim', type=int, default=16,
                    help='rank of the interaction term in fox_gn')
    ap.add_argument('--model', default='std',
                    choices=['std', 'fox', 'fox_gn'],
                    help='std = data-driven v(im); fox = Fox+VFT, '
                         'single-molecule parameters only; '
                         'fox_gn = plus an explicit pair term')
    args = ap.parse_args()

    # 单 seed 时文件名带 seed，便于同一模型并行多个 seed 而不互相覆盖
    tag = (f'fox_{args.model}_s{args.seeds[0]}.csv' if len(args.seeds) == 1
           else f'fox_{args.model}_results.csv')
    out = os.path.join(HERE, tag)
    f = open(out, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['task', 'seed', 'R', 'MAE'])
    summary = []
    for key in args.tasks:
        summary.append(run_task(key, args.data_base, args.seeds,
                                args.epochs, args.batch_size, w,
                                model=args.model, enc_dim=args.enc_dim,
                                gn_dim=args.gn_dim,
                                early_stop=not args.no_early_stop))
        f.flush()
    f.close()

    print(f'\n===== 外推汇总（model={args.model}）=====')
    for key, rm, rs_, mm, ms in summary:
        print(f'{TASKS[key]["name"]:20s} R = {rm:.4f} ± {rs_:.4f} | '
              f'MAE = {mm:.4f} ± {ms:.4f}')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
