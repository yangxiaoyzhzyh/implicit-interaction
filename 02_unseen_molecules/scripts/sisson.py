# -*- coding: utf-8 -*-
"""train_sisson_permol_cls.py — Sisson 74 类协议下的 v(im) 分类模型

实验 12 — Sisson olfactory task（补材 S3，表 S12）
路线 v(im)（另含 v(ex) 臂的对照配置）
产出 results/sisson/（results.csv、results_multiseed.csv、curves/）；logs/olfactory/
拐点选 ckpt 由 olfactory/knee_detect.py 完成；描述符线性对照见 baselines/run_sisson_rdkit_linear.py

以全混合物编码基线为基线，只改架构：

  老（全混合物 / wm-cls74，test 0.7635）
      两分子原子拼成一张图 → GIN×3 → 跨分子 TFM → 全体原子 masked-mean → Linear→74
  新（逐分子 / per-molecule）
      pad(31→64) → GIN×3(64) → Linear(64→dim)
      → 按 mol_id 拆成「每分子一条序列」→ 分子内 TFM(dim, layers×heads)（无跨分子注意力）
      → 分子内 masked-mean → 按权重凸组合（摩尔分数；Sisson 无浓度记录 → w=1/n_mol）
      → MLP(dim→dim→dim) → Linear(dim, 74) 分类头

与老脚本严格保持一致的部分（控制变量）：
  - 数据：trainpairs74.pt / testpairs74.pt（分子级 novel-molecule carving，test 从不参与选择）
  - 损失：74-way BCEWithLogits
  - 指标：macro-AUROC（Mann-Whitney 秩统计，并列取平均秩，只在有正有负的类上算）
  - 优化：Adam + warmup 3 + CosineAnnealingLR
与老脚本的 deliberate 差异：
  1) 架构（上面那一条）
  2) **不设早停**：固定跑满 --epochs（默认 200），用最后一个 epoch 的权重直接评估 test；
     val 只打印监控曲线，不参与任何模型选择。

用法：
  # 冒烟 / 测速
  python train_sisson_permol_cls.py --limit 2000 --epochs 2
  # 正式（新分子外推）
  python train_sisson_permol_cls.py --epochs 200
"""
import argparse, os, sys, random, time, json, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument('--odor-pair-dir', required=True,
                help='含 pairing/data.py 的 odor-pair 目录（unpickle PairData 需要）')
ap.add_argument('--pairing-dir', default=None,
                help='pairing 数据目录（默认 <本目录>/../dataset，即 sisson_repro/dataset）')
ap.add_argument('--suffix', default='74')
# ── 架构 ──
ap.add_argument('--feat', default='permol', choices=['permol', 'sisson'],
                help="原子特征：permol=新架构 31 维（含手性）；sisson=老脚本 30 维（对齐老实现）")
ap.add_argument('--variant', default='full', choices=['full', 'gin_only', 'tfm_only'],
                help='full=GIN+分子内TFM（正身）；gin_only/tfm_only 为消融')
ap.add_argument('--combine', default='wmean', choices=['wmean', 'sum', 'concat'])
ap.add_argument('--head', default='mlp', choices=['mlp', 'none'], help='组合后的 MLP 消费端')
# ── 模型超参（默认 = 老脚本最优配置：gin 3×64 / dim256 / 2 层 / 4 头 / ffn×2 / do0.1）──
ap.add_argument('--gin-hidden', type=int, default=64)
ap.add_argument('--gin-layers', type=int, default=3)
ap.add_argument('--dim', type=int, default=256)
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--heads', type=int, default=4)
ap.add_argument('--ffn-mult', type=int, default=2)
ap.add_argument('--dropout', type=float, default=0.1)
# ── 训练超参 ──
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--bsz', type=int, default=64)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--wd', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--warmup', type=int, default=3)
ap.add_argument('--eval-every', type=int, default=1, help='每 N 个 epoch 跑一次 val（仅监控）')
ap.add_argument('--save-each', type=int, default=0,
                help='>0 时每 N 个 epoch 存一份 ckpt 到 <ckpt-dir>/ep###.pt（1=每 epoch 都存）')
ap.add_argument('--test-every', type=int, default=0,
                help='>0 时每 N 个 epoch 跑一次 test 并记入曲线（仅诊断过拟合，不用于选模）')
ap.add_argument('--ckpt-dir', default=None, help='--save-each 的输出目录（默认 <本目录>/ckpts_<tag>）')
ap.add_argument('--eval-only', default=None,
                help='给定 ckpt 路径：只加载并评估 val/test，不训练（用于一次性报告选定 ckpt）')
# ── 在线拐点早停（只用 val）──
ap.add_argument('--early-stop', default='none', choices=['none', 'knee'],
                help='knee = val 曲线拐点早停（headroom 变换+因果MA+一阶导），停在拐点并回载 best-val ckpt')
ap.add_argument('--knee-window', type=int, default=5, help='因果移动平均窗口')
ap.add_argument('--knee-ratio', type=float, default=0.1, help='一阶导阈值（相对历史峰值斜率）')
ap.add_argument('--knee-streak', type=int, default=2, help='需连续满足的 epoch 数')
ap.add_argument('--min-epochs', type=int, default=10, help='至少训练这么多 epoch 才允许早停')
ap.add_argument('--seeds', default=None, help='多种子，逗号分隔（数据只建一次），如 42,43,44,45,46')
# ── 模式 ──
ap.add_argument('--n-val', type=int, default=1000, help='val 对数（从 train 随机切，seed=42）')
ap.add_argument('--val-split-file', default=None,
                help='分子级 val 切分 JSON（提供则覆盖随机 val；本脚本 val 不做模型选择）')
ap.add_argument('--limit', type=int, default=0, help='冒烟：仅取前 N 对 train 与 N//4 对 test')
ap.add_argument('--tag', default='')
ap.add_argument('--device', default=None, help='cuda / cpu（默认自动）')
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
PAIRING = args.pairing_dir or os.path.join(HERE, '..', 'dataset')
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

TAG = args.tag or f'{args.variant}_{args.combine}_{args.head}_s{args.seed}'
print(f'Device: {DEVICE} | pairing: {PAIRING} | feat={args.feat} tag={TAG}', flush=True)

sys.path.insert(0, args.odor_pair_dir)
try:
    import pairing.data  # noqa: F401  注册 PairData 类（unpickle 需要）
    print('pairing.data: imported', flush=True)
except Exception as e:                      # 服务器缺 ogb 等依赖 → 只注册最小 PairData 供 unpickle
    import types as _types
    _pkg = sys.modules.setdefault('pairing', _types.ModuleType('pairing'))
    if not hasattr(_pkg, '__path__'):
        _pkg.__path__ = [args.odor_pair_dir]
    _mod = _types.ModuleType('pairing.data')

    class PairData(Data):                   # 只需承载 y / smiles_s / smiles_t，无需 ogb
        pass

    _mod.PairData = PairData
    sys.modules['pairing.data'] = _mod
    print(f'pairing.data: stub (import failed: {type(e).__name__}: {e})', flush=True)

with open(os.path.join(PAIRING, f'notes{args.suffix}.csv'), encoding='utf-8') as f:
    NOTES = [x.strip() for x in f.read().split(',') if x.strip()]
N = len(NOTES)
print(f'Sisson classes: {N}', flush=True)

# ═══════════ 图构建（分子边界保留：mol_id + 仅分子内共价键）═══════════
FEAT_DIM = 31


def smi2g_permol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    t = [6, 7, 8, 9, 16, 17, 35, 53]
    c = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
    hyb = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
           Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
           Chem.HybridizationType.SP3D2]
    nf = []
    for a in mol.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in t]
        f += [1 if a.GetChiralTag() == x else 0 for x in c]
        deg = min(a.GetDegree(), 5); f += [1.0 if deg == i else 0.0 for i in range(6)]
        h = min(a.GetTotalNumHs(), 4); f += [1.0 if h == i else 0.0 for i in range(5)]
        f += [1.0 if a.GetHybridization() == y else 0.0 for y in hyb]
        f += [1.0 if a.GetIsAromatic() else 0.0]
        fc = int(a.GetFormalCharge())
        f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0, 1.0 if fc > 0 else 0.0]
        f += [1.0 if a.IsInRing() else 0.0]
        nf.append(f[:FEAT_DIM])
    ei = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[u, v], [v, u]]
    if not ei:
        ei = [[0, 0]]
    return {'x': np.array(nf, dtype=np.float32), 'ei': np.array(ei, dtype=np.int64).T}


def smi2g_sisson(smi):
    """老脚本 wm-cls74 的 30 维特征（不足 31 维补零，保证 pad 语义一致）。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    nf = []
    for a in mol.GetAtoms():
        f = [a.GetAtomicNum() == 6, a.GetAtomicNum() == 7, a.GetAtomicNum() == 8,
             a.GetAtomicNum() == 9, a.GetAtomicNum() == 15, a.GetAtomicNum() == 16,
             a.GetAtomicNum() == 17, a.GetAtomicNum() == 35, a.GetAtomicNum() == 53,
             a.GetFormalCharge() == 1, a.GetFormalCharge() == -1,
             a.GetHybridization() == Chem.HybridizationType.SP,
             a.GetHybridization() == Chem.HybridizationType.SP2,
             a.GetHybridization() == Chem.HybridizationType.SP3,
             a.GetIsAromatic(),
             a.GetDegree() == 1, a.GetDegree() == 2, a.GetDegree() == 3, a.GetDegree() == 4,
             a.GetTotalNumHs() == 0, a.GetTotalNumHs() == 1, a.GetTotalNumHs() == 2,
             a.GetTotalNumHs() == 3, a.IsInRing(),
             len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]) > 0,
             len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]) > 0,
             len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]) > 0,
             a.GetMass() > 30, a.GetMass() > 60, a.GetMass() > 90, a.GetMass() > 120]
        f = [1.0 if x else 0.0 for x in f[:FEAT_DIM]]
        f += [0.0] * (FEAT_DIM - len(f))
        nf.append(f)
    ei = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[u, v], [v, u]]
    if not ei:
        ei = [[0, 0]]
    return {'x': np.array(nf, dtype=np.float32), 'ei': np.array(ei, dtype=np.int64).T}


SMI2G = smi2g_permol if args.feat == 'permol' else smi2g_sisson
_gc = {}


class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'n_mol'):
            return None
        return super().__cat_dim__(key, value, *a, **kw)


def build(smis):
    """分子边界保留：mol_id 标记每个原子属于第几个分子；edge_index 仅含分子内共价键。"""
    ax, aei, amid = [], [], []
    off = 0
    for i, s in enumerate(smis):
        if s not in _gc:
            g = SMI2G(s)
            if g is None:
                return None
            _gc[s] = g
        g = _gc[s]
        ax.append(g['x']); aei.append(g['ei'] + off)
        amid += [i] * len(g['x'])
        off += len(g['x'])
    if not ax:
        return None
    return GData(x=torch.tensor(np.concatenate(ax), dtype=torch.float),
                 edge_index=torch.tensor(np.concatenate(aei, axis=1), dtype=torch.long),
                 mol_id=torch.tensor(amid, dtype=torch.long),
                 n_mol=torch.tensor(len(smis), dtype=torch.float))


# ═══════════ 数据（与老脚本一致）═══════════
def load_pairs(fname):
    data, _ = torch.load(fname, weights_only=False)
    n = data.y.size(0)
    assert data.y.size(1) == N, f'class mismatch: pt has {data.y.size(1)}, notes.csv has {N}'
    out = []
    for i in range(n):
        y = data.y[i].float()
        if y.sum() > 0:
            out.append((data.smiles_s[i], data.smiles_t[i], y))
    return out


print(f'Loading trainpairs{args.suffix}.pt ...', flush=True)
train_raw = load_pairs(os.path.join(PAIRING, f'trainpairs{args.suffix}.pt'))
print(f'  train pairs: {len(train_raw)}', flush=True)
test_raw = load_pairs(os.path.join(PAIRING, f'testpairs{args.suffix}.pt'))
print(f'  test pairs : {len(test_raw)}', flush=True)

if args.limit > 0:
    train_raw = train_raw[:args.limit]
    test_raw = test_raw[:max(1, args.limit // 4)]
    print(f'[smoke] limited to train {len(train_raw)} / test {len(test_raw)}', flush=True)

if args.val_split_file:
    with open(args.val_split_file, encoding='utf-8') as f:
        SP = json.load(f)
    val_idx = SP['val_indices']
    vmols = set(SP['molecules'])
    tr_idx = [i for i in range(len(train_raw)) if i not in set(val_idx) and
              train_raw[i][0] not in vmols and train_raw[i][1] not in vmols]
    va_rows = [train_raw[i] for i in val_idx]
    tr_rows = [train_raw[i] for i in tr_idx]
    print(f'val split loaded: {args.val_split_file} | train {len(tr_rows)} | '
          f'val {len(va_rows)} | test {len(test_raw)}', flush=True)
else:
    idx = list(range(len(train_raw))); random.Random(42).shuffle(idx)
    n_val = min(args.n_val, max(1, len(train_raw) // 2))
    va_rows = [train_raw[i] for i in idx[:n_val]]
    tr_rows = [train_raw[i] for i in idx[n_val:]]
    print(f'split(pair): train {len(tr_rows)} | val {len(va_rows)} | test {len(test_raw)}',
          flush=True)


def report_cover(name, rows):
    if not rows:
        print(f'  [{name}] empty'); return
    s = torch.stack([y for _, _, y in rows]).sum(0)
    missing = [NOTES[i] for i in range(N) if s[i] == 0]
    print(f'  [{name}] label coverage: {N - len(missing)}/{N}' +
          (f' | missing: {missing[:8]}' if missing else ''), flush=True)


report_cover('train-after', tr_rows); report_cover('val', va_rows)


def make_graphs(rows):
    out = []
    for sa, sb, y in rows:
        d = build([sa, sb])
        if d is not None:
            d.y = y
            out.append(d)
    return out


print('Building graphs ...', flush=True)
tr_d = make_graphs(tr_rows); va_d = make_graphs(va_rows); te_d = make_graphs(test_raw)
print(f'graphs: train {len(tr_d)} | val {len(va_d)} | test {len(te_d)}', flush=True)
va_loader = DataLoader(va_d, batch_size=256)
te_loader = DataLoader(te_d, batch_size=256)


# ═══════════ 模型：逐分子新架构 + 74 类分类头 ═══════════
class GINConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                                 nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))


class PerMolClassifier(nn.Module):
    """逐分子路线 + 分类头：分子内 GIN(+TFM) → 分子内 masked-mean → 凸组合 → MLP → 74 logits。"""

    def __init__(self):
        super().__init__()
        gh, dim = args.gin_hidden, args.dim
        self.dim = dim
        self.variant, self.combine = args.variant, args.combine
        self.use_mlp = (args.head == 'mlp')
        self.pad = nn.ZeroPad2d((0, gh - FEAT_DIM, 0, 0))
        self.convs = nn.ModuleList([GINConv(gh) for _ in range(args.gin_layers)])
        self.proj = nn.Linear(gh, dim)
        if args.variant != 'gin_only':
            enc = nn.TransformerEncoderLayer(d_model=dim, nhead=args.heads,
                                             dim_feedforward=args.ffn_mult * dim,
                                             batch_first=True, dropout=args.dropout)
            self.tfm = nn.TransformerEncoder(enc, args.layers)
        in_dim = dim * 2 if args.combine == 'concat' else dim
        if args.head == 'mlp':
            self.mlp = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head = nn.Linear(in_dim if args.head == 'none' else dim, N)

    def _mol_vectors(self, h, batch, mol_id):
        """每分子单独成序列 → 分子级向量。返回 (v, sample_of, mol_of)。"""
        M = int(mol_id.max().item()) + 1
        key = batch * M + mol_id
        order = torch.argsort(key, stable=True)
        h_s, key_s = h[order], key[order]
        uniq, inv = torch.unique_consecutive(key_s, return_inverse=True)
        dense, mask = to_dense_batch(h_s, inv)          # [n_mol, max_atoms, dim]
        if self.variant != 'gin_only':
            dense = self.tfm(dense, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        v = (dense * m).sum(1) / m.sum(1).clamp(min=1)  # 分子内 masked-mean
        return v, uniq // M, uniq % M

    def forward(self, x, ei, batch, mol_id, n_mol):
        x = self.pad(x)
        if self.variant != 'tfm_only':
            for c in self.convs:
                x = c(x, ei)
        h = self.proj(x)
        v, sample_of, mol_of = self._mol_vectors(h, batch, mol_id)
        S = int(batch.max().item()) + 1
        if self.combine == 'concat':                     # 非尺寸不变，仅作对照
            slot = sample_of * 2 + mol_of.clamp(max=1)
            mat = scatter(v, slot, dim=0, dim_size=2 * S, reduce='sum').view(S, 2, self.dim)
            z = torch.cat([mat[:, 0], mat[:, 1]], dim=-1)
        else:
            if self.combine == 'wmean':                  # 摩尔分数加权（Sisson 无浓度 → 1/n）
                w = 1.0 / n_mol[sample_of].clamp(min=1).unsqueeze(-1)
            else:                                        # sum
                w = torch.ones_like(n_mol[sample_of].unsqueeze(-1))
            z = scatter(v * w, sample_of, dim=0, dim_size=S, reduce='sum')
        if self.use_mlp:
            z = self.mlp(z)
        return self.head(z)                              # logits [S, 74]


def _macro_auc(Y, Z):
    """macro-AUROC（Mann-Whitney 秩统计，并列取平均秩），只在有正有负的类上算。"""
    n = Y.shape[0]
    aucs = []
    for i in range(Y.shape[1]):
        y = Y[:, i]; z = Z[:, i]
        pos = y.sum(); neg = n - pos
        if pos == 0 or neg == 0:
            continue
        order = np.argsort(z, kind='mergesort')
        sz = z[order]
        r = np.empty(n, dtype=np.float64)
        j = 0
        while j < n:
            k = j
            while k + 1 < n and sz[k + 1] == sz[j]:
                k += 1
            r[j:k + 1] = (j + k + 2) / 2.0
            j = k + 1
        ranks = np.empty(n); ranks[order] = r
        s = ranks[y > 0].sum()
        aucs.append((s - pos * (pos + 1) / 2.0) / (pos * neg))
    return float(np.mean(aucs)), len(aucs)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    zs, ys = [], []
    for b in loader:
        b = b.to(DEVICE)
        zs.append(model(b.x, b.edge_index, b.batch, b.mol_id, b.n_mol).cpu())
        ys.append(b.y.cpu())
    return _macro_auc(torch.cat(ys).numpy(), torch.cat(zs).numpy())


m = PerMolClassifier().to(DEVICE)          # 用于 n_params 统计与 --eval-only
n_params = sum(p.numel() for p in m.parameters())
print(f'Params: {n_params:,} | variant={args.variant} combine={args.combine} head={args.head}',
      flush=True)

if args.eval_only:
    m.load_state_dict(torch.load(args.eval_only, weights_only=True, map_location=DEVICE))
    v_auc, _ = evaluate(m, va_loader)
    t_auc, n_auc = evaluate(m, te_loader)
    print(f'[eval-only] {args.eval_only}\n  val {v_auc:.4f} | TEST {t_auc:.4f} '
          f'({n_auc}/{N} classes)', flush=True)
    sys.exit(0)


# ═══════════ 在线拐点早停（只读 val，绝不看 test）═══════════
class KneeStopper:
    """val 曲线 → headroom 变换 → 因果移动平均 → 一阶导 → 定位学习收益拐点。

    AUROC 饱和在 1.0，直接求导会被天花板压缩，故先做 h = -log10(1 - val)，
    把饱和段拉直；再用因果 MA(window) 去逐 epoch 噪声；一阶导 d1 降到
    「历史峰值斜率 × ratio」并连续 streak 个 epoch → 判定到达拐点。
    停止后回载 [0, knee] 内 val 最优的 ckpt（等价于该点早停 + best-val 回载）。
    """

    def __init__(self, window, ratio, streak, min_epochs, eps=1e-6):
        self.w, self.ratio, self.streak = window, ratio, streak
        self.min_epochs, self.eps = min_epochs, eps
        self.hist, self.prev_hs, self.d1_max = [], None, 0.0
        self.bad, self.knee = 0, None

    def step(self, ep, val):
        h = -math.log10(max(self.eps, 1.0 - val))
        self.hist.append(h)
        hs = float(np.mean(self.hist[-self.w:]))
        d1 = 0.0 if self.prev_hs is None else hs - self.prev_hs
        self.prev_hs = hs
        self.d1_max = max(self.d1_max, d1)
        if ep < self.min_epochs:
            return False, d1, self.d1_max
        if d1 < self.ratio * self.d1_max:
            self.bad += 1
            if self.bad >= self.streak and self.knee is None:
                self.knee = ep
                return True, d1, self.d1_max
        else:
            self.bad = 0
        return False, d1, self.d1_max


def run_one(seed, tag):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    mm = PerMolClassifier().to(DEVICE)
    tr_loader = DataLoader(tr_d, batch_size=args.bsz, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(mm.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup))
    WARMUP = args.warmup
    warmup_lrs = [args.lr * (ep + 1) / max(1, WARMUP) for ep in range(WARMUP)]
    stopper = (KneeStopper(args.knee_window, args.knee_ratio, args.knee_streak, args.min_epochs)
               if args.early_stop == 'knee' else None)

    ckpt = os.path.join(HERE, f'sisson_permol_cls{args.suffix}_{tag}.pt')
    CKPT_DIR = args.ckpt_dir or os.path.join(HERE, f'ckpts_{tag}')
    if args.save_each > 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    curve_path = os.path.join(HERE, f'curve_{tag}.csv')
    _f = open(curve_path, 'w', newline='', encoding='utf-8')
    _w = csv.DictWriter(_f, fieldnames=['ep', 'lr', 'loss', 'val', 'd1', 'd1_peak', 'sec'])
    _w.writeheader()

    t0 = time.time(); best_val, best_ep, best_sd, stop_ep = -1.0, -1, None, None
    for ep in range(args.epochs):
        if ep < WARMUP:
            for pg in opt.param_groups:
                pg['lr'] = warmup_lrs[ep]
        mm.train(); tl = 0.0
        for b in tr_loader:
            b = b.to(DEVICE)
            z = mm(b.x, b.edge_index, b.batch, b.mol_id, b.n_mol)
            loss = F.binary_cross_entropy_with_logits(z, b.y.float())
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        if ep >= WARMUP:
            sch.step()
        if args.save_each > 0 and (ep % args.save_each == 0 or ep == args.epochs - 1):
            torch.save(mm.state_dict(), os.path.join(CKPT_DIR, f'ep{ep:03d}.pt'))
        vauc, _ = evaluate(mm, va_loader)
        if vauc > best_val:
            best_val, best_ep = vauc, ep
            best_sd = {k: v.detach().cpu().clone() for k, v in mm.state_dict().items()}
        d1 = d1p = ''
        stopped = False
        if stopper is not None:
            stopped, d1, d1p = stopper.step(ep, vauc)
        msg = (f'[{tag}] ep {ep:3d} | lr {opt.param_groups[0]["lr"]:.1e} | '
               f'loss {tl / len(tr_loader):.4f} | val {vauc:.4f} (best {best_val:.4f}@{best_ep})'
               + (f' | d1 {d1:.4f}/{d1p:.4f}' if d1 != '' else '')
               + f' | {time.time() - t0:.0f}s')
        print(msg, flush=True)
        _w.writerow(dict(ep=ep, lr=f'{opt.param_groups[0]["lr"]:.3e}',
                         loss=round(tl / len(tr_loader), 6), val=round(vauc, 4),
                         d1=round(d1, 5) if d1 != '' else '',
                         d1_peak=round(d1p, 5) if d1p != '' else '',
                         sec=round(time.time() - t0)))
        _f.flush()
        if stopped:
            stop_ep = ep
            print(f'>> [{tag}] 拐点触发: 停于 ep{ep}，回载 best-val ckpt ep{best_ep} '
                  f'(val {best_val:.4f})', flush=True)
            break

    if best_sd is not None:
        mm.load_state_dict(best_sd)
    torch.save(mm.state_dict(), ckpt)
    t_auc, n_auc = evaluate(mm, te_loader)
    elapsed = time.time() - t0
    _f.close()
    print(f'>> [{tag}] ckpt ep{best_ep} | TEST macro-AUROC = {t_auc:.4f} '
          f'({n_auc}/{N} classes) | val {best_val:.4f} | '
          f'stop_ep {stop_ep if stop_ep is not None else args.epochs - 1} | {elapsed:.0f}s\n',
          flush=True)
    return dict(tag=tag, seed=seed, early_stop=args.early_stop,
                knee_window=args.knee_window, knee_ratio=args.knee_ratio,
                knee_streak=args.knee_streak, min_epochs=args.min_epochs,
                epochs=args.epochs, bsz=args.bsz, lr=args.lr, dim=args.dim,
                layers=args.layers, heads=args.heads, gin_layers=args.gin_layers,
                gin_hidden=args.gin_hidden, dropout=args.dropout, params=n_params,
                best_val=round(best_val, 4), best_ep=best_ep,
                stop_ep=stop_ep if stop_ep is not None else args.epochs - 1,
                test_auc=round(t_auc, 4), n_classes=n_auc, seconds=round(elapsed))


SEEDS = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]
rows = []
for sd in SEEDS:
    _tag = f'{TAG}_s{sd}' if len(SEEDS) > 1 else TAG
    rows.append(run_one(sd, _tag))

csv_path = os.path.join(HERE, 'results_multiseed.csv' if len(SEEDS) > 1 else 'results.csv')
write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if write_header:
        wr.writeheader()
    for r in rows:
        wr.writerow(r)

aucs = [r['test_auc'] for r in rows]
print('=' * 70, flush=True)
print(f'>> early-stop={args.early_stop} | seeds={SEEDS}', flush=True)
for r in rows:
    print(f"   seed {r['seed']}: ckpt ep{r['best_ep']} (stop@{r['stop_ep']}) "
          f"| val {r['best_val']:.4f} | TEST {r['test_auc']:.4f}", flush=True)
if len(aucs) > 1:
    print(f'>> macro-AUROC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (n={len(aucs)})',
          flush=True)
print('>> 锚点: Sisson MPNN-GNN 0.7627±0.0047 | GIN-GNN 0.7359±0.0222 | '
      '老架构 wm-cls74（全混合物） 0.7635', flush=True)
print(f'>> 结果: {csv_path}', flush=True)
