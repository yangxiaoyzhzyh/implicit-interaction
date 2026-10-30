"""run_synth_routes.py — 合成组合回归：v(im) 与 v(ex) 的组分数外推（机理验证）

实验 13 — Synthetic families A / B / C（补材 S4）
路线 两条（v(im) 与 v(ex)，同一生成数据、同协议）
产出 results/synth_routes_results.csv（列名 vp_r2 / vw_r2 为历史标签，分别对应 v(im) / v(ex) 的 R²）；logs/synth_routes.log

**定位**：在一个**与 K 无关**的生成函数上，检验"组合规则给定（加权和）vs 学出来（有交互层）"
在组分数外推上的差异。自造数据只用于**机理**，不承担泛化证据（那由真实领域实验负责）。

**三条写作约定（一并声明，避免循环论证）**：

1. **生成函数与 K 无关**——同一函数对任意 K 都有定义，形式不随 K 变；K 只体现在**训练集的
   覆盖范围**（只喂 K≤2）。因此外推是"分布的性质"，不是"函数的性质"。
2. **方向性预测事先声明**——随交互强度 λ 增大，v(ex) 的组分数外推应当**单调变差**；λ=0 时两者
   应当持平。若事后挑 λ 即为 p-hacking。
3. **多族 + 一个对抗性族**——对抗族的组合规则**不可写成"元素贡献的加权和"**，用来检验
   v(ex) 是否会翻盘（若翻盘，则边界被找到；若不翻盘，结论更强）。

**生成器族**：

    add_pair     y = Σᵢ wᵢ·uᵢ + λ Σ_{i<j} wᵢwⱼ·uᵢuⱼ        （可加 + 成对交互，λ 控制强度）
    add_hidden   y = tanh( c · Σᵢ wᵢ·(A xᵢ) )                （可加但经隐藏非线性）
    pair_hidden  y = Σ_{i<j} wᵢwⱼ·h(xᵢ,xⱼ)   h 为随机 MLP   （纯成对交互；对抗性：不可分解）

**模型（参数量对齐）**：

    v(im)   y = MLP( Σᵢ wᵢ · E(xᵢ) )                    E 为逐元素 MLP，元素之间**无交互**
    v(ex)   y = MLP( meanpool( Attn( {[xᵢ, wᵢ]} ) ) )    元素之间有自注意力（组合规则学出来）

用法：
    python run_synth_routes.py --smoke
    python run_synth_routes.py --seeds 42 43 44 --epochs 300

归档版本（repro_v2/04_synthetic/scripts）：只增加了 --out 参数，模型与数据生成
逻辑逐字未改，因此与旧归档（家族 A）同源可比。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 8                      # 每个"元素"的固有特征维
TRAIN_KS = (1, 2)            # 训练只覆盖 K≤2
TEST_KS = (3, 5, 10, 20)     # 外推
LAMBDAS = (0.0, 0.1, 0.5, 1.0, 2.0)
FAMILIES = ('add_pair', 'add_hidden', 'pair_hidden')


# --------------------------------------------------------------------------- #
# 生成器
# --------------------------------------------------------------------------- #
class Generator:
    """固定一套随机参数，对任意 K 生成数据（函数本身与 K 无关）。"""

    def __init__(self, family, d=DIM, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.family = family
        self.d = d
        self.A = torch.randn(d, d, generator=g) * (1.0 / np.sqrt(d))
        self.a = torch.randn(d, generator=g) * (1.0 / np.sqrt(d))
        self.b = torch.randn(d, generator=g) * (1.0 / np.sqrt(d))
        # pair_hidden 的 h：把 (xᵢ, xⱼ) 映到标量
        self.h1 = torch.randn(2 * d, 32, generator=g) * (1.0 / np.sqrt(2 * d))
        self.h2 = torch.randn(32, 1, generator=g) * (1.0 / np.sqrt(32))

    def sample(self, n, ks, lam=0.0, gen=None):
        """返回 (X [n,maxK,d], W [n,maxK], mask [n,maxK], y [n])。"""
        maxk = max(ks)
        xs, ws, ys = [], [], []
        for _ in range(n):
            k = int(ks[np.random.randint(len(ks))])
            x = torch.randn(k, self.d)
            w = torch.rand(k)
            w = w / w.sum()                       # 摩尔分数（无序集合的权重）
            xs.append(x)
            ws.append(w)
            ys.append(self.forward(x, w, lam))
        X = torch.zeros(n, maxk, self.d)
        W = torch.zeros(n, maxk)
        M = torch.zeros(n, maxk, dtype=torch.bool)
        y = torch.stack(ys)
        for i, (x, w) in enumerate(zip(xs, ws)):
            k = x.shape[0]
            X[i, :k], W[i, :k], M[i, :k] = x, w, True
        return X, W, M, y

    def forward(self, x, w, lam):
        k = x.shape[0]
        if self.family == 'add_pair':
            u = x @ self.a                                   # [k]
            lin = (w * u).sum()
            if lam > 0 and k > 1:
                # Σ_{i<j} wᵢwⱼ uᵢuⱼ = ½[(Σ wᵢuᵢ)² − Σ wᵢ²uᵢ²]
                pair = 0.5 * (lin ** 2 - (w.pow(2) * u.pow(2)).sum())
            else:
                pair = torch.zeros(())
            return lin + lam * pair
        if self.family == 'add_hidden':
            z = (w.unsqueeze(1) * (x @ self.A.T)).sum(0)     # [d]
            return torch.tanh(z @ self.a)
        if self.family == 'pair_hidden':
            if k < 2:
                return torch.zeros(())
            xi = x.unsqueeze(0).expand(k, k, self.d)
            xj = x.unsqueeze(1).expand(k, k, self.d)
            pair_in = torch.cat([xi, xj], dim=-1)
            h = torch.tanh(pair_in @ self.h1) @ self.h2      # [k,k,1]
            h = h.squeeze(-1)
            ww = w.unsqueeze(0) * w.unsqueeze(1)
            tri = torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)
            return (ww * h)[tri].sum()
        raise ValueError(self.family)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class Vp(nn.Module):
    """v(im)：逐元素编码 → 摩尔分数加权和 → MLP。元素之间无交互。"""

    def __init__(self, d=DIM, hid=120):
        super().__init__()
        self.E = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, X, W, M):
        z = self.E(X)                       # [n,k,hid]
        z = (z * W.unsqueeze(-1)).sum(1)    # 加权和（组合规则给定）
        return self.head(z).squeeze(-1)


class Vw(nn.Module):
    """v(ex)：整锅 token 一起过自注意力 → CLS 读出 → MLP。组合规则学出来。

    用 CLS token 读出（而非 mean-pool），是为了消除"池化时除以 K"这一与 K 相关的
    人为尺度问题——否则即使在纯可加任务上也会退化，会污染"组合规则学不学得会"这个结论。
    """

    def __init__(self, d=DIM, hid=64, nhead=2, layers=1):
        super().__init__()
        self.emb = nn.Linear(d + 1, hid)
        self.cls = nn.Parameter(torch.zeros(1, 1, hid))
        enc = nn.TransformerEncoderLayer(hid, nhead, dim_feedforward=hid,
                                         dropout=0.0, batch_first=True,
                                         norm_first=False)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, X, W, M):
        n, k, _ = X.shape
        tok = torch.cat([X, W.unsqueeze(-1)], dim=-1)        # [n,k,d+1]
        h = self.emb(tok)
        h = torch.cat([self.cls.expand(n, -1, -1), h], dim=1)  # 前置 CLS
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool), ~M], dim=1)
        h = self.enc(h, src_key_padding_mask=pad)             # 元素之间有交互
        return self.head(h[:, 0]).squeeze(-1)                 # CLS 读出（与 K 无关）


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# 训练 / 评测
# --------------------------------------------------------------------------- #
def r2(p, y):
    return float(1.0 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run_one(family, lam, seed, epochs, n_train, n_test, device, verbose=False,
            train_ks=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = Generator(family, seed=1000 + seed)                 # 生成器固定（不随 λ 变）
    Xtr, Wtr, Mtr, ytr = gen.sample(n_train, train_ks or TRAIN_KS, lam)
    te = {k: gen.sample(n_test, (k,), lam) for k in TEST_KS}
    # 标准化 y（用训练集统计量）
    ym, ys = ytr.mean(), ytr.std()
    ytr_n = (ytr - ym) / ys
    te_n = {k: ((t[3] - ym) / ys, t) for k, t in te.items()}

    out = {}
    for name, build in (('v(im)', Vp), ('v(ex)', Vw)):
        torch.manual_seed(seed)
        model = build().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        n = Xtr.shape[0]
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                Xb, Wb, Mb, yb = Xtr[idx], Wtr[idx], Mtr[idx], ytr_n[idx]
                Xb, Wb, Mb, yb = Xb.to(device), Wb.to(device), Mb.to(device), yb.to(device)
                opt.zero_grad()
                loss = F.mse_loss(model(Xb, Wb, Mb), yb)
                loss.backward()
                opt.step()
        model.eval()
        res = {}
        with torch.no_grad():
            for k in TEST_KS:
                yn, t = te_n[k]
                X, W, M, _ = t
                p = model(X.to(device), W.to(device), M.to(device)).cpu()
                res[k] = r2(p, yn)
        out[name] = res
        if verbose:
            print(f'    {name}: ' + ' | '.join(f'K={k} R²={res[k]:+.3f}' for k in TEST_KS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--n-train', type=int, default=5000)
    ap.add_argument('--n-test', type=int, default=2000)
    ap.add_argument('--families', nargs='+', default=list(FAMILIES))
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.5, 2.0])
    ap.add_argument('--train-ks', type=int, nargs='+', default=None,
                    help='覆盖 TRAIN_KS，例如 2 3；测试集仍为 3,5,10,20')
    ap.add_argument('--out', default=None,
                    help='结果 CSV 路径；默认写到旧 results/，归档时请显式指定')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.n_test = 3, 600, 200
        args.seeds = args.seeds[:1]
        args.families = ['add_pair']
        args.lambdas = [0.0, 1.0]

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ks = tuple(args.train_ks) if args.train_ks else TRAIN_KS
    print(f'device={dev} | families={args.families} | lambdas={args.lambdas} | '
          f'seeds={args.seeds} | epochs={args.epochs} | train_ks={train_ks}',
          flush=True)
    print(f'参数量：v(im)={n_params(Vp())} | v(ex)={n_params(Vw())}', flush=True)

    t0 = time.time()
    rows = []
    for fam in args.families:
        for lam in args.lambdas:
            acc = {k: {'v(im)': [], 'v(ex)': []} for k in TEST_KS}
            for s in args.seeds:
                r = run_one(fam, lam, s, args.epochs, args.n_train, args.n_test, dev,
                            train_ks=train_ks)
                for k in TEST_KS:
                    acc[k]['v(im)'].append(r['v(im)'][k])
                    acc[k]['v(ex)'].append(r['v(ex)'][k])
            print(f'\n=== {fam}  λ={lam} ===')
            print(f'{"K":>4}{"v(im) R²":>12}{"v(ex) R²":>12}{"gap":>10}')
            for k in TEST_KS:
                a = float(np.mean(acc[k]['v(im)']))
                b = float(np.mean(acc[k]['v(ex)']))
                print(f'{k:>4}{a:>12.3f}{b:>12.3f}{a - b:>10.3f}')
                rows.append((fam, lam, k, a, b, a - b))

    out = args.out or os.path.join(HERE, '..', '..', 'results',
                                   'synth_routes_results.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write('family,lambda,K,vp_r2,vw_r2,gap\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n')
    print(f'\nsaved {out} | 总用时 {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
