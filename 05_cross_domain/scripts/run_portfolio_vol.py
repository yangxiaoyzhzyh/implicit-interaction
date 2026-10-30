"""run_portfolio_vol.py — 组合波动率（金融）跨领域复现

实验 14 — Portfolio volatility（补材 S5，表 S13）
路线 两条（v(im) 与 v(ex)）
产出 results/portfolio_vol_results.csv；logs/portfolio_vol.log、.err
价格数据由脚本下载，不随仓库分发

**目的只有一个**：验证"跨领域性"——在非化学领域复现
"v(im) 在组分轴外推上稳住、v(ex) 掉下去"。

**不做** ρ 分层、**不做**解析解对照（那些属于"深入探究交互强度"，超范围）。

任务：给定组合（K 个资产 + 权重 w，w 是**输入**而非决策变量）→ 预测持有期已实现波动率。
加性基准：Σ wᵢσᵢ（加权平均波动率，"分散化之前"）。
切分：训练 K ≤ 3 → 测试 K ∈ {5, 10, 20}；**多折时序切分**（expanding window），
      每折独立标准化并留一个持有期的 gap，避免标签窗口跨折泄漏。

⚠ 为什么不用单次 70/30：实测后 30% 落在单一低波动 regime 里，测试段标签方差仅为训练段的
1/5（std 0.013 vs 0.071），R² 会被数学性地压塌（加性基准 R² = −29）。多折能让各折 regime 不同。

数据：Yahoo chart API（走代理 127.0.0.1:7897），缓存到 results/portfolio_prices.csv。

用法：
    python run_portfolio_vol.py --download           # 只下载数据
    python run_portfolio_vol.py --smoke
    python run_portfolio_vol.py --epochs 200
"""
import argparse
import csv
import json
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(PAPER, 'results')
CACHE = os.path.join(RES, 'portfolio_prices.csv')

PROXY = 'http://127.0.0.1:7897'

# 资产池（S&P 100 主要成分）
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'V',
    'UNH', 'XOM', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV',
    'MRK', 'KO', 'PEP', 'AVGO', 'COST', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
    'LIN', 'ADBE', 'CRM', 'DHR', 'NKE', 'TXN', 'NEE', 'WFC', 'PM', 'BMY',
    'UNP', 'RTX', 'QCOM', 'AMGN', 'LOW', 'HON', 'ORCL', 'INTU', 'COP', 'IBM',
    'CAT', 'AMAT', 'GE', 'BA', 'SBUX', 'GS', 'BLK', 'ISRG', 'MDT', 'AXP',
    'DE', 'BKNG', 'T', 'VZ', 'PLD', 'MMM', 'ADI', 'GILD', 'MDLZ', 'ADP',
    'CVS', 'LMT', 'SYK', 'TJX', 'CB', 'MO', 'ZTS', 'CI', 'PGR', 'USB',
    'SO', 'DUK', 'BDX', 'ITW', 'NOC', 'FIS', 'EMR', 'ETN', 'APD', 'SHW',
]

WIN_FEAT = 60      # 特征窗口（日）
WIN_HOLD = 20      # 持有期（日），即标签窗口
STEP = 5           # 采样步长
N_COMB = 25        # 每个时间点生成的随机组合数
N_FOLDS = 4        # 时序折数


# --------------------------------------------------------------------------- #
def download():
    os.makedirs(RES, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))
    series = {}
    t0 = time.time()
    for i, tk in enumerate(TICKERS):
        url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{tk}'
               f'?range=10y&interval=1d')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            d = json.loads(opener.open(req, timeout=30).read())
            r = d['chart']['result'][0]
            ser = {t: c for t, c in zip(r['timestamp'],
                                        r['indicators']['quote'][0]['close'])
                   if c is not None}
            series[tk] = ser
            print(f'  [{i + 1}/{len(TICKERS)}] {tk}: {len(ser)} 天', flush=True)
        except Exception as e:
            print(f'  [{i + 1}/{len(TICKERS)}] {tk}: 失败 {type(e).__name__}', flush=True)
    common = None
    for ser in series.values():
        common = set(ser) if common is None else (common & set(ser))
    common = sorted(common)
    print(f'公共交易日 {len(common)} 天，成功 {len(series)}/{len(TICKERS)} 个资产，'
          f'用时 {time.time() - t0:.0f}s', flush=True)
    with open(CACHE, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date'] + list(series.keys()))
        for t in common:
            w.writerow([t] + [f'{series[tk][t]:.6f}' for tk in series])
    print(f'saved {CACHE}', flush=True)


def load():
    with open(CACHE, encoding='utf-8') as f:
        r = csv.reader(f)
        head = next(r)
        rows = [row for row in r]
    dates = np.array([int(row[0]) for row in rows])
    P = np.array([[float(x) for x in row[1:]] for row in rows])
    return head[1:], dates, P


def build_all(P, ks, n_comb, rng, step=STEP, maxk=None):
    """一次性生成全部样本并附时间索引（供多折时序切分）。

    特征窗口 [t-WIN_FEAT, t) 不含未来；标签窗口 [t, t+WIN_HOLD)。
    返回 X[n,k,d], W[n,k], M[n,k], Y[n], A[n], Tidx[n]。
    """
    R = np.diff(np.log(P), axis=0)
    T, N = R.shape
    Xs, Ws, Ms, Ys, As, Ts = [], [], [], [], [], []
    maxk = maxk or max(ks)
    for t in range(WIN_FEAT, T - WIN_HOLD, step):
        hist = R[t - WIN_FEAT:t]
        fut = R[t:t + WIN_HOLD]
        sd = hist.std(0) + 1e-12
        skew = ((hist - hist.mean(0)) ** 3).mean(0) / sd ** 3
        kurt = ((hist - hist.mean(0)) ** 4).mean(0) / sd ** 4
        fmat = np.stack([sd, skew, kurt], 1)
        for _ in range(n_comb):
            k = int(rng.choice(ks))
            idx = rng.choice(N, size=k, replace=False)
            w = rng.dirichlet(np.ones(k))
            port = fut[:, idx] @ w
            y = float(np.log(port.std() + 1e-12))
            X = np.zeros((maxk, fmat.shape[1]), dtype=np.float32)
            W = np.zeros(maxk, dtype=np.float32)
            M = np.zeros(maxk, dtype=bool)
            X[:k], W[:k], M[:k] = fmat[idx], w, True
            Xs.append(X); Ws.append(W); Ms.append(M); Ys.append(y)
            As.append(float((w * sd[idx]).sum()))
            Ts.append(t)
    return (np.stack(Xs), np.stack(Ws), np.stack(Ms),
            np.array(Ys, dtype=np.float32), np.array(As, dtype=np.float32),
            np.array(Ts))


def make_models(D, torch, nn):
    class Vp(nn.Module):
        """可塑编码器 + 给定组合（加权和）。"""
        def __init__(self, hid=96):
            super().__init__()
            self.E = nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

        def forward(self, X, W, M):
            z = self.E(X)
            z = (z * W.unsqueeze(-1)).sum(1)
            return self.head(z).squeeze(-1)

    class Vw(nn.Module):
        """整组 token 一起过自注意力，CLS 读出；组合规则学出来。"""
        def __init__(self, hid=48, nhead=2):
            super().__init__()
            self.emb = nn.Linear(D + 1, hid)
            self.cls = nn.Parameter(torch.zeros(1, 1, hid))
            enc = nn.TransformerEncoderLayer(hid, nhead, dim_feedforward=hid,
                                             dropout=0.0, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, 1)
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

        def forward(self, X, W, M):
            n, k, _ = X.shape
            h = self.emb(torch.cat([X, W.unsqueeze(-1)], -1))
            h = torch.cat([self.cls.expand(n, -1, -1), h], 1)
            pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool), ~M], 1)
            h = self.enc(h, src_key_padding_mask=pad)
            return self.head(h[:, 0]).squeeze(-1)

    return Vp, Vw


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    ap.add_argument('--n-comb', type=int, default=N_COMB)
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.download:
        download()
        return
    if not os.path.exists(CACHE):
        download()
    if args.smoke:
        args.epochs, args.seeds = 3, [42]

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    tk, dates, P = load()
    print(f'载入 {P.shape[0]} 天 × {P.shape[1]} 个资产', flush=True)
    rng = np.random.default_rng(0)

    Xtr_all, Wtr_all, Mtr_all, Ytr_all, Atr_all, Ttr = build_all(
        P, (1, 2, 3), args.n_comb, rng, maxk=3)
    Xte_all, Wte_all, Mte_all, Yte_all, Ate_all, Tte = build_all(
        P, (5, 10, 20), args.n_comb, rng, maxk=20)
    print(f'样本池：训练 K≤3 = {len(Ytr_all)} | 测试 K∈{{5,10,20}} = {len(Yte_all)}',
          flush=True)

    D = Xtr_all.shape[-1]
    Vp, Vw = make_models(D, torch, nn)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={dev} | v(im) params={sum(p.numel() for p in Vp().parameters())} |'
          f' v(ex) params={sum(p.numel() for p in Vw().parameters())}', flush=True)

    t_lo, t_hi = int(Tte.min()), int(Tte.max())
    edges = np.linspace(t_lo, t_hi, args.folds + 1).astype(int)
    print(f'时序折：t∈[{t_lo}, {t_hi}] → edges={list(edges)}', flush=True)

    rows = []
    for fi in range(1, args.folds):
        cut = edges[fi]
        tr = Ttr < cut - WIN_HOLD                    # 标签窗口完全在 cut 之前
        te = (Tte >= edges[fi - 1]) & (Tte < cut)
        if tr.sum() < 500 or te.sum() < 50:
            continue
        Xtr, Wtr, Mtr, Ytr, Atr = (Xtr_all[tr], Wtr_all[tr], Mtr_all[tr],
                                   Ytr_all[tr], Atr_all[tr])
        Xte, Wte, Mte, Yte, Ate, Kte = (Xte_all[te], Wte_all[te], Mte_all[te],
                                        Yte_all[te], Ate_all[te], Tte[te])
        Kte = np.array([int(Mte[i].sum()) for i in range(len(Mte))])

        # 该折独立标准化（只用该折训练段统计量）
        mu, sd = Xtr[Mtr].mean(0), Xtr[Mtr].std(0) + 1e-8
        Xtr = Xtr.copy(); Xte = Xte.copy()
        Xtr[Mtr] = (Xtr[Mtr] - mu) / sd
        Xte[Mte] = (Xte[Mte] - mu) / sd
        ym, ysd = Ytr.mean(), Ytr.std()
        Ytr_n, Yte_n = (Ytr - ym) / ysd, (Yte - ym) / ysd

        A = np.stack([Atr, np.ones_like(Atr)], 1)
        coef = np.linalg.lstsq(A, Ytr_n, rcond=None)[0]
        a_pred = np.stack([Ate, np.ones_like(Ate)], 1) @ coef

        print(f'\n--- 折 {fi}: 训练 {int(tr.sum())} | 测试 {int(te.sum())} '
              f'(y_train std={Ytr.std():.3f}, y_test std={Yte.std():.3f}) ---', flush=True)
        print('  加性基准 Σwᵢσᵢ: ' +
              ' | '.join(f'K={k} R²={r2(a_pred[Kte == k], Yte_n[Kte == k]):+.3f}'
                         for k in (5, 10, 20)), flush=True)
        for name, build in (('v(im)', Vp), ('v(ex)', Vw)):
            for seed in args.seeds:
                torch.manual_seed(seed)
                m = build().to(dev)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                n = len(Ytr)
                for ep in range(args.epochs):
                    m.train()
                    perm = torch.randperm(n)
                    for i in range(0, n, 512):
                        idx = perm[i:i + 512]
                        Xb = torch.tensor(Xtr[idx]).to(dev)
                        Wb = torch.tensor(Wtr[idx]).to(dev)
                        Mb = torch.tensor(Mtr[idx]).to(dev)
                        yb = torch.tensor(Ytr_n[idx]).to(dev)
                        opt.zero_grad()
                        F.mse_loss(m(Xb, Wb, Mb), yb).backward()
                        opt.step()
                m.eval()
                with torch.no_grad():
                    p = m(torch.tensor(Xte).to(dev), torch.tensor(Wte).to(dev),
                          torch.tensor(Mte).to(dev)).cpu().numpy()
                line = f'  {name} seed{seed}:'
                for k in (5, 10, 20):
                    v = r2(p[Kte == k], Yte_n[Kte == k])
                    line += f'  K={k} R²={v:+.3f}'
                    rows.append((fi, name, seed, k, v))
                print(line, flush=True)

    print(f'\n=== 汇总（跨折 × 跨 seed 均值）===')
    print(f'{"K":>4}{"v(im)":>10}{"v(ex)":>10}{"gap":>10}')
    summ = []
    for k in (5, 10, 20):
        a = np.mean([r[4] for r in rows if r[1] == 'v(im)' and r[3] == k])
        b = np.mean([r[4] for r in rows if r[1] == 'v(ex)' and r[3] == k])
        print(f'{k:>4}{a:>10.3f}{b:>10.3f}{a - b:>10.3f}')
        summ.append((k, a, b))

    out = os.path.join(RES, 'portfolio_vol_results.csv')
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write('fold,model,seed,K,r2\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f}\n')
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
