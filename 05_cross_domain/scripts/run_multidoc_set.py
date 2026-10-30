"""run_multidoc_set.py — 多文档推理（20 Newsgroups）跨领域复现

实验 15 — Multi-document reasoning（补材 S5，表 S14/S15）
路线 两条（v(im) 与 v(ex)）
产出 results/multidoc_set_results.csv、multidoc_set_homo_results.csv、multidoc_emb_20ng.npz；logs/multidoc_full.log

任务：k 篇文档 → 集合级标签（主导主题）。
结构对应：文档↔分子、k↔组分数、文档权重↔摩尔分数、集合标签↔混合物性质。

- v(im)：MLP( Σ wᵢ·MLP_doc(embᵢ) )       文档之间**无交互**
- v(ex)：MLP( Attn({MLP_doc(embᵢ)}) )     文档之间有注意力
- 加性基准：(1/k)Σ embᵢ → 分类器          （"平均文档嵌入"，最自然的朴素基线）

切分：训练 k ∈ {1,2} → 测试 k ∈ {5,10,20}。

为在 CPU 上跑得动：文档嵌入用冻结的 MiniLM 预先算好并缓存，训练只在小向量上进行
（MLP_doc 仍是可训练的，即"可塑编码器"这一角色保留）。

用法：
    python run_multidoc_set.py --build-emb    # 先算嵌入并缓存（约 1-3 分钟）
    python run_multidoc_set.py --smoke
    python run_multidoc_set.py --epochs 60
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(PAPER, 'results')
EMB_CACHE = os.path.join(RES, 'multidoc_emb_20ng.npz')

LM_BASE = 'http://localhost:1234/v1/embeddings'   # LM Studio（本地，不走代理）
EMB_MODEL = 'text-embedding-bge-m3'
N_CLASS = 20
MIX = 0.7          # 集合中主导类文档的占比
MAXCHARS = 512     # 文档截断长度（真实文档比短文本慢 10 倍，必须截短）
PER_CLASS = 0      # 每类最多取多少篇文档去编码；0 = 不限（全量）


_OPENER = None


def _opener():
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _OPENER


def _embed(texts, batch=64, retries=3):
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:MAXCHARS] for t in texts[i:i + batch]]
        body = json.dumps({'model': EMB_MODEL, 'input': chunk}).encode()
        for a in range(retries):
            try:
                req = urllib.request.Request(LM_BASE, data=body,
                                             headers={'Content-Type': 'application/json'})
                d = json.loads(_opener().open(req, timeout=180).read())
                out.append(np.array([x['embedding'] for x in d['data']], dtype=np.float32))
                break
            except Exception:
                if a == retries - 1:
                    raise
                time.sleep(3)
        if (i // batch) % 20 == 0:
            print(f'    embed {i}/{len(texts)}', flush=True)
    return np.concatenate(out)


def build_emb():
    from sklearn.datasets import fetch_20newsgroups

    t0 = time.time()
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    print(f'20NG: train {len(tr.data)} / test {len(te.data)}，类别 {len(tr.target_names)}',
          flush=True)

    rng = np.random.default_rng(0)

    def subsample(data, target):
        keep = []
        for c in range(N_CLASS):
            idx = np.where(target == c)[0]
            if PER_CLASS and len(idx) > PER_CLASS:
                idx = rng.choice(idx, PER_CLASS, replace=False)
            keep.extend(idx.tolist())
        keep = sorted(keep)
        return [data[i] for i in keep], np.array([target[i] for i in keep])

    tr_txt, tr_y = subsample(tr.data, tr.target)
    te_txt, te_y = subsample(te.data, te.target)
    print(f'取样后 train {len(tr_txt)} / test {len(te_txt)} 篇（每类 ≤{PER_CLASS}）', flush=True)
    print(f'BGE-M3 编码（LM Studio，{EMB_MODEL}，截断 {MAXCHARS} 字符）...', flush=True)
    Etr = _embed(tr_txt)
    print(f'  train done {Etr.shape}，用时 {time.time() - t0:.0f}s', flush=True)
    Ete = _embed(te_txt)
    print(f'  test done {Ete.shape}，总用时 {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(EMB_CACHE, Etr=Etr, ytr=tr_y, Ete=Ete, yte=te_y)
    print(f'缓存 → {EMB_CACHE}', flush=True)


def load_emb():
    d = np.load(EMB_CACHE)
    return d['Etr'], d['ytr'], d['Ete'], d['yte']


def build_sets(E, y, ks, n_per, rng, maxk=None, mix=MIX, task='majority'):
    """生成集合样本：X[n,maxk,D], W[n,maxk], M[n,maxk], Y[n]。

    task='majority'：标签 = 集合的主导主题（难度随 k **下降**——多数投票更可靠）
    task='homo'    ：标签 = 集合是否**全部同一主题**（难度随 k **上升**——一篇异类更难发现）
    """
    N, D = E.shape
    maxk = maxk or max(ks)
    X = np.zeros((len(ks) * n_per, maxk, D), dtype=np.float32)
    W = np.zeros((len(ks) * n_per, maxk), dtype=np.float32)
    M = np.zeros((len(ks) * n_per, maxk), dtype=bool)
    Y = np.zeros(len(ks) * n_per, dtype=np.int64)
    idx_by_class = [np.where(y == c)[0] for c in range(N_CLASS)]
    r = 0
    for k in ks:
        for j in range(n_per):
            if task == 'majority':
                c = int(rng.integers(N_CLASS))
                n_main = min(k, max(1, int(round(k * mix))))
                picks = list(rng.choice(idx_by_class[c], size=n_main, replace=False))
                rest = [i for i in range(N) if y[i] != c]
                picks += list(rng.choice(rest, size=k - n_main, replace=False))
                label = c
            else:                                   # 'homo'
                c = int(rng.integers(N_CLASS))
                if j % 2 == 0:                      # 正例：全同类
                    picks = list(rng.choice(idx_by_class[c], size=k, replace=False))
                    label = 1
                else:                               # 负例：恰好含 1 篇异类
                    picks = list(rng.choice(idx_by_class[c], size=k - 1, replace=False))
                    c2 = int((c + 1 + rng.integers(N_CLASS - 1)) % N_CLASS)
                    picks.append(int(rng.choice(idx_by_class[c2], size=1)[0]))
                    label = 0
            rng.shuffle(picks)
            X[r, :k] = E[picks]
            W[r, :k] = 1.0 / k                     # 等权（不提供额外权重信息）
            M[r, :k] = True
            Y[r] = label
            r += 1
    return X, W, M, Y


def make_models(D, torch, nn, n_out=N_CLASS):
    class Vp(nn.Module):
        """可塑编码器 + 给定组合（加权和），文档间无交互。"""
        def __init__(self, hid=256):
            super().__init__()
            self.E = nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n_out))

        def forward(self, X, W, M):
            z = self.E(X)
            z = (z * W.unsqueeze(-1)).sum(1)
            return self.head(z)

    class Vw(nn.Module):
        """全部文档一起过自注意力，CLS 读出；文档间有交互。"""
        def __init__(self, hid=192, nhead=4):
            super().__init__()
            self.emb = nn.Linear(D, hid)
            self.cls = nn.Parameter(torch.zeros(1, 1, hid))
            enc = nn.TransformerEncoderLayer(hid, nhead, dim_feedforward=hid,
                                             dropout=0.0, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, 1)
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n_out))

        def forward(self, X, W, M):
            n, k, _ = X.shape
            h = self.emb(X)
            h = torch.cat([self.cls.expand(n, -1, -1), h], 1)
            pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool), ~M], 1)
            h = self.enc(h, src_key_padding_mask=pad)
            return self.head(h[:, 0])

    class Additive(nn.Module):
        """加性基准：(1/k)Σ embᵢ → 线性分类器。"""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(D, n_out)

        def forward(self, X, W, M):
            m = M.unsqueeze(-1).float()
            z = (X * m).sum(1) / m.sum(1).clamp(min=1)
            return self.fc(z)

    class VpSmall(nn.Module):
        """v(im) 的小容量版本：用于检验"参数多反而差"是否为过拟合。"""
        def __init__(self, hid=32):
            super().__init__()
            self.E = nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(hid, n_out))

        def forward(self, X, W, M):
            z = self.E(X)
            z = (z * W.unsqueeze(-1)).sum(1)
            return self.head(z)

    return Vp, Vw, Additive, VpSmall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-emb', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    ap.add_argument('--n-train', type=int, default=4000, help='每个训练 k 的样本数')
    ap.add_argument('--n-test', type=int, default=600, help='每个测试 k 的样本数')
    ap.add_argument('--task', default='majority', choices=['majority', 'homo'],
                    help='majority=主导主题（难度随k下降）；homo=是否全同主题（难度随k上升）')
    ap.add_argument('--test-ks', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.build_emb or not os.path.exists(EMB_CACHE):
        build_emb()
    if args.smoke:
        args.epochs, args.seeds, args.n_train, args.n_test = 3, [42], 300, 150

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    Etr, ytr, Ete, yte = load_emb()
    D = Etr.shape[1]
    print(f'嵌入 train {Etr.shape} / test {Ete.shape}', flush=True)

    n_out = N_CLASS if args.task == 'majority' else 2
    train_ks = (1, 2) if args.task == 'majority' else (2, 3)
    rng = np.random.default_rng(0)
    Xtr, Wtr, Mtr, Ytr = build_sets(Etr, ytr, train_ks, args.n_train, rng,
                                    maxk=max(train_ks), task=args.task)
    print(f'训练集合样本 {len(Ytr)}（k∈{train_ks}，task={args.task}，n_out={n_out}）', flush=True)
    tests = {}
    for k in args.test_ks:
        tests[k] = build_sets(Ete, yte, (k,), args.n_test, rng, maxk=max(args.test_ks),
                              task=args.task)[:4]

    dev = 'cpu'
    Vp, Vw, Additive, VpSmall = make_models(D, torch, nn, n_out)
    for nm, cls in (('v(im)', Vp), ('v(im)-small', VpSmall), ('v(ex)', Vw), ('加性基准', Additive)):
        print(f'  {nm} params = {sum(p.numel() for p in cls().parameters())}', flush=True)

    def acc(logits, y):
        return float((logits.argmax(1) == torch.tensor(y)).float().mean())

    rows = []
    for name, build in (('加性基准', Additive), ('v(im)', Vp), ('v(im)-small', VpSmall), ('v(ex)', Vw)):
        for seed in args.seeds:
            torch.manual_seed(seed)
            m = build()
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            n = len(Ytr)
            Xt = torch.tensor(Xtr); Wt = torch.tensor(Wtr); Mt = torch.tensor(Mtr)
            Yt = torch.tensor(Ytr)
            for ep in range(args.epochs):
                m.train()
                perm = torch.randperm(n)
                for i in range(0, n, 128):
                    b = perm[i:i + 128]
                    opt.zero_grad()
                    F.cross_entropy(m(Xt[b], Wt[b], Mt[b]), Yt[b]).backward()
                    opt.step()
            m.eval()
            line = f'  {name:<8} seed{seed}:'
            with torch.no_grad():
                for k in args.test_ks:
                    Xk, Wk, Mk, Yk = tests[k]
                    a = acc(m(torch.tensor(Xk), torch.tensor(Wk), torch.tensor(Mk)), Yk)
                    line += f'  k={k} acc={a:.3f}'
                    rows.append((name, seed, k, a))
            print(line, flush=True)

    print('\n=== 汇总（跨 seed 均值）===')
    names = ['加性基准', 'v(im)', 'v(im)-small', 'v(ex)']
    print(f'{"k":>4}' + ''.join(f'{n:>13}' for n in names))
    for k in args.test_ks:
        f = lambda nm: np.mean([r[3] for r in rows if r[0] == nm and r[2] == k])
        print(f'{k:>4}' + ''.join(f'{f(n):>13.3f}' for n in names))

    out = os.path.join(RES, f'multidoc_set_{args.task}_results.csv')
    with open(out, 'w', encoding='utf-8', newline='') as fh:
        fh.write('model,seed,k,acc\n')
        for r in rows:
            fh.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.4f}\n')
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
