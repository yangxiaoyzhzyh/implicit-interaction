# 05_cross_domain — 非化学领域

同一形式 $y=R_\phi(\sum_i w_i v_i)$ 在两个互不相关的领域上复现，用来支持
“这是多体系统的结构，不是化学的特殊性”。两个实验各一个脚本，两条路线都在脚本内。

## 脚本

| 脚本 | 领域 | 设定 |
|---|---|---|
| `scripts/run_multidoc_set.py` | 文档集合 | $k$ 篇文档 → 集合级标签；冻结 BGE-M3 嵌入 + 可训练塑形模块；训练 $k\in\{1,2\}$，测试 $k\in\{5,10,20\}$ |
| `scripts/run_portfolio_vol.py` | 组合波动率 | $K$ 个资产 + 给定权重（**权重是输入，不是决策变量**）→ 持有期已实现波动率；训练 $K\le3$，测试 $K\in\{5,10,20\}$，扩展窗口时序划分并留一个持有期 gap |

文档集合有两种标签，同一脚本用 `--task` 切换：

- `set`（主导主题，20 类，chance ≈ 0.05）—— 难度随 $k$ **下降**（多数投票更可靠）；
- `homo`（是否全部同主题，chance = 0.5）—— 难度随 $k$ **上升**，与化学侧可比。

## 两个实验各自说明什么

- **同质文档集合**是最干净的一组：冻结嵌入平均后准确率停在 0.51–0.53（随机水平），
  而同样一阶加权和、只换成被集合塑形的编码器就到 0.68；同一批元素、同一聚合、
  同一读出，唯一变量是编码器。
- **组合波动率**的加性规则是 $\sum_i w_i\sigma_i$（分散化之前的加权平均波动率），
  分散化效应正是它够不到的残差；v(im) 在低 $K$ 领先（$K=5$ 时 +0.068，超过 0.047 的散布），
  $K=20$ 时两者持平。

## 本目录的待办

1. **输出路径未改**：两个脚本仍把结果写到旧 `results/`，重跑时需改到本目录 `summary/`。
2. **文档实验需重跑确认 seed 数**：现表为两 seed 均值，正文标注为“two-seed means”，
   若要 3 seed 需重跑。
3. **嵌入缓存未归档**：`multidoc_emb_20ng.npz`（BGE-M3 冻结嵌入）是重跑的前提，
   体积较大，建议放入 `ckpt/` 并注明来源与维度（18,846 文档 × 1024）。
4. **组合实验的价格缓存未归档**：`portfolio_prices.csv`（89 资产 × 2,514 交易日）
   同样需随 `ckpt/` 归档，否则无法复现同一划分。

## 归档清单（2026-10-30 补齐）

| 文件 | 内容 |
|---|---|
| `scripts/run_multidoc_set.py` | 文档集合：`--task set/homo`，冻结 BGE-M3 + 可训练塑形模块 |
| `scripts/run_portfolio_vol.py` | 组合波动率：K 个资产 + 给定权重 → 已实现波动率 |
| `runs/multidoc_set_results.csv` | 主导主题标签（`set`），acc 随 k 上升 |
| `runs/multidoc_set_homo_results.csv` | 同质判定标签（`homo`），acc 随 k 下降 |
| `runs/portfolio_vol_results.csv` | 3 折 × 2 seed × K∈{5,10,20} 的 R²，含 v(p)/v(w) |
| `logs/multidoc_full.log`、`logs/portfolio_vol.log` | 运行日志 |

20 Newsgroups 的嵌入缓存（`multidoc_emb_20ng.npz`，71 MB）未纳入归档，
由 `run_multidoc_set.py` 首次运行时重建。
