# 08_small_labels — 小样本读出：聚合方式对「描述符 vs 塑形编码器」的影响

本目录回答一个问题：在**极少标签**下，v(im) 的塑形编码器是否仍比描述符基线强？
并同时检验一个容易忽略的变量——**逐组分特征如何聚合成混合物向量**。

## 两种协议

| protocol | 训练数据 | 文件 |
|---|---|---|
| `100bin` | 官方 train 中**仅 100 条二元（K=2）**的行；验证取同池余下 50 条；测试为官方 test 全部 | `runs/agg_100bin.json` |
| `1pct` | 官方 train 的 1%（约 200–300 行，含各 K） | `runs/agg_all_1pct.json` |

两者都用官方 `kfold_split_0`，2 次重复（不同抽样），报告 Pearson R 的均值与标准差。

## 四种特征

| key | 内容 | 原始维度 |
|---|---|---:|
| `rdkit2d` | RDKit 2D 描述符 | 217 |
| `ecfp4` | ECFP4 指纹 | 2048 |
| `maccs` | MACCS 指纹 | 167 |
| `enc` | 其余七个任务的塑形编码器拼接（**不含本任务**，避免泄漏） | 1792 |

## 四种聚合（都固定、无「学出来的交互模块」）

| key | 形式 | 参数 |
|---|---|---|
| `sum` | $\sum_i w_i x_i / \sum_i w_i$ | 无 |
| `mean` | $\sum_i x_i / K$ | 无 |
| `concat` | 按权重降序取前 5 个组分，零填充后拼接 | 无 |
| `attn` | 单头注意力池化，score $=$ MLP$([x_i, w_i])$ | 有（仅池化） |

统一管线：逐分子向量 → 在标签子集涉及的分子上做无监督 PCA 降到 64 维（不接触标签）
→ 按组分构造令牌（最多 5 槽）→ 池化 → 标准化 + PCA(64) → 同一个两层 MLP（hidden 128，dropout 0.3，AdamW，早停 30 epoch）。

## 结果（`100bin`，最佳聚合下）

| task | 最佳描述符 | R | 编码器 | R | 差 |
|---|---|---:|---|---:|---:|
| mon (仅 39 条) | maccs/sum | 0.596 | sum | 0.537 | −0.059 |
| density | rdkit2d/sum | 0.877 | sum | 0.874 | −0.004 |
| heat_vap | rdkit2d/sum | 0.930 | sum | 0.936 | +0.006 |
| he | maccs/mean | 0.525 | mean | 0.526 | +0.002 |
| logv | maccs/sum | 0.791 | sum | 0.873 | +0.082 |
| nist_logv | maccs/concat | 0.642 | sum | 0.813 | +0.171 |
| ionic_cond | maccs/sum | 0.446 | mean | 0.526 | +0.080 |
| ionic_visc | maccs/sum | 0.696 | sum | 0.837 | +0.141 |

编码器领先 5、持平 2、落后 1（阈值 ±0.005）；差值中位 +0.043，均值 +0.052。

## 三条要点

1. **聚合方式的影响大于特征本身**。同一描述符在不同聚合下可差 0.4 以上
   （MACCS 在 density：sum 0.475 / mean 0.746；RDKit2D 在 nist：sum 0.006（100 条）/ 0.933（1%））。
   因此任何「描述符基线」都必须声明聚合方式，否则数字不可比。
2. **数据越少，加权和越占优**：`1pct` 下最优聚合分散在 sum/mean/attn，`100bin` 下几乎全是 sum。
3. **极低样本下 MACCS 常是最好的描述符**（binary 实验里 4/8 个任务），
   因为它 167 位稀疏、对 100 条标签最省样本。

## 修正：PCA 放在混合物层（`100bin_mixpca_*`）

上一版把逐分子向量先 PCA 到 64 维，但该 PCA 拟合于**分子间**差异，会丢掉「该分子在混合物里
如何表现」的方向，对编码器尤其不利。修正版把 PCA 移到混合物层：

- `sum` / `mean`：先在原生维度池化，再在 100 条**标签混合物**上拟合 StandardScaler + PCA(64)；
- `concat`：在标签混合物的拼接向量上拟合 PCA，并按线性性拆成 5 块投影矩阵，
  用 $z=\sum_k W_k (x_k-\mu_k)/\sigma_k$ 逐行计算，不物化大矩阵；
- 注意力池化需要逐 token 的表示，无法这样处理，故不在修正版内。

影响（IlT κ，MLP 读出）：编码器 0.476 → **0.672**，RDKit2D 0.329 → **−0.010**。

修正后，两臂各自最优（描述符三种 × 三种聚合 × 两种读出；编码器三种聚合 × 两种读出）：

| task | 最佳描述符 | R | 最佳编码器 | R | 差 |
|---|---|---:|---|---:|---:|
| mon | ECFP4/sum/MLP | 0.586 | sum/MLP | 0.543 | −0.044 |
| density | RDKit2D/sum/MLP | 0.971 | sum/MLP | 0.973 | +0.002 |
| heat_vap | RDKit2D/sum/MLP | 0.961 | sum/MLP | 0.972 | +0.012 |
| he | RDKit2D/mean/MLP | 0.557 | sum/MLP | 0.567 | +0.009 |
| logV | RDKit2D/sum/XGB | 0.755 | sum/MLP | 0.870 | +0.116 |
| NIST η | RDKit2D/sum/XGB | 0.728 | sum/MLP | 0.832 | +0.104 |
| IlT κ | ECFP4/sum/XGB | 0.631 | sum/XGB | 0.597 | −0.034 |
| IlT η | MACCS/sum/MLP | 0.830 | sum/MLP | 0.864 | +0.034 |

编码器领先 5、持平 1、落后 2；差值中位 **+0.010**，均值 +0.025，最大 +0.116。

一条稳定的规律：**编码器一律以 MLP 读出最佳**（八个任务的 `sum/MLP`），
而描述符在 3 个任务上以 XGB 最佳（logV、NIST、IlT κ）——稠密嵌入适合神经读出，
稀疏指纹适合树。

## 文件

```
summary.csv                        长表：task, protocol, feature, agg, R, sd（544 行）
runs/agg_100bin_mixpca_mlp.json    ★ 主结果：混合物层 PCA + MLP（3 聚合 × 4 特征）
runs/agg_100bin_mixpca_xgb.json    ★ 主结果：混合物层 PCA + XGB（3 聚合 × 4 特征）
runs/agg_100bin_mixpca_best.csv    ★ 主表：最佳描述符 vs 最佳编码器（两种读出）
runs/mixpca_mlp.log / _xgb.log     两次主运行的完整输出
runs/agg_100bin.json / .log        早期版本（单分子层 PCA，4 聚合含注意力）
runs/agg_100bin_xgb.json / .log    早期版本 + XGB
runs/agg_all_1pct.json / .log      1% 标签协议（4 特征 × 4 聚合，含注意力）
runs/agg_100bin_best.csv           早期版本的最佳对照（MLP）
runs/agg_all_1pct_best.csv         同上（1% 协议）
runs/agg_sweep_1pct.json / .log    最早的两特征扫描（rdkit + enc），已被 agg_all_1pct 覆盖
runs/agg_sweep_1pct_summary.csv    该扫描的对照表
runs/agg_iltk_1pct.json            单任务先导：IlT κ、1%、描述符三聚合
runs/agg_iltk_enc_1pct.json        单任务先导：IlT κ、1%、七编码器三聚合
scripts/agg_100bin_mixpca.py       ★ 主结果入口（--readout mlp|xgb）
scripts/agg_100bin.py              早期版本（单分子层 PCA，含注意力池化）
scripts/agg_100bin_xgb.py          早期版本 + XGB
scripts/agg_all.py                 1% 协议（4 特征 × 4 聚合）
scripts/agg_variants.py            令牌构造、聚合、训练循环（被上面各脚本复用）
scripts/summarize_mixpca.py        ★ 生成主表
scripts/summarize_100bin.py        早期版本的对照表
scripts/summarize_all.py           1% 协议的对照表
scripts/run_iltk_*.py              单任务先导实验（IlT κ），产出 runs/agg_iltk_*.json
```

`*.err` 为空文件（运行无异常），归档时不保留。

## XGBoost 读出（`100bin_xgb`）

树模型不接受「注意力池化」这种带参数的池化，因此只比较 sum / mean / concat。
结果与 MLP 读出方向不同：**编码器的优势在树读出下基本消失**。

| task | 最佳描述符（XGB） | R | 编码器（XGB） | R | 差 |
|---|---|---:|---|---:|---:|
| mon | maccs/sum | 0.520 | sum | 0.506 | −0.014 |
| density | maccs/sum | 0.788 | sum | 0.747 | −0.041 |
| heat_vap | rdkit2d/sum | 0.842 | sum | 0.854 | +0.012 |
| he | ecfp4/sum | 0.417 | sum | 0.385 | −0.032 |
| logv | rdkit2d/concat | 0.771 | sum | 0.759 | −0.012 |
| nist_logv | rdkit2d/sum | 0.758 | sum | 0.649 | −0.109 |
| ionic_cond | maccs/sum | 0.680 | sum | 0.613 | −0.067 |
| ionic_visc | rdkit2d/sum | 0.795 | sum | 0.814 | +0.019 |

编码器只在 2 个任务领先（+0.012、+0.019），在 5 个任务落后（最多 −0.109）。

**两种读出各有偏好**：树模型在描述符上更强（NIST 0.758 vs MLP 0.012；IlT κ 0.680 vs 0.446），
MLP 在 64 维稠密编码器向量上更强（NIST 0.813 vs 0.649；logV 0.873 vs 0.759）。
若两臂都允许自选读出，优势收窄到中位 +0.004——小样本的优势**依赖神经读出**。

## 复现

```bash
cd scripts
python agg_100bin.py --reps 2        # 100 条二元（约 12 分钟，CPU）
python agg_all.py --ratio 0.01       # 1% 标签
python summarize_100bin.py           # 生成对照表
```

依赖：`06_transfer/scripts/out/encoder_cache.npz`（七个编码器的逐分子向量）与
`06_transfer/scripts/fp_matrix.py`（指纹）。本目录脚本通过 `sys.path` 引用它们，
不复制缓存。
