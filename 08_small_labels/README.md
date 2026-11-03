# 08_small_labels — 小样本读出：聚合方式对「描述符 vs 塑形编码器」的影响

本目录回答一个问题：在**极少标签**下，v(im) 的塑形编码器是否仍比描述符基线强？
并同时检验一个容易忽略的变量——**逐组分特征如何聚合成混合物向量**。

## 协议

| protocol | 训练数据 | 文件 |
|---|---|---|
| `100bin_mixpca` | 官方 train 中**仅 100 条二元（K=2）**的行；验证取同池余下 50 条；测试为官方 test 全部 | `runs/agg_100bin_mixpca_mlp.json`、`_xgb.json` |

用官方 `kfold_split_0`，2 次重复（不同抽样），报告 Pearson R 的均值与标准差。

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

## 两条要点

1. **聚合方式的影响大于特征本身**。同一描述符在不同聚合下可差 0.4 以上
   （MACCS 在 density：sum 0.475 / mean 0.746；RDKit2D 在 nist：sum 0.006（100 条）/ 0.933（1%））。
   因此任何「描述符基线」都必须声明聚合方式，否则数字不可比。
2. **极低样本下 MACCS 常是最好的描述符**（binary 实验里 4/8 个任务），
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
summary.csv                        长表：task, protocol, feature, agg, R, sd（192 行）
runs/agg_100bin_mixpca_mlp.json    ★ 主结果：混合物层 PCA + MLP（3 聚合 × 4 特征）
runs/agg_100bin_mixpca_xgb.json    ★ 主结果：混合物层 PCA + XGB（3 聚合 × 4 特征）
runs/agg_100bin_mixpca_best.csv    ★ 主表：最佳描述符 vs 最佳编码器（两种读出）
runs/mixpca_mlp.log / _xgb.log     两次主运行的完整输出
scripts/agg_100bin_mixpca.py       ★ 主结果入口（--readout mlp|xgb）
scripts/agg_variants.py            令牌构造、聚合、训练循环（被主脚本复用）
scripts/summarize_mixpca.py        ★ 生成主表
```

`*.err` 为空文件（运行无异常），归档时不保留。

## 复现

```bash
cd scripts
python agg_100bin_mixpca.py --readout mlp --reps 2   # 100 条二元（约 15 分钟，CPU）
python agg_100bin_mixpca.py --readout xgb --reps 2
python summarize_mixpca.py                           # → runs/agg_100bin_mixpca_best.csv
```

依赖：`06_transfer/scripts/out/encoder_cache.npz`（七个编码器的逐分子向量）与
`06_transfer/scripts/fp_matrix.py`（指纹）。本目录脚本通过 `sys.path` 引用它们，
不复制缓存。
