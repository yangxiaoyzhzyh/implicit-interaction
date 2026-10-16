# scripts/analysis — 归档脚本说明

本目录存放**产生论文数字的分析脚本**。每个脚本头部都写明了口径，改动数字前请先读头部。

## make_molaxis_stats.py

分子（身份）轴统计：逐任务的 ω、含新组分样本比例、新分子数。

### ω 的定义（正文与补充材料同此口径）

对每个**测试样本**：

```
ω_i = Σ (该样本中"训练集未出现过的组分"的摩尔分数)
```

对每个**任务**：

```
ω = median( ω_i )，仅在 ω_i > 0 的样本上取中位数
```

**⚠ 不要用全部测试样本取中位数。** 例如 SPE κ 约 60% 的测试样本不含新组分，
那样会得到 ω = 0，把高 ω 任务误判为零压力任务，整条论证失效。

两种口径的实测差异：

| 任务 | 全部样本取中位数 | 只含新组分的样本取中位数（采用） |
|---|---:|---:|
| SPE κ | 0.000 | **0.570** |
| IlT κ | 0.000053 | **0.000900** |
| IlT η | 0.000180 | **0.001419** |
| NIST η | 0.000 | **0.015800** |

### 划分协议

`train` = 组分数 ≤ `train_max` 的样本，`test` = 其余。`train_max` 按原协议：
drug / SPE κ / IlT κ / IlT η 取 2，NIST logV 取 1。

组分数的定义按数据实际情况：drug 的组分 = **溶剂 + 药物**两列合并
（`cmp_ids_solvent` + `cmp_ids_drug`）；其余任务用 `cmp_ids`。

### 数据与运行

- 数据：`literature/CheMixHub/datasets/<dataset>/processed_data/*.csv`（仓库根相对路径，
  可用环境变量 `REPO_ROOT` 覆盖）。
- 运行：

```bash
python scripts/analysis/make_molaxis_stats.py                  # 打印，并与归档基准值对比
python scripts/analysis/make_molaxis_stats.py --out results/molaxis_stats.csv
```

### 复现状态

| 任务 | 本脚本 | 基准值 | |
|---|---:|---:|---|
| drug | 0.4000 | 0.400 | ✅ |
| SPE κ | 0.5700 | 0.570 | ✅ |
| IlT κ | 0.0009 | 0.001 | ✅ |
| IlT η | 0.0014 | 0.001 | ✅ |
| NIST η | 0.0158 | 0.016 | ✅ |

全部五个任务可复现。

### drug 的口径（已定）

drug 的数据结构与其余任务不同：一个样本 = 溶剂混合物 + 一个药物，
且 `cmp_mole_fractions_drug` 恒为占位值 `[1.0]`、不参与归一化。因此：

- **组分数**按"溶剂 + 药物"计（若只数溶剂，K 恒为 1–2，没有测试样本）；
- **ω 只按溶剂计算**（药物那一列是占位值，并进去会把 ω 撑到 1.0，不合理）；
- 汇总仍为"在 ω > 0 的测试样本上取中位数"，得 **0.400**。

旧归档 `results/molaxis_stats.csv` 里 drug 记的是 0.500，该值无法由数据复现，
已按上面口径改为 0.400（正文 Table 1、补材 Table S1 同步更新）。
注意旧文件里 drug 的 `unseen_pct`（24.6%）是按"含新分子样本比例"算的，
与本脚本按溶剂口径得到的值（3.3%）不是同一个量，不要互相对照。

### 相关脚本

- `newmol_contamination.py` — 同族统计，算的是"含新分子的样本比例"，不含 ω。
- `scripts/per_molecule/run_spe_vim_groups.py`、`scripts/whole_mixture/run_spe_vex_groups.py`
  — 逐样本 ω 的计算处（变量名 `te_omega`）。
