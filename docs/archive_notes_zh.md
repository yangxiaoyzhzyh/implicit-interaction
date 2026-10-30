# repro_v2 — 统一复现归档

建立日期：2026-10-26

**结构原则**：脚本只有一份，任务与划分都由参数选择。四条路线各一个脚本，
划分一个脚本；不存在“每个任务一个脚本”或“每个轴一个脚本”的情况。

```
repro_v2/
├── common/            引擎、任务注册表、编码器、分子切割生成器
├── scripts/           四个建模脚本 + 划分 + 汇总
├── splits/            *.npz（make_splits.py 产出，四个臂共用同一份索引）；logs/ 是划分生成日志
├── 01_composition/    组分数外推（Table 1、S3；scripts/run_k3.sh 为放宽上限那次，产出 runs_k3/）
├── 02_unseen_molecules/  新分子外推（S1，含 Sisson 单独脚本）
├── 03_target_value/   极值外推（S2）
├── 04_synthetic/      合成任务（Table 2、S4.2）
├── 05_cross_domain/   非化学领域（S7：文档集合 + 组合波动率）
├── 06_transfer/       编码器迁移（S8；8 个 .pt + 留一读出）
├── 07_indistribution/ 官方 5-fold 随机 CV（S6）
└── 08_small_labels/   100 条标签的小样本比较（S8）
```

编号与文章的对应关系即上表，**目录编号连续（01–08），不跳号**。
两项未被正文引用的分析放在 `../repro_extra/`（合成非线性扫描、IlT κ 去水实验），
其中后者的分布统计是正文 IlT κ 段那几个百分数的来源。

每个外推目录一律三层（有不存权重的目录就没有 `ckpt/`），含义固定：

| 层 | 内容 | 是否可用于论文 |
|---|---|---|
| `summary.csv` | 归一后的总表（task × model × split，含 mean/sd/n/per_seed） | **是** |
| `runs/` | 脚本直接写出的原始行（每 seed 一行 / 每臂一文件） | 是（原始凭证） |
| `logs/` `ckpt/` | 运行日志、权重 | — |

外推目录只放**数据、日志、权重与说明**，不放脚本（Sisson 例外，见 02）。

## 六个脚本

| 脚本 | 作用 | 任务 | 划分 |
|---|---|---|---|
| `scripts/make_splits.py` | 生成划分 | `--task` | `--split composition/unseen/target` |
| `scripts/vim.py` | v(im)：一阶聚合 + 塑形编码器 | `--tag` | `--split-file` |
| `scripts/vex.py` | v(ex)：跨组分注意力 | `--tag` | `--split-file` |
| `scripts/additive.py` | 线性可加基线（闭式解，无 seed） | `--task` | `--split-file` |
| `scripts/frozen.py` | 冻结编码器臂（RDKit 2D 217 维） | `--task` | `--split-file` |
| `scripts/collect_summary.py` | 把 `runs/*.csv` 归一成 `summary.csv` | `--dir` | — |

四个建模脚本读**同一份** `splits/*.npz`，因此同一划分下四个臂的比较是严格可控的。
四个脚本的原始表列名不同（per-seed 行 / 单行 / 已汇总），`collect_summary.py`
把它们统一成 `task, model, split, n_seeds, R_mean, R_sd, per_seed, source` 一张表。

## 协议

四条路线共享：`epochs=200`、`batch=128`、`Adam(lr=1e-4, wd=1e-6)`、前 3 epoch 线性 warmup、
`CosineAnnealingLR(T_max=197)`、目标按训练集均值/标准差标准化、seed 42/43/44。

按外推方式只差一处——**是否早停**：

| 外推 | 划分 | 早停 | 理由 |
|---|---|---|---|
| 组分数 | `--split composition`（`K<=train_max` 训练） | **关**（`--no-early-stop`） | 留出验证集会吃掉一个组分数档位；train/test 只差组分数，无分布外泛化需要抑制 |
| 新分子 | `--split unseen`（分子**二分**切割） | **开**（knee） | train/test 不共享分子；val 从 train 内部按行留出（1/8）。不作三分：再切一刀会显著缩小可用数据，而 knee 规则本就是为“val 由 train 切出”设计的 |
| 极值 | `--split target`（四分位带训练、两端测试） | **开**（knee） | 训练带内部可留出，测试端极端值不泄漏 |

`val_idx` 一律从训练侧留出 1/8（最多 600，`RandomState(0)`），测试侧不受影响。

## 任务注册表

`common/spec.py` 是任务定义的唯一来源，`--task` / `--tag` 都取这里的键：

| 键 | 数据集 | property | 条件列 | `train_max` |
|---|---|---|---|---|
| `drug` | drug-solubility | Log solubility | Temperature | 2 |
| `spe` | polymer-electrolyte | log Conductivity | Temperature | 2 |
| `iltk` | ionic-liquids | Electrical conductivity | T, P, Frequency | 2 |
| `ilteta` | ionic-liquids | Viscosity | T, P, Frequency | 2 |
| `nist` | nist-logV | （无 property 列） | T, P | 1 |
| `miscible` | miscible-solvent | Enthalpy of mixing | — | 2 |
| `miscible_hvap` | miscible-solvent | Heat of vaporization | — | 2 |
| `miscible_density` | miscible-solvent | Density | — | 2 |
| `mon` | MON | Motor octane number | — | 2 |
| `logv` | logV | Log viscosity | T | 2 |

数据根：`literature/CheMixHub/datasets`。

## 各轴的任务集不同

同一个任务不一定三条轴都能做。下表由 `common/spec.py` + 数据实测得出（2026-10-26），
**不能假设三轴任务集相同**。

| 任务 | 样本 | $K_{\max}$ | 唯一分子 | 组分数轴 | 新分子轴 | 极值轴 |
|---|---:|---:|---:|:--:|:--:|:--:|
| `drug` | 26,713 | 3 | 169 | ✓ 2,490 / 24,223 | ✓ | ✓ |
| `spe` | 11,350 | 5 | 268 | ✓ 8,944 / 2,406 | ✓ | ✓ |
| `iltk` | 40,904 | 3 | 479 | ✓ 24,688 / 16,216 | ✓ | ✓ |
| `ilteta` | 75,992 | 3 | 699 | ✓ 59,685 / 16,307 | ✓ | ✓ |
| `nist` | 239,200 | 2 | 1,644 | ✓ 29,563 / 209,637 | ✓ | ✓ |
| `miscible`（ΔHmix） | 30,061 | 5 | **81** | ✓ 3,580 / 26,481 | **✗** | ✓ |
| `miscible_density` | 30,061 | 5 | **81** | ✓ 3,580 / 26,481 | **✗** | ✓ |
| `miscible_hvap` | 30,061 | 5 | **81** | ✓ 3,580 / 26,481 | **✗** | ✓ |
| `logv` | 34,374 | **2** | 1,628 | **✗** | ✓ | ✓ |
| `mon` | 684 | 121 | 416 | ✓ 386 / 298 | ✓（薄） | ✓ 68 / 69 |

两条排除理由：

- **`logv` 不能做组分数轴**：$K_{\max}=2$，训练侧（$K\le2$）已覆盖全部 34,374 条，
  测试集为空。
- **`miscible*` 三个任务不能做新分子轴**：只有 81 个唯一分子，二分分子切割后每一侧
  约 40 个分子，无法在保持组分数分布对齐的前提下构造可信划分。旧归档里这三个任务
  在新分子轴上有数值（ΔHmix 0.13 等），正是分子数太少的产物，不应引用。

`mon` 在新分子轴上只有 416 个分子、684 条样本，属于“薄”但可行；引用时需注明方差较大。

## 快速开始

```bash
cd repro_v2
DATA="C:/Users/yangx/v3-mixture-token/literature/CheMixHub/datasets"

# 1) 划分（每个任务 × 每种外推各一次）
python scripts/make_splits.py --task mon --split composition --data-base "$DATA"

# 2) 四条路线（组分数轴不早停），原始行写进 runs/
python scripts/vim.py      --tag mon --split-file splits/mon_composition.npz \
                           --data-base "$DATA" --no-early-stop \
                           --out 01_composition/runs
python scripts/vex.py      --tag mon --split-file splits/mon_composition.npz \
                           --data-base "$DATA" --no-early-stop \
                           --out 01_composition/runs
python scripts/additive.py --task mon --split-file splits/mon_composition.npz \
                           --data-base "$DATA" --out 01_composition/runs
python scripts/frozen.py   --task mon --split-file splits/mon_composition.npz \
                           --data-base "$DATA" --out 01_composition/runs

# 3) 归一成该外推的总表
python scripts/collect_summary.py --dir 01_composition
```

已验证可跑（2026-10-26，MON 组分数划分）：train 322 / val 64 / test 298，
加性基线 R=0.9195、冻结臂 R=0.9190 ± 0.0071。

汇总 CSV 由脚本自己写：`vim.py` / `vex.py` 每个 seed 追加一行到
`<out>/<tag>_<model>_<split>.csv`（列含 knee / best_ep / val_R / test_R / test_R2 / MAE），
`additive.py` 写 `additive_<split>.csv`（单行，无 seed），
`frozen.py` 写 `frozen_<split>.csv`（mean / sd / per_seed）。

## 待办

1. **权重只有 `06_transfer` 有**：`01`–`05` 只用于出表格数字，本批不保存权重。
   表示层面的分析（隐空间有效秩、PCA、迁移热图）需要权重，应单独跑并显式保存。
