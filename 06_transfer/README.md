# 06_transfer — 编码器作为可迁移资产

编码器在**单任务全部数据**上训练一次，之后冻结，只训一个小读出。检验两件事：
(a) 能不能用别的任务的编码器（留一任务）；(b) 标签很少时是否优于通用描述符。

## 结构

```
06_transfer/
├── scripts/   9 个脚本（训练 / 留一读出 / 小样本 / 可加基线 / 编码器缓存）
├── ckpt/      8 个编码器权重（每个 4.7 MB）
├── runs/      5 个结果 JSON（原始产出）
├── logs/      6 个运行日志
└── summary.csv  归一后的总表（116 行）
```

## 编码器训练

| 项 | 值 |
|---|---|
| 脚本 | `scripts/train_all_permol.py` |
| 任务 | 8 个：`mon` `density` `heat_vap` `he` `logv` `nist_logv` `ionic_cond` `ionic_visc` |
| 数据 | 每个任务的**全部**数据（不做外推划分） |
| epoch | 固定 100，**无早停、无验证集** |
| seed | 42 |
| 权重 | 已保存到 `ckpt/permol_<task>_full_all.pt` |

权重里含 `state_dict` 与配置，可直接被下游读出复用——这是「编码器当资产」的实际含义。

## 三组下游实验

| 脚本 | 做什么 | 覆盖 |
|---|---|---|
| `leave_one_task_out.py` | 留一任务：8 个编码器轮流留一个，用其余 7 个拼接 + 线性读出 | **8 任务完整** |
| `mlp_small_data.py` + `mlp_ionic.json` | 1% / 5% 标签下的 **MLP** 读出 | `mon`、`ionic_cond`、`ionic_visc` |
| `additive_baseline.py` | 线性可加基线（闭式，无 seed） | **8 任务完整** |

## 与论文的一致性

**留一任务（补充材料 S18）完全对上**：以 `mon` 为例，本归档 `own 0.9546 / others 0.8696 / ECFP4 0.1031 / RDKit2D 0.7786 / MACCS 0.7521`，与 S18 的 0.955 / 0.870 / 0.103 / 0.779 / 0.752 逐项一致；`density` 的 own 与 others 都是 0.9920，也与 S18 一致。

**1% / 5% 小样本（S19）对不上**：本归档只有 3 个任务，且数值与 S19 不同（如 `mon` 1% 的 others 为 0.1035，S19 写 0.256）。说明 S19 来自另一次运行，本地缺 5 个任务的结果。

## 待办

1. **核对该批与 S19 的差异**：本归档 `mon` 1% others = 0.104，S19 写 0.256，差一倍。
   两者用的是同一协议（7 编码器 concat → PCA64 → MLP），差异需归因到数据集版本、
   编码器训练或读出实现，确认后 S19 才能以本表为唯一来源。
2. **`he` / `heat_vap` 键名对照**：`he` = 混合焓（= `miscible`）、
   `heat_vap` = 汽化焓（= `miscible_hvap`），跨目录引用时按 `common/spec.py` 对照。

## 重跑命令

```bash
python scripts/train_all_permol.py --root <CheMixHub 根> --out ckpt/   # 8 个编码器
python scripts/cache_encoders.py                                      # 3540 分子 × 256
python scripts/leave_one_task_out.py                                  # 留一（S18）
python scripts/mlp_small_data.py --out ../runs/mlp_small_data.json    # 1%/5%（S19）
python scripts/additive_baseline.py                                   # 可加基线
python scripts/collect_transfer.py                                    # 汇总
```
