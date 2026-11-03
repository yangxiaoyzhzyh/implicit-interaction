# 07_indistribution — 官方 5-fold 随机 CV

在基准官方随机五分折上的表现。这一节的作用**不是**比较两条路线，而是确认双方都没有
在比较开始前就被削弱：若连分布内都打不过基线，后面的外推结果就无从解释。

## 结构

```
07_indistribution/
├── scripts/
│   ├── train_5fold_vim.py           v(im)：本次运行（paper_v2 的 train_task_permol.py）
│   ├── train_5fold_vex_current.py   v(ex)：本次重跑，用 common/engine.py 的 Vex
│   └── train_5fold_vex.py           旧 AtomFormer 整图版（= v(ex)-simple），留档
├── runs/
│   ├── vim_5fold.csv                41 行（含 1 行废运行，见下）
│   ├── vex_5fold_current.csv        **40 行 = 8 任务 × 5 折，正文所用**
│   ├── vex_5fold.csv                旧 AtomFormer 结果（40 行），仅留档
│   └── baselines.csv                8 行官方最强基线（CheMixHub Table 2 / Table 8）
├── logs_vex_current/                40 个 v(ex) 逐折日志（本次重跑）
├── logs/                            8 个旧 v(ex) 逐折日志（留档）
└── summary.csv                      16 行 = 8 任务 × 2 路线（v(ex) 行已换新值）
```

无 checkpoint：两个脚本都是训练—评估—丢弃权重。

## 两个臂：已按同一协议重跑

| 臂 | 来源 | 协议 |
|---|---|---|
| v(im) | `runs/vim_5fold.csv` | 200 epochs + `patience=20` 早停（官方 val），`best_ep` 落在 19–60 |
| **v(ex)** | **本次重跑**：`scripts/train_5fold_vex_current.py` → `runs/vex_5fold_current.csv`，逐折日志 `logs_vex_current/` | 同上：200 epochs + patience 20，官方 val 早停 |
| v(ex)，旧 | `runs/vex_5fold.csv`（上一篇文章 AtomFormer 已发表结果） | 40 epochs 固定；**仅留档**，已由上一行取代 |

两臂现在共享全部协议（同一份官方 fold、同一早停规则、同一 lr/batch/warmup），
唯一差别是模型结构：v(ex) 在逐组分编码之后多一层跨组分 transformer + CLS
（2.22 M vs 1.23 M 参数）。旧 `scripts/train_5fold_vex.py`（AtomFormer 整图版）
保留但不再用于正文——它的数值正是补材 Table S3 中的 **v(ex)-simple** 一列。

## 结果

| 任务 | v(im) | v(ex) | 官方最强 | v(im) 差 | v(ex) 差 |
|:---|---:|---:|---:|---:|---:|
| density | **0.9992** ± 0.0001 | 0.9990 ± 0.0001 | 0.999 | +0.0002 | 0.0000 |
| ΔHmix (`he`) | **0.9773** ± 0.0021 | 0.9749 ± 0.0037 | 0.976 | +0.0013 | −0.0011 |
| ΔHvap (`heat_vap`) | **0.9996** ± 0.0001 | 0.9994 ± 0.0001 | 0.999 | +0.0006 | +0.0004 |
| logV (`logv`) | 0.9606 ± 0.0041 | **0.9917** ± 0.0019 | 0.991 | −0.0304 | +0.0007 |
| NIST η (`nist_logv`) | 0.9940 ± 0.0006 | **0.9956** ± 0.0010 | 0.995 | −0.0010 | +0.0006 |
| IlT κ (`ionic_cond`) | 0.9288 ± 0.0053 | **0.9750** ± 0.0011 | 0.997 | **−0.0682** | −0.0220 |
| IlT η (`ionic_visc`) | 0.9834 ± 0.0029 | **0.9969** ± 0.0007 | 0.995 | −0.0116 | +0.0019 |
| MON | **0.8994** ± 0.0271 | 0.8743 ± 0.0226 | 0.913 | −0.0136 | −0.0387 |

与官方最强基线的最大差距：v(im) 在 IlT κ 上的 0.068，v(ex) 在 MON 上的 0.039，其余都在 0.031 以内。

## 两臂的整体差异

| 指标 | 值 |
|---|---:|
| 中位差（v(im) − v(ex)） | −0.0007 |
| 平均绝对差 | 0.015 |
| v(im) 更高的任务数 | 4 / 8 |
| 最大单任务差 | −0.0462（IlT κ） |
| 16 个数值的范围 | 0.874 – 0.9996 |

四胜四负，中位差基本为零。

逐任务的差值与折间标准差的对比：

| 任务 | v(im) − v(ex) | 折间 sd（较大者） | 倍数 |
|:---|---:|---:|---:|
| IlT κ | −0.0462 | 0.0053 | 9× |
| logV | −0.0311 | 0.0041 | 8× |
| IlT η | −0.0135 | 0.0029 | 5× |
| MON | +0.0251 | 0.0271 | <1× |
| NIST η | −0.0016 | 0.0010 | 2× |
| ΔHmix | +0.0024 | 0.0037 | <1× |
| 其余两项 | ≤0.0002 | — | — |

所以表述应是"整体接近、四胜四负"，而不是"不可区分"：IlT κ、logV、IlT η 三处的差距
确实大于折间散布（5–9 倍），方向都是 v(ex) 更好；MON 与 ΔHmix 上 v(im) 略高但落在散度内。

## 已知瑕疵

1. **1 行废运行**：`mon` fold 0 只跑了 2 epoch（`test_r = −0.067`），已在 `summary.csv`
   中剔除（按 `epochs < 100` 过滤），但原样保留在 `runs/vim_5fold.csv` 里。
2. **基线列是引用值**：来自 CheMixHub 官方 Table 2（MAE）与 Table 8（Pearson ρ），
   不是本仓库重跑；`runs/baselines.csv` 记录了每个任务对应的最强方法名。
3. **两套 v(ex) 并存**：`runs/vex_5fold.csv` 是旧 AtomFormer（40 轮固定）的历史结果，
   `runs/vex_5fold_current.csv` 是本次按 v(im) 协议重跑的结果。**正文与补材用的是后者**；
   前者对应补材 Table S3 的 v(ex)-simple 列。
4. **无 checkpoint**：两套脚本都是训练—评估—丢弃权重，`ckpt/` 为空。

## 待办

1. `he` 与 `heat_vap` 的任务键命名与其它目录（`miscible` / `miscible_hvap`）不同，
   跨目录引用时需对照 `common/spec.py`。
