# 06_transfer — 编码器作为可迁移资产

编码器在**单任务全部数据**上训练一次，之后冻结，只训一个小读出。检验两件事：
(a) 能不能用别的任务的编码器（留一任务）；(b) 标签很少时是否优于通用描述符。

## 结构

```
06_transfer/
├── scripts/   18 个脚本（训练 / 留一读出 / 小样本 / 可加基线 / 编码器缓存）
├── ckpt/      8 个编码器权重（每个 4.7 MB）
├── runs/      3 个结果 JSON（原始产出）
├── logs/      6 个运行日志
└── summary.csv  归一后的总表（120 行）
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
| `mlp_small_data.py` + `runs/mlp_small_data.json` | 1% / 5% 标签下的 **MLP** 读出（运行记录见 `logs/mlp_ionic.log`） | **8 个任务全覆盖** |
| `additive_baseline.py` | 线性可加基线（闭式，无 seed） | **8 任务完整** |

## 待办

2. **`he` / `heat_vap` 键名对照**：`he` = 混合焓（= `miscible`）、
   `heat_vap` = 汽化焓（= `miscible_hvap`），跨目录引用时按 `common/spec.py` 对照。

## 重跑命令

```bash
python scripts/train_all_permol.py --root <CheMixHub 根> --out ckpt/   # 8 个编码器
python scripts/cache_encoders.py                                      # 3540 分子 × 256
python scripts/leave_one_task_out.py                                  # 留一任务
python scripts/mlp_small_data.py --out ../runs/mlp_small_data.json    # 1%/5% 小样本
python scripts/additive_baseline.py                                   # 可加基线
python scripts/collect_transfer.py                                    # 汇总
```
