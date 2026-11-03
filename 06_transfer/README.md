# 06_transfer — 编码器训练与缓存特征

本目录只保留「编码器」这一可迁移资产：八个任务各训练一个编码器，并把逐分子向量
缓存下来，供 `08_small_labels` 直接复用。正文的 Table 3 与 table S7 全部由 08 产出，
因此原先的留一读出、1%/5% 小样本与可加基线三组下游实验**并未进入正文**，已从归档移除。

## 结构

```
06_transfer/
├── scripts/
│   ├── train_all_permol.py    八个任务的编码器训练
│   ├── train_task_permol.py   单任务训练（train_all 逐个调用它）
│   ├── encoder.py             GIN×3 + TFM×2 + mean pooling（被 08 引用）
│   ├── fp_matrix.py           ECFP4 / MACCS / RDKit2D 指纹（被 08 引用）
│   ├── leave_one_task_out.py  任务表 `TASKS`、`load_task`、`CKPT`/`load_model`、
│   │                          `mol_vectors`——**共有 08 与 cache_encoders 复用**；
│   │                          它自身的留一读出实验不在正文，结果已移除
│   ├── cache_encoders.py      逐分子向量缓存
│   └── out/encoder_cache.npz  3540 分子 × 256（25 MB）
└── ckpt/                      8 个编码器权重（每个 4.7 MB）
```

**没有实验日志与结果表**：本目录不产出正文数字，只产出权重与缓存。

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

> **特征维度**：`encoder.py` 与 `train_task_permol.py` 现已改用正文的 44 维原子特征词表
> （`pad_from=45`）。归档的 `ckpt/*.pt` 与 `out/encoder_cache.npz` 由早期的 31 维版本训练
> 与缓存，宽度分别为 32 与 45，**与新默认不兼容**；用 `train_all_permol.py` +
> `cache_encoders.py` 重跑一次即可得到 44 维版本。

## 被谁使用

| 使用者 | 依赖 |
|---|---|
| `08_small_labels/scripts/agg_100bin_mixpca.py` | `from leave_one_task_out import TASKS, load_task`；`fp_matrix.py`；`out/encoder_cache.npz` |
| `08_small_labels/scripts/agg_variants.py` | 同上 |
| `scripts/cache_encoders.py` | `from leave_one_task_out import CKPT, load_model, mol_vectors, DS, TASKS`；`fp_matrix.py`（`eval_encoders` 只是历史兜底，主路径不会触发） |
| 任何想重跑 08 的读者 | `ckpt/*.pt` 或 `encoder_cache.npz` |

因此 `ckpt/`、`encoder.py`、`fp_matrix.py`、`leave_one_task_out.py`、`cache_encoders.py`
与缓存文件**不可删**——删掉 `leave_one_task_out.py` 会让 08 的主结果脚本与
`cache_encoders.py` **直接 import 失败**。

## 重跑命令

```bash
python scripts/train_all_permol.py --root <CheMixHub 根> --out ckpt/   # 8 个编码器
python scripts/cache_encoders.py                                      # 3540 分子 × 256
```

## 待办

1. **`he` / `heat_vap` 键名对照**：`he` = 混合焓（= `miscible`）、
   `heat_vap` = 汽化焓（= `miscible_hvap`），跨目录引用时按 `common/spec.py` 对照。
