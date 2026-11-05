# 02_unseen_molecules — 新分子外推

测试集里的分子在训练侧完全不出现。划分由分子级切割生成，两侧的组分数分布对齐。

> 下文所说的「本轴」一律指**新分子外推**；另两条轴是 `01_composition`（组分数外推）
> 与 `03_target_value`（极值外推）。

## 脚本

本轴没有专属脚本——`vim.py` / `vex.py` / `additive.py` / `frozen.py` 都在 `../scripts/`，
用 `--split-file ../splits/<task>_unseen.npz` 指向本轴的划分即可。

| 脚本 | 建模 | 本轴用法 |
|---|---|---|
| `../scripts/vim.py` | v(im) | `--split-file ../splits/<task>_unseen.npz`（早停默认开启） |
| `../scripts/vex.py` | v(ex) | 同上 |

**本轴的任务集比组分数轴小**：

| 可用（6 个） | 不可用 |
|---|---|
| `nist` 1,644 分子、`logv` 1,628、`ilteta` 699、`iltk` 479、`spe` 268、`drug` 169 | `miscible` / `miscible_density` / `miscible_hvap`：各只有 **81 个唯一分子** |

81 个分子二分切割后每侧约 40 个，无法在保持组分数分布对齐的前提下得到可信划分。
旧归档里这三个任务在本轴上有数值（ΔHmix 0.13），是分子数不足的产物，不应引用。

**为什么本轴早停**：切割保证 train/test 不共享分子；val 从训练侧按行留出 1/8，
只用于 knee 早停。**不做三分**——再切一刀会显著缩小本已不大的训练侧，而 knee 规则
本来就是为“val 由 train 内部切出”设计的（它检测验证曲线的拐点，不依赖 val 是独立同分布）。

## 数据现状

6 个任务 × 2 路线 × 3 seed 已齐。数据取自 CheMixHub（归档不附数据副本），
划分用 `python ../scripts/make_splits.py --task <task> --split unseen --data-base <数据根>` 现场生成。

