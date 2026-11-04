# 09_vex_simple — v(ex)-simple（整图路线）

补充材料 Table S1 的 **v(ex)-simple** 一列由本目录产出。这一列回答的是
“参数与 v(im) 严格一致时，把交互放到原子级的整图上会不会更好”。

本目录**独立于 `common/` 引擎**：脚本自带特征与模型，不 import `engine.py`，
以免与现有流水线的改动相互牵连。

## 目录结构

```
09_vex_simple/
├── vsimple.py                        复现脚本（自包含，44 维特征）
├── results/
│   └── vsimple_composition_44d.csv   Table S1 的现行数据（9 任务 × 3 seeds）
├── runs/                             逐任务逐 seed 原始输出（27 个）
│   └── <task>_s<seed>.csv            task, seed, test_R, test_R2, MAE, n_train, n_test
└── logs/                             逐任务逐 seed 训练日志（27 个）
    └── <task>_s<seed>.log            每 20 轮一次的 loss、末尾的 test R / 参数量
```

**没有 checkpoint**：该脚本每次运行从头训练、评估后即丢弃权重（与
`01_composition` 的其它路线一致），因此归档中不含 `.pt` 文件。若需复现某一行，
用 `runs/` 里的参数设定重跑即可——seed 固定为 42/43/44。

## 模型

一个混合物的全部原子进入**同一张图**：

- 只在分子内连边，分子之间不加边；分子身份只由批索引在池化时区分；
- 每个原子带 44 维化学特征（18 元素 one-hot、手性 2、度 9、氢 5、杂化 5、
  芳香 1、电荷 3、成环 1），第 45 维是该组分的**摩尔分数**（组内常量）；
- `GIN×3(64) → Linear(64→256) → TransformerEncoder×2(256, 4 heads) → 全体原子 masked mean`
- 最后接 `Linear(256+ncond → 256) → ReLU → Dropout → 256 → ReLU → 1`

没有逐组分编码器，也没有 CLS token；交互完全由跨分子的原子级注意力承担。

## 参数量

| 路线 | 参数量 |
|---|---:|
| v(im) | 1.23 M |
| **v(ex)-simple** | **1.23 M** |
| v(ex) | 2.28 M |

因此这一列是**严格参数匹配**的对照：v(ex)-simple 与 v(im) 只差“交互放在哪里”，
不比 v(im) 多一个参数（输入统一补零到 64 维，首层宽度不变）。

## 运行

```bash
# 单任务（推荐：直接读 01_composition 用的同一份 npz，保证行级一致）
python vsimple.py --data-base /path/to/chemidata \
    --task mon --seeds 42 43 44 \
    --split-file ../splits/mon_composition.npz \
    --out results/mon.csv

# 多任务、自行按 K 划分（不读 npz）
python vsimple.py --data-base /path/to/chemidata \
    --tasks drug spe iltk ilteta nist --seeds 42 43 44
```

- `--data-base` 指向含 `drug-solubility/`、`polymer-electrolyte/` 等子目录的
  CheMixHub 数据根；每个子目录下取 `processed_data/processed_*.csv` 与
  `compounds.csv`。
- `--split-file` 给定时直接用其中的 `train_idx` / `test_idx`；**`val_idx` 会并回训练**
  （与 `engine.py` 在组分轴“不早停”的行为一致）。否则按 `K ≤ train_max` 现算。
- 训练：Adam 1e-4、weight decay 1e-6、batch 128、200 epochs、前 3 epoch 线性
  warmup 后 cosine，**无早停**（组分轴不切验证集）。

输出 CSV 列：`task, seed, test_R, test_R2, MAE, n_train, n_test`。

## 已归档的结果

| 文件 | 覆盖 | 说明 |
|---|---|---|
| **`results/vsimple_composition_44d.csv`** | **9 个任务 × 3 seeds** | **Table S1 的现行数据**（44 维词表，与 `01_composition` 同一份划分） |

现行结果（Pearson $R$，三个 seed 的均值 ± 标准差）：

| 任务 | v(ex)-simple | v(im) | v(ex) |
|---|---:|---:|---:|
| density | 0.960 ± 0.017 | **0.999** | 0.949 |
| ΔHvap | 0.977 ± 0.005 | **0.998** | 0.979 |
| MON | 0.752 ± 0.086 | **0.922** | 0.840 |
| NIST η | 0.690 ± 0.032 | **0.874** | 0.747 |
| ΔHmix | 0.764 ± 0.039 | **0.929** | 0.874 |
| drug lnS | 0.608 ± 0.030 | **0.629** | 0.581 |
| IlT η | 0.831 ± 0.010 | **0.844** | 0.811 |
| SPE κ | 0.552 ± 0.150 | 0.733 | **0.756** |
| IlT κ | 0.762 ± 0.012 | 0.732 | **0.829** |

九项中 v(ex)-simple 有八项低于 v(im)（−0.013 到 −0.184），仅 IlT κ 高出 0.030；
相对 v(ex) 则有六项明显更低，三项更高：drug +0.027、IlT η +0.020、density +0.011。
其中 drug 与 density 落在 v(ex) 的种子散度内，**IlT η 超出**（v(ex) 三个种子 0.803–0.819，
本轮 0.831 ± 0.010），因此正文写作“comparable or slightly better on the rest”。

## 与当前流水线的差异（引用时必须注明）

- 本目录**不调用** `common/engine.py`，模型、训练循环、数据装配各自独立；
  与 `01_composition` 只共享**划分文件**（npz）与原始数据。
- 模型构造顺序与初始化 RNG 流和 `engine.py` 不同，同一 seed 不保证逐位复现。

## 复现命令（服务器上实际使用的）

```bash
for t in drug spe iltk ilteta nist miscible miscible_density miscible_hvap mon; do
  for s in 42 43 44; do
    python vsimple.py --data-base /root/autodl-fs/chemixhub/datasets \
      --task $t --seeds $s --epochs 200 \
      --split-file splits/${t}_composition.npz \
      --out runs/${t}_s${s}.csv
  done
done
```
