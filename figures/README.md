# figures —— 正文 Figure 2–5 的图件与生成脚本

`make_figures.py` 一条命令生成正文四张数据图（矢量 PDF 与位图 PNG 两份）。
Figure 1（两条路线的示意图）是手绘的，不在本目录。

## 编号对照

脚本内部按制作顺序命名（`Figure1`–`Figure4`），正文按出现顺序编号（Figure 2–5），
两者差一位。本目录的文件名已按**正文编号**命名，括号内是脚本写出的名字：

| 本目录文件 | 脚本内名 | 正文 | 内容 |
|---|---|---|---|
| `Fig2_composition.pdf/.png` | `Figure1` | Figure 2 | 九个任务上的组分数外推，v(im) / v(ex) / 可加基线 |
| `Fig3_coverage.pdf/.png` | `Figure2` | Figure 3 | 覆盖率：(a) SPE κ 抬高训练上限；(b) IlT κ 两种划分 |
| `Fig4_additive_target.pdf/.png` | `Figure3` | Figure 4 | 纯可加合成目标，v(im) 与 v(ex) 随 $K$ 的表现 |
| `Fig5_transfer.pdf/.png` | `Figure4` | Figure 5 | 一百条标签的迁移：留一任务编码器与小样本 |

## 数据来源

| 面板 | 读取 | 对应表 |
|---|---|---|
| Fig2（组分数外推） | `../01_composition/summary.csv` | Table 1 |
| Fig3(a) SPE κ | 脚本内常量 `SPE`，取自补材 Table S6 | Table S6 |
| Fig3(b) IlT κ | 脚本内常量 `ILTK`，组分轴取 Table 1、五折取 Table S3 | Table 1 / S3 |
| Fig4（合成） | `../04_synthetic/runs/synth_add_pair_3seeds_summary.csv` | Table 2 |
| Fig5（迁移） | `../08_small_labels/summary.csv`（只含 `100bin_mixpca` 网格） | Table 3 / S7 |

Fig2、Fig4、Fig5 的数值全部来自本归档的 CSV；Fig3 的两个面板是常量，
与补材对应表逐项一致，改表时需同步改这里的常量。

## 重跑

```bash
cd repro_v2/figures
python make_figures.py          # 需要 matplotlib；输出写到本目录
```

脚本写出的是 `Figure1.pdf` … `Figure4.pdf`（即上表的“脚本内名”），与本目录现有的
`Fig2_*`–`Fig5_*` 内容相同、只是命名不同；重跑后按上表对照即可。

依赖：`numpy`、`matplotlib`（`Agg` 后端，无需显示设备）。字体用默认 DejaVu，图内数学符号为 mathtext。
