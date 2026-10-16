# v(ex) 臂 — per-task scripts（实验 1 / 补材 S1 表 S2 的 v(ex) 列）

Scripts behind the released v(ex) logs and the corresponding rows of the result tables.
They are **per-task variants** of the v(ex) arm (the 8-task loop lives in `../train_task.py`);
file names were normalized for this repository, **contents are unchanged**.

| File | Purpose | Produced artifact |
|---|---|---|
| `vex_he_5fold.py` | ΔHmix, 5-fold — also defines the ± convention used in the manuscript (fold-to-fold std, `ddof=0`) | `../../../logs/chemixhub_5fold_whole/he_5fold.log` |
| `vex_mon_5fold.py` | MON, 5-fold and variant runs (`full` / `gin_only` / `tfm_only`) | `../../../logs/chemixhub_5fold_whole/mon_best5fold.log` |
| `vex_mon_e2b_5fold.py` | MON, component-count extrapolation variant | — |

Exploratory scripts that the paper does not report (MON hyper-parameter search, a 20-seed
ΔHmix rerun, ensemble evaluation, the RDKit2D descriptor control) have been moved to
`../../../_unused/scripts/vw_legacy/`（历史归档，未随本次统一改名）.

Original location: the legacy reproduction bundle of the preceding study; copies here keep this
repository self-contained for the v(ex) column.
