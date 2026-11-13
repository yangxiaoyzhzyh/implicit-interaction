# Reproduction archive — interaction without an interaction module

Code, logs and result tables for the paper

> **Interaction without an interaction module: a fixed first-order aggregate for many-body property prediction**

This repository contains the **code, logs and result tables**; the datasets themselves are
read from the CheMixHub collection by path.

Every file kept here either produces a number reported in the paper, or documents how such a
number was produced. Working notes are not part of the archive.

## What the experiments are about

A many-body system is a group of components combined at some composition, whose property
belongs to the group rather than to any component alone. Chemical mixtures are its canonical
instance. Two ways of modelling them are compared:

- **v(ex)** — *explicit interaction*: the per-component representations are consumed by a
  transformer over the components and a CLS token is read out, so a module sees several
  components at once. A cheaper variant, **v(ex)-simple** (`09_vex_simple/`), instead puts
  all atoms of a mixture into one graph and pools them by a masked mean.
- **v(im)** — *implicit interaction*: each component is encoded on its own, the vectors are
  combined by a weighted sum fixed in advance, and a small readout consumes the sum; no
  operation ever sees two components at once.

The claim is that the combination rule, rather than the interaction, is what fails when the
component count rises: a fixed first-order aggregate matches the explicit route under random
splits and leads it under component-count extrapolation, while the excess is carried by the
encoder that training shapes.

## Layout

| Directory | Contents | Where it appears |
|---|---|---|
| `common/` | engine, task registry, encoder, molecule-carving generator | — |
| `scripts/` | modelling scripts, split generation, table collection | — |
| `splits/` | the index is written on demand by `make_splits.py` from your own data root; every arm then reads that one `*.npz` | — |
| `01_composition/` | component-count extrapolation (nine tasks, four arms) | Table 1; Sections S2, S6 |
| `02_unseen_molecules/` | component-identity extrapolation (six tasks) | Section S4 |
| `03_target_value/` | target-value (extreme) extrapolation (ten tasks) | Section S5 |
| `04_synthetic/` | synthetic exactly-additive target, the label-noise sweep and the aggregation-order sweep | Table 2; Tables S11--S13 |
| `05_cross_domain/` | portfolio volatility and document sets | Sections S8--S10 |
| `06_transfer/` | eight per-task encoder checkpoints and their vector cache | feeds `08` |
| `07_indistribution/` | official five-fold random CV, eight tasks | Section S3 |
| `08_small_labels/` | hundred-label comparison of encoders and descriptors | Table 3; Section S7 |
| `09_vex_simple/` | parameter-matched whole-graph variant | Section S1 |
| `10_seed_stability/` | twenty seeds of both arms on MON, with the CLS attention and the per-molecule encoding stored | Table S11 |
| `figures/` | the four data figures as PDF and PNG, plus `make_figures.py` | Figs. 2--5 |

Each extrapolation directory has the same layers: `summary.csv` (the normalised table),
`runs/` (raw rows as the scripts write them) and `logs/` (one log per run). `09_vex_simple/`
writes one CSV per task and seed into `runs/` and the aggregated table into `results/`.

## Notes that are not obvious from the file names

**`01_composition/`** — the nine tasks are `drug`, `spe`, `iltk`, `ilteta`, `nist`, `miscible`,
`miscible_hvap`, `miscible_density`, `mon`; `logv` is absent because its mixtures are at most
binary, so training on $K\le2$ leaves no test rows. This axis uses no validation set and no
early stopping. The additive arm is ridge ($\alpha=10^{-2}$) rather than plain least squares:
mole fractions sum to one, so the component columns are collinear with the intercept, and the
unregularised solution diverges where one solvent carries almost all the weight — it changes
only the two IlT tasks (IlT $\kappa$ $-0.133\to+0.737$, IlT $\eta$ $0.007\to+0.818$) and moves
the rest by $\le 0.013$; `scripts/additive.py --plain` reproduces the unregularised numbers.
`scripts/run_k3.sh` produces the relaxed-ceiling row of Table S6 (`runs_k3/`), and
`scripts/split_statistics.py` writes `runs/split_statistics.csv`, the source of the solvent and
molar-mass fractions quoted in the text.

**`02_unseen_molecules/`** — six tasks; the three `miscible*` tasks are absent because they hold
only 81 unique molecules, too few for a partition that keeps the component-count distribution
aligned on both sides.

**`03_target_value/`** — both arms are produced by the engine's own threshold split
(`--split-by y --y-train 0.25 0.75 --y-test 0.10 0.90`), not from a stored `*.npz`; the two
arms therefore share one partition by construction.

**`06_transfer/`** — the encoders here are trained on **31** atom features (`pad_from=32`), not
the 44-feature vocabulary of the main pipeline (`common/engine.py` uses `pad_from=45`); the
shipped `ckpt/*.pt` and `scripts/out/encoder_cache.npz` belong to that 31-feature variant and
are what `08_small_labels/` reads. Training uses each task's full data, 100 epochs, seed 42.

**`07_indistribution/`** — three result files, one per arm: `runs/vim_5fold.csv` (v(im)),
`runs/vex_5fold_current.csv` (v(ex), the column reported in Table S3) and `runs/vex_5fold.csv`
(the AtomFormer whole-graph model, which is the v(ex)-simple column of Table S3).
`runs/baselines*.csv` hold the published benchmark values these are compared against.

**`08_small_labels/`** — `summary.csv` is the full grid, 192 rows = 2 protocols × 4 feature sets
× 3 aggregations × 8 tasks; Table 3 reports the best cell per task and Section S7 breaks the
encoder side down by aggregation.

**`09_vex_simple/`** — the variant holds the parameter count equal to v(im) (1.23 million each);
`results/vsimple_composition_44d.csv` is the nine-task, three-seed table of Section S1.

**`figures/`** — `make_figures.py` writes `Figure2`--`Figure5` as PDF and PNG, one per data
figure, under the same numbers as the captions.

**`10_seed_stability/`** — the engine writes the CLS attention over component tokens
(`--save-attn`, `--save-attn-all` also dumps the training mixtures) and one encoding per
unique component (`--save-emb`); `scripts/compare_seeds.py` turns those files into the
per-seed-pair table. `runs/summary_seeds.csv` holds the twenty test $R$ values per arm
(`v(im)` 0.913 $\pm$ 0.022, `v(ex)` 0.833 $\pm$ 0.027) and `runs/pair_summary.csv` the 190
seed pairs of Table S11. Nothing in this directory is needed to reproduce the main tables.

**`04_synthetic/scripts/synth_noise.py`** — the label-noise sweep of Table S12; it reuses the
same additive generator and writes `runs/synth_noise.csv`. The aggregation used for the
`linear` rows is available as `--agg linear` (weight $\alpha\propto\epsilon+w\,s$, both learnt).

**`04_synthetic/scripts/synth_order.py`** — the aggregation-order sweep of Table S13; the same embedding and readout, with the weight taken as the mole fraction, learnt linearly, or learnt with a quadratic term. Writes `runs/synth_order.csv` (three seeds).

## Reproducing a table

```bash
DATA=/path/to/CheMixHub/datasets

# 1) split (once per task and axis)
python scripts/make_splits.py --task iltk --split composition --data-base "$DATA"

# 2) the four arms, sharing the same split file
python scripts/vim.py      --tag iltk --split-file splits/iltk_composition.npz \
                           --data-base "$DATA" --no-early-stop --out 01_composition/runs
python scripts/vex.py      --tag iltk --split-file splits/iltk_composition.npz \
                           --data-base "$DATA" --no-early-stop --out 01_composition/runs
python scripts/additive.py --task iltk --split-file splits/iltk_composition.npz \
                           --data-base "$DATA" --out 01_composition/runs
python scripts/frozen.py   --task iltk --split-file splits/iltk_composition.npz \
                           --data-base "$DATA" --out 01_composition/runs

# 3) normalise into the table
python scripts/collect_summary.py --dir 01_composition
```

The other axes differ only in the split: `--split unseen` for component identity, and
`--split-by y` on the engine for the target-value axis (no split file). Early stopping is off
for component-count extrapolation, because holding out a validation fold would cost one whole
component-count step, and on (knee rule) for the other two, where the validation set is carved
from the training side and the test values stay out of it.

Protocol: 200 epochs, batch 128, Adam (lr 1e-4, weight decay 1e-6), three-epoch linear warmup,
cosine annealing, targets standardised with training statistics, seeds 42/43/44.

## Requirements

`torch`, `torch-geometric`, `rdkit`, `numpy`, `pandas`, `xgboost`. The task registry
(`common/spec.py`) is the single source of task definitions.

## License

MIT — see `LICENSE`.
