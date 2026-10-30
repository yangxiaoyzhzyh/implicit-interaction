# Reproduction archive — interaction without an interaction module

Code, logs and result tables for the paper

> **Interaction without an interaction module: a fixed first-order aggregate for many-body property prediction**

This repository contains the **code, logs and result tables only**; the manuscript and its
supplementary materials are not included, and the underlying datasets are not redistributed
here (they are read from the CheMixHub collection by path).

## What the experiments are about

A many-body system is a group of components combined at some composition, whose property
belongs to the group rather than to any component alone. Chemical mixtures are its canonical
instance. Two ways of modelling them are compared:

- **v(ex)** — *explicit interaction*: the per-component representations are consumed by
  attention, or all atoms of a mixture are placed in one graph, so a module sees several
  components at once.
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
| `splits/` | `*.npz` index files shared by all arms; `splits/logs/` is their generation log | — |
| `01_composition/` | component-count extrapolation (`scripts/run_k3.sh` is the relaxed-ceiling run) | Table 1, Section S3 |
| `02_unseen_molecules/` | component-identity extrapolation, with the Sisson carving script | Section S1 |
| `03_target_value/` | target-value (extreme) extrapolation | Section S2 |
| `04_synthetic/` | synthetic additive and interaction families | Table 2, Section S4.2 |
| `05_cross_domain/` | portfolio volatility and document sets | Section S7 |
| `06_transfer/` | eight encoder checkpoints and the leave-one-out readout | Section S8 |
| `07_indistribution/` | official five-fold random CV | Section S6 |
| `08_small_labels/` | the hundred-label comparison | Section S8 |
| `docs/archive_notes_zh.md` | working notes in Chinese: protocol table, task registry, which axes each task supports | — |

Each extrapolation directory has the same three layers: `summary.csv` (the normalised table),
`runs/` (raw rows written by the scripts), and `logs/` and `ckpt/` (outputs and weights).
Directories are numbered contiguously, and each maps to one table or supplementary section.

Two further analyses are **not** part of this archive because the manuscript does not use
them: a synthetic scan over the strength of a nonlinear term, and a water-removal study of the
ionic-liquid conductivity task. The second is the source of the solvent-fraction statistics
quoted in the IlT κ discussion.

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

The additive baseline is fitted by ridge regression (`alpha=1e-2`) rather than plain least
squares: the mole fractions sum to one, so the component columns are collinear with the
intercept, and where a solvent carries most of the weight the unregularised solution is
unstable out of sample. Pass `--plain` to reproduce the unregularised numbers.

Protocol: 200 epochs, batch 128, Adam (lr 1e-4, weight decay 1e-6), three-epoch linear warmup,
cosine annealing, targets standardised with training statistics, seeds 42/43/44. Early stopping
is off for component-count extrapolation and on (knee rule) for the other two axes; the reasons
are recorded in `docs/archive_notes_zh.md`.

## Requirements

`torch`, `torch-geometric`, `rdkit`, `numpy`, `pandas`, `xgboost`. The task registry
(`common/spec.py`) is the single source of task definitions.

## License

MIT — see `LICENSE`.
