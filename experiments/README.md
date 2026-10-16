# Experiments — two axes of extrapolation in mixture property prediction

Scripts, logs and result tables behind every experiment reported in

> *Two extrapolation axes decide whether interaction needs a module in many-body systems*

and its supplementary materials. The index below maps each table of the paper to the script
that produces it, the result file it writes, and the log it leaves behind; every script
header repeats that mapping. Nothing outside this folder is needed to reproduce a table.

## Layout

```
experiments/
├── scripts/
│   ├── per_molecule/     v(im) implementation and its experiments
│   ├── whole_mixture/    v(ex) implementation and its experiments
│   │   └── vex_legacy/   per-task scripts behind the released v(ex) logs
│   ├── baselines/        RDKit2D+XGB, Sisson descriptor-linear, XGB controls
│   ├── analysis/         dataset audits and the tables behind Figure 1 / Table 1
│   ├── olfactory/        Sisson odor model and training-curve analysis
│   ├── synthetic/        synthetic generating families
│   └── crossdomain/      portfolio and multi-document experiments
├── logs/                 training and evaluation logs, by experiment
└── results/              result CSVs, per-sample predictions, split metadata
```

## Route naming

Two modelling routes appear throughout this folder. Earlier notes and file names used
several labels for the same two things; they are unified here as **v(im)** and **v(ex)**:

| Route | Definition | Labels still used in file names and data columns |
|---|---|---|
| **v(im)** | per-molecule encoder → composition-weighted sum → readout; no operation sees two components at once | `e2`, `v(p)`, `per_molecule/` |
| **v(ex)** | all atoms of a mixture in one graph → cross-molecule attention → readout | `e1`, `v(w)`, `whole_mixture/`, `vex_*` (legacy prefix) |

Result and log files keep their original names and their internal `route` labels
(`e1` / `e2`), so that archived numbers stay traceable; every script header states which
files it produces. RDKit2D+XGBoost and the additive baseline are **external references**,
not routes — they are listed under Baselines.

## Experiment index

| # | Experiment | Reported in | Scripts | Results | Logs |
|---|---|---|---|---|---|
| 1 | CheMixHub 8 tasks, official 5-fold (in-distribution anchors) | S1, S2 in-dist anchors | `scripts/per_molecule/train_task_permol.py`, `scripts/whole_mixture/train_task.py` | `results/chemixhub_permol_results.csv` | `logs/chemixhub_5fold_permol/`, `logs/chemixhub_5fold_whole/` |
| 2 | ΔHmix / ΔHvap four-cell grids (train ≤ 2 or ≤ 3 → test 5, val 4) | Table 1, S2 | `scripts/per_molecule/run_hmix_kext.py` | `results/hmix_e*_traink*.csv`, `results/hvap_e*_traink*.csv` (+ `_meta.json`) | `logs/extrapolation/hmix_*`, `logs/extrapolation/hvap_*` |
| 3 | ΔHmix / ΔHvap per-composition-count profile | S2 | `scripts/per_molecule/run_perk_thermo.py`, `scripts/analysis/summarize_perk.py` | `results/perk_hmix_e{1,2}.csv`, `results/perk_hvap_e{1,2}.csv` | `logs/extrapolation/perk_*` |
| 4 | MON, both routes (train ≤ 2 → test ≥ 3, up to 121) | Table 1, S2 | `scripts/per_molecule/run_mon_routes.py` | `results/mon_routes_final_results.csv`, `results/mon_routes_pred.csv` | `logs/extrapolation/mon_routes_final.log` |
| 5 | Five tasks, both routes (drug lnS, SPE κ, IlT κ, IlT η, NIST η) | Table 1 | `scripts/per_molecule/run_5tasks_vim.py`, `scripts/whole_mixture/run_5tasks_vex.py` | `results/tasks5_e{1,2}_results.csv` | `logs/extrapolation/tasks5_e{1,2}.log` |
| 6 | SPE κ, subset decomposition and relaxed ceiling | Table 1, S2, S3 | `scripts/per_molecule/run_spe_vim_groups.py`, `scripts/whole_mixture/run_spe_vex_groups.py`, `scripts/per_molecule/run_spe_vim_k3.py` | `results/spe_e{1,2}_groups_results.csv`, `results/spe_e2_k3_results.csv`, `results/spe_e{1,2}_pred.csv` | `logs/extrapolation/spe_*` |
| 7 | NIST η, molecular-axis pressure ladder | S3 | `scripts/per_molecule/run_nist_ladder.py` | `results/nist_ladder_k{1,2,5}_e{1,2}.csv`, `results/nist_e{1,2}_results.csv`, `results/nist_e{1,2}_pred.csv` | `logs/extrapolation/nist_*` |
| 8 | Additive baseline (`y = Σ cₘxₘ + Σ βⱼcondⱼ + b`) | Table 1 and throughout | `scripts/per_molecule/run_linear_additive.py` | `results/linear_additive_results.csv` | — |
| 9 | RDKit2D+XGBoost reference (five tasks, SPE κ, coverage sweep) | S2 external baseline | `scripts/baselines/run_5tasks_rdkit_xgb.py`, `scripts/baselines/run_spe_xgb_k3.py`, `scripts/baselines/run_xgb_coverage.py` | `results/tasks5_rdkit_xgb_results.csv`, `results/spe_xgb_k3_results.csv`, `results/xgb_coverage_results.csv` | `logs/extrapolation/tasks5_xgb.log` |
| 10 | Two-axis statistics (ω, 1 − a, nearest-neighbour similarity) | Table 1, S3 | `scripts/analysis/make_molaxis_stats.py` | `results/molaxis_stats.csv` | — |
| 11 | Dataset audits (coverage, contamination, similarity, sub-pair coverage) | S3 | `scripts/analysis/spe_coverage_by_trainmax.py`, `scripts/analysis/newmol_contamination.py`, `scripts/analysis/spe_nn_similarity.py`, `scripts/analysis/spe_react_evidence.py`, `scripts/analysis/spe_subpair_coverage.py` | `results/analysis/*.out` | — |
| 12 | Sisson olfactory task (both routes + descriptor-linear baseline) | S3 | `scripts/olfactory/train_sisson_permol_cls.py`, `scripts/baselines/run_sisson_rdkit_linear.py` | `results/sisson/` | `logs/olfactory/`, `logs/sisson_rdkit_linear2.log` |
| 13 | Synthetic families A / B / C | S4 | `scripts/synthetic/run_synth_routes.py` | `results/synth_routes_results.csv` | `logs/synth_routes.log` |
| 14 | Portfolio volatility (finance) | S5 | `scripts/crossdomain/run_portfolio_vol.py` | `results/portfolio_vol_results.csv` (price data are downloaded by the script and not redistributed) | `logs/portfolio_vol.log` |
| 15 | Multi-document reasoning | S5 | `scripts/crossdomain/run_multidoc_set.py` | `results/multidoc_set_results.csv`, `results/multidoc_set_homo_results.csv`, `results/multidoc_emb_20ng.npz` | `logs/multidoc_full.log` |
| 16 | Figure 1 / Table 1 summary tables | Figure 1, Table 1 | `scripts/analysis/summarize_fig1.py`, `scripts/analysis/list_k_span.py` | — | — |

## Environment

- Python 3.10+, PyTorch, PyTorch Geometric, RDKit, scikit-learn, XGBoost, pandas, scipy.
- All models are small (≈1.2 M parameters), pretraining-free, and trainable on a single
  consumer GPU.

## Data

We use the public **CheMixHub** benchmark (8 tasks). The composition-axis experiments use two
properties of its miscible-solvent dataset: ΔHmix (enthalpy of mixing) and ΔHvap (heat of
vaporization). Expected layout:

```
<data-root>/
├── <task>/processed_data/processed_<Task>Data.csv     # drug-solubility, polymer-electrolyte,
│                                                      # ionic-liquids, nist-logV, miscible-solvent, ...
└── MON/processed_MON.csv
```

Each script takes `--data-base` (five-task extrapolation) or `--root` / `--data-root`
(CheMixHub 5-fold) and points at this tree. Compound ids are resolved to SMILES from the
per-dataset `compounds.csv` shipped with the benchmark.

## Usage

Every script supports `--help`. Representative calls (a few scripts carry machine-specific
defaults for `--root` / `--data-root` / `--data-base`; pass explicit paths):

**1. CheMixHub 8 tasks, official 5-fold (both routes)**

```bash
python scripts/whole_mixture/train_task.py --task mon --folds 0 1 2 3 4 \
    --data-root <data-root> --epochs 200 --seed 42
python scripts/per_molecule/train_task_permol.py --task mon --folds 0 1 2 3 4 \
    --data-root <data-root> --epochs 200 --seed 42
```

**2. Extrapolation experiments**

```bash
# five tasks, v(im) and v(ex) arms (train ≤ 2 → test ≥ 3)
python scripts/per_molecule/run_5tasks_vim.py --data-base <data-root> \
    --tasks drug spe iltk ilteta nist --seeds 42 43 44 --epochs 200
python scripts/whole_mixture/run_5tasks_vex.py --data-base <data-root> \
    --tasks drug spe iltk ilteta nist --seeds 42 43 44 --epochs 200

# MON, both routes with a shared protocol
python scripts/per_molecule/run_mon_routes.py --data-root <data-root>/MON \
    --protocol final --seeds 42 43 44

# ΔHmix / ΔHvap four-cell grid (train ≤ 2 or 3 → test = 5, val = 4)
python scripts/per_molecule/run_hmix_kext.py --route e2 \
    --train-max-k 2 --val-k 4 --test-k 5 --seeds 42 43 44
python scripts/per_molecule/run_hmix_kext.py --prop "Heat of vaporization" --route e2 \
    --train-max-k 2 --val-k 4 --test-k 5 --seeds 42 43 44

# per-composition-count profile (K = 3, 4, 5 evaluated separately)
python scripts/per_molecule/run_perk_thermo.py --prop "Enthalpy of mixing" --route e2 \
    --seeds 42 43 44

# SPE κ subset decomposition and relaxed ceiling, NIST ladder
python scripts/per_molecule/run_spe_vim_groups.py --data-base <data-root> \
    --configs 2 3 --seeds 42 43 44 --dump-pred results/spe_e2_pred.csv
python scripts/per_molecule/run_nist_ladder.py --route e2 --drop-top-k 2 --seed 42
```

**3. Baselines**

```bash
python scripts/baselines/run_5tasks_rdkit_xgb.py --data-base <data-root> --seeds 42 43 44
python scripts/per_molecule/run_linear_additive.py --out results/
python scripts/baselines/run_xgb_coverage.py            # coverage sweep (S2 table)
```

**4. Dataset audits**

```bash
python scripts/analysis/spe_coverage_by_trainmax.py     # coverage by train_max ∈ {2,3}
python scripts/analysis/newmol_contamination.py         # new-molecule fraction per task
python scripts/analysis/spe_nn_similarity.py            # nearest-neighbour Tanimoto
python scripts/analysis/summarize_perk.py               # per-K table from results/perk_*.csv
```

## Result files

| File | Contents |
|---|---|
| `results/chemixhub_permol_results.csv` | 8 tasks × 5 folds, v(im) arm (per-fold r / MAE) |
| `results/mon_routes_final_results.csv` | MON, both routes × seen/unseen |
| `results/{nist,tasks5,spe}_*_results.csv` | five-task and SPE extrapolation summaries |
| `results/hmix_e*_traink*.csv` · `hvap_e*_traink*.csv` (+ `_meta.json`) | ΔHmix / ΔHvap four-cell grids, per seed, with split metadata |
| `results/perk_{hmix,hvap}_e{1,2}.csv` | per-composition-count profile (test K = 3/4/5, per seed) |
| `results/{tasks5_rdkit_xgb,spe_xgb_k3,xgb_coverage}_results.csv` | RDKit2D+XGBoost reference scores |
| `results/spe_e*_pred.csv` | per-sample predictions (`sample_idx, K, omega, pred, true`) |
| `results/linear_additive_results.csv` | linear additive baseline |
| `results/analysis/*.out` | dataset-audit outputs |

## Conventions

- **Splits.** Domain benchmarks use the official CheMixHub k-fold split
  (`KFold(shuffle=True)`, sample-level 70/10/20). Extrapolation experiments define
  `train = {K ≤ k}` / `test = {K ≥ k+1}` explicitly, with the exact values recorded in the
  `_meta.json` files alongside each result CSV.
- **Error bars.** ± is one standard deviation with `ddof = 0`. For 5-fold runs it is the
  fold-to-fold spread of test r at a single seed; for multi-seed runs it is the seed-to-seed
  spread; for binned dose–response curves it is the bootstrap standard error of the within-bin
  correlation (B = 200).
- **Model selection.** The best epoch is chosen on validation only; test is never used for
  model selection. The ΔHmix / ΔHvap grids reload the best epoch on the K = 4 validation split,
  Sisson stops on an online knee criterion, and the MON composition-axis protocol has no
  validation split at all and trains a fixed 200 epochs.

