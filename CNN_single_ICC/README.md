# ICC mass-spectrometry classification pipeline

This subproject provides a reproducible pipeline for binary classification of
one-dimensional ICC mass-spectrometry data. Source code, executable scripts,
data, model checkpoints, and results have separate responsibilities.

## Project layout

```text
CNN_single_ICC/
├── data/
│   ├── raw/                 # Source CSV files
│   ├── processed/           # One NumPy file per spectrum
│   └── splits/              # Generated train/test arrays (ignored by Git)
├── models/
│   └── legacy_checkpoints/  # Ten tracked CNN checkpoints
├── results/
│   ├── legacy/              # Tracked historical results
│   ├── generated/           # New evaluation and training results
│   ├── figures/             # Generated plots
│   └── runtime/             # Generated benchmark results
├── scripts/
│   ├── data/                # Preprocessing, splitting, augmentation
│   ├── train/               # Neural-network training
│   ├── search/              # Hyperparameter searches
│   ├── evaluate/            # Fixed-model and ensemble evaluation
│   ├── benchmark/           # Runtime benchmarks
│   └── plot/                # Result visualization
├── src/icc_pipeline/        # Shared importable Python package
└── pyproject.toml
```

## Environment setup

The tested environment is Python 3.14 with CPU-only PyTorch. All subprojects
share the repository-level environment. From this directory:

```bash
source ../.venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
```

No CUDA or NVIDIA packages are required.

## Typical workflow

```bash
python scripts/data/preprocess_spectra.py
python scripts/data/create_stratified_split.py
python scripts/train/train_cnn.py --smoke-test
python scripts/evaluate/evaluate_random_forest.py --smoke-test
python scripts/evaluate/evaluate_cnn_ensemble.py --smoke-test
python scripts/plot/plot_bootstrap_metrics.py
```

Search, training, evaluation, and benchmark scripts retain their full
experiment defaults. Pass `--smoke-test` to expensive scripts when checking a
code path with a small workload.

## Terminal execution and path handling

The original scripts used machine-specific Windows paths:

```python
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_ICC\stratified_split_data"
```

All paths are now derived in `src/icc_pipeline/paths.py` from the package
location:

```python
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LEGACY_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "legacy_checkpoints"
```

The local package is installed in editable mode, so scripts import shared code
without modifying `sys.path`. Paths do not depend on the current working
directory. For example, this command works from `/tmp` or any other directory:

```bash
/absolute/path/EQLC-EC/.venv/bin/python \
  /absolute/path/CNN_single_ICC/scripts/evaluate/evaluate_random_forest.py \
  --smoke-test
```

New outputs never overwrite tracked historical results. Historical bootstrap
CSVs are in `results/legacy/bootstrap`; new results are written to
`results/generated`.

## Historical-result verification

Replay deterministic legacy baselines with:

```bash
python scripts/evaluate/verify_legacy_results.py
```

The verification report is written to
`results/generated/legacy_verification.json`.
