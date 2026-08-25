# CHD cross-batch mass-spectrometry pipeline

This subproject is an engineering-preserved version of the published CHD
experiments. It separates code, processed data, cross-batch splits, model
checkpoints, and results while retaining the original numerical workflow.

## Project layout

```text
CNN_CHD/
├── data/
│   ├── raw/                         # Raw CSVs (not included)
│   ├── processed/                   # 6,009 spectra plus labels.npy
│   ├── batch_labels/                # Historical y_1.npy through y_4.npy
│   ├── splits/<experiment>/         # 12 tracked published ComBat splits
│   └── generated_splits/            # Newly generated splits (ignored by Git)
├── models/
│   ├── legacy_checkpoints/<experiment>/  # 120 tracked published CNNs
│   └── generated/                   # Newly trained models (ignored by Git)
├── results/
│   ├── legacy/experiments/<experiment>/  # Published CSV results
│   ├── legacy/training_logs/        # Published training logs
│   ├── generated/                   # New runs (ignored by Git)
│   └── figures/                     # New figures (ignored by Git)
├── scripts/
│   ├── data/                        # Preprocessing and batch splitting
│   ├── train/                       # Parameterized legacy CNN training
│   ├── evaluate/                    # CNN, EQLC, voting, and SVM evaluation
│   └── plot/                        # Dataset and result visualization
└── src/chd_pipeline/                # Shared importable package
```

An experiment identifier uses `<training batches>_<test batch>`. For example,
`34_2` means that batches 3 and 4 form the training set and batch 2 forms the
test set. The 12 tracked identifiers are `12_3`, `12_4`, `13_2`, `13_4`,
`14_2`, `14_3`, `23_1`, `23_4`, `24_1`, `24_3`, `34_1`, and `34_2`.

## Environment setup

The tested setup uses Python 3.14 and CPU-only PyTorch. All subprojects share
the repository-level environment. Activate it from this directory:

```bash
source ../.venv/bin/activate
```

For a fresh checkout, create the environment from the repository root and
install the shared locked dependencies:

```bash
cd ..
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e CNN_CHD
```

No CUDA or NVIDIA packages are required.

## Published-method preservation notice

The published preprocessing concatenates the selected training and test
batches, fits `StandardScaler` on the combined matrix, and then applies ComBat
to that combined matrix. This exposes test-distribution information during
preprocessing and therefore is data leakage under a strict prospective
evaluation protocol.

That behavior is intentionally retained in
`scripts/data/apply_joint_combat.py` so the published comparison between base
models and voting models is not silently changed. The training entry point also
retains the original per-epoch test evaluation and model-mode behavior. This
repository organization is an engineering cleanup, not a corrected
methodological reanalysis. A future prospective pipeline should fit every
learned preprocessing step using training data only.

## Data availability

The repository contains the processed spectra and all 12 published ComBat
splits. The four source CSV files are not present. Therefore, evaluation,
plotting, and training from the frozen splits work immediately; rerunning raw
preprocessing requires placing the original CSVs in `data/raw/`.

## Typical commands

Evaluate the complete published `34_2` checkpoint set:

```bash
python scripts/evaluate/evaluate_cnn_models.py --experiment 34_2
python scripts/evaluate/evaluate_eqlc.py --experiment 34_2 --smoke-test
python scripts/evaluate/evaluate_cnn_ensemble.py --experiment 34_2 --smoke-test
```

Train one model with the original root-script defaults:

```bash
python scripts/train/train_cnn.py --experiment 34_2 --smoke-test
```

The ten historical ensemble configurations are represented without duplicate
source files by `--learning-rate` and `--seed`:

```bash
python scripts/train/train_cnn.py \
  --experiment 34_2 --batch-size 256 --learning-rate 0.00006 --seed 41
python scripts/train/train_cnn.py \
  --experiment 34_2 --batch-size 256 --learning-rate 0.00005 --seed 45
```

Recreate the original uncorrected `13_4` split and run its legacy SVM script:

```bash
python scripts/data/create_batch_split.py --train-batches 1 3 --test-batch 4
python scripts/evaluate/evaluate_svm.py --experiment 13_4 --smoke-test
```

Reproduce the published joint ComBat operation for `34_2` without overwriting
the tracked historical split:

```bash
python scripts/data/apply_joint_combat.py --train-batches 3 4 --test-batch 2
```

Create figures:

```bash
python scripts/plot/plot_experiment_accuracy.py --experiment 34_2
python scripts/plot/plot_batch_distribution.py --batch 1
python scripts/plot/plot_cross_batch_performance.py
```

## Terminal execution and paths

All paths are derived from `src/chd_pipeline/paths.py`; scripts do not depend on
the current working directory and do not modify `sys.path`. After the editable
installation, this works from `/tmp` or any other directory:

```bash
/absolute/path/EQLC-EC/.venv/bin/python \
  /absolute/path/CNN_CHD/scripts/evaluate/evaluate_cnn_models.py \
  --experiment 34_2 --smoke-test
```

New outputs never overwrite tracked published artifacts.
