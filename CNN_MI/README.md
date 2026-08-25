# Three-batch MI mass-spectrometry pipeline

This subproject evaluates 204-feature myocardial-infarction classifiers under
three leave-one-batch-out experiments. The engineering structure has been
standardized without changing batch membership, model hyperparameters,
bootstrap sizes, or metric definitions.

## Layout

```text
CNN_MI/
├── data/
│   ├── raw/                 # Batch 1.csv, 2.csv, and 3.csv
│   ├── processed/           # 3,980 NumPy spectra plus labels.npy
│   ├── batch_labels/        # Tracked per-batch labels
│   └── splits/              # Generated arrays keyed by experiment
├── models/
│   ├── legacy_checkpoints/experiments/{12_3,13_2,23_1}
│   ├── legacy_ensemble_selection/
│   └── generated/
├── results/
│   ├── legacy/experiments/{12_3,13_2,23_1}
│   ├── generated/
│   └── figures/
├── scripts/{data,train,evaluate,plot}
├── src/mi_pipeline/
└── pyproject.toml
```

Experiment names encode training and test batches: `12_3` trains on batches 1
and 2 and tests on batch 3; `13_2` trains on 1 and 3 and tests on 2; `23_1`
trains on 2 and 3 and tests on 1. The original root split script corresponds
to `23_1`, which remains the default.

## Environment and commands

Use the repository-wide virtual environment:

```bash
source ../.venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
python scripts/data/create_batch_split.py --experiment 23_1
python scripts/evaluate/evaluate_classical.py RandomForest --experiment 23_1 --smoke-test
python scripts/evaluate/evaluate_cnn_ensemble.py --experiment 23_1 --smoke-test --seed 42
python scripts/plot/plot_experiment_metrics.py --experiment 23_1
```

Scripts can be invoked by absolute path from any working directory. Full
published workloads remain the default; `--smoke-test` reduces only the trial
workload.

## Path and hierarchy changes

The previous code used paths such as:

```python
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_MI\stratified_split_data"
```

It now resolves paths from the package location:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
```

Concrete mappings include `data/*.csv` → `data/raw/*.csv`,
`output_for_cnn/` → `data/processed/`, `stratified_split_data/` →
`data/splits/<experiment>/`, `models_temp/<experiment>/` →
`models/legacy_checkpoints/experiments/<experiment>/`, and
`test/result/temp/<experiment>/` → `results/legacy/experiments/<experiment>/`.

Nine experiment-specific plotting scripts were consolidated into parameterized
plot commands, eliminating duplicate source while preserving each experiment's
recorded axis limits and reference values.

## Preserved behavior and reviewer notes

- Raw batch sizes are 1,330, 1,305, and 1,345; all spectra have 204 features.
- Splits are strict leave-one-batch-out splits. No batch-effect correction is
  performed in this subproject.
- CNN training remains 2,000 epochs by default with seed 52, batch size 64,
  dropout 0.2, learning rate 0.00015, and no weight decay.
- CNN ensemble evaluation retains 100 bootstrap repetitions of 1,200 samples.
- The five classical scripts were copied from HIP/CER and previously pointed
  to HIP/CER arrays. They now target the selected MI experiment but retain the
  copied first-240-sample bootstrap behavior. This is documented rather than
  silently redesigned in an engineering-only pass.
- `models/legacy_ensemble_selection/` preserves the exact checkpoint selected
  by the old root ensemble directory, even though an identical copy also
  exists in the complete `23_1` checkpoint set.
