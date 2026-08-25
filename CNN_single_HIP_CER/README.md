# HIP/CER mass-spectrometry classification pipeline

This subproject contains the binary cerebellar-versus-hippocampal experiment.
The engineering layout has been standardized without changing the published
data split, model parameters, bootstrap sample count, or metric definitions.

## Layout

```text
CNN_single_HIP_CER/
├── data/
│   ├── raw/                 # Source CSV
│   ├── processed/           # One NumPy file per spectrum
│   └── splits/              # Generated train/test arrays
├── models/
│   └── legacy_checkpoints/  # Seven tracked CNN checkpoints
├── results/
│   ├── legacy/              # Historical bootstrap, search, summary, runtime data
│   ├── generated/           # New evaluation and training tables
│   ├── figures/             # Generated plots
│   └── runtime/             # New benchmark output
├── scripts/
│   ├── data/
│   ├── train/
│   ├── evaluate/
│   ├── benchmark/
│   └── plot/
├── src/hip_cer_pipeline/    # Shared package code and path definitions
└── pyproject.toml
```

## Environment and execution

All EQLC subprojects use the repository-level environment:

```bash
source ../.venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
```

The scripts can be launched from this directory or by absolute script path
from any other working directory:

```bash
python scripts/data/create_stratified_split.py
python scripts/evaluate/evaluate_random_forest.py --smoke-test
python scripts/evaluate/evaluate_cnn_ensemble.py --smoke-test --seed 42
python scripts/plot/plot_bootstrap_metrics.py
```

Full experiment defaults remain active unless `--smoke-test` is supplied.
Generated output is separated from tracked historical results.

## Path changes

The original scripts embedded paths from one Windows workstation, for example:

```python
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_single_HIP_CER\stratified_split_data"
```

Paths are now anchored to the installed package location:

```python
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LEGACY_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "legacy_checkpoints"
```

Concrete mappings include:

- `output_for_cnn/` → `data/processed/`
- `stratified_split_data/` → `data/splits/`
- `models_for_ensemble/` → `models/legacy_checkpoints/`
- `test/result/` → `results/legacy/bootstrap/`
- new bootstrap CSVs → `results/generated/`

No path depends on the terminal's current directory or a developer username.

## Preserved experimental behavior

- The source shape assertion remains 1201 samples by 2463 features.
- The split remains stratified 80/20 with `random_state=46`.
- The classical model parameters are unchanged.
- Classical evaluation retains the historical 100 bootstrap repetitions and
  the fixed index universe/sample size of 240, including its exclusion of the
  final sample in the 241-sample test split.
- CNN training still evaluates on the test split after every epoch. This is a
  methodological limitation of the published workflow and is intentionally
  preserved here because this pass is engineering-only.
