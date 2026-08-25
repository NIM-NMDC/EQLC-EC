# Tomato mass-spectrometry classification pipeline

This subproject classifies organic and non-organic tomato spectra. The
engineering structure and path handling have been standardized while retaining
the original preprocessing alternatives, grouped split, model parameters, and
evaluation definitions.

## Layout and workflow

```text
CNN_tomato/
├── data/{raw,processed,splits}
├── models/{legacy_checkpoints,generated}
├── results/{legacy,generated,figures,runtime}
├── scripts/{data,train,evaluate,benchmark,plot}
├── src/tomato_pipeline/
└── pyproject.toml
```

Use the single repository environment:

```bash
source ../.venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
python scripts/data/create_group_stratified_split.py
python scripts/evaluate/evaluate_random_forest.py --smoke-test
python scripts/plot/plot_bootstrap_metrics.py
```

After training new checkpoints, evaluate them without moving files:

```bash
python scripts/evaluate/evaluate_cnn_ensemble.py \
  --checkpoint-dir models/generated --smoke-test --seed 42
```

The full-spectrum representation in `data/processed/` is the canonical input
used by the 16,227-feature CNN. The 256-bin and selected-m/z-pair alternatives
write to separate directories so they cannot overwrite the canonical data.

## Path changes

Before:

```python
data_folder = r"D:\work_GuoLin\machine_learning\machinelearning\src\guolin\CNN_TOMATO\stratified_split_data"
```

After:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
```

Directory mappings are `processed_data/` → `data/processed/`,
`stratified_split_data/` → `data/splits/`, `test/result/` →
`results/legacy/bootstrap/`, and new output → `results/generated/`.

## Preserved behavior and available assets

- 1,600 tracked spectra contain 16,227 features, balanced 800/800.
- The split is made at sample-ID level, stratified by label, with a 29% test
  fraction and `random_state=42`; this prevents ten replicate spectra from the
  same sample ID appearing on both sides.
- Baseline scripts retain their original model settings and fixed-first-240
  bootstrap procedure. Their old paths incorrectly pointed to HIP/CER data;
  they now correctly use the tomato split.
- The raw `raw.csv` and tomato `.pt` checkpoints are not present in this
  repository. Therefore preprocessing can be rerun once the raw CSV is placed
  in `data/raw/`, and ensemble evaluation can be rerun once checkpoints are
  placed in `models/legacy_checkpoints/`. Existing processed spectra and
  historical ensemble tables remain usable now.
