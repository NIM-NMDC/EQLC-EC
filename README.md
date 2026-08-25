# EQLC-EC

Engineering-preserved pipelines for ensemble classification of one-dimensional
mass-spectrometry data. The repository contains five independent experiments
that share one reproducible Python environment.

## Repository layout

```text
EQLC-EC/
├── CNN_single_ICC/       # ICC binary classification
├── CNN_CHD/              # CHD cross-batch classification
├── CNN_single_HIP_CER/   # Hippocampal versus cerebellar classification
├── CNN_tomato/           # Organic versus non-organic tomato classification
├── CNN_MI/               # Three-batch myocardial-infarction experiments
├── requirements.txt      # Shared, fully pinned CPU dependencies
└── README.md
```

Each subproject separates raw and processed data, model checkpoints,
historical and generated results, executable scripts, and reusable source code.
Its README documents the exact workflow and known methodological limitations.

## Shared Python environment

The tested setup is Python 3.14 on Ubuntu/WSL2 with CPU-only PyTorch. From the
repository root, create the only virtual environment used by this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation \
  -e CNN_single_ICC \
  -e CNN_CHD \
  -e CNN_single_HIP_CER \
  -e CNN_tomato \
  -e CNN_MI
```

For the existing checkout, only activation is normally required:

```bash
source .venv/bin/activate
```

No subproject should contain a second virtual environment. CUDA, NVIDIA Linux
drivers, Conda, Jupyter, and unrelated Python packages are not required.

## Subprojects

| Subproject | Experiment design | Documentation |
| --- | --- | --- |
| ICC | Stratified train/test split | [`CNN_single_ICC/README.md`](CNN_single_ICC/README.md) |
| CHD | Published cross-batch workflow with documented pooled ComBat | [`CNN_CHD/README.md`](CNN_CHD/README.md) |
| HIP/CER | Stratified 80/20 split | [`CNN_single_HIP_CER/README.md`](CNN_single_HIP_CER/README.md) |
| Tomato | Sample-ID-grouped, label-stratified split | [`CNN_tomato/README.md`](CNN_tomato/README.md) |
| MI | Three leave-one-batch-out experiments | [`CNN_MI/README.md`](CNN_MI/README.md) |

## Working conventions

- Run experiment entry points from the relevant `scripts/` directory tree.
- Paths are resolved from package locations and do not depend on the terminal's
  current directory or a developer-specific absolute path.
- Tracked historical artifacts are read-only inputs. New models, tables,
  figures, runtime measurements, and generated splits use ignored output
  directories.
- This cleanup preserves published numerical workflows. Documented leakage or
  evaluation limitations are not silently changed during engineering work.
- Add a dependency to the root `requirements.txt` only when a project actually
  requires it; all subprojects continue to share the same `.venv`.
