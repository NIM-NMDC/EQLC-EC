"""Project-relative paths and batch-experiment definitions for MI."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BATCH_LABELS_DIR = DATA_DIR / "batch_labels"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_CHECKPOINTS_DIR = MODELS_DIR / "legacy_checkpoints" / "experiments"
LEGACY_ENSEMBLE_DIR = MODELS_DIR / "legacy_ensemble_selection"
GENERATED_MODELS_DIR = MODELS_DIR / "generated"
RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_EXPERIMENTS_DIR = RESULTS_DIR / "legacy" / "experiments"
LEGACY_PUBLISHED_DIR = RESULTS_DIR / "legacy" / "published_default"
LEGACY_SUMMARY_DIR = RESULTS_DIR / "legacy" / "summary"
GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"
FIGURES_DIR = RESULTS_DIR / "figures"

EXPERIMENTS = {
    "12_3": ((1, 2), 3),
    "13_2": ((1, 3), 2),
    "23_1": ((2, 3), 1),
}
DEFAULT_EXPERIMENT = "23_1"


def split_dir(experiment: str) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {tuple(EXPERIMENTS)}")
    return SPLITS_DIR / experiment


def checkpoint_dir(experiment: str) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {tuple(EXPERIMENTS)}")
    return EXPERIMENT_CHECKPOINTS_DIR / experiment


def legacy_result_dir(experiment: str) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose from {tuple(EXPERIMENTS)}")
    return LEGACY_EXPERIMENTS_DIR / experiment


def generated_result_dir(experiment: str) -> Path:
    return GENERATED_RESULTS_DIR / experiment
