"""Project-relative paths for the CHD experiment."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BATCH_LABELS_DIR = DATA_DIR / "batch_labels"
LEGACY_SPLITS_DIR = DATA_DIR / "splits"
GENERATED_SPLITS_DIR = DATA_DIR / "generated_splits"

MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_CHECKPOINTS_DIR = MODELS_DIR / "legacy_checkpoints"
GENERATED_MODELS_DIR = MODELS_DIR / "generated"

RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_RESULTS_DIR = RESULTS_DIR / "legacy" / "experiments"
LEGACY_TRAINING_LOGS_DIR = RESULTS_DIR / "legacy" / "training_logs"
GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"
FIGURES_DIR = RESULTS_DIR / "figures"

EXPERIMENTS = (
    "12_3",
    "12_4",
    "13_2",
    "13_4",
    "14_2",
    "14_3",
    "23_1",
    "23_4",
    "24_1",
    "24_3",
    "34_1",
    "34_2",
)
DEFAULT_EXPERIMENT = "34_2"


def validate_experiment(experiment: str) -> str:
    if experiment not in EXPERIMENTS:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiment {experiment!r}; choose one of: {choices}")
    return experiment


def experiment_name(train_batches: list[int] | tuple[int, ...], test_batch: int) -> str:
    name = f"{''.join(str(batch) for batch in train_batches)}_{test_batch}"
    return validate_experiment(name)


def legacy_split_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return LEGACY_SPLITS_DIR / validate_experiment(experiment)


def generated_split_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return GENERATED_SPLITS_DIR / validate_experiment(experiment)


def legacy_model_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return LEGACY_CHECKPOINTS_DIR / validate_experiment(experiment)


def generated_model_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return GENERATED_MODELS_DIR / validate_experiment(experiment)


def legacy_result_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return LEGACY_RESULTS_DIR / validate_experiment(experiment)


def generated_result_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return GENERATED_RESULTS_DIR / validate_experiment(experiment)
