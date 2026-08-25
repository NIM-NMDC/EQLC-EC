"""Project-relative paths for the tomato experiment."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BINNED_DIR = DATA_DIR / "processed_binned"
MZ_PAIR_DIR = DATA_DIR / "processed_mz_pairs"
SPLIT_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_CHECKPOINTS_DIR = MODELS_DIR / "legacy_checkpoints"
GENERATED_MODELS_DIR = MODELS_DIR / "generated"
RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_RESULTS_DIR = RESULTS_DIR / "legacy" / "bootstrap"
LEGACY_SUMMARY_DIR = RESULTS_DIR / "legacy" / "summary"
GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNTIME_DIR = RESULTS_DIR / "runtime"


def ensure_output_dirs() -> None:
    for directory in (
        PROCESSED_DIR,
        SPLIT_DIR,
        GENERATED_MODELS_DIR,
        GENERATED_RESULTS_DIR,
        FIGURES_DIR,
        RUNTIME_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
