"""Project-relative paths for the ICC experiment.

Every path is anchored to the installed source tree rather than the current
working directory or a developer-specific machine path.
"""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_CHECKPOINTS_DIR = MODELS_DIR / "legacy_checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_RESULTS_ROOT = RESULTS_DIR / "legacy"
LEGACY_RESULTS_DIR = LEGACY_RESULTS_ROOT / "bootstrap"
LEGACY_SEARCH_RESULTS_DIR = LEGACY_RESULTS_ROOT / "hyperparameter_search"
LEGACY_SUMMARY_DIR = LEGACY_RESULTS_ROOT / "summary"
GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"
CROSS_VALIDATION_DIR = GENERATED_RESULTS_DIR / "search"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNTIME_DIR = RESULTS_DIR / "runtime"


def ensure_output_dirs() -> None:
    """Create directories used only for generated artifacts."""

    for directory in (
        SPLIT_DIR,
        GENERATED_RESULTS_DIR,
        CROSS_VALIDATION_DIR,
        FIGURES_DIR,
        RUNTIME_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
