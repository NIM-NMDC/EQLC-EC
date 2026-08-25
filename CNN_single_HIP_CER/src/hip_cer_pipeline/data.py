"""Data loading and legacy evaluation helpers for HIP/CER."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .paths import GENERATED_RESULTS_DIR, SPLIT_DIR


def load_split(*, flatten: bool = False) -> tuple[np.ndarray, ...]:
    """Load the fixed 80/20 train/test split."""

    paths = [
        SPLIT_DIR / "X_train.npy",
        SPLIT_DIR / "y_train.npy",
        SPLIT_DIR / "X_test.npy",
        SPLIT_DIR / "y_test.npy",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split arrays. Run scripts/data/create_stratified_split.py first:\n"
            + "\n".join(missing)
        )
    X_train, y_train, X_test, y_test = (np.load(path) for path in paths)
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_scores: np.ndarray,
) -> dict[str, float]:
    """Compute the five metrics recorded by the published workflow."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "AUC": float(roc_auc_score(y_true, positive_scores)),
    }


def legacy_bootstrap(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_bootstraps: int = 100,
    seed: int = 42,
    legacy_test_count: int = 240,
) -> pd.DataFrame:
    """Reproduce the original fixed-240-sample bootstrap procedure."""

    if len(y_test) < legacy_test_count:
        raise ValueError(
            f"The legacy procedure requires at least {legacy_test_count} test samples; "
            f"found {len(y_test)}."
        )
    random_state = np.random.RandomState(seed)
    rows: list[dict[str, float | int]] = []
    for number in range(1, n_bootstraps + 1):
        indices = random_state.choice(
            legacy_test_count, legacy_test_count, replace=True
        )
        X_bootstrap = X_test[indices]
        y_bootstrap = y_test[indices]
        predictions = model.predict(X_bootstrap)
        probabilities = model.predict_proba(X_bootstrap)[:, 1]
        row: dict[str, float | int] = {"Bootstrap Sample": number}
        row.update(classification_metrics(y_bootstrap, predictions, probabilities))
        rows.append(row)
    return pd.DataFrame(rows)


def timestamped_result_path(algorithm: str, output_dir: Path | None = None) -> Path:
    directory = output_dir or GENERATED_RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{algorithm}_{timestamp}.csv"
