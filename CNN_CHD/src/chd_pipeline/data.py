"""Data and metric helpers shared by CHD scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .paths import DEFAULT_EXPERIMENT, generated_result_dir, generated_split_dir, legacy_split_dir


def load_split(
    experiment: str = DEFAULT_EXPERIMENT,
    *,
    corrected: bool = True,
    generated: bool = False,
    flatten: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one frozen cross-batch split or a newly generated equivalent."""

    directory = generated_split_dir(experiment) if generated else legacy_split_dir(experiment)
    suffix = "_combat" if corrected else ""
    paths = (
        directory / f"X_train{suffix}.npy",
        directory / f"y_train{suffix}.npy",
        directory / f"X_test{suffix}.npy",
        directory / f"y_test{suffix}.npy",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        source = "generated" if generated else "legacy"
        raise FileNotFoundError(
            f"Missing {source} split files for experiment {experiment}:\n"
            + "\n".join(missing)
        )
    X_train, y_train, X_test, y_test = (np.load(path) for path in paths)
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    positive_scores: np.ndarray,
) -> dict[str, float]:
    """Calculate the five metrics used by the published scripts."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "AUC": float(roc_auc_score(labels, positive_scores)),
    }


def timestamped_result_path(algorithm: str, experiment: str) -> Path:
    directory = generated_result_dir(experiment)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{algorithm}_{timestamp}.csv"


def save_bootstrap_rows(
    rows: list[dict[str, float | int]], algorithm: str, experiment: str
) -> Path:
    path = timestamped_result_path(algorithm, experiment)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
