"""Data loading and evaluation helpers for the tomato experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .paths import GENERATED_RESULTS_DIR, SPLIT_DIR


def load_split(*, flatten: bool = False) -> tuple[np.ndarray, ...]:
    paths = [
        SPLIT_DIR / "X_train.npy",
        SPLIT_DIR / "y_train.npy",
        SPLIT_DIR / "X_test.npy",
        SPLIT_DIR / "y_test.npy",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split arrays. Run scripts/data/create_group_stratified_split.py first:\n"
            + "\n".join(missing)
        )
    X_train, y_train, X_test, y_test = (np.load(path) for path in paths)
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "AUC": float(roc_auc_score(y_true, scores)),
    }


def legacy_bootstrap(model: object, X_test: np.ndarray, y_test: np.ndarray, *, n_bootstraps: int = 100, seed: int = 42) -> pd.DataFrame:
    """Retain the fixed first-240-sample bootstrap used by baseline scripts."""

    if len(y_test) < 240:
        raise ValueError("The legacy baseline requires at least 240 test samples")
    random_state = np.random.RandomState(seed)
    rows = []
    for number in range(1, n_bootstraps + 1):
        indices = random_state.choice(240, 240, replace=True)
        y_true = y_test[indices]
        y_pred = model.predict(X_test[indices])
        scores = model.predict_proba(X_test[indices])[:, 1]
        rows.append({"Bootstrap Sample": number, **metrics(y_true, y_pred, scores)})
    return pd.DataFrame(rows)


def timestamped_result_path(name: str, output_dir: Path | None = None) -> Path:
    directory = output_dir or GENERATED_RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
