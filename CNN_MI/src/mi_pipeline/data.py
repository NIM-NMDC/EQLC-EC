"""MI split loading, metrics, and historical bootstrap behavior."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .paths import DEFAULT_EXPERIMENT, generated_result_dir, split_dir


def load_split(experiment: str = DEFAULT_EXPERIMENT, *, flatten: bool = False) -> tuple[np.ndarray, ...]:
    directory = split_dir(experiment)
    paths = [directory / name for name in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing arrays for experiment {experiment}. Run scripts/data/create_batch_split.py --experiment {experiment}:\n"
            + "\n".join(missing)
        )
    X_train, y_train, X_test, y_test = (np.load(path) for path in paths)
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1); X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"Accuracy": float(accuracy_score(y_true, y_pred)), "Precision": float(precision), "Recall": float(recall), "F1 Score": float(f1), "AUC": float(roc_auc_score(y_true, scores))}


def legacy_baseline_bootstrap(model: object, X_test: np.ndarray, y_test: np.ndarray, *, repetitions: int = 100, seed: int = 42) -> pd.DataFrame:
    """Retain the copied baseline scripts' first-240-sample bootstrap."""
    random_state = np.random.RandomState(seed); rows = []
    for number in range(1, repetitions + 1):
        indices = random_state.choice(240, 240, replace=True)
        predictions = model.predict(X_test[indices]); scores = model.predict_proba(X_test[indices])[:, 1]
        rows.append({"Bootstrap Sample": number, **metrics(y_test[indices], predictions, scores)})
    return pd.DataFrame(rows)


def result_path(name: str, experiment: str) -> Path:
    directory = generated_result_dir(experiment); directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
