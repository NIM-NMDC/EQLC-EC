"""Reusable classical-model search and evaluation for ICC."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate

from .data import (
    bootstrap_predictions,
    classification_metrics,
    load_split,
    timestamped_result_path,
)
from .paths import CROSS_VALIDATION_DIR


SEED = 42


def evaluate_estimator(
    name: str,
    estimator: Any,
    *,
    n_bootstraps: int = 100,
    seed: int = SEED,
) -> tuple[dict[str, float], pd.DataFrame, Path]:
    """Fit on training data, evaluate once, then bootstrap the fixed predictions."""

    X_train, y_train, X_test, y_test = load_split(flatten=True)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    point_metrics = classification_metrics(y_test, predictions, probabilities)

    bootstrap = bootstrap_predictions(
        y_test,
        lambda indices: (predictions[indices], probabilities[indices]),
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    output_path = timestamped_result_path(name)
    bootstrap.to_csv(output_path, index=False)

    formatted = ", ".join(f"{key}={value:.4f}" for key, value in point_metrics.items())
    print(f"{name} point estimate: {formatted}")
    print(f"Bootstrap results saved to {output_path}")
    return point_metrics, bootstrap, output_path


def search_estimator(
    name: str,
    estimator_factory: Callable[[dict[str, Any]], Any],
    parameter_grid: dict[str, list[Any]],
    *,
    smoke_test: bool = False,
    seed: int = SEED,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    """Select parameters by stratified CV and touch the test set only once."""

    X_train, y_train, X_test, y_test = load_split(flatten=True)
    parameters = list(ParameterGrid(parameter_grid))
    if smoke_test:
        parameters = parameters[:1]
    cv = StratifiedKFold(n_splits=2 if smoke_test else 5, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []

    for params in parameters:
        estimator = estimator_factory(params)
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1", "auc": "roc_auc"},
            n_jobs=1 if smoke_test else -1,
        )
        row = {
            **params,
            "CV Accuracy": float(scores["test_accuracy"].mean()),
            "CV F1": float(scores["test_f1"].mean()),
            "CV AUC": float(scores["test_auc"].mean()),
        }
        rows.append(row)
        print(row)

    best_row = max(rows, key=lambda row: row["CV AUC"])
    # Keep original Python values (especially None); a DataFrame may coerce
    # mixed None/integer parameter columns to NaN.
    best_params = {key: best_row[key] for key in parameter_grid}
    table = pd.DataFrame(rows).sort_values("CV AUC", ascending=False).reset_index(drop=True)
    best_estimator = estimator_factory(best_params)
    best_estimator.fit(X_train, y_train)
    predictions = best_estimator.predict(X_test)
    probabilities = best_estimator.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, predictions, probabilities)

    CROSS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke_test else ""
    output_path = CROSS_VALIDATION_DIR / f"{name}_search{suffix}.csv"
    table.to_csv(output_path, index=False)
    print(f"Best parameters: {best_params}")
    print(f"Final untouched-test metrics: {test_metrics}")
    print(f"Search table saved to {output_path}")
    return best_params, table, test_metrics
