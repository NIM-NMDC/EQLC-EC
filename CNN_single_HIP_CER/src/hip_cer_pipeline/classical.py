"""Classical models and published parameters for HIP/CER."""

from __future__ import annotations

import inspect
from typing import Any

import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data import legacy_bootstrap, load_split, timestamped_result_path


SEED = 42
MODEL_NAMES = ("RandomForest", "SVM", "AdaBoost", "XGBoost", "LightGBM")


def build_estimator(name: str, *, smoke_test: bool = False) -> Any:
    """Build a model with the parameters used by the original script."""

    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=10 if smoke_test else 500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=SEED,
        )
    if name == "SVM":
        return SVC(
            C=10,
            gamma=0.01,
            kernel="rbf",
            random_state=SEED,
            probability=True,
            cache_size=2000,
        )
    if name == "AdaBoost":
        parameters: dict[str, Any] = {
            "n_estimators": 5 if smoke_test else 200,
            "learning_rate": 1.0,
            "random_state": SEED,
        }
        # Older scikit-learn versions exposed this parameter. Current versions
        # implement the same discrete SAMME algorithm without the switch.
        if "algorithm" in inspect.signature(AdaBoostClassifier).parameters:
            parameters["algorithm"] = "SAMME"
        return AdaBoostClassifier(**parameters)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=5 if smoke_test else 500,
            learning_rate=0.1,
            random_state=SEED,
            eval_metric="logloss",
        )
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            num_leaves=31,
            max_depth=20,
            learning_rate=0.1,
            n_estimators=5 if smoke_test else 200,
            random_state=SEED,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def evaluate_estimator(
    name: str,
    *,
    smoke_test: bool = False,
    n_bootstraps: int | None = None,
) -> None:
    """Fit one published baseline and write its bootstrap results."""

    X_train, y_train, X_test, y_test = load_split(flatten=True)
    estimator = build_estimator(name, smoke_test=smoke_test)
    estimator.fit(X_train, y_train)
    count = n_bootstraps if n_bootstraps is not None else (2 if smoke_test else 100)
    results = legacy_bootstrap(
        estimator, X_test, y_test, n_bootstraps=count, seed=SEED
    )
    output_path = timestamped_result_path(name)
    results.to_csv(output_path, index=False)
    print(results.mean(numeric_only=True).to_string())
    print(f"Results saved to {output_path}")
