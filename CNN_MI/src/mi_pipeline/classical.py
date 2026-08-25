"""Classical baseline definitions retained from the original MI directory."""

import inspect
from typing import Any
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .data import legacy_baseline_bootstrap, load_split, result_path


def build_estimator(name: str, *, smoke_test: bool = False) -> Any:
    seed = 42
    if name == "RandomForest": return RandomForestClassifier(n_estimators=10 if smoke_test else 500, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=seed)
    if name == "SVM": return SVC(C=10, gamma=0.01, kernel="rbf", random_state=seed, probability=True, cache_size=2000)
    if name == "AdaBoost":
        parameters: dict[str, Any] = {"n_estimators": 5 if smoke_test else 200, "learning_rate": 1.0, "random_state": seed}
        if "algorithm" in inspect.signature(AdaBoostClassifier).parameters: parameters["algorithm"] = "SAMME"
        return AdaBoostClassifier(**parameters)
    if name == "XGBoost": return XGBClassifier(n_estimators=5 if smoke_test else 500, learning_rate=0.1, random_state=seed, eval_metric="logloss")
    if name == "LightGBM": return lgb.LGBMClassifier(num_leaves=31, max_depth=20, learning_rate=0.1, n_estimators=5 if smoke_test else 200, random_state=seed, verbosity=-1)
    raise ValueError(f"Unknown model: {name}")


def evaluate_estimator(name: str, experiment: str, *, smoke_test: bool = False) -> None:
    X_train, y_train, X_test, y_test = load_split(experiment, flatten=True)
    estimator = build_estimator(name, smoke_test=smoke_test); estimator.fit(X_train, y_train)
    frame = legacy_baseline_bootstrap(estimator, X_test, y_test, repetitions=2 if smoke_test else 100)
    output_path = result_path(name, experiment); frame.to_csv(output_path, index=False)
    print(frame.mean(numeric_only=True).to_string()); print(f"Results saved to {output_path}")
