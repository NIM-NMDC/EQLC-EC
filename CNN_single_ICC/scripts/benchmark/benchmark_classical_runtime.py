"""Benchmark independent classical-model fits on stratified folds."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from icc_pipeline.data import load_split
from icc_pipeline.paths import RUNTIME_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.repeats = 2
    X_train, y_train, _, _ = load_split(flatten=True)
    seed = 42
    small = args.smoke_test
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=10 if small else 500,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "SVM": SVC(C=10, gamma=0.01, kernel="rbf", probability=True, cache_size=2000),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
            n_estimators=10 if small else 200,
            learning_rate=1.0,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=10 if small else 500,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    cv = StratifiedKFold(n_splits=2 if small else 5, shuffle=True, random_state=seed)
    rows = []
    for name, template in models.items():
        repeat_train_times, repeat_prediction_times = [], []
        for repeat in range(args.repeats):
            fold_train_times, fold_prediction_times = [], []
            for train_indices, validation_indices in cv.split(X_train, y_train):
                model = clone(template)
                start = time.perf_counter()
                model.fit(X_train[train_indices], y_train[train_indices])
                fold_train_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                model.predict(X_train[validation_indices])
                fold_prediction_times.append(time.perf_counter() - start)
            if repeat > 0:
                repeat_train_times.append(float(np.mean(fold_train_times)))
                repeat_prediction_times.append(float(np.mean(fold_prediction_times)))
        rows.append(
            {
                "Model": name,
                "Train_Time_Mean": float(np.mean(repeat_train_times)),
                "Train_Time_Std": float(np.std(repeat_train_times)),
                "Predict_Time_Mean": float(np.mean(repeat_prediction_times)),
                "Predict_Time_Std": float(np.std(repeat_prediction_times)),
            }
        )
        print(rows[-1])
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUNTIME_DIR / "model_runtime_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
