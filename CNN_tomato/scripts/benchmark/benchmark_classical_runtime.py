#!/usr/bin/env python3
"""Benchmark the four classical tomato baselines."""

import argparse, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tomato_pipeline.classical import build_estimator
from tomato_pipeline.data import load_split
from tomato_pipeline.paths import RUNTIME_DIR


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke-test", action="store_true"); args = parser.parse_args()
    X_train, y_train, _, _ = load_split(flatten=True)
    cv = StratifiedKFold(n_splits=2 if args.smoke_test else 5, shuffle=True, random_state=42)
    repeats = 2 if args.smoke_test else 11; rows = []
    for name in ("RandomForest", "SVM", "AdaBoost", "XGBoost"):
        train_times, prediction_times = [], []
        for repeat in range(repeats):
            fold_train, fold_prediction = [], []
            model = build_estimator(name, smoke_test=args.smoke_test)
            for train_indices, validation_indices in cv.split(X_train, y_train):
                started = time.time(); model.fit(X_train[train_indices], y_train[train_indices]); fold_train.append(time.time() - started)
                started = time.time(); model.predict(X_train[validation_indices]); fold_prediction.append(time.time() - started)
            if repeat > 0: train_times.append(np.mean(fold_train)); prediction_times.append(np.mean(fold_prediction))
        rows.append({"Model": name, "Train_Time_Mean": np.mean(train_times), "Train_Time_Std": np.std(train_times), "Predict_Time_Mean": np.mean(prediction_times), "Predict_Time_Std": np.std(prediction_times), "All_Train_Times": train_times, "All_Predict_Times": prediction_times})
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True); output_path = RUNTIME_DIR / "classical_model_runtime.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False); print(f"Results saved to {output_path}")


if __name__ == "__main__": main()
