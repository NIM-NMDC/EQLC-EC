"""Replay deterministic legacy ICC baselines and compare their stored CSVs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from icc_pipeline.data import classification_metrics, load_split
from icc_pipeline.paths import GENERATED_RESULTS_DIR, LEGACY_RESULTS_DIR


def legacy_bootstrap(y_test, predictions, probabilities):
    np.random.seed(42)
    rows = []
    # This deliberately replays the old bug: only indices 0..299 were sampled.
    for bootstrap_number in range(1, 101):
        indices = np.random.choice(300, 300, replace=True)
        row = {"Bootstrap Sample": bootstrap_number}
        row.update(classification_metrics(y_test[indices], predictions[indices], probabilities[indices]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    X_train, y_train, X_test, y_test = load_split(flatten=True)
    models = {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=200,
            learning_rate=1.0,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(C=10, gamma=0.01, kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    report = {}
    columns = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        replay = legacy_bootstrap(y_test, predictions, probabilities)
        candidates = sorted(LEGACY_RESULTS_DIR.glob(f"{name}_*.csv"))
        if not candidates:
            report[name] = {"status": "missing legacy CSV"}
            continue
        legacy = pd.read_csv(candidates[-1])
        max_difference = float(np.abs(replay[columns] - legacy[columns]).to_numpy().max())
        report[name] = {
            "legacy_file": candidates[-1].name,
            "exact_replay": bool(max_difference < 1e-12),
            "maximum_absolute_difference": max_difference,
            "replayed_accuracy_mean": float(replay["Accuracy"].mean()),
            "legacy_accuracy_mean": float(legacy["Accuracy"].mean()),
            "replayed_auc_mean": float(replay["AUC"].mean()),
            "legacy_auc_mean": float(legacy["AUC"].mean()),
        }
        print(name, report[name])
    GENERATED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_RESULTS_DIR / "legacy_verification.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verification report saved to {output_path}")


if __name__ == "__main__":
    main()
