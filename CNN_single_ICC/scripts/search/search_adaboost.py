"""Stratified cross-validated AdaBoost parameter search."""

import argparse

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from icc_pipeline.classical import SEED, search_estimator


def build(params):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=int(params["max_depth"]), random_state=SEED),
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        random_state=SEED,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    search_estimator(
        "AdaBoost",
        build,
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
            "max_depth": [1, 2, 3],
        },
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
