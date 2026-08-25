"""Stratified cross-validated random-forest parameter search."""

import argparse

from sklearn.ensemble import RandomForestClassifier

from icc_pipeline.classical import SEED, search_estimator


def build(params):
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=SEED,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    search_estimator(
        "RandomForest",
        build,
        {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        },
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
