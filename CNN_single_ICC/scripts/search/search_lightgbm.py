"""Stratified cross-validated LightGBM parameter search."""

import argparse

from lightgbm import LGBMClassifier

from icc_pipeline.classical import SEED, search_estimator


def build(params):
    return LGBMClassifier(
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    search_estimator(
        "LightGBM",
        build,
        {
            "num_leaves": [31, 50, 100],
            "max_depth": [-1, 10, 20, 30],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
        },
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
