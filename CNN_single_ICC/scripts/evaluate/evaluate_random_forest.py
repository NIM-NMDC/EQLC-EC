"""Bootstrap evaluation of the recorded random-forest configuration."""

import argparse

from sklearn.ensemble import RandomForestClassifier

from icc_pipeline.classical import SEED, evaluate_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model = RandomForestClassifier(
        n_estimators=10 if args.smoke_test else 500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    evaluate_estimator("RandomForest", model, n_bootstraps=5 if args.smoke_test else 100)


if __name__ == "__main__":
    main()
