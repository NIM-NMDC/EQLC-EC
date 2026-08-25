"""Bootstrap evaluation of the recorded AdaBoost configuration."""

import argparse

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from icc_pipeline.classical import SEED, evaluate_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
        n_estimators=10 if args.smoke_test else 200,
        learning_rate=1.0,
        random_state=SEED,
    )
    evaluate_estimator("AdaBoost", model, n_bootstraps=5 if args.smoke_test else 100)


if __name__ == "__main__":
    main()
