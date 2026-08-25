"""Bootstrap evaluation of the recorded XGBoost configuration."""

import argparse

from xgboost import XGBClassifier

from icc_pipeline.classical import SEED, evaluate_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model = XGBClassifier(
        n_estimators=10 if args.smoke_test else 500,
        learning_rate=0.1,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )
    evaluate_estimator("XGBoost", model, n_bootstraps=5 if args.smoke_test else 100)


if __name__ == "__main__":
    main()
