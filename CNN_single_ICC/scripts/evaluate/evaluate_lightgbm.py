"""Bootstrap evaluation of the recorded LightGBM configuration."""

import argparse

from lightgbm import LGBMClassifier

from icc_pipeline.classical import SEED, evaluate_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model = LGBMClassifier(
        num_leaves=31,
        max_depth=20,
        learning_rate=0.1,
        n_estimators=10 if args.smoke_test else 200,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )
    evaluate_estimator("LightGBM", model, n_bootstraps=5 if args.smoke_test else 100)


if __name__ == "__main__":
    main()
