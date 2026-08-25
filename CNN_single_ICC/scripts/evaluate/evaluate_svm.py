"""Bootstrap evaluation of the recorded SVM configuration."""

import argparse

from sklearn.svm import SVC

from icc_pipeline.classical import SEED, evaluate_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model = SVC(
        C=10,
        gamma=0.01,
        kernel="rbf",
        random_state=SEED,
        probability=True,
        cache_size=2000,
    )
    evaluate_estimator("SVM", model, n_bootstraps=5 if args.smoke_test else 100)


if __name__ == "__main__":
    main()
