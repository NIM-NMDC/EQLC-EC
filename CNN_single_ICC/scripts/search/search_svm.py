"""Stratified cross-validated support-vector-machine parameter search."""

import argparse

from sklearn.svm import SVC

from icc_pipeline.classical import SEED, search_estimator


def build(params):
    kernel = str(params["kernel"])
    return SVC(
        C=float(params["C"]),
        gamma=float(params["gamma"]) if kernel == "rbf" else "scale",
        kernel=kernel,
        random_state=SEED,
        probability=True,
        cache_size=2000,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    search_estimator(
        "SVM",
        build,
        {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1],
            "kernel": ["linear", "rbf"],
        },
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
