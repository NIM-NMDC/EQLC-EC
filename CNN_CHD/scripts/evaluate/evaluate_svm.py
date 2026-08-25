"""Run the legacy SVM script on a generated uncorrected CHD batch split."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.svm import SVC

from chd_pipeline.data import classification_metrics, load_split, save_bootstrap_rows
from chd_pipeline.paths import EXPERIMENTS


SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="13_4")
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.bootstrap_samples = 2
        args.sample_size = 50
    np.random.seed(SEED)

    X_train, y_train, X_test, y_test = load_split(
        args.experiment, corrected=False, generated=True, flatten=True
    )
    model = SVC(
        C=10,
        gamma=0.01,
        kernel="rbf",
        random_state=SEED,
        probability=True,
        cache_size=2000,
    )
    model.fit(X_train, y_train)
    rows = []
    for number in range(1, args.bootstrap_samples + 1):
        # Preserve the original script's first-N-samples bootstrap index range.
        indices = np.random.choice(args.sample_size, args.sample_size, replace=True)
        labels = y_test[indices]
        predictions = model.predict(X_test[indices])
        probabilities = model.predict_proba(X_test[indices])[:, 1]
        rows.append(
            {
                "Bootstrap Sample": number,
                **classification_metrics(labels, predictions, probabilities),
            }
        )
    output_path = save_bootstrap_rows(rows, "SVM", args.experiment)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
