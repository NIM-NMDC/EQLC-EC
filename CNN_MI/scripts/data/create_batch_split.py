#!/usr/bin/env python3
"""Create one of the three leave-one-batch-out MI splits."""

import argparse
import numpy as np
from mi_pipeline.paths import DEFAULT_EXPERIMENT, EXPERIMENTS, PROCESSED_DIR, split_dir


def combine(filenames: list[str]) -> np.ndarray: return np.array([np.load(PROCESSED_DIR / filename) for filename in filenames])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT); args = parser.parse_args()
    labels_data = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True); filenames = labels_data[:, 0]; labels = labels_data[:, 1].astype(int); batches = labels_data[:, 2].astype(int)
    actual = {path.name for path in PROCESSED_DIR.glob("*.npy") if path.name != "labels.npy"}; assert set(filenames) == actual, "labels.npy does not match processed spectra"
    training_batches, testing_batch = EXPERIMENTS[args.experiment]; train_mask = np.isin(batches, training_batches); test_mask = batches == testing_batch
    directory = split_dir(args.experiment); directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "X_train.npy", combine(filenames[train_mask].tolist())); np.save(directory / "X_test.npy", combine(filenames[test_mask].tolist())); np.save(directory / "y_train.npy", labels[train_mask]); np.save(directory / "y_test.npy", labels[test_mask])
    print(f"Experiment {args.experiment}: train batches {training_batches}, test batch {testing_batch}")
    print(f"Training samples: {train_mask.sum()}; testing samples: {test_mask.sum()}")
    print(f"Split arrays saved to {directory}")


if __name__ == "__main__": main()
