"""Create the original uncorrected, manually selected cross-batch split."""

from __future__ import annotations

import argparse

import numpy as np

from chd_pipeline.paths import PROCESSED_DIR, experiment_name, generated_split_dir


def load_and_combine(file_names: list[str]) -> np.ndarray:
    return np.array([np.load(PROCESSED_DIR / name) for name in file_names])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batches", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--test-batch", type=int, default=4)
    args = parser.parse_args()
    experiment = experiment_name(args.train_batches, args.test_batch)

    labels_data = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)
    file_names = labels_data[:, 0]
    labels = labels_data[:, 1].astype(int)
    batches = labels_data[:, 2].astype(int)
    data_files = sorted(
        path.name for path in PROCESSED_DIR.glob("*.npy") if path.name != "labels.npy"
    )
    if set(file_names) != set(data_files):
        raise ValueError("File names in labels.npy do not match the processed spectra")

    train_files, train_labels, test_files, test_labels = [], [], [], []
    for index, batch in enumerate(batches):
        if batch in args.train_batches:
            train_files.append(str(file_names[index]))
            train_labels.append(labels[index])
        elif batch == args.test_batch:
            test_files.append(str(file_names[index]))
            test_labels.append(labels[index])

    output_dir = generated_split_dir(experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train.npy": load_and_combine(train_files),
        "X_test.npy": load_and_combine(test_files),
        "y_train.npy": np.array(train_labels),
        "y_test.npy": np.array(test_labels),
    }
    for name, array in arrays.items():
        np.save(output_dir / name, array)
    print(f"Saved uncorrected experiment {experiment} to {output_dir}")


if __name__ == "__main__":
    main()
