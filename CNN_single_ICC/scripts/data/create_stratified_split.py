"""Create the legacy-compatible, stratified 80/20 ICC split."""

from __future__ import annotations

import csv

import numpy as np
from sklearn.model_selection import train_test_split

from icc_pipeline.paths import PROCESSED_DIR, SPLIT_DIR


SEED = 46


def load_and_combine(file_names: np.ndarray) -> np.ndarray:
    return np.asarray([np.load(PROCESSED_DIR / str(name)) for name in file_names])


def main() -> None:
    labels_path = PROCESSED_DIR / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError("Run scripts/data/preprocess_spectra.py first")
    labels_data = np.load(labels_path)
    file_names = labels_data[:, 0]
    labels = labels_data[:, 1].astype(int)
    data_files = sorted(path.name for path in PROCESSED_DIR.glob("*.npy") if path.name != "labels.npy")
    if set(file_names) != set(data_files):
        missing = sorted(set(file_names) - set(data_files))
        extra = sorted(set(data_files) - set(file_names))
        raise ValueError(f"Processed files do not match labels; missing={missing}, extra={extra}")

    train_files, test_files, train_labels, test_labels = train_test_split(
        file_names,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train.npy": load_and_combine(train_files),
        "X_test.npy": load_and_combine(test_files),
        "y_train.npy": train_labels,
        "y_test.npy": test_labels,
    }
    for name, array in arrays.items():
        np.save(SPLIT_DIR / name, array)

    with (SPLIT_DIR / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "label", "split"])
        writer.writerows((name, int(label), "train") for name, label in zip(train_files, train_labels))
        writer.writerows((name, int(label), "test") for name, label in zip(test_files, test_labels))

    train_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
    test_counts = dict(zip(*np.unique(test_labels, return_counts=True)))
    print(f"Saved split to {SPLIT_DIR}")
    print(f"Train: {len(train_labels)}, class counts: {train_counts}")
    print(f"Test: {len(test_labels)}, class counts: {test_counts}")


if __name__ == "__main__":
    main()
