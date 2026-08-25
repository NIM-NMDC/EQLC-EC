#!/usr/bin/env python3
"""Create the published stratified 80/20 HIP/CER split."""

import numpy as np
from sklearn.model_selection import train_test_split

from hip_cer_pipeline.paths import PROCESSED_DIR, SPLIT_DIR


def load_and_combine(filenames: np.ndarray) -> np.ndarray:
    return np.array([np.load(PROCESSED_DIR / str(name)) for name in filenames])


def main() -> None:
    labels_data = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)
    filenames = labels_data[:, 0]
    labels = labels_data[:, 1].astype(int)
    actual_files = {
        path.name for path in PROCESSED_DIR.glob("*.npy") if path.name != "labels.npy"
    }
    assert set(filenames) == actual_files, "labels.npy does not match processed spectra"

    train_files, test_files, train_labels, test_labels = train_test_split(
        filenames,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=46,
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLIT_DIR / "X_train.npy", load_and_combine(train_files))
    np.save(SPLIT_DIR / "X_test.npy", load_and_combine(test_files))
    np.save(SPLIT_DIR / "y_train.npy", np.asarray(train_labels))
    np.save(SPLIT_DIR / "y_test.npy", np.asarray(test_labels))
    print(f"Training samples: {len(train_labels)}")
    print(f"Testing samples: {len(test_labels)}")
    print(f"Split arrays saved to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
