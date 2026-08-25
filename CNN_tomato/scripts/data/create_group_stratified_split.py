#!/usr/bin/env python3
"""Create the original label-stratified split at the tomato sample-ID level."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tomato_pipeline.paths import PROCESSED_DIR, SPLIT_DIR


def combine(filenames: np.ndarray) -> np.ndarray:
    return np.array([np.load(PROCESSED_DIR / str(filename)) for filename in filenames])


def main() -> None:
    labels_data = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)
    frame = pd.DataFrame({
        "filename": labels_data[:, 0],
        "label": labels_data[:, 1].astype(int),
        "sample_id": labels_data[:, 2],
    })
    unique_ids = frame[["sample_id", "label"]].drop_duplicates()
    assert unique_ids["sample_id"].nunique() == len(unique_ids), "A sample ID has conflicting labels"
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=0.29, stratify=unique_ids["label"], random_state=42
    )
    train = frame[frame["sample_id"].isin(train_ids["sample_id"])]
    test = frame[frame["sample_id"].isin(test_ids["sample_id"])]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLIT_DIR / "X_train.npy", combine(train["filename"].to_numpy()))
    np.save(SPLIT_DIR / "X_test.npy", combine(test["filename"].to_numpy()))
    np.save(SPLIT_DIR / "y_train.npy", train["label"].to_numpy())
    np.save(SPLIT_DIR / "y_test.npy", test["label"].to_numpy())
    print(f"Training spectra: {len(train)} from {train['sample_id'].nunique()} sample IDs")
    print(f"Testing spectra: {len(test)} from {test['sample_id'].nunique()} sample IDs")
    print(f"Split arrays saved to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
