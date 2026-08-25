"""Reproduce the published joint train/test ComBat preprocessing.

This intentionally fits StandardScaler and ComBat after concatenating the
training and test batches. The behavior is retained for historical
reproducibility and must not be interpreted as a leakage-free pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.preprocessing import StandardScaler

from chd_pipeline.paths import PROCESSED_DIR, experiment_name, generated_split_dir


SEED = 52


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batches", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--test-batch", type=int, default=2)
    args = parser.parse_args()
    np.random.seed(SEED)
    experiment = experiment_name(args.train_batches, args.test_batch)

    labels_data = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)
    file_names = labels_data[:, 0]
    labels = labels_data[:, 1].astype(int)
    batches = labels_data[:, 2].astype(int)
    train_indices = [index for index, batch in enumerate(batches) if batch in args.train_batches]
    test_indices = [index for index, batch in enumerate(batches) if batch == args.test_batch]

    train_data = np.array([np.load(PROCESSED_DIR / file_names[index]) for index in train_indices])
    train_labels = np.array([labels[index] for index in train_indices])
    test_data = np.array([np.load(PROCESSED_DIR / file_names[index]) for index in test_indices])
    test_labels = np.array([labels[index] for index in test_indices])

    train_2d = train_data.reshape(train_data.shape[0], -1)
    test_2d = test_data.reshape(test_data.shape[0], -1)
    combined_data = np.concatenate((train_2d, test_2d), axis=0)
    batch_labels = pd.Series(np.concatenate((batches[train_indices], batches[test_indices])))
    covariates = pd.DataFrame({"batch": batch_labels})

    # Deliberately preserve the published joint fitting order.
    combined_scaled = StandardScaler().fit_transform(pd.DataFrame(combined_data))
    combat_result = neuroCombat(dat=combined_scaled.T, covars=covariates, batch_col="batch")
    combined_combat = combat_result["data"].T

    train_combat = combined_combat[: train_data.shape[0], :].reshape(train_data.shape)
    test_combat = combined_combat[train_data.shape[0] :, :].reshape(test_data.shape)
    output_dir = generated_split_dir(experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train_combat.npy": train_combat,
        "y_train_combat.npy": train_labels,
        "X_test_combat.npy": test_combat,
        "y_test_combat.npy": test_labels,
    }
    for name, array in arrays.items():
        np.save(output_dir / name, array)
    print("WARNING: reproduced the published joint train/test ComBat preprocessing.")
    print(f"Saved corrected experiment {experiment} to {output_dir}")


if __name__ == "__main__":
    main()
