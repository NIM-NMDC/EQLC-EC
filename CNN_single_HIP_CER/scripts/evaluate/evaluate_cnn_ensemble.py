#!/usr/bin/env python3
"""Evaluate hard and soft voting across the tracked CNN checkpoints."""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from hip_cer_pipeline.data import load_split, timestamped_result_path
from hip_cer_pipeline.paths import LEGACY_CHECKPOINTS_DIR
from hip_cer_pipeline.torch_models import ArrayDataset, Simple1DCNN


def metric_row(number: int, labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "Bootstrap Sample": number,
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": roc_auc_score(labels, scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    _, _, X_test, y_test = load_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = sorted(LEGACY_CHECKPOINTS_DIR.glob("*.pt"))
    if args.smoke_test:
        checkpoint_paths = checkpoint_paths[:1]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {LEGACY_CHECKPOINTS_DIR}")

    models = []
    for checkpoint_path in checkpoint_paths:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model.to(device))

    # The historical script did not set a seed for ensemble resampling.
    random_state = np.random.RandomState(args.seed) if args.seed is not None else np.random
    hard_rows: list[dict[str, float | int]] = []
    soft_rows: list[dict[str, float | int]] = []
    repetitions = 2 if args.smoke_test else 100

    for number in range(1, repetitions + 1):
        indices = random_state.choice(len(X_test), 240, replace=True)
        bootstrap_data = X_test[indices]
        bootstrap_labels = y_test[indices]
        loader = DataLoader(ArrayDataset(bootstrap_data, bootstrap_labels), batch_size=1024)
        hard_batches = []
        soft_batches = []
        with torch.no_grad():
            for inputs, _ in loader:
                outputs = [model(inputs.to(device)) for model in models]
                hard_batches.append(
                    np.array([output.argmax(dim=1).cpu().numpy() for output in outputs]).T
                )
                soft_batches.append(
                    np.mean([torch.softmax(output, dim=1).cpu().numpy() for output in outputs], axis=0)
                )
        votes = np.concatenate(hard_batches)
        hard_predictions = np.array([np.argmax(np.bincount(row)) for row in votes])
        soft_probabilities = np.concatenate(soft_batches)
        soft_predictions = soft_probabilities.argmax(axis=1)
        hard_rows.append(metric_row(number, bootstrap_labels, hard_predictions, hard_predictions))
        soft_rows.append(metric_row(number, bootstrap_labels, soft_predictions, soft_probabilities[:, 1]))

    hard_path = timestamped_result_path("HardVoting")
    soft_path = timestamped_result_path("SoftVoting")
    pd.DataFrame(hard_rows).to_csv(hard_path, index=False)
    pd.DataFrame(soft_rows).to_csv(soft_path, index=False)
    print(f"Hard-voting results saved to {hard_path}")
    print(f"Soft-voting results saved to {soft_path}")


if __name__ == "__main__":
    main()
