"""Evaluate hard and soft voting for the tracked ICC CNN checkpoints.

The tracked checkpoints are legacy full-object PyTorch pickles. Load them only
because they are project-owned artifacts; new training saves state dictionaries.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from icc_pipeline.data import (
    bootstrap_predictions,
    classification_metrics,
    load_split,
    timestamped_result_path,
)
from icc_pipeline.paths import LEGACY_CHECKPOINTS_DIR
from icc_pipeline.torch_models import ArrayDataset, Simple1DCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.bootstrap_samples = 5

    _, _, X_test, y_test = load_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = sorted(LEGACY_CHECKPOINTS_DIR.glob("*.pt"))
    if args.smoke_test:
        model_paths = model_paths[:2]
    if not model_paths:
        raise FileNotFoundError(f"No .pt models found in {LEGACY_CHECKPOINTS_DIR}")

    models = []
    for path in model_paths:
        model = torch.load(path, map_location=device, weights_only=False)
        model.to(device).eval()
        models.append(model)
    print(f"Loaded {len(models)} trusted legacy models")

    loader = DataLoader(ArrayDataset(X_test, y_test), batch_size=args.batch_size)
    member_predictions, member_probabilities = [], []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            batch_predictions, batch_probabilities = [], []
            for model in models:
                outputs = model(inputs)
                batch_predictions.append(outputs.argmax(1).cpu().numpy())
                batch_probabilities.append(torch.softmax(outputs, 1).cpu().numpy())
            member_predictions.append(np.stack(batch_predictions, axis=1))
            member_probabilities.append(np.stack(batch_probabilities, axis=1))

    member_predictions = np.concatenate(member_predictions)
    member_probabilities = np.concatenate(member_probabilities)
    positive_vote_fraction = member_predictions.mean(axis=1)
    hard_predictions = (positive_vote_fraction > 0.5).astype(int)
    mean_probabilities = member_probabilities.mean(axis=1)
    soft_predictions = mean_probabilities.argmax(axis=1)

    evaluations = {
        "HardVoting": (hard_predictions, positive_vote_fraction),
        "SoftVoting": (soft_predictions, mean_probabilities[:, 1]),
    }
    for name, (predictions, scores) in evaluations.items():
        point_metrics = classification_metrics(y_test, predictions, scores)
        bootstrap = bootstrap_predictions(
            y_test,
            lambda indices, p=predictions, s=scores: (p[indices], s[indices]),
            n_bootstraps=args.bootstrap_samples,
            seed=args.seed,
        )
        output_path = timestamped_result_path(name)
        bootstrap.to_csv(output_path, index=False)
        print(f"{name} point estimate: {point_metrics}")
        print(f"{name} bootstrap results saved to {output_path}")


if __name__ == "__main__":
    main()
