"""Bootstrap hard and soft voting for a legacy CHD CNN ensemble."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from chd_pipeline.data import classification_metrics, load_split, save_bootstrap_rows
from chd_pipeline.paths import DEFAULT_EXPERIMENT, EXPERIMENTS, legacy_model_dir
from chd_pipeline.torch_models import ArrayDataset, Simple1DCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.bootstrap_samples = 2
        args.sample_size = 50

    _, _, X_test, y_test = load_split(args.experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in legacy_model_dir(args.experiment).iterdir():
        if path.suffix == ".pt":
            model = torch.load(path, map_location=device, weights_only=False).to(device)
            model.eval()
            models.append(model)
    if args.smoke_test:
        models = models[:2]
    if not models:
        raise FileNotFoundError(f"No checkpoints found for experiment {args.experiment}")

    hard_rows, soft_rows = [], []
    for number in range(1, args.bootstrap_samples + 1):
        indices = np.random.choice(X_test.shape[0], args.sample_size, replace=True)
        data, labels = X_test[indices], y_test[indices]
        hard_batches, soft_batches = [], []
        loader = DataLoader(ArrayDataset(data, labels), batch_size=1024, shuffle=False)
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                member_predictions, member_probabilities = [], []
                for model in models:
                    outputs = model(inputs)
                    member_predictions.append(outputs.argmax(1).cpu().numpy())
                    member_probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
                hard_batches.append(np.array(member_predictions).T)
                soft_batches.append(np.mean(member_probabilities, axis=0))
        member_predictions = np.concatenate(hard_batches, axis=0)
        hard_predictions = np.array(
            [np.argmax(np.bincount(predictions)) for predictions in member_predictions]
        )
        soft_probabilities = np.concatenate(soft_batches, axis=0)
        soft_predictions = np.argmax(soft_probabilities, axis=1)
        hard_rows.append(
            {
                "Bootstrap Sample": number,
                **classification_metrics(labels, hard_predictions, hard_predictions),
            }
        )
        soft_rows.append(
            {
                "Bootstrap Sample": number,
                **classification_metrics(labels, soft_predictions, soft_probabilities[:, 1]),
            }
        )

    hard_path = save_bootstrap_rows(hard_rows, "HardVoting", args.experiment)
    soft_path = save_bootstrap_rows(soft_rows, "SoftVoting", args.experiment)
    print(f"Loaded {len(models)} trusted legacy models")
    print(f"Hard-voting results saved to {hard_path}")
    print(f"Soft-voting results saved to {soft_path}")


if __name__ == "__main__":
    main()
