"""Bootstrap the published single-model EQLC result."""

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
    matching = [
        path
        for path in legacy_model_dir(args.experiment).iterdir()
        if "Simple1DCNN_5e-05_256_200_0.50_seed41" in path.name and path.suffix == ".pt"
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one EQLC checkpoint, found {len(matching)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(matching[0], map_location=device, weights_only=False).to(device)
    model.eval()
    rows = []
    for number in range(1, args.bootstrap_samples + 1):
        indices = np.random.choice(X_test.shape[0], args.sample_size, replace=True)
        data, labels = X_test[indices], y_test[indices]
        predictions, probabilities = [], []
        loader = DataLoader(ArrayDataset(data, labels), batch_size=1024, shuffle=False)
        with torch.no_grad():
            for inputs, _ in loader:
                outputs = model(inputs.to(device))
                predictions.append(outputs.argmax(1).cpu().numpy())
                probabilities.append(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        metrics = classification_metrics(
            labels, np.concatenate(predictions), np.concatenate(probabilities)
        )
        rows.append({"Bootstrap Sample": number, **metrics})
    output_path = save_bootstrap_rows(rows, "EQLC", args.experiment)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
