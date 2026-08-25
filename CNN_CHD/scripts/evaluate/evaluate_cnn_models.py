"""Evaluate each legacy CNN plus hard and soft voting on one experiment."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from chd_pipeline.data import load_split, timestamped_result_path
from chd_pipeline.paths import DEFAULT_EXPERIMENT, EXPERIMENTS, legacy_model_dir
from chd_pipeline.torch_models import ArrayDataset, Simple1DCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_test, y_test = load_split(args.experiment)
    test_dataset = ArrayDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model_paths = [path for path in legacy_model_dir(args.experiment).iterdir() if path.suffix == ".pt"]
    if args.smoke_test:
        model_paths = model_paths[:2]
    models, model_names = [], []
    for path in model_paths:
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        models.append(model.to(device))
        model_names.append(path.name)
    if not models:
        raise FileNotFoundError(f"No checkpoints found for experiment {args.experiment}")

    rows = []
    with torch.no_grad():
        for model, model_name in zip(models, model_names):
            predictions, probabilities, labels = [], [], []
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predictions.append(outputs.argmax(1).cpu().numpy())
                probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
                labels.append(targets.cpu().numpy())
            predictions_array = np.concatenate(predictions)
            probabilities_array = np.concatenate(probabilities)
            labels_array = np.concatenate(labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_array, predictions_array, average="binary"
            )
            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy_score(labels_array, predictions_array),
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "AUC": roc_auc_score(labels_array, probabilities_array[:, 1]),
                }
            )

    hard_batches, soft_batches, label_batches = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            member_predictions, member_probabilities = [], []
            for model in models:
                outputs = model(inputs)
                member_predictions.append(outputs.argmax(1).cpu().numpy())
                member_probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
            hard_batches.append(np.array(member_predictions).T)
            soft_batches.append(np.mean(member_probabilities, axis=0))
            label_batches.append(targets.cpu().numpy())

    hard_member_predictions = np.concatenate(hard_batches, axis=0)
    hard_predictions = np.array(
        [np.argmax(np.bincount(predictions)) for predictions in hard_member_predictions]
    )
    soft_probabilities = np.concatenate(soft_batches, axis=0)
    soft_predictions = np.argmax(soft_probabilities, axis=1)
    labels_array = np.concatenate(label_batches)
    for name, predictions, scores in (
        ("Hard Voting", hard_predictions, hard_predictions),
        ("Soft Voting", soft_predictions, soft_probabilities[:, 1]),
    ):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_array, predictions, average="binary"
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(labels_array, predictions),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUC": roc_auc_score(labels_array, scores),
            }
        )

    output_path = timestamped_result_path("ModelPerformance", args.experiment)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Loaded {len(models)} trusted legacy models")
    print(f"Model performance saved to {output_path}")


if __name__ == "__main__":
    main()
