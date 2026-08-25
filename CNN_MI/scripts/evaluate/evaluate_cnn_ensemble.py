#!/usr/bin/env python3
"""Evaluate hard and soft CNN voting for one MI batch experiment."""

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from mi_pipeline.data import load_split, result_path
from mi_pipeline.paths import DEFAULT_EXPERIMENT, EXPERIMENTS, checkpoint_dir
from mi_pipeline.torch_models import ArrayDataset, Simple1DCNN


def metric_row(number: int, labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"Bootstrap Sample": number, "Accuracy": accuracy_score(labels, predictions), "Precision": precision, "Recall": recall, "F1 Score": f1, "AUC": roc_auc_score(labels, scores)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--seed", type=int); args = parser.parse_args()
    _, _, X_test, y_test = load_split(args.experiment); checkpoints = sorted(checkpoint_dir(args.experiment).glob("*.pt")); checkpoints = checkpoints[:1] if args.smoke_test else checkpoints
    if not checkpoints: raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir(args.experiment)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); models = [torch.load(path, map_location=device, weights_only=False).to(device).eval() for path in checkpoints]
    random_state = np.random.RandomState(args.seed) if args.seed is not None else np.random; hard_rows, soft_rows = [], []
    for number in range(1, (2 if args.smoke_test else 100) + 1):
        indices = random_state.choice(len(X_test), 1200, replace=True); labels = y_test[indices]; hard_batches, soft_batches = [], []
        with torch.no_grad():
            for inputs, _ in DataLoader(ArrayDataset(X_test[indices], labels), batch_size=1024):
                outputs = [model(inputs.to(device)) for model in models]; hard_batches.append(np.array([output.argmax(1).cpu().numpy() for output in outputs]).T); soft_batches.append(np.mean([torch.softmax(output, 1).cpu().numpy() for output in outputs], axis=0))
        votes = np.concatenate(hard_batches); hard_predictions = np.array([np.argmax(np.bincount(vote)) for vote in votes]); probabilities = np.concatenate(soft_batches); soft_predictions = probabilities.argmax(1)
        hard_rows.append(metric_row(number, labels, hard_predictions, hard_predictions)); soft_rows.append(metric_row(number, labels, soft_predictions, probabilities[:, 1]))
    hard_path, soft_path = result_path("HardVoting", args.experiment), result_path("SoftVoting", args.experiment); pd.DataFrame(hard_rows).to_csv(hard_path, index=False); pd.DataFrame(soft_rows).to_csv(soft_path, index=False); print(f"Hard-voting results saved to {hard_path}"); print(f"Soft-voting results saved to {soft_path}")


if __name__ == "__main__": main()
