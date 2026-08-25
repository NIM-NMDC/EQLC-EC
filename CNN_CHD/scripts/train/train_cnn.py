"""Train the published CHD 1D CNN without changing its experiment logic."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from chd_pipeline.data import load_split
from chd_pipeline.paths import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    generated_model_dir,
    generated_result_dir,
)
from chd_pipeline.torch_models import ArrayDataset, Simple1DCNN, seed_everything


def print_dataset_info(dataset: ArrayDataset, name: str) -> None:
    labels = [dataset[index][1].item() for index in range(len(dataset))]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"{name} size: {len(dataset)} samples")
    print(f"{name} shape (first sample): {dataset[0][0].shape}")
    print(f"{name} class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")


def evaluate_model(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    predictions, labels, probabilities = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
            probabilities.append(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    predictions_array = np.concatenate(predictions)
    labels_array = np.concatenate(labels)
    probabilities_array = np.concatenate(probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array, predictions_array, average="binary"
    )
    return (
        loss / len(loader),
        accuracy_score(labels_array, predictions_array),
        precision,
        recall,
        f1,
        roc_auc_score(labels_array, probabilities_array),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--use-generated-split", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = 1

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_split(
        args.experiment, corrected=True, generated=args.use_generated_split
    )
    train_dataset = ArrayDataset(X_train, y_train)
    test_dataset = ArrayDataset(X_test, y_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print_dataset_info(train_dataset, "Training set")
    print_dataset_info(test_dataset, "Testing set")

    class_counts = np.unique(y_train, return_counts=True)[1]
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = Simple1DCNN(input_size=199, num_classes=2, dropout_value=args.dropout).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model.train()
    log_rows = []
    test_accuracy = float("nan")

    for epoch in range(args.epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # The published script evaluates the test set after every epoch and leaves
        # the model in eval mode. This behavior is intentionally preserved.
        test_loss, test_accuracy, precision, recall, f1, auc = evaluate_model(
            model, test_loader, criterion, device
        )
        row = {
            "Epoch": epoch + 1,
            "Train Loss": running_loss / len(train_loader),
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc,
        }
        log_rows.append(row)
        print(row)
        if (epoch + 1) % 5 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.95
            print(f"Learning rate updated to {optimizer.param_groups[0]['lr']:.6f}")

    result_dir = generated_result_dir(args.experiment) / "training_logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = str(args.learning_rate)
    log_path = result_dir / (
        f"training_log_b{args.batch_size}_do{args.dropout}_lr{learning_rate}_"
        f"wd{args.weight_decay}_seed{args.seed}.csv"
    )
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    model_dir = generated_model_dir(args.experiment)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (
        f"Simple1DCNN_{learning_rate}_{args.batch_size}_{args.epochs}_"
        f"{args.dropout:.2f}_seed{args.seed}_acc{test_accuracy:.4f}.pt"
    )
    torch.save(model, model_path)
    print(f"Training log saved to {log_path}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
