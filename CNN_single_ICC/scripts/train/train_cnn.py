"""Train the ICC 1D CNN with validation-only model selection."""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from icc_pipeline.data import classification_metrics, load_split
from icc_pipeline.paths import GENERATED_RESULTS_DIR
from icc_pipeline.torch_models import ArrayDataset, Simple1DCNN, seed_everything


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels, predictions, probabilities = [], [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            losses.append(criterion(outputs, targets).item())
            probabilities.append(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    labels_array = np.concatenate(labels)
    predictions_array = np.concatenate(predictions)
    probabilities_array = np.concatenate(probabilities)
    return float(np.mean(losses)), classification_metrics(
        labels_array, predictions_array, probabilities_array
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = 1

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_full, y_train_full, X_test, y_test = load_split()
    train_indices, validation_indices = train_test_split(
        np.arange(len(y_train_full)),
        test_size=0.2,
        stratify=y_train_full,
        random_state=args.seed,
    )
    train_dataset = ArrayDataset(X_train_full[train_indices], y_train_full[train_indices])
    validation_dataset = ArrayDataset(
        X_train_full[validation_indices], y_train_full[validation_indices]
    )
    test_dataset = ArrayDataset(X_test, y_test)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Simple1DCNN(
        input_size=X_train_full.shape[-1], num_classes=2, dropout_value=args.dropout
    ).to(device)
    class_counts = np.bincount(y_train_full[train_indices], minlength=2)
    class_weights = torch.as_tensor(1.0 / class_counts, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()  # evaluate() switches to eval mode, so restore this every epoch.
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        validation_loss, validation_metrics = evaluate(
            model, validation_loader, criterion, device
        )
        row = {
            "Epoch": epoch,
            "Train Loss": float(np.mean(train_losses)),
            "Validation Loss": validation_loss,
            **{f"Validation {key}": value for key, value in validation_metrics.items()},
        }
        log_rows.append(row)
        print(row)
        if validation_metrics["AUC"] > best_auc:
            best_auc = validation_metrics["AUC"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Final untouched test loss: {test_loss:.4f}")
    print(f"Final untouched test metrics: {test_metrics}")

    run_dir = GENERATED_RESULTS_DIR / "training_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"cnn_seed{args.seed}_training_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    model_path = run_dir / f"Simple1DCNN_seed{args.seed}_state_dict.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": int(X_train_full.shape[-1]),
            "num_classes": 2,
            "dropout": args.dropout,
            "seed": args.seed,
            "validation_auc": best_auc,
            "test_metrics": test_metrics,
        },
        model_path,
    )
    print(f"Training log saved to {log_path}")
    print(f"Safe state-dict checkpoint saved to {model_path}")


if __name__ == "__main__":
    main()
