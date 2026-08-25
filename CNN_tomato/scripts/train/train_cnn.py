#!/usr/bin/env python3
"""Train the original full-spectrum tomato one-dimensional CNN."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from tomato_pipeline.data import load_split
from tomato_pipeline.paths import GENERATED_MODELS_DIR, GENERATED_RESULTS_DIR
from tomato_pipeline.torch_models import ArrayDataset, Simple1DCNN, seed_everything


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, ...]:
    model.eval(); loss = 0.0; predictions = []; labels_all = []; probabilities = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss += criterion(outputs, labels).item()
            predictions.append(outputs.argmax(1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            probabilities.append(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
    y_pred, y_true, scores = map(np.concatenate, (predictions, labels_all, probabilities))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return loss / len(loader), accuracy_score(y_true, y_pred), precision, recall, f1, roc_auc_score(y_true, scores)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(); seed = 46; seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_split()
    batch_size = 256
    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ArrayDataset(X_test, y_test), batch_size=batch_size)
    weights = torch.as_tensor(1.0 / np.unique(y_train, return_counts=True)[1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    dropout, learning_rate, weight_decay = 0.2, 0.00001, 0.001
    model = Simple1DCNN(16227, 2, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs = 1 if args.smoke_test else args.epochs; rows = []
    for epoch in range(epochs):
        model.train(); running_loss = 0.0
        for batch_number, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
            loss = criterion(model(inputs), labels); loss.backward(); optimizer.step(); running_loss += loss.item()
            if args.smoke_test and batch_number == 0: break
        test_loss, accuracy, precision, recall, f1, auc = evaluate(model, test_loader, criterion, device)
        rows.append([epoch + 1, running_loss / (batch_number + 1), test_loss, accuracy, precision, recall, f1, auc])
        print(f"Epoch {epoch + 1}/{epochs}: accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        if (epoch + 1) % 30 == 0:
            for group in optimizer.param_groups: group["lr"] *= 0.95
    GENERATED_RESULTS_DIR.mkdir(parents=True, exist_ok=True); GENERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = GENERATED_RESULTS_DIR / f"training_log_b{batch_size}_do{dropout}_lr{learning_rate}_wd{weight_decay}.csv"
    pd.DataFrame(rows, columns=["Epoch", "Train Loss", "Test Loss", "Test Accuracy", "Precision", "Recall", "F1", "AUC"]).to_csv(log_path, index=False)
    model_path = GENERATED_MODELS_DIR / f"Simple1DCNN_{learning_rate}_{batch_size}_{epochs}_{dropout:.2f}_seed{seed}_acc{accuracy:.4f}.pt"
    torch.save(model, model_path); print(f"Training log saved to {log_path}"); print(f"Model saved to {model_path}")


if __name__ == "__main__": main()
