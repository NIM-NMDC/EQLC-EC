"""Benchmark independent ICC 1D-CNN training and prediction runs."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from icc_pipeline.data import load_split
from icc_pipeline.paths import RUNTIME_DIR
from icc_pipeline.torch_models import ArrayDataset, Simple1DCNN, seed_everything


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs, args.repeats = 1, 2
    seed_everything(46)
    X_train, y_train, X_test, y_test = load_split()
    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=256, shuffle=True)
    test_loader = DataLoader(ArrayDataset(X_test, y_test), batch_size=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for repeat in range(args.repeats):
        # A fresh model and optimizer make repeats independent and comparable.
        seed_everything(46 + repeat)
        model = Simple1DCNN(511, 2, 0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()
        epoch_times = []
        for _ in range(args.epochs):
            model.train()
            synchronize(device)
            start = time.perf_counter()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            synchronize(device)
            epoch_times.append(time.perf_counter() - start)
        model.eval()
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            for inputs, _ in test_loader:
                model(inputs.to(device))
        synchronize(device)
        prediction_time = time.perf_counter() - start
        rows.append(
            {
                "Run": repeat + 1,
                "Avg_Train_Epoch_Time": float(np.mean(epoch_times)),
                "Predict_Time": prediction_time,
            }
        )
        print(rows[-1])

    # The first run is warm-up when enough repeats are requested.
    frame = pd.DataFrame(rows[1:] if len(rows) > 1 else rows)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUNTIME_DIR / "deep_learning_model_runtime_with_prediction.csv"
    frame.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
