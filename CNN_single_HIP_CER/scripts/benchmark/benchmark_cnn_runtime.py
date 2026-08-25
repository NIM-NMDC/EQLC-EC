#!/usr/bin/env python3
"""Benchmark CNN training-epoch and prediction runtime."""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hip_cer_pipeline.data import load_split
from hip_cer_pipeline.paths import RUNTIME_DIR
from hip_cer_pipeline.torch_models import ArrayDataset, Simple1DCNN, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    seed_everything(46)
    X_train, y_train, X_test, y_test = load_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=256, shuffle=True)
    test_loader = DataLoader(ArrayDataset(X_test, y_test), batch_size=256)
    model = Simple1DCNN(2463, 2, 0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 1 if args.smoke_test else 100
    repeats = 2 if args.smoke_test else 11
    train_times = []
    predict_times = []

    for repeat in range(repeats):
        started = time.perf_counter()
        model.train()
        for _ in range(epochs):
            for batch_number, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                if args.smoke_test and batch_number == 0:
                    break
        epoch_time = (time.perf_counter() - started) / epochs
        model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            for inputs, _ in test_loader:
                model(inputs.to(device))
        prediction_time = time.perf_counter() - started
        if repeat > 0:
            train_times.append(epoch_time)
            predict_times.append(prediction_time)

    frame = pd.DataFrame({
        "Run": range(2, repeats + 1),
        "Avg_Train_Epoch_Time": train_times,
        "Predict_Time": predict_times,
    })
    frame["Avg_Train_Epoch_Time_Mean"] = frame["Avg_Train_Epoch_Time"].mean()
    frame["Avg_Train_Epoch_Time_Std"] = frame["Avg_Train_Epoch_Time"].std()
    frame["Predict_Time_Mean"] = frame["Predict_Time"].mean()
    frame["Predict_Time_Std"] = frame["Predict_Time"].std()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUNTIME_DIR / "cnn_runtime_with_prediction.csv"
    frame.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
