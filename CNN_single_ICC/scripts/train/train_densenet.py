"""Exploratory DenseNet121 baseline for ICC spectra.

The one-dimensional spectrum is repeated vertically to avoid the zero-height
failure in the original implementation. ImageNet weights are opt-in because a
mass spectrum is not a natural image and downloading weights must be explicit.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import DenseNet121_Weights, densenet121

from icc_pipeline.data import classification_metrics, load_split
from icc_pipeline.torch_models import seed_everything


class SpectrumImageDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        spectrum = torch.as_tensor(self.data[index], dtype=torch.float32).squeeze()
        image = spectrum.unsqueeze(0).unsqueeze(0).repeat(1, 32, 1)
        return image, torch.as_tensor(self.labels[index], dtype=torch.long)


def build_model(pretrained: bool) -> nn.Module:
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    labels, predictions, scores, losses = [], [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            losses.append(criterion(outputs, targets).item())
            labels.append(targets.cpu().numpy())
            predictions.append(outputs.argmax(1).cpu().numpy())
            scores.append(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
    labels = np.concatenate(labels)
    return float(np.mean(losses)), classification_metrics(
        labels, np.concatenate(predictions), np.concatenate(scores)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    seed = 42
    seed_everything(seed)
    X_train, y_train, X_test, y_test = load_split()
    train_indices, validation_indices = train_test_split(
        np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=seed
    )
    train_data = SpectrumImageDataset(X_train, y_train)
    validation_data = SpectrumImageDataset(X_train, y_train)
    test_data = SpectrumImageDataset(X_test, y_test)
    if args.smoke_test:
        args.epochs = 1
        train_indices = train_indices[:16]
        validation_indices = validation_indices[:8]
        test_indices = np.concatenate(
            [np.flatnonzero(y_test == class_index)[:4] for class_index in range(2)]
        )
        test_data = Subset(test_data, test_indices)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(validation_data, validation_indices), batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        validation_loss, validation_metrics = evaluate(
            model, validation_loader, criterion, device
        )
        print(f"Epoch {epoch}: validation loss={validation_loss:.4f}, {validation_metrics}")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}, metrics={test_metrics}")


if __name__ == "__main__":
    main()
