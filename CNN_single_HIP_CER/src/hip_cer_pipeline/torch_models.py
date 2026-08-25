"""PyTorch dataset and model definitions shared by HIP/CER scripts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = torch.as_tensor(self.data[index], dtype=torch.float32).squeeze()
        label = torch.as_tensor(self.labels[index], dtype=torch.long)
        return sample, label


class Simple1DCNN(nn.Module):
    """The exact architecture used by the tracked HIP/CER checkpoints."""

    def __init__(self, input_size: int, num_classes: int, dropout_value: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_out_size = input_size + 2
        self.pool1_out_size = self.conv1_out_size // 2
        self.conv2_out_size = self.pool1_out_size + 2
        self.pool2_out_size = self.conv2_out_size // 2
        self.fc1 = nn.Linear(64 * self.pool2_out_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.unsqueeze(1)
        outputs = self.pool(torch.relu(self.conv1(outputs)))
        outputs = self.pool(torch.relu(self.conv2(outputs)))
        outputs = outputs.reshape(outputs.shape[0], 64 * self.pool2_out_size)
        outputs = self.dropout(outputs)
        outputs = torch.relu(self.fc1(outputs))
        return self.fc2(outputs)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
