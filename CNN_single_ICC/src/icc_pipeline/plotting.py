"""Headless plotting helpers for the recorded ICC bootstrap results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES_DIR, LEGACY_RESULTS_DIR


ALGORITHM_ORDER = [
    "RandomForest",
    "SVM",
    "AdaBoost",
    "XGBoost",
    "EQLC",
    "SoftVoting",
    "HardVoting",
]


def load_legacy_metric(metric: str) -> tuple[list[str], list[np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    for path in sorted(LEGACY_RESULTS_DIR.glob("*.csv")):
        algorithm = path.stem.split("_")[0]
        frame = pd.read_csv(path)
        if metric in frame:
            values[algorithm] = frame[metric].to_numpy()
    names = [name for name in ALGORITHM_ORDER if name in values]
    if not names:
        raise FileNotFoundError(f"No CSV with column {metric!r} in {LEGACY_RESULTS_DIR}")
    return names, [values[name] for name in names]


def draw_metric(metric: str, output_name: str) -> Path:
    names, values = load_legacy_metric(metric)
    figure, axis = plt.subplots(figsize=(10, 5))
    boxes = axis.boxplot(values, tick_labels=names, patch_artist=True)
    colors = ["#C79DC9", "#C2B4D3", "#CCD3E5", "#0076B9", "#EF607A", "#EC3E31", "#DB2834"]
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
    axis.set_ylabel(metric, fontsize=16)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(False)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / output_name
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    for name, metric_values in zip(names, values):
        print(f"{name}: variance={np.var(metric_values):.8f}, std={np.std(metric_values):.4f}")
    print(f"Figure saved to {output_path}")
    return output_path
