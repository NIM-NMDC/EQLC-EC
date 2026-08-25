"""Headless plotting support for tracked tomato bootstrap tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES_DIR, LEGACY_RESULTS_DIR


ORDER = ["RandomForest", "SVM", "AdaBoost", "XGBoost", "EQLC", "SoftVoting", "HardVoting"]


def load_metric(metric: str) -> tuple[list[str], list[np.ndarray]]:
    values = {}
    for path in sorted(LEGACY_RESULTS_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        if metric in frame:
            values[path.stem.split("_")[0]] = frame[metric].to_numpy()
    names = [name for name in ORDER if name in values]
    if not names:
        raise FileNotFoundError(f"No tracked table contains {metric!r}")
    return names, [values[name] for name in names]


def draw_metric(metric: str, filename: str) -> None:
    names, values = load_metric(metric)
    figure, axis = plt.subplots(figsize=(10, 5))
    boxes = axis.boxplot(values, tick_labels=names, patch_artist=True)
    for patch, color in zip(boxes["boxes"], ["#C79DC9", "#C2B4D3", "#CCD3E5", "#0076B9", "#EF607A", "#EC3E31", "#DB2834"]):
        patch.set_facecolor(color)
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(False)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    for name, samples in zip(names, values):
        print(f"{name}: variance={np.var(samples):.8f}, std={np.std(samples):.4f}")
    print(f"Figure saved to {output_path}")
