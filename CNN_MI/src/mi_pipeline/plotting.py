"""Plotting helpers for the three tracked MI batch experiments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .paths import FIGURES_DIR, legacy_result_dir

ORDER = ["RandomForest", "SVM", "XGBoost", "EQLC", "SoftVoting", "HardVoting"]
PRESETS = {"12_3": {"Accuracy": (0.771, (0.70, 0.85)), "F1 Score": (0.825, (0.74, 0.89))}, "13_2": {"Accuracy": (0.807, (0.73, 0.88)), "F1 Score": (0.853, (0.76, 0.91))}, "23_1": {"Accuracy": (0.876, (0.75, 0.90)), "F1 Score": (0.876, (0.77, 0.92))}}


def load_metric(experiment: str, metric: str) -> tuple[list[str], list[np.ndarray]]:
    values = {}
    for path in sorted(legacy_result_dir(experiment).glob("*.csv")):
        frame = pd.read_csv(path)
        if metric in frame: values[path.stem.split("_")[0]] = frame[metric].to_numpy()
    names = [name for name in ORDER if name in values]
    if not names: raise FileNotFoundError(f"No {metric!r} results for {experiment}")
    return names, [values[name] for name in names]


def draw_metric(experiment: str, metric: str) -> None:
    names, values = load_metric(experiment, metric); median, limits = PRESETS[experiment][metric]
    figure, axis = plt.subplots(figsize=(6, 4)); boxes = axis.boxplot(values, tick_labels=names, patch_artist=True)
    for patch, color in zip(boxes["boxes"], ["#0076B9", "#EC3E31", "#DB2834"]): patch.set_facecolor(color)
    axis.axhline(median, linestyle="--", color="gray"); axis.set_ylim(limits); axis.set_ylabel(metric, fontsize=18); axis.tick_params(axis="x", rotation=30); axis.grid(False); figure.tight_layout()
    directory = FIGURES_DIR / experiment; directory.mkdir(parents=True, exist_ok=True); filename = "accuracy_boxplot.png" if metric == "Accuracy" else "f1_score_boxplot.png"; output_path = directory / filename
    figure.savefig(output_path, dpi=300); plt.close(figure)
    for name, samples in zip(names, values): print(f"{name}: variance={np.var(samples):.8f}, std={np.std(samples):.4f}")
    print(f"Figure saved to {output_path}")
