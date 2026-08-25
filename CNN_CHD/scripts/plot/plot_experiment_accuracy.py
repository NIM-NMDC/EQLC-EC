"""Plot legacy bootstrap accuracy for one train/test batch experiment."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chd_pipeline.paths import EXPERIMENTS, FIGURES_DIR, legacy_result_dir


Y_LIMITS = {
    "12_3": (0.74, 0.82),
    "12_4": (0.65, 0.80),
    "13_2": (0.68, 0.83),
    "13_4": (0.70, 0.85),
    "14_2": (0.68, 0.83),
    "14_3": (0.69, 0.84),
    "23_1": (0.74, 0.89),
    "23_4": (0.71, 0.86),
    "24_1": (0.71, 0.86),
    "24_3": (0.71, 0.86),
    "34_1": (0.69, 0.84),
    "34_2": (0.67, 0.82),
}
DESIRED_ORDER = ["RandomForest", "SVM", "XGBoost", "EQLC", "SoftVoting", "HardVoting"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="34_2")
    args = parser.parse_args()
    result_dir = legacy_result_dir(args.experiment)
    accuracy_by_name = {}
    for path in result_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        if "Accuracy" in frame:
            accuracy_by_name[path.name.split("_")[0]] = frame["Accuracy"].values
    names = [name for name in DESIRED_ORDER if name in accuracy_by_name]
    values = [accuracy_by_name[name] for name in names]
    if not values:
        raise ValueError(f"No bootstrap accuracy CSVs found in {result_dir}")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.axvspan(0.5, 3.5, facecolor="white", alpha=0.3)
    axis.axvspan(3.5, len(names) + 0.5, facecolor="lightpink", alpha=0.3)
    boxes = axis.boxplot(values, tick_labels=names, patch_artist=True)
    for patch, color in zip(boxes["boxes"], ["#0076B9", "#EC3E31", "#DB2834"]):
        patch.set_facecolor(color)
    axis.set_xlim(0.5, len(names) + 0.5)
    axis.set_ylim(*Y_LIMITS[args.experiment])
    axis.yaxis.set_major_formatter("{x:.2f}")
    axis.set_ylabel("Accuracy", fontsize=24)
    axis.grid(False)
    axis.tick_params(labelsize=16)
    plt.xticks(rotation=30)
    plt.subplots_adjust(left=0.18, right=0.99, top=0.95, bottom=0.25)
    train_batches, test_batch = args.experiment.split("_")
    figure.text(
        0.6,
        0.32,
        f"Train Set: Batch{'&'.join(train_batches)}, Test Set: Batch{test_batch}",
        ha="center",
        va="center",
        fontsize=16,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"accuracy_boxplot_{args.experiment}.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    for name, accuracies in zip(names, values):
        print(
            f"{name} Variance: {np.var(accuracies):.8f}, "
            f"Standard Deviation: {np.std(accuracies):.4f}"
        )
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
