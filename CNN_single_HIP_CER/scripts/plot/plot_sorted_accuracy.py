#!/usr/bin/env python3
"""Plot accuracy distributions sorted by their median."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hip_cer_pipeline.paths import FIGURES_DIR
from hip_cer_pipeline.plotting import load_metric


def main() -> None:
    names, values = load_metric("Accuracy")
    order = np.argsort([np.median(samples) for samples in values])
    ordered_names = [names[index] for index in order]
    ordered_values = [values[index] for index in order]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.boxplot(ordered_values, tick_labels=ordered_names)
    axis.set_title("Accuracy by model, sorted by median")
    axis.set_ylabel("Accuracy")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y")
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "sorted_accuracy_boxplot.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
