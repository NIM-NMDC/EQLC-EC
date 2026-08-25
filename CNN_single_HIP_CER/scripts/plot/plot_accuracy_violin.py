#!/usr/bin/env python3
"""Plot the legacy accuracy distributions as violin plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hip_cer_pipeline.paths import FIGURES_DIR
from hip_cer_pipeline.plotting import load_metric


def main() -> None:
    names, values = load_metric("Accuracy")
    figure, axis = plt.subplots(figsize=(10, 6))
    violins = axis.violinplot(values, showmeans=False, showextrema=False, showmedians=True, widths=0.5)
    colors = ["skyblue", "lightgreen", "whitesmoke", "lightpink", "seagreen", "dodgerblue"]
    for violin, color in zip(violins["bodies"], colors):
        violin.set_facecolor(color)
        violin.set_alpha(0.6)
    axis.set_xticks(range(1, len(names) + 1), names, rotation=30)
    axis.set_ylabel("Accuracy")
    axis.grid(False)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "accuracy_violinplot.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    for name, samples in zip(names, values):
        print(f"{name}: variance={np.var(samples):.8f}, std={np.std(samples):.4f}")
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
