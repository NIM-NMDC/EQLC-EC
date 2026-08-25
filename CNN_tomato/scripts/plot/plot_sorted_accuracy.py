#!/usr/bin/env python3
"""Plot tracked accuracy distributions sorted by median."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tomato_pipeline.paths import FIGURES_DIR
from tomato_pipeline.plotting import load_metric

if __name__ == "__main__":
    names, values = load_metric("Accuracy"); order = np.argsort([np.median(samples) for samples in values]); names = [names[index] for index in order]; values = [values[index] for index in order]
    figure, axis = plt.subplots(figsize=(10, 6)); axis.boxplot(values, tick_labels=names); axis.set_title("Accuracy by model, sorted by median"); axis.set_ylabel("Accuracy"); axis.tick_params(axis="x", rotation=45); axis.grid(axis="y"); figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True); output_path = FIGURES_DIR / "sorted_accuracy_boxplot.png"; figure.savefig(output_path, dpi=300); plt.close(figure); print(f"Figure saved to {output_path}")
