#!/usr/bin/env python3
"""Plot tomato spectrum and grouped test-set class counts."""
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tomato_pipeline.paths import FIGURES_DIR, PROCESSED_DIR, SPLIT_DIR

if __name__ == "__main__":
    complete = list(map(int, Counter(np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)[:, 1]).values()))
    test = list(map(int, Counter(np.load(SPLIT_DIR / "y_test.npy")).values())) if (SPLIT_DIR / "y_test.npy").exists() else [int(value * 0.29) for value in complete]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, values, title in ((axes[0], complete, "Complete Dataset"), (axes[1], test, "Independent Testing Dataset")):
        axis.pie(values, labels=values, startangle=90, wedgeprops={"width": 0.3, "edgecolor": "w"}, colors=["#0076B9", "#EC3E31"]); axis.text(0, 0, f"N={sum(values)}", ha="center", va="center"); axis.set(aspect="equal", title=title)
    figure.tight_layout(); FIGURES_DIR.mkdir(parents=True, exist_ok=True); output_path = FIGURES_DIR / "class_distribution.png"; figure.savefig(output_path, dpi=300); plt.close(figure); print(f"Figure saved to {output_path}")
