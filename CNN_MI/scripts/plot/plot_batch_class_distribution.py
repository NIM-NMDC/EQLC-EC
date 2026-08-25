#!/usr/bin/env python3
"""Plot the class distribution of one raw MI batch."""
import argparse
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mi_pipeline.paths import BATCH_LABELS_DIR, FIGURES_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--batch", type=int, choices=[1, 2, 3], required=True); args = parser.parse_args(); counts = Counter(np.load(BATCH_LABELS_DIR / f"y_{args.batch}.npy"))
    figure, axis = plt.subplots(figsize=(6, 6)); axis.pie(list(counts.values()), labels=[f"Class {key}: {value}" for key, value in counts.items()], startangle=90, wedgeprops={"width": 0.35, "edgecolor": "w"}); axis.set(aspect="equal", title=f"Batch {args.batch} class distribution")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True); output_path = FIGURES_DIR / f"batch_{args.batch}_class_distribution.png"; figure.savefig(output_path, dpi=300); plt.close(figure); print(f"Figure saved to {output_path}")
