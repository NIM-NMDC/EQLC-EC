#!/usr/bin/env python3
"""Plot complete and nominal 20% test-set class counts."""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hip_cer_pipeline.paths import FIGURES_DIR, PROCESSED_DIR


def main() -> None:
    labels = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)[:, 1]
    counts = list(map(int, Counter(labels).values()))
    subsets = [int(count * 0.2) for count in counts]
    figure, axes = plt.subplots(figsize=(12, 6), ncols=2)
    for axis, values, title, radius in (
        (axes[0], counts, "Complete Dataset", 1.0),
        (axes[1], subsets, "Independent Testing Dataset", 0.6),
    ):
        axis.pie(
            values,
            labels=values,
            startangle=90,
            wedgeprops={"width": 0.3 if radius == 1.0 else 0.15, "edgecolor": "w"},
            colors=["#0076B9", "#EC3E31"],
            radius=radius,
        )
        axis.text(0, 0, f"N={sum(values)}", ha="center", va="center")
        axis.set(aspect="equal", title=title)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "class_distribution.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
