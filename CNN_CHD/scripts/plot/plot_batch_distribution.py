"""Plot the label distribution for one of the four acquisition batches."""

from __future__ import annotations

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chd_pipeline.paths import BATCH_LABELS_DIR, FIGURES_DIR


COLORS = {
    1: ["#0076B954", "#0076B9"],
    2: ["#EC3E3154", "#EC3E31"],
    3: ["#6BBC4754", "#6BBC47"],
    4: ["#FCC41E54", "#FCC41E"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    labels = np.load(BATCH_LABELS_DIR / f"y_{args.batch}.npy")
    label_counts = Counter(labels)
    counts = [label_counts.get(0, 0), label_counts.get(1, 0)]
    total = sum(counts)
    percentages = [f"{count / total * 100:.1f}%" for count in counts]

    figure, axis = plt.subplots(figsize=(8, 8))
    wedges, _ = axis.pie(
        counts,
        startangle=90,
        wedgeprops={"width": 0.3, "edgecolor": "w"},
        textprops={"fontsize": 24},
        colors=COLORS[args.batch],
    )
    for index, wedge in enumerate(wedges):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        axis.text(x * 1.4, y * 0.5, str(counts[index]), ha="center", va="center", fontsize=36)
    legend_labels = [
        f"{name} : {percentage}"
        for name, percentage in zip(["Control", "CHD"], percentages)
    ]
    axis.legend(
        wedges,
        legend_labels,
        loc="upper center",
        fontsize=36,
        bbox_to_anchor=(0.5, 0.0),
        handletextpad=2,
    )
    axis.set(aspect="equal", title=f"Batch {args.batch}")
    axis.text(0, 0, f"N={total}", ha="center", va="center", fontsize=36)
    axis.title.set_fontsize(48)
    figure.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"batch_{args.batch}_distribution.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
