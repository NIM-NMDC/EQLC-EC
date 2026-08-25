"""Plot the complete CHD label distribution and the published test subset counts."""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chd_pipeline.paths import FIGURES_DIR, PROCESSED_DIR


def main() -> None:
    labels = np.load(PROCESSED_DIR / "labels.npy", allow_pickle=True)[:, 1]
    counts = Counter(labels)
    count_values = list(map(int, counts.values()))
    total_samples = sum(count_values)

    figure, axes = plt.subplots(figsize=(12, 6), ncols=2)
    axes[0].pie(
        count_values,
        labels=count_values,
        startangle=90,
        wedgeprops={"width": 0.3, "edgecolor": "w"},
        textprops={"fontsize": 48},
        colors=["#0076B9", "#EC3E31"],
    )
    axes[0].set(aspect="equal", title="Complete Dataset")
    axes[0].text(0, 0, f"N={total_samples}", ha="center", va="center", fontsize=48)
    axes[0].title.set_fontsize(36)

    # Preserve the manually specified counts used in the original figure script.
    subset_counts = [240, 230]
    axes[1].pie(
        subset_counts,
        labels=subset_counts,
        startangle=90,
        wedgeprops={"width": 0.15, "edgecolor": "w"},
        textprops={"fontsize": 48},
        colors=["#0076B9", "#EC3E31"],
        radius=0.6,
    )
    axes[1].set(aspect="equal", title="Independent Testing Dataset")
    axes[1].text(0, 0, f"N={sum(subset_counts)}", ha="center", va="center", fontsize=48)
    axes[1].title.set_fontsize(36)

    handles, _ = axes[0].get_legend_handles_labels()
    class_names = ["Control", "CHD"]
    axes[0].legend(handles, class_names, loc="upper right", fontsize=24, bbox_to_anchor=(1.2, 1))
    axes[1].legend(handles, class_names, loc="upper right", fontsize=24, bbox_to_anchor=(1.2, 1))
    figure.text(0.5, 0.12, "Number of features per sample:", ha="center", fontsize=48)
    figure.text(0.83, 0.12, "199", ha="center", fontsize=48, color="red")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "dataset_distribution.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
