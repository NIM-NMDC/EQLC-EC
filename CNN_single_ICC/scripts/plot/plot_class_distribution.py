"""Plot actual complete-data and test-split class counts."""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from icc_pipeline.paths import FIGURES_DIR, PROCESSED_DIR, SPLIT_DIR


def main() -> None:
    labels = np.load(PROCESSED_DIR / "labels.npy")[:, 1].astype(int)
    test_labels = np.load(SPLIT_DIR / "y_test.npy").astype(int)
    complete_counts = Counter(labels)
    test_counts = Counter(test_labels)
    class_names = ["Astrocytes", "Neurons"]
    figure, axes = plt.subplots(figsize=(12, 6), ncols=2)
    for axis, counts, title in (
        (axes[0], complete_counts, "Complete Dataset"),
        (axes[1], test_counts, "Independent Testing Dataset"),
    ):
        values = [counts[index] for index in range(2)]
        axis.pie(values, labels=values, startangle=90, wedgeprops={"width": 0.3})
        axis.set_title(f"{title}\nN={sum(values)}")
        axis.legend(class_names)
    figure.suptitle("511 features per sample")
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "sample_numbers.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
