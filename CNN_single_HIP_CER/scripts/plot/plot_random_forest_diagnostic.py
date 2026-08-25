#!/usr/bin/env python3
"""Plot the tracked HIP/CER random-forest bootstrap accuracy."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hip_cer_pipeline.paths import FIGURES_DIR, LEGACY_RESULTS_DIR


def main() -> None:
    paths = sorted(LEGACY_RESULTS_DIR.glob("RandomForest_*.csv"))
    if not paths:
        raise FileNotFoundError("No tracked random-forest result was found")
    frame = pd.read_csv(paths[-1])
    print(frame.head().to_string(index=False))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.boxplot(frame["Accuracy"])
    axis.set_title("Random-forest bootstrap accuracy")
    axis.set_ylabel("Accuracy")
    axis.grid(axis="x")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "random_forest_accuracy_diagnostic.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
