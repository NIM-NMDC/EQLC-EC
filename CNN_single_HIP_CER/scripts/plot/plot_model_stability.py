#!/usr/bin/env python3
"""Plot standard deviations recorded in the legacy workbook."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hip_cer_pipeline.paths import FIGURES_DIR, LEGACY_SUMMARY_DIR


def main() -> None:
    frame = pd.read_excel(LEGACY_SUMMARY_DIR / "standard_deviation.xlsx").set_index("models")
    color_map = {
        "HardVoting": "#EC3E31",
        "SoftVoting": "#EC3E31",
        "EQLC": "#A6D0E6",
        "XGBoost": "#4F99C9",
        "AdaBoost": "#A6D0E6",
        "SVM": "#A6D0E6",
        "RandomForest": "#A6D0E6",
    }
    colors = [color_map.get(model, "gray") for model in frame.index]
    limits = {
        "Accuracy": (0.0, 0.02),
        "Precision": (0.0, 0.016),
        "Recall": (0.0, 0.031),
        "F1_Score": (0.0, 0.02),
    }
    figure = plt.figure(figsize=(10, 8))
    for index, metric in enumerate(limits, start=1):
        axis = figure.add_subplot(2, 2, index)
        axis.barh(frame.index, frame[metric], color=colors)
        axis.set_title(metric)
        axis.set_xlabel("Standard Deviation")
        axis.set_ylabel("Models")
        axis.set_xlim(limits[metric])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "model_stability_metrics.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
