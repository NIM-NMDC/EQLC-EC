"""Draw the tracked standard-deviation summary without a GUI dependency."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from icc_pipeline.paths import FIGURES_DIR, LEGACY_SUMMARY_DIR


def main() -> None:
    data = pd.read_excel(LEGACY_SUMMARY_DIR / "standard_deviation.xlsx").set_index("models")
    color_map = {
        "HardVoting": "#EC3E31",
        "SoftVoting": "#EC3E31",
        "EQLC": "#A6D0E6",
        "XGBoost": "#4F99C9",
    }
    colors = [color_map.get(model, "#A6D0E6") for model in data.index]
    figure = plt.figure(figsize=(10, 8))
    for index, metric in enumerate(["Accuracy", "Precision", "Recall", "F1_Score"], start=1):
        axis = figure.add_subplot(2, 2, index)
        axis.barh(data.index, data[metric], color=colors)
        axis.set_title(metric)
        axis.set_xlabel("Standard deviation")
        axis.grid(False)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "model_stability_metrics.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
