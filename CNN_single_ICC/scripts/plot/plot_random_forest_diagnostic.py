"""Small diagnostic plot for the tracked random-forest bootstrap result."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from icc_pipeline.paths import FIGURES_DIR, LEGACY_RESULTS_DIR


def main() -> None:
    candidates = sorted(LEGACY_RESULTS_DIR.glob("RandomForest_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No random-forest result found in {LEGACY_RESULTS_DIR}")
    data = pd.read_csv(candidates[-1])
    print(data.head())
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.boxplot(data["Accuracy"])
    axis.set_title("RandomForest bootstrap accuracy")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "random_forest_accuracy_diagnostic.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
