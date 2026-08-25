"""Summarize legacy model accuracy across all 12 batch experiments."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chd_pipeline.paths import EXPERIMENTS, FIGURES_DIR, legacy_result_dir


def main() -> None:
    data: dict[str, list[float]] = {}
    for experiment in EXPERIMENTS:
        matching = list(legacy_result_dir(experiment).glob("ModelPerformance*.csv"))
        if len(matching) != 1:
            raise RuntimeError(f"Expected one ModelPerformance CSV for {experiment}")
        frame = pd.read_csv(matching[0])
        for _, row in frame.iterrows():
            model_name = str(row.iloc[0])
            if "_" in model_name:
                model_name = "_".join(model_name.split("_")[:-1])
            data.setdefault(model_name, []).append(float(row.iloc[1]))

    statistics = {
        name: (sum(values) / len(values), np.std(values)) for name, values in data.items()
    }
    for name, (average, standard_deviation) in statistics.items():
        print(
            f"{name}: Average Accuracy = {average:.8f}, "
            f"Std Deviation = {standard_deviation:.8f}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    statistics_path = FIGURES_DIR / "model_performance_statistics.csv"
    statistics_frame = pd.DataFrame(statistics).T
    statistics_frame.columns = ["Average Accuracy", "Std Deviation"]
    statistics_frame.index.name = "Model Name"
    statistics_frame.to_csv(statistics_path)

    figure, axis = plt.subplots(figsize=(10, 4))
    for model_name, accuracies in data.items():
        marker = "s" if model_name == "ComBat" else "o"
        axis.plot(EXPERIMENTS, accuracies, marker=marker, label=model_name)
    axis.set_title("Model Accuracy", fontsize=18)
    axis.set_xlabel(
        "12_3 means Batch 1 and Batch 2 are training data; Batch 3 is test data",
        fontsize=18,
        color="red",
    )
    axis.set_ylabel("Accuracy", fontsize=18)
    axis.tick_params(axis="x", rotation=36, labelsize=16)
    axis.tick_params(axis="y", labelsize=16)
    axis.legend(title="Model Name", fontsize=12, title_fontsize=14)
    axis.grid(False)
    figure.tight_layout()
    output_path = FIGURES_DIR / "cross_batch_model_accuracy.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Statistics saved to {statistics_path}")
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
