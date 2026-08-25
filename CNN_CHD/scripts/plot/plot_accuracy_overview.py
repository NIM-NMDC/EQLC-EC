"""Plot experiment CSV accuracy distributions sorted by median."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chd_pipeline.paths import EXPERIMENTS, FIGURES_DIR, legacy_result_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="34_2")
    args = parser.parse_args()
    accuracy_data, algorithm_names = [], []
    for path in legacy_result_dir(args.experiment).glob("*.csv"):
        frame = pd.read_csv(path)
        if "Accuracy" in frame:
            accuracy_data.append(frame["Accuracy"].values)
            algorithm_names.append(path.name.split("_")[0])
    if not accuracy_data:
        raise ValueError(f"No accuracy data found for {args.experiment}")
    order = np.argsort([np.median(values) for values in accuracy_data])
    sorted_data = [accuracy_data[index] for index in order]
    sorted_names = [algorithm_names[index] for index in order]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.boxplot(sorted_data, tick_labels=sorted_names)
    axis.set_title("Boxplot of Accuracy across Models (Sorted by Median)")
    axis.set_ylabel("Accuracy")
    axis.set_xlabel("Models")
    axis.grid(axis="y")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"accuracy_overview_{args.experiment}.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    for name, values in zip(sorted_names, sorted_data):
        print(f"{name} Variance: {np.var(values):.8f}")
        print(f"{name} Standard Deviation: {np.std(values):.4f}")
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
