#!/usr/bin/env python3
"""Compare paper, hard-voting, and soft-voting results across MI splits."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mi_pipeline.paths import FIGURES_DIR, LEGACY_SUMMARY_DIR


def draw_metric(workbook: pd.ExcelFile, metric: str, limits: tuple[float, float]) -> None:
    sheets = workbook.sheet_names
    frames = {sheet: pd.read_excel(workbook, sheet_name=sheet) for sheet in sheets}
    labels = frames[sheets[0]]["distribution"].astype(str).str.replace(" ", "\n").to_list()
    values = {sheet: frames[sheet][metric].to_numpy() for sheet in sheets}
    colors = ["#6BBC47", "#EC3E31", "#0076B9"]
    x = np.arange(len(sheets))
    width = 0.22
    figure, axis = plt.subplots(figsize=(12, 6))
    for label_index, label in enumerate(labels):
        axis.bar(
            x + (label_index - 1) * width,
            [values[sheet][label_index] for sheet in sheets],
            width,
            color=colors[label_index],
            label=label,
        )
    axis.axhline(np.mean(values[sheets[0]]), color="gray", linestyle="--", label="Average of paper")
    axis.scatter(x[1], np.mean(values[sheets[1]]), color="black", marker="o", s=80, label="Average of HardVoting")
    axis.scatter(x[2], np.mean(values[sheets[2]]), color="black", marker="X", s=80, label="Average of SoftVoting")
    axis.set_xticks(x, sheets)
    axis.set_ylim(limits)
    axis.set_xlabel("Source of Model")
    axis.set_ylabel("F1 Score" if metric == "F1_Score" else metric)
    axis.legend(fontsize=9)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"{metric.lower()}_comparison.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    workbook = pd.ExcelFile(LEGACY_SUMMARY_DIR / "standard_deviation.xlsx")
    draw_metric(workbook, "Accuracy", (0.75, 0.90))
    draw_metric(workbook, "F1_Score", (0.80, 0.90))
