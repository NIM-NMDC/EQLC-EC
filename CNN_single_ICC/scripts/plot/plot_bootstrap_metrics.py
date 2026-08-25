"""Create box plots for one or all recorded bootstrap metrics."""

from __future__ import annotations

import argparse

from icc_pipeline.plotting import draw_metric


METRICS = {
    "accuracy": ("Accuracy", "accuracy_boxplot.png"),
    "precision": ("Precision", "precision_boxplot.png"),
    "recall": ("Recall", "recall_boxplot.png"),
    "f1": ("F1 Score", "f1_score_boxplot.png"),
    "auc": ("AUC", "auc_boxplot.png"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=["all", *METRICS],
        default="all",
        help="Metric to plot; the default creates all five plots.",
    )
    args = parser.parse_args()
    selected = METRICS.values() if args.metric == "all" else [METRICS[args.metric]]
    for metric, output_name in selected:
        draw_metric(metric, output_name)


if __name__ == "__main__":
    main()
