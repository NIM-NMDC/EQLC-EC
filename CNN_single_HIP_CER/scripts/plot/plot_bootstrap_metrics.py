#!/usr/bin/env python3
"""Plot boxplots for the five recorded bootstrap metrics."""

import argparse

from hip_cer_pipeline.plotting import draw_metric


METRICS = {
    "accuracy": ("Accuracy", "accuracy_boxplot.png"),
    "precision": ("Precision", "precision_boxplot.png"),
    "recall": ("Recall", "recall_boxplot.png"),
    "f1": ("F1 Score", "f1_score_boxplot.png"),
    "auc": ("AUC", "auc_boxplot.png"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["all", *METRICS], default="all")
    args = parser.parse_args()
    selected = METRICS if args.metric == "all" else {args.metric: METRICS[args.metric]}
    for metric, output_name in selected.values():
        draw_metric(metric, output_name)


if __name__ == "__main__":
    main()
