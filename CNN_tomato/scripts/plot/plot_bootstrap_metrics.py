#!/usr/bin/env python3
"""Plot tracked tomato bootstrap metrics."""
import argparse
from tomato_pipeline.plotting import draw_metric

METRICS = {"accuracy": ("Accuracy", "accuracy_boxplot.png"), "precision": ("Precision", "precision_boxplot.png"), "recall": ("Recall", "recall_boxplot.png"), "f1": ("F1 Score", "f1_score_boxplot.png"), "auc": ("AUC", "auc_boxplot.png")}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--metric", choices=["all", *METRICS], default="all"); selected = parser.parse_args().metric
    for metric, filename in (METRICS.values() if selected == "all" else [METRICS[selected]]): draw_metric(metric, filename)
