#!/usr/bin/env python3
"""Plot accuracy and/or F1 for one MI batch experiment."""
import argparse
from mi_pipeline.paths import DEFAULT_EXPERIMENT, EXPERIMENTS
from mi_pipeline.plotting import draw_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT); parser.add_argument("--metric", choices=["all", "accuracy", "f1"], default="all"); args = parser.parse_args()
    if args.metric in ("all", "accuracy"): draw_metric(args.experiment, "Accuracy")
    if args.metric in ("all", "f1"): draw_metric(args.experiment, "F1 Score")
