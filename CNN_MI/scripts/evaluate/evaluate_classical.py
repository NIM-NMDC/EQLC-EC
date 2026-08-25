#!/usr/bin/env python3
"""Evaluate one copied classical baseline on an MI batch experiment."""
import argparse
from mi_pipeline.classical import evaluate_estimator
from mi_pipeline.paths import DEFAULT_EXPERIMENT, EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("model", choices=["RandomForest", "SVM", "AdaBoost", "XGBoost", "LightGBM"]); parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT); parser.add_argument("--smoke-test", action="store_true"); args = parser.parse_args(); evaluate_estimator(args.model, args.experiment, smoke_test=args.smoke_test)
