#!/usr/bin/env python3
"""Evaluate the published XGBoost baseline."""

import argparse
from hip_cer_pipeline.classical import evaluate_estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    evaluate_estimator("XGBoost", smoke_test=args.smoke_test)
