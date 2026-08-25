#!/usr/bin/env python3
"""Evaluate the random-forest baseline on tomato data."""
import argparse
from tomato_pipeline.classical import evaluate_estimator
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke-test", action="store_true")
    evaluate_estimator("RandomForest", smoke_test=parser.parse_args().smoke_test)
