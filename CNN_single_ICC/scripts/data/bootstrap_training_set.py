"""Generate deterministic bootstrap copies of the ICC training split."""

from __future__ import annotations

import argparse

import numpy as np

from icc_pipeline.paths import SPLIT_DIR


def bootstrap_sample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=len(X), replace=True)
    return X[indices], y[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    count = 1 if args.smoke_test else args.count

    X_train = np.load(SPLIT_DIR / "X_train.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    output_dir = SPLIT_DIR / "bootstrap_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(42, 42 + count):
        X_sample, y_sample = bootstrap_sample(X_train, y_train, seed)
        np.save(output_dir / f"X_train_bootstrap_{seed}.npy", X_sample)
        np.save(output_dir / f"y_train_bootstrap_{seed}.npy", y_sample)
        print(f"Saved bootstrap seed {seed} to {output_dir}")


if __name__ == "__main__":
    main()
