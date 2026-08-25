"""Apply label-preserving augmentation suitable for one-dimensional spectra."""

from __future__ import annotations

import argparse

import numpy as np

from icc_pipeline.paths import SPLIT_DIR


def augment_spectra(spectra: np.ndarray, seed: int) -> np.ndarray:
    """Use small intensity scaling and noise; never flip the m/z axis."""

    rng = np.random.default_rng(seed)
    scales = rng.normal(1.0, 0.02, size=(len(spectra), 1, 1))
    feature_scale = np.std(spectra, axis=-1, keepdims=True)
    noise = rng.normal(0.0, 0.01, size=spectra.shape) * feature_scale
    return (spectra * scales + noise).astype(spectra.dtype, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    X_train = np.load(SPLIT_DIR / "X_train.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    augmented = augment_spectra(X_train, args.seed)
    np.save(SPLIT_DIR / "X_train_augmented.npy", np.concatenate([X_train, augmented]))
    np.save(SPLIT_DIR / "y_train_augmented.npy", np.concatenate([y_train, y_train]))
    print(f"Saved scientifically valid 1D augmentation to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
