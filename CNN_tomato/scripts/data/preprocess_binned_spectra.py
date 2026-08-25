#!/usr/bin/env python3
"""Create the original 256-bin alternative tomato representation."""

import argparse

import numpy as np
import pandas as pd

from tomato_pipeline.paths import BINNED_DIR, RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RAW_DATA_DIR / "raw.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, header=None)
    filenames = frame.iloc[0, 1:].str.replace(".RAW", "", regex=False) + ".npy"
    label_names = frame.iloc[1, 1:]
    label_mapping = {label: index for index, label in enumerate(label_names.unique())}
    labels = label_names.map(label_mapping)
    mz_values = frame.iloc[2:, 0].astype(float).to_numpy()
    intensities = frame.iloc[2:, 1:].astype(float).to_numpy()
    bins = np.arange(150, 150 + 3 * 256, 3)
    digitized = np.digitize(mz_values, bins) - 1
    rows = []
    BINNED_DIR.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(filenames):
        binned = np.zeros_like(bins, dtype=float)
        for bin_index, intensity in zip(digitized, intensities[:, index]):
            if 0 <= bin_index < len(binned):
                binned[bin_index] += intensity
        np.save(BINNED_DIR / filename, binned)
        rows.append((filename, int(labels.iat[index])))
    np.save(BINNED_DIR / "labels.npy", np.array(rows, dtype=object))
    (BINNED_DIR / "label_mapping.txt").write_text(
        "".join(f"{label}: {index}\n" for label, index in label_mapping.items()), encoding="utf-8"
    )
    print(f"Saved {len(rows)} binned spectra to {BINNED_DIR}")


if __name__ == "__main__":
    main()
