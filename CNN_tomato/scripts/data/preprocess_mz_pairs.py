#!/usr/bin/env python3
"""Create the original selected-m/z-pair alternative representation."""

import argparse

import numpy as np
import pandas as pd

from tomato_pipeline.paths import MZ_PAIR_DIR, RAW_DATA_DIR


MZ_PAIRS = [
    (151.0478, 185.0420), (151.0478, 210.0760), (151.0478, 230.9900), (151.0478, 257.0740), (151.0478, 311.1160),
    (159.0628, 230.9900), (159.0628, 237.0748), (159.0628, 249.0380), (159.0628, 257.0740), (159.0628, 287.1370),
    (160.0756, 230.9900), (160.0756, 237.0748), (160.0756, 249.0370), (160.0756, 251.1622), (160.0756, 406.1320),
    (189.1597, 210.0760), (189.1597, 237.0748), (189.1602, 311.1160), (192.0767, 287.1370), (210.0760, 257.0740),
    (230.9900, 237.0748), (230.9900, 406.1320), (230.9910, 251.1622), (237.0748, 311.1160), (249.0370, 311.1160),
    (249.0380, 257.0740), (251.1622, 311.1160), (287.1370, 406.1320),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RAW_DATA_DIR / "raw.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, header=None, low_memory=False)
    frame.iloc[2:, 0] = pd.to_numeric(frame.iloc[2:, 0], errors="coerce")
    filenames = frame.iloc[0, 1:].str.replace(".RAW", "", regex=False) + ".npy"
    label_names = frame.iloc[1, 1:]
    label_mapping = {label: index for index, label in enumerate(label_names.unique())}
    labels = label_names.map(label_mapping)
    intensities = frame.iloc[2:, 1:].astype(float)
    rows = []
    MZ_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    for sample_index, filename in enumerate(filenames):
        extracted = []
        for mz1, mz2 in MZ_PAIRS:
            mz1_index = frame.iloc[2:, 0].sub(mz1).abs().idxmin()
            mz2_index = frame.iloc[2:, 0].sub(mz2).abs().idxmin()
            # This positional lookup intentionally retains the original script's behavior.
            extracted.append(intensities.iloc[mz1_index, sample_index] + intensities.iloc[mz2_index, sample_index])
        np.save(MZ_PAIR_DIR / filename, np.asarray(extracted))
        sample_id = "_".join(filename.split("_")[-3:-1])
        rows.append((filename, int(labels.iat[sample_index]), sample_id))
    np.save(MZ_PAIR_DIR / "labels.npy", np.array(rows, dtype=object))
    (MZ_PAIR_DIR / "label_mapping.txt").write_text(
        "".join(f"{label}: {index}\n" for label, index in label_mapping.items()), encoding="utf-8"
    )
    print(f"Saved {len(rows)} m/z-pair spectra to {MZ_PAIR_DIR}")


if __name__ == "__main__":
    main()
