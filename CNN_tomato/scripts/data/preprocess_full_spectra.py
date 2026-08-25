#!/usr/bin/env python3
"""Extract all 16,227 intensity features from the raw tomato CSV."""

import argparse
import logging

import numpy as np
import pandas as pd

from tomato_pipeline.paths import PROCESSED_DIR, RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RAW_DATA_DIR / "raw.csv"))
    args = parser.parse_args()
    input_path = args.input
    frame = pd.read_csv(input_path, header=None)
    filenames = frame.iloc[0, 1:].str.replace(".RAW", "", regex=False) + ".npy"
    label_names = frame.iloc[1, 1:]
    label_mapping = {label: index for index, label in enumerate(label_names.unique())}
    labels = label_names.map(label_mapping)
    intensities = frame.iloc[2:, 1:].astype(float)
    rows = []
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(filenames):
        np.save(PROCESSED_DIR / filename, intensities.iloc[:, index].to_numpy())
        sample_id = "_".join(filename.split("_")[-3:-1])
        rows.append((filename, int(labels.iat[index]), sample_id))
    np.save(PROCESSED_DIR / "labels.npy", np.array(rows, dtype=object))
    (PROCESSED_DIR / "label_mapping.txt").write_text(
        "".join(f"{label}: {index}\n" for label, index in label_mapping.items()),
        encoding="utf-8",
    )
    logging.info("Saved %d full spectra to %s", len(rows), PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
