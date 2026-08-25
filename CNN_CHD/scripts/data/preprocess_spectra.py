"""Convert the four raw CHD batch CSVs into one NumPy file per spectrum."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from chd_pipeline.paths import PROCESSED_DIR, RAW_DATA_DIR


EXPECTED_FEATURES = 199


def main() -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw CSV files are not included. Place them in {RAW_DATA_DIR} before preprocessing."
        )
    csv_files = [name for name in os.listdir(RAW_DATA_DIR) if name.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    label_mapping: dict[object, int] = {}
    all_labels: list[tuple[str, int, str]] = []
    for csv_file in csv_files:
        frame = pd.read_csv(RAW_DATA_DIR / csv_file)
        raw_labels = frame.iloc[:, 0].values
        data = frame.iloc[:, 2:].values
        if data.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"{csv_file}: expected {EXPECTED_FEATURES} features, got {data.shape[1]}"
            )

        for label in np.unique(raw_labels):
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
        numeric_labels = np.array([label_mapping[label] for label in raw_labels])
        batch_name = os.path.splitext(csv_file)[0]

        for index, spectrum in enumerate(data, start=1):
            file_name = f"{batch_name}_spectrum_{index}.npy"
            np.save(PROCESSED_DIR / file_name, spectrum.reshape(1, -1))
            all_labels.append((file_name, int(numeric_labels[index - 1]), batch_name))

    np.save(PROCESSED_DIR / "labels.npy", all_labels)
    print(f"Saved {len(all_labels)} spectra to {PROCESSED_DIR}")
    print("Label mapping:")
    for label, numeric in label_mapping.items():
        print(f"  {label!r} -> {numeric}")


if __name__ == "__main__":
    main()
