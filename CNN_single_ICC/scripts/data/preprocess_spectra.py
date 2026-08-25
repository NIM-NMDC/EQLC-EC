"""Convert the ICC CSV into one NumPy spectrum per row."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from icc_pipeline.paths import PROCESSED_DIR, RAW_DATA_DIR


EXPECTED_SAMPLES = 1544
EXPECTED_FEATURES = 511


def main() -> None:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(path) for path in csv_files]
    raw_labels = np.concatenate([frame.iloc[:, -1].astype(str).to_numpy() for frame in frames])
    label_names = sorted(np.unique(raw_labels).tolist())
    label_mapping = {label: index for index, label in enumerate(label_names)}
    all_labels: list[tuple[str, int]] = []

    total = 0
    for csv_path, frame in zip(csv_files, frames):
        data = frame.iloc[:, :-1].to_numpy()
        labels = frame.iloc[:, -1].astype(str).to_numpy()
        if data.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"{csv_path.name}: expected {EXPECTED_FEATURES} features, got {data.shape[1]}"
            )
        base_name = csv_path.stem
        for index, spectrum in enumerate(data, start=1):
            file_name = f"{base_name}_spectrum_{index}.npy"
            np.save(PROCESSED_DIR / file_name, spectrum.reshape(1, -1))
            all_labels.append((file_name, label_mapping[labels[index - 1]]))
        total += len(data)

    if total != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} samples, got {total}")
    np.save(PROCESSED_DIR / "labels.npy", np.asarray(all_labels))
    (PROCESSED_DIR / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {total} spectra with {EXPECTED_FEATURES} features to {PROCESSED_DIR}")
    print(f"Label mapping: {label_mapping}")


if __name__ == "__main__":
    main()
