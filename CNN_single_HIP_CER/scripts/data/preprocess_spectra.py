#!/usr/bin/env python3
"""Convert the source HIP/CER CSV into one NumPy tensor per spectrum."""

import numpy as np
import pandas as pd

from hip_cer_pipeline.paths import PROCESSED_DIR, RAW_DATA_DIR


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    label_mapping: dict[object, int] = {}
    all_labels: list[tuple[str, int]] = []

    csv_paths = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        raw_labels = frame.iloc[:, -1].to_numpy()
        spectra = frame.iloc[:, :-1].to_numpy()
        assert spectra.shape == (1201, 2463), (
            f"Unexpected data shape for {csv_path.name}: {spectra.shape}"
        )

        for label in np.unique(raw_labels):
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
        numeric_labels = np.array([label_mapping[label] for label in raw_labels])

        for index, spectrum in enumerate(spectra, start=1):
            filename = f"{csv_path.stem}_spectrum_{index}.npy"
            np.save(PROCESSED_DIR / filename, spectrum.reshape(1, -1))
            all_labels.append((filename, int(numeric_labels[index - 1])))

    np.save(PROCESSED_DIR / "labels.npy", all_labels)
    print(f"Saved {len(all_labels)} spectra to {PROCESSED_DIR}")
    print("Label mapping:")
    for label, numeric in label_mapping.items():
        print(f"  {label!r} -> {numeric}")


if __name__ == "__main__":
    main()
