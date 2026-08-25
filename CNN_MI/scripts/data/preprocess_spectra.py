#!/usr/bin/env python3
"""Convert all three raw MI batch CSVs into 204-feature NumPy tensors."""

import numpy as np
import pandas as pd
from mi_pipeline.paths import BATCH_LABELS_DIR, PROCESSED_DIR, RAW_DATA_DIR


def main() -> None:
    csv_paths = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_paths: raise FileNotFoundError(f"No batch CSV files found in {RAW_DATA_DIR}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True); BATCH_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    label_mapping = {}; rows = []; per_batch: dict[str, list[int]] = {}
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path); raw_labels = frame.iloc[:, 0].to_numpy(); spectra = frame.iloc[:, 2:].to_numpy(); assert spectra.shape[1] == 204, f"{csv_path.name} has {spectra.shape[1]} features"
        for label in np.unique(raw_labels):
            if label not in label_mapping: label_mapping[label] = len(label_mapping)
        numeric_labels = np.array([label_mapping[label] for label in raw_labels]); batch = csv_path.stem; per_batch[batch] = numeric_labels.tolist()
        for index, spectrum in enumerate(spectra, 1):
            filename = f"{batch}_spectrum_{index}.npy"; np.save(PROCESSED_DIR / filename, spectrum.reshape(1, -1)); rows.append((filename, int(numeric_labels[index - 1]), batch))
    np.save(PROCESSED_DIR / "labels.npy", rows)
    for batch, labels in per_batch.items(): np.save(BATCH_LABELS_DIR / f"y_{batch}.npy", np.asarray(labels))
    print(f"Saved {len(rows)} spectra from {len(csv_paths)} batches to {PROCESSED_DIR}")
    print(f"Label mapping: {label_mapping}")


if __name__ == "__main__": main()
