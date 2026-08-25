#!/usr/bin/env python3
"""Plot batch sizes and train/test roles for one MI experiment."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mi_pipeline.paths import BATCH_LABELS_DIR, DEFAULT_EXPERIMENT, EXPERIMENTS, FIGURES_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=EXPERIMENTS, default=DEFAULT_EXPERIMENT); args = parser.parse_args(); training, testing = EXPERIMENTS[args.experiment]
    counts = [len(np.load(BATCH_LABELS_DIR / f"y_{batch}.npy")) for batch in (1, 2, 3)]; roles = ["Testing Set" if batch == testing else "Training Set" for batch in (1, 2, 3)]; colors = ["#6BBC47" if role == "Testing Set" else color for role, color in zip(roles, ["#0076B927", "#EC3E3127", "#0076B927"])]
    figure, axis = plt.subplots(figsize=(8, 8)); axis.pie(counts, labels=[f"Batch {batch}\n{role}" for batch, role in zip((1, 2, 3), roles)], startangle=90, wedgeprops={"width": 0.5, "edgecolor": "w"}, colors=colors); axis.set(aspect="equal"); figure.tight_layout()
    directory = FIGURES_DIR / args.experiment; directory.mkdir(parents=True, exist_ok=True); output_path = directory / "batch_roles.png"; figure.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(figure); print(f"Figure saved to {output_path}")
