import os
import sys
import numpy as np
from h5py import File
import matplotlib.pyplot as plt


def main(input_dir: str, output_file: str):
    results_path = os.path.join(input_dir, "results.h5")

    with File(results_path, "r") as f:
        cell = f["cell"][:]
        nuc = f["nuc"][:]

    n_t, rows, cols = cell.shape
    mid_col = cols // 2

    cell_mid = cell[:, :, mid_col].T
    nuc_mid = nuc[:, :, mid_col].T

    fig, ax = plt.subplots(figsize=(n_t / 50, rows / 50), dpi=100)
    ax.set_aspect("equal")

    cell_color = np.zeros((rows, n_t, 3))
    cell_color[cell_mid == 1] = [0.8, 0.4, 0.4]

    ax.imshow(cell_color, interpolation="nearest")

    nuc_overlay = np.zeros((rows, n_t, 4))
    nuc_overlay[nuc_mid == 1] = [1.0, 1.0, 0.0, 0.8]

    ax.imshow(nuc_overlay, interpolation="nearest")

    tick_interval = max(1, n_t // 10)
    tick_positions = list(range(0, n_t, tick_interval))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position (pixels)")
    ax.set_title(f"Nucleus Evolution on Cell (t=0 to {n_t - 1})")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Expected 2 arguments, found {len(sys.argv) - 1}")
        print("Usage: visualize_nuc_evolution.py <input_dir> <output_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(input_dir, output_file)
