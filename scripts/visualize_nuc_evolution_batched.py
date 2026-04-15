import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from h5py import File


def progress_bar(current: int, total: int, prefix: str = "", bar_len: int = 40):
    filled = int(bar_len * current / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    pct = f"{100 * current / total:.1f}%"
    sys.stdout.write(f"\r{prefix} [{bar}] {pct} ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        print()


def main(input_dir: str, output_file: str, batch_size: int = 100):
    results_path = os.path.join(input_dir, "results.h5")

    with File(results_path, "r") as f:
        n_t = f["cell"].shape[0]
        rows = f["cell"].shape[1]
        cols = f["cell"].shape[2]

    mid_col = cols // 2

    cell_mid = np.zeros((rows, n_t), dtype=np.int8)
    nuc_mid = np.zeros((rows, n_t), dtype=np.int8)

    print(f"Processing {n_t} frames (batch_size={batch_size})...")

    with File(results_path, "r") as f:
        for start in range(0, n_t, batch_size):
            end = min(start + batch_size, n_t)
            cell_batch = f["cell"][start:end, :, mid_col].astype(np.int8)
            nuc_batch = f["nuc"][start:end, :, mid_col].astype(np.int8)
            cell_mid[:, start:end] = cell_batch.T
            nuc_mid[:, start:end] = nuc_batch.T
            progress_bar(end, n_t, prefix="  Loading")

    fig_width = 8
    fig_height = 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.set_aspect("equal")

    cell_color = np.zeros((rows, n_t, 3))
    cell_color[cell_mid == 1] = [0.8, 0.4, 0.4]

    ax.imshow(cell_color, interpolation="nearest")

    nuc_overlay = np.zeros((rows, n_t, 4))
    nuc_overlay[nuc_mid == 1] = [1.0, 1.0, 0.0, 0.8]

    ax.imshow(nuc_overlay, interpolation="nearest")

    with File(results_path, "r") as f:
        t = f["t"][:]

    font_size = min(12, max(6, 120 // n_t))
    tick_interval = max(1, n_t // 10)
    tick_indices = list(range(0, n_t, tick_interval))
    tick_labels = [f"{t[i]:.1f}" for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, fontsize=font_size, rotation=45)
    ax.set_xlabel("Time (t)", fontsize=font_size)
    ax.set_ylabel("Position (pixels)", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Expected at least 2 arguments, found {len(sys.argv) - 1}")
        print(
            "Usage: visualize_nuc_evolution_chunked.py <input_dir> <output_file> [batch_size]"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    main(input_dir, output_file, batch_size)
