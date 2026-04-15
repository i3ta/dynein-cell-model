import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.patches import Patch


def progress_bar(current: int, total: int, prefix: str = "", bar_len: int = 40):
    filled = int(bar_len * current / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    pct = f"{100 * current / total:.1f}%"
    sys.stdout.write(f"\r{prefix} [{bar}] {pct} ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        print()


def main(dir: str, output: str, batch_size: int = 100):
    results_path = os.path.join(dir, "results.h5")
    env_path = os.path.join(dir, "env.png")

    with File(results_path, "r") as f:
        t_len = f["cell"].shape[0]
        frames = [0, t_len // 3, 2 * t_len // 3, t_len - 1]

    print(f"Pass 1: Computing min/max for color scaling...")
    signal_min, signal_max = np.inf, -np.inf
    ac_min, ac_max = np.inf, -np.inf

    with File(results_path, "r") as f:
        for start in range(0, t_len, batch_size):
            end = min(start + batch_size, t_len)
            A_batch = f["A"][start:end]
            AC_batch = f["AC"][start:end]
            signal_min = min(signal_min, A_batch.min())
            signal_max = max(signal_max, A_batch.max())
            ac_min = min(ac_min, AC_batch.min())
            ac_max = max(ac_max, AC_batch.max())
            progress_bar(end, t_len, prefix="  Pass 1")

    print(f"A range: [{signal_min:.4f}, {signal_max:.4f}]")
    print(f"AC range: [{ac_min:.4f}, {ac_max:.4f}]")
    print(f"\nPass 2: Loading {len(frames)} frames...")

    env_img = plt.imread(env_path)
    env_mask = (env_img > 0).astype(np.float32)

    selected_data = {}
    with File(results_path, "r") as f:
        for i, fr in enumerate(frames):
            cell = f["cell"][fr].astype(np.int32)
            nuc = f["nuc"][fr].astype(np.int32)
            A = f["A"][fr]
            AC = f["AC"][fr]
            t = f["t"][fr]

            combined = cell + nuc
            combined = np.maximum(combined, 0.25 * env_mask)
            selected_data[fr] = {
                "t": t,
                "cell": combined,
                "A": A,
                "AC": AC,
            }
            progress_bar(i + 1, len(frames), prefix="  Pass 2")

    fig, axs = plt.subplots(
        3, 5, figsize=(12, 16), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.2]}
    )

    for i, fr in enumerate(frames):
        data = selected_data[fr]
        t = data["t"]
        cell = data["cell"]
        signal = data["A"]
        ac_signal = data["AC"]

        ax_cell = axs[0, i]
        ax_signal = axs[1, i]
        ax_ac = axs[2, i]

        ax_cell.imshow(cell, cmap="inferno")
        ax_cell.set_title(f"t = {t} ms")
        im_signal = ax_signal.imshow(
            signal, cmap="inferno", vmin=signal_min, vmax=signal_max
        )
        im_ac = ax_ac.imshow(ac_signal, cmap="inferno", vmin=ac_min, vmax=ac_max)

    legend_elements = [
        Patch(color=plt.cm.inferno(0.5), label="Cell"),
        Patch(color=plt.cm.inferno(1.0), label="Nucleus"),
    ]
    axs[0, -1].axis("off")
    axs[0, -1].legend(handles=legend_elements, loc="center", frameon=False)
    axs[1, -1].axis("off")
    axs[2, -1].axis("off")
    cax = axs[1, -1].inset_axes([0, 0, 0.2, 1.0])
    cbar = fig.colorbar(im_signal, cax=cax, orientation="vertical")
    cbar.set_label("Actin factor")
    cbar.ax.tick_params()
    cax_ac = axs[2, -1].inset_axes([0, 0, 0.2, 1.0])
    cbar_ac = fig.colorbar(im_ac, cax=cax_ac, orientation="vertical")
    cbar_ac.set_label("AC factor")
    cbar_ac.ax.tick_params()

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Expected at least 2 arguments, found {len(sys.argv) - 1}")
        print("Usage: gen_fig1_chunked.py <input_dir> <output_file> [batch_size]")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    main(results_dir, output_file, batch_size)
