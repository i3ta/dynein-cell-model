import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.patches import Patch


def main(dir: str, output: str):
    with File(os.path.join(dir, "results.h5"), "r") as file:
        dset_t = file["t"]
        dset_cell = file["cell"]
        dset_nuc = file["nuc"]
        dset_A = file["A"]

        data_t = dset_t[:]
        data_cell = dset_cell[:, :, :] + dset_nuc[:, :, :]
        data_A = dset_A[:, :, :]
        t_len = dset_cell.shape[0]

    frames = [0, t_len // 3, 2 * t_len // 3, t_len - 1]
    fig, axs = plt.subplots(
        2, 5, figsize=(12, 11), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.2]}
    )

    signal_min = np.min(data_A)
    signal_max = np.max(data_A)

    for fr, i in zip(frames, range(len(frames))):
        t = data_t[fr]
        cell = data_cell[fr]
        signal = data_A[fr]

        ax_cell = axs[0, i]
        ax_signal = axs[1, i]

        ax_cell.imshow(cell, cmap="inferno")
        ax_cell.set_title(f"t = {t} ms")
        im_signal = ax_signal.imshow(
            signal, cmap="inferno", vmin=signal_min, vmax=signal_max
        )

    legend_elements = [
        Patch(color=plt.cm.inferno(0.5), label="Cell"),
        Patch(color=plt.cm.inferno(1.0), label="Nucleus"),
    ]
    axs[0, -1].axis("off")
    axs[0, -1].legend(handles=legend_elements, loc="center", frameon=False)
    axs[1, -1].axis("off")
    cax = axs[1, -1].inset_axes([0, 0, 0.2, 1.0])
    cbar = fig.colorbar(im_signal, cax=cax, orientation="vertical")
    cbar.set_label("Actin factor")
    cbar.ax.tick_params()

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Expected 2 arguments, found {len(sys.argv) - 1}")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(results_dir, output_file)
