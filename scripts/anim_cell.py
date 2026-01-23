import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.animation import FuncAnimation


def main(dir: str, output: str):
    file = File(os.path.join(dir, "results.h5"), "r")

    dset_t = file["t"]
    dset_cell = file["cell"]
    dset_nuc = file["nuc"]
    dset_A = file["A"]

    data_cell = dset_cell[:, :, :].astype(np.int32) + dset_nuc[:, :, :].astype(np.int32)
    data_A = dset_A[:, :, :]

    t_len: int = dset_cell.shape[0]

    fig, axs = plt.subplots(1, 2)
    im0 = axs[0].imshow(data_cell[0], interpolation="nearest", cmap="inferno")
    im1 = axs[1].imshow(data_A[0], interpolation="nearest", cmap="inferno", vmin=0.0, vmax=1.0)
    fig.suptitle(f"t = {dset_t[0]}")

    def animate(i):
        im0.set_array(data_cell[i])
        im1.set_array(data_A[i])
        fig.suptitle(f"t = {dset_t[i]}")
        return (im0, im1)

    anim = FuncAnimation(fig, animate, frames=range(0, t_len), blit=True)
    anim.save(output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Expected 2 arguments, found {len(sys.argv) - 1}")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(results_dir, output_file)
