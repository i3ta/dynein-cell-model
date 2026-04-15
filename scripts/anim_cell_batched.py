import os
import shutil
import sys
import tempfile

import imageio
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


def main(dir: str, output: str, batch_size: int = 50, downsample: int = 1):
    results_path = os.path.join(dir, "results.h5")
    env_path = os.path.join(dir, "env.png")

    with File(results_path, "r") as f:
        t_len = f["cell"].shape[0]
        rows = f["cell"].shape[1]
        cols = f["cell"].shape[2]

    print(f"Dataset: {t_len} frames, {rows}x{cols} grid")
    print(f"Downsample factor: {downsample}")
    print(f"Batch size: {batch_size}")

    slice_rows = slice(None, None, downsample)
    slice_cols = slice(None, None, downsample)

    env_img = plt.imread(env_path)
    env_mask = 0.25 * (env_img > 0).astype(np.float32)
    if downsample > 1:
        env_mask = env_mask[slice_rows, slice_cols]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axs:
        ax.axis("off")

    temp_dir = tempfile.mkdtemp(prefix="anim_cell_")
    print(f"Writing frames to {temp_dir}")

    try:
        for start in range(0, t_len, batch_size):
            end = min(start + batch_size, t_len)

            with File(results_path, "r") as f:
                t_batch = f["t"][start:end]
                cell_batch = f["cell"][start:end, slice_rows, slice_cols].astype(
                    np.int32
                ) + f["nuc"][start:end, slice_rows, slice_cols].astype(np.int32)
                A_batch = f["A"][start:end, slice_rows, slice_cols]
                AC_batch = f["AC"][start:end, slice_rows, slice_cols]

            for i, (t, cell, A, AC) in enumerate(
                zip(t_batch, cell_batch, A_batch, AC_batch)
            ):
                frame_idx = start + i
                cell = np.maximum(cell, env_mask)

                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(cell, interpolation="nearest", cmap="inferno")
                axs[0].set_title(f"t = {t:.1f}", fontsize=10)
                axs[0].axis("off")

                axs[1].imshow(
                    A, interpolation="nearest", cmap="inferno", vmin=0.0, vmax=1.0
                )
                axs[1].set_title("A", fontsize=10)
                axs[1].axis("off")

                axs[2].imshow(
                    AC, interpolation="nearest", cmap="inferno", vmin=0.0, vmax=1.0
                )
                axs[2].set_title("AC", fontsize=10)
                axs[2].axis("off")

                fig.tight_layout()
                fig.savefig(
                    os.path.join(temp_dir, f"frame_{frame_idx:06d}.png"),
                    dpi=80,
                    bbox_inches="tight",
                )
                plt.close(fig)

            progress_bar(end, t_len, prefix="  Rendering")

        print("Encoding GIF...")

        images = []
        for frame_idx in range(t_len):
            img_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            images.append(imageio.imread(img_path))

        imageio.mimsave(output, images, fps=20, loop=0)
        print(f"Saved to {output}")

    finally:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up {temp_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Expected at least 2 arguments, found {len(sys.argv) - 1}")
        print(
            "Usage: anim_cell_chunked.py <input_dir> <output.gif> [batch_size] [downsample]"
        )
        print("  batch_size: frames per batch (default: 50)")
        print(
            "  downsample: spatial downsampling factor, e.g. 2 = half resolution (default: 1)"
        )
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    downsample = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    main(results_dir, output_file, batch_size, downsample)
