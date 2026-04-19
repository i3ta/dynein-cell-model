import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from h5py import File
from scipy.ndimage import binary_dilation, binary_erosion

from tqdm import tqdm


def load_data(
    data_path: str,
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool], npt.NDArray[np.float32]]:
    with File(data_path, "r") as f:
        dset_nuc = np.array(f["nuc"], dtype=np.bool)
        dset_cell = np.array(f["cell"], dtype=np.bool)
        dset_AC = np.array(f["AC"], dtype=np.float32)

    return dset_nuc, dset_cell, dset_AC


def generate_outlines(mask):
    """
    Generate 8-outline from mask
    mask: (samples, rows, cols) or (rows, cols)
    """
    if mask.ndim == 3:
        outlines = []
        for i in range(mask.shape[0]):
            kernel = np.ones((3, 3), dtype=bool)
            dilated = binary_dilation(mask[i], structure=kernel)
            outlines.append(dilated & ~mask[i])
        return np.array(outlines)
    else:
        kernel = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(mask, structure=kernel)
        return dilated & ~mask


def generate_inner_outlines(mask):
    """
    Generate inner outline from mask
    mask: (samples, rows, cols) or (rows, cols)
    """
    if mask.ndim == 3:
        outlines = []
        for i in range(mask.shape[0]):
            eroded = binary_erosion(mask[i])
            outlines.append(mask[i] & ~eroded)
        return np.array(outlines)
    else:
        eroded = binary_erosion(mask)
        return mask & ~eroded


def generate_dyn_field_old(
    nuc_outline: npt.NDArray[np.bool],
    cell_inner_outline: npt.NDArray[np.bool],
    AC: npt.NDArray[np.float32],
    retract=False,
) -> npt.NDArray[np.float32]:
    dyn_f = np.zeros_like(AC, dtype=np.float32)
    scaling = np.zeros_like(AC, dtype=np.int32)
    n = int(np.count_nonzero(nuc_outline) / (6 if retract else 30))

    nuc_outline_coords = np.argwhere(nuc_outline > 0)

    for row, col in np.argwhere(cell_inner_outline > 0):
        if AC[row, col] <= 0.1:
            continue

        dist2 = np.sum((nuc_outline_coords - np.array([row, col])) ** 2, axis=1)
        min_r, min_c = nuc_outline_coords[np.argmin(dist2)]
        dist_f = np.sqrt(np.min(dist2)) * (AC[row, col] - 0.1)

        for r in range(min_r - n, min_r + n):
            for c in range(min_c - n, min_c + n):
                if nuc_outline[r, c]:
                    dyn_f[r, c] += dist_f
                    scaling[r, c] += 1

    for row, col in np.argwhere(scaling > 0):
        if retract:
            dyn_f[row, col] = max(1 - dyn_f[row, col] / scaling[row, col] / 60, 0)
        else:
            dyn_f[row, col] = min(dyn_f[row, col] / scaling[row, col] / 60, 1)

    return dyn_f


def preprocess_bfs(
    cell: npt.NDArray[np.bool],
    cell_inner_outline: npt.NDArray[np.bool],
    nuc_inner_outline: npt.NDArray[np.bool],
    AC: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    DR = [-1, 0, 1, 0]
    DC = [0, -1, 0, 1]

    sim_rows, sim_cols = cell.shape
    dyn_f = np.zeros_like(AC, dtype=np.float32)
    dist = np.full((sim_rows, sim_cols), -1.0, dtype=np.float32)
    scaling = np.full((sim_rows, sim_cols), 0, dtype=np.int32)
    visited = np.zeros_like(AC, dtype=np.bool)
    visited[nuc_inner_outline] = True
    rev = []
    backtrack = {}

    queue = list(np.argwhere(nuc_inner_outline))
    dist[nuc_inner_outline] = 0.0

    while len(queue) > 0:
        r, c = queue.pop(0)
        rev.append((r, c))
        for dr, dc in zip(DR, DC):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < sim_rows
                and 0 <= nc < sim_cols
                and cell[nr, nc]
                and not visited[nr, nc]
            ):
                visited[nr, nc] = True
                backtrack[(nr, nc)] = (r, c)
                dist[nr, nc] = dist[r, c] + 1.0
                queue.append((nr, nc))
        if cell_inner_outline[r, c] and AC[r, c] > 0.1:
            dyn_f[r, c] = (AC[r, c] - 0.1) * dist[r, c]
            scaling[r, c] = 1

    for r, c in reversed(rev):
        if dyn_f[r, c] == 0:
            continue
        parent = backtrack.get((r, c))
        if parent is not None:
            dyn_f[parent] += dyn_f[r, c]
            scaling[parent] += scaling[r, c]

    mask = (scaling > 0) & nuc_inner_outline
    dyn_f[mask] /= scaling[mask]
    dyn_f[~mask] = 0.0

    return dyn_f


class DynFieldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.dyn_sigma = nn.Parameter(torch.tensor(8.5))
        self.scale_factor = nn.Parameter(torch.tensor(0.55))

    def forward(self, dyn_f_raw, mode="protrude"):
        """
        Args:
            dyn_f_raw: (batch, rows, cols) - precomputed BFS values, centered
            mode: 'protrude' or 'retract'
        Returns:
            pred: (batch, rows, cols) - normalized and smoothed dyn_f values
        """
        kernel = self._gaussian_kernel()
        dyn_f = F.conv2d(
            dyn_f_raw.unsqueeze(1),
            kernel.repeat(1, 1, 1, 1),
            padding=self.kernel_size // 2,
        ).squeeze(1)
        dyn_f = dyn_f * self.scale_factor

        if mode == "retract":
            dyn_f = 1.0 - dyn_f

        return dyn_f

    def _gaussian_kernel(self):
        sigma = torch.abs(self.dyn_sigma)
        size = 2 * int(3 * sigma.item()) + 1
        if size % 2 == 0:
            size += 1
        size = max(3, size)

        device = next(self.parameters()).device

        ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        self.kernel_size = size
        return kernel.unsqueeze(0).unsqueeze(0)


def fit_model(data_path: str, epochs: int = 20):
    # Load data
    print("Loading data...")
    dset_nuc, dset_cell, dset_AC = load_data(data_path)
    dset_nuc, dset_cell, dset_AC = dset_nuc[1:], dset_cell[1:], dset_AC[1:]
    nuc_outline = generate_outlines(dset_nuc)
    nuc_inner_outline = generate_inner_outlines(dset_nuc)
    cell_inner_outline = generate_inner_outlines(dset_cell)
    samples = dset_nuc.shape[0]
    print(f"Loaded {samples} samples!")

    print("Generating old samples...")
    dyn_f_old_protr = []
    dyn_f_old_retr = []
    for i in tqdm(range(samples)):
        dyn_f_old_protr.append(
            generate_dyn_field_old(nuc_outline[i], cell_inner_outline[i], dset_AC[i])
        )
        dyn_f_old_retr.append(
            generate_dyn_field_old(
                nuc_inner_outline[i], cell_inner_outline[i], dset_AC[i], True
            )
        )
    dyn_f_old_protr = torch.tensor(np.array(dyn_f_old_protr), dtype=torch.float32)
    dyn_f_old_retr = torch.tensor(np.array(dyn_f_old_retr), dtype=torch.float32)

    print("Generating BFS samples...")
    dyn_f_pre = []
    for _ in tqdm(range(samples)):
        dyn_f_pre.append(
            preprocess_bfs(
                dset_cell[i], cell_inner_outline[i], nuc_inner_outline[i], dset_AC[i]
            )
        )
    dyn_f_pre = torch.tensor(np.array(dyn_f_pre), dtype=torch.float32)

    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move tensors to device
    dyn_f_pre = dyn_f_pre.to(device)
    nuc_outline_mask = torch.tensor(nuc_outline, dtype=torch.float32).to(device)
    nuc_inner_outline_mask = torch.tensor(nuc_inner_outline, dtype=torch.float32).to(
        device
    )
    dyn_f_old_protr = dyn_f_old_protr.to(device)
    dyn_f_old_retr = dyn_f_old_retr.to(device)

    # Initialize model and move to device
    model = DynFieldModel().to(device)
    optimizer = optim.Adam([model.dyn_sigma, model.scale_factor], lr=0.01)

    print(f"Training for {epochs} epochs...")

    loss_history = []
    pbar = tqdm(total=epochs, desc="Training")
    for _ in range(epochs):
        optimizer.zero_grad()

        pred_protr = model(dyn_f_pre, "protrude")
        mask_bool_protr = nuc_outline_mask.bool()
        pred_protr_masked = pred_protr[mask_bool_protr]
        target_protr_masked = dyn_f_old_protr[mask_bool_protr]
        loss_protr = F.mse_loss(pred_protr_masked, target_protr_masked)

        pred_retr = model(dyn_f_pre, "retract")
        mask_bool_retr = nuc_inner_outline_mask.bool()
        pred_retr_masked = pred_retr[mask_bool_retr]
        target_retr_masked = dyn_f_old_retr[mask_bool_retr]
        loss_retr = F.mse_loss(pred_retr_masked, target_retr_masked)

        total_loss = loss_protr + loss_retr
        loss_history.append(total_loss.item())
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(
            {
                "σ": f"{model.dyn_sigma.item():.3f}",
                "s": f"{model.scale_factor.item():.3f}",
                "loss": f"{total_loss.item():.4f}",
            }
        )
        pbar.update(1)
    pbar.close()

    print("Training complete!")
    print(f"Final dyn_sigma: {model.dyn_sigma.item()}")
    print(f"Final scale_factor: {model.scale_factor.item()}")

    # Plot loss curve
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.savefig(output_path)
        print(f"Saved loss curve to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Expected at least 1 argument, found {len(sys.argv) - 1}")
        print("Usage: fit_dyn_f.py <data_path> [epochs] [output_plot]")
        sys.exit(1)

    data_path = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    fit_model(data_path, epochs)
