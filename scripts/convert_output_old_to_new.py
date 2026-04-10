import os
import sys
import numpy as np
from h5py import File


def parse_txt_file(filepath):
    """Parse old txt format, return (env_rows, env_cols, fr_rows, fr_cols, fr_rows_pos, fr_cols_pos, data)."""
    with open(filepath, "r") as f:
        header = f.readline().split()
        fr_rows, fr_cols, fr_rows_pos, fr_cols_pos, env_rows, env_cols = (
            int(header[0]),
            int(header[1]),
            int(header[2]),
            int(header[3]),
            int(header[4]),
            int(header[5]),
        )
        data = np.fromstring(f.read(), sep=" ")
    return (
        env_rows,
        env_cols,
        fr_rows,
        fr_cols,
        fr_rows_pos,
        fr_cols_pos,
        data.reshape(fr_rows, fr_cols),
    )


def get_time_steps(dataset_dir):
    """Get list of time step indices from dataset directory."""
    files = os.listdir(dataset_dir)
    indices = []
    for f in files:
        if f.endswith(".txt"):
            try:
                idx = int(f.replace(".txt", ""))
                indices.append(idx)
            except ValueError:
                pass
    return sorted(indices)


def read_dataset_to_array(base_dir, dataset_name, n_t, env_rows, env_cols):
    """Read all time steps of a dataset into a 3D numpy array placed on full environment."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        return None

    sample_file = os.path.join(dataset_dir, "0.txt")
    if not os.path.exists(sample_file):
        sample_file = os.path.join(dataset_dir, "1.txt")
    if not os.path.exists(sample_file):
        return None

    arr = np.zeros((n_t, env_rows, env_cols), dtype=np.float64)

    for t in range(n_t):
        filepath = os.path.join(dataset_dir, f"{t}.txt")
        if os.path.exists(filepath):
            _, _, rows, cols, fr_r, fr_c, data = parse_txt_file(filepath)
            arr[t, fr_r : fr_r + rows, fr_c : fr_c + cols] = data

    return arr


def main(input_dir, output_file):
    text_dir = os.path.join(input_dir, "text")
    if not os.path.isdir(text_dir):
        text_dir = input_dir

    datasets_to_check = ["Im", "A", "AC", "I", "IC"]
    n_t = 0
    env_rows, env_cols = 0, 0

    for ds in datasets_to_check:
        ds_dir = os.path.join(text_dir, ds)
        if os.path.isdir(ds_dir):
            indices = get_time_steps(ds_dir)
            if indices:
                n_t = max(indices) + 1
                sample_file = os.path.join(ds_dir, f"{indices[0]}.txt")
                env_rows, env_cols, _, _, _, _, _ = parse_txt_file(sample_file)
                break

    if n_t == 0:
        print(f"Error: No valid dataset directories found in {text_dir}")
        sys.exit(1)

    print(f"Detected: {n_t} time steps, {env_rows}x{env_cols} environment")

    datasets = {}
    dataset_mapping = {
        "Im": "cell",
        "Im_nuc": "nuc",
    }

    for src_name, dst_name in dataset_mapping.items():
        arr = read_dataset_to_array(text_dir, src_name, n_t, env_rows, env_cols)
        if arr is not None:
            datasets[dst_name] = arr
            print(f"  {src_name} -> {dst_name}: shape {arr.shape}")

    for ds in ["A", "AC", "I", "IC"]:
        arr = read_dataset_to_array(text_dir, ds, n_t, env_rows, env_cols)
        if arr is not None:
            datasets[ds] = arr
            print(f"  {ds}: shape {arr.shape}")

    if "cell" not in datasets:
        arr = read_dataset_to_array(text_dir, "Im", n_t, env_rows, env_cols)
        if arr is not None:
            datasets["cell"] = arr

    if "nuc" not in datasets:
        nuc_dir = os.path.join(text_dir, "Im_nuc")
        if os.path.isdir(nuc_dir):
            print("  Warning: Im_nuc not found, skipping nuc dataset")
        else:
            print("  Warning: Im_nuc not found, skipping nuc dataset")

    t_values = np.arange(n_t, dtype=np.float64)

    with File(output_file, "w") as f:
        f.create_dataset("t", data=t_values)
        print(f"  t: shape {t_values.shape}")

        for name, data in datasets.items():
            f.create_dataset(name, data=data)
            print(f"  {name}: shape {data.shape}")

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Expected 2 arguments, found {len(sys.argv) - 1}")
        print("Usage: convert_output_old_to_new.py <input_dir> <output.h5>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(input_dir, output_file)
