import os
import sys

from h5py import File


def main(data_file: str, output_dir: str):
    """
    Convert the output from the new model format to old model format

    Args:
        data_file (str): Path to the data file
        output_dir (str): Folder to output files to
    """
    with File(data_file, "r") as file:
        n_t = file["t"].shape[0]
        dset = {
            "Im": file["cell"],
            "Im_nuc": file["nuc"],
            "A": file["A"],
            "AC": file["AC"],
            "I": file["I"],
            "IC": file["IC"],
        }

        for dir_name, data in dset.items():
            dir = os.path.join(output_dir, dir_name)
            os.makedirs(dir, exist_ok=True)

            for t in range(n_t):
                target = os.path.join(dir, f"{t}.txt")

                with open(target, "w") as file:
                    file.write(f"{data[t].shape[0]} ")  # Number of rows
                    file.write(f"{data[t].shape[1]} ")  # Number of cols
                    file.write(f"0 0 ")  # Top left of frame
                    file.write(f"{data[t].shape[0]} ")  # Number of rows
                    file.write(f"{data[t].shape[1]} ")  # Number of cols

                    for i in range(data[t].shape[0]):
                        for j in range(data[t].shape[1]):
                            file.write(f"{int(data[t][i][j])} ")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Expected 2 arguments, found {len(sys.argv) - 1}")
        sys.exit(1)

    data_file, output_dir = sys.argv[1:]
    main(data_file, output_dir)
