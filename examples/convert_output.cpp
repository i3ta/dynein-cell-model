#include "highfive/H5DataSet.hpp"
#include <filesystem>
#include <fstream>
#include <highfive/H5File.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
template <typename T>
void output_dataset(const HighFive::File &file, const fs::path &output_dir,
                    const std::string &name, const int n_t);
}

/**
 * Convert the hdf5 output to the old text-based output format.
 */
int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Expected 2 arguments, found " << argc - 1 << std::endl;
    return 1;
  }

  fs::path results_file{argv[1]};
  fs::path output_dir{argv[2]};

  fs::create_directories(output_dir);

  HighFive::File file{results_file, HighFive::File::ReadOnly};
  HighFive::DataSet dset_t = file.getDataSet("t");
  const int n_t = dset_t.getSpace().getDimensions()[0];

  const std::string datasets[] = {"A", "AC", "I", "IC", "nuc", "cell"};
  output_dataset<double>(file, output_dir, "A", n_t);
  output_dataset<double>(file, output_dir, "AC", n_t);
  output_dataset<double>(file, output_dir, "I", n_t);
  output_dataset<double>(file, output_dir, "IC", n_t);
  output_dataset<int>(file, output_dir, "cell", n_t);
  output_dataset<int>(file, output_dir, "nuc", n_t);
}

namespace {
template <typename T>
void output_dataset(const HighFive::File &file, const fs::path &output_dir,
                    const std::string &name, const int n_t) {
  HighFive::DataSet dset = file.getDataSet(name);
  fs::path dset_dir = output_dir / name;
  fs::create_directories(dset_dir);

  for (size_t t = 0; t < n_t; ++t) {
    fs::path output = dset_dir / (std::to_string(t) + ".txt");

    std::vector<size_t> offset = {t, 0, 0};
    std::vector<size_t> dims = dset.getDimensions();
    std::vector<size_t> count = {1, dims[1], dims[2]};

    std::vector<std::vector<std::vector<T>>> data;
    dset.select(offset, count).read(data);

    std::ofstream file{output};

    file << dims[1] << " " << dims[2]
         << " ";    // frame columns and rows (entire simulation)
    file << "0 0 "; // top left of frame (entire simulation)
    file << dims[1] << " " << dims[2] << " "; // simulation columns and rows
    file << "\n";

    for (int i = 0; i < dims[1]; i++) {
      for (int j = 0; j < dims[2]; j++) {
        file << data[0][i][j] << " ";
      }
      file << "\n";
    }

    file.close();
  }
};
} // namespace
