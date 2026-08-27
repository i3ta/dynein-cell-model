#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>
#include <tqdm.hpp>

#include "dynein_cell_model/dynein_cell_model.h"
#include "dynein_cell_model/io.h"
#include "dynein_cell_model/simulate.h"
#include "metric_utils/metric_utils.h"

namespace dcm = dynein_cell_model;

int main(int argc, char *argv[]) {
  metrics::ScopedTimer auto_timer("Total Elapsed Time");

  if (argc != 2) {
    std::cerr << "Expected 1 argument, found " << argc - 1 << std::endl;
    return 1;
  }

  std::filesystem::path root{argv[1]};

  metrics::ScopedTimer timer("sections", false);
  std::cout << "Starting setup..." << std::endl;

  // file paths
  std::filesystem::path configFile = root / "config.yaml";
  std::filesystem::path cellFile = root / "cell.png";
  std::filesystem::path envFile = root / "env.png";
  std::filesystem::path aFile = root / "A.png";
  std::filesystem::path acFile = root / "AC.png";
  std::filesystem::path iFile = root / "I.png";
  std::filesystem::path icFile = root / "IC.png";
  std::filesystem::path results = root / "results.h5";

  // read files
  dcm::CellModelConfig config(configFile.string());

  // parse masks
  dcm::Mat_i nucleus_mask =
      dcm::matrixFromMask(cellFile.string(), cv::Vec3b(127, 127, 127));
  dcm::Mat_i cell_mask =
      dcm::matrixFromMask(cellFile.string(), cv::Vec3b(255, 255, 255)) +
      nucleus_mask;
  dcm::Mat_i env_mask =
      dcm::matrixFromMask(envFile.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i A_init =
      dcm::matrixFromMask(aFile.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i AC_init =
      dcm::matrixFromMask(acFile.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i I_init =
      dcm::matrixFromMask(iFile.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_d IC_init =
      dcm::matrixFromMask(icFile.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();

  // Preserve the metrics-runner initial polarization: inactive cytosolic
  // signal starts at 0.75, while active cytosolic signal occupies the front
  // (higher-row) half of the cell.
  IC_init *= 0.75;
  dcm::Mat_d AC_polarized = AC_init.cast<double>();
  int minRow = config.simRows;
  int maxRow = 0;
  for (int c = 0; c < config.simCols; ++c) {
    for (int r = 0; r < config.simRows; ++r) {
      if (cell_mask(r, c) == 1) {
        minRow = std::min(minRow, r);
        maxRow = std::max(maxRow, r);
      }
    }
  }
  const int midRow = (minRow + maxRow) / 2;
  for (int c = 0; c < config.simCols; ++c) {
    for (int r = 0; r < config.simRows; ++r) {
      if (nucleus_mask(r, c) == 1) {
        AC_polarized(r, c) = 0;
        IC_init(r, c) = 0;
      } else if (cell_mask(r, c) == 1 && r > midRow) {
        AC_polarized(r, c) = 0.75;
        IC_init(r, c) = 0;
      }
    }
  }

  dcm::CellState state = dcm::initializeState(config, cell_mask, nucleus_mask,
                                              env_mask.sparseView());
  state.A = A_init.cast<double>();
  state.AC = AC_polarized;
  state.I = I_init.cast<double>();
  state.IC = IC_init;
  dcm::setOutput(state, results.string());
  dcm::initializeAdhesions(state);
  dcm::saveState(state);

  std::cout << "Setup done. (" << timer.elapsed().count() << " ms)"
            << std::endl;
  std::vector<double> iter_times;

  std::cout << "Running iterations: " << config.numIters << " iterations"
            << std::endl;
  auto A = tq::trange(config.numIters);
  for (int i : A) {
    timer.reset();
    dcm::step(state);
    iter_times.push_back(timer.elapsed().count());

    Eigen::Map<dcm::Arr_d> iter_arr(iter_times.data(), iter_times.size());
    double mean = iter_arr.mean();
    A << mean << " ms / it ";
  }
  std::cout << std::endl;

  Eigen::Map<dcm::Arr_d> iter_arr(iter_times.data(), iter_times.size());
  double mean = iter_arr.mean();
  double stdev =
      sqrt((iter_arr - mean).square().sum() / (iter_times.size() - 1));
  std::cout << "----- Summary -----\n";
  std::cout << "Mean: " << mean << " ms / it\n";
  std::cout << "Stdev: " << stdev << " ms / it\n";
}
