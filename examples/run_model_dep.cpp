#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>

#include "dynein_cell_model/dynein_cell_model.h"
#include "dynein_cell_model/io.h"
#include "dynein_cell_model/simulate.h"
#include "metric_utils/metric_utils.h"
#include <tqdm.hpp>

namespace dcm = dynein_cell_model;

// Run the retained deprecated algorithm variants for regression comparison.
int main(int argc, char *argv[]) {
  metrics::ScopedTimer totalTimer("Total Elapsed Time");
  if (argc != 2) {
    std::cerr << "Expected 1 argument, found " << argc - 1 << std::endl;
    return 1;
  }

  const std::filesystem::path root{argv[1]};
  const auto configFile = root / "config.yaml";
  const auto cellFile = root / "cell.png";
  const auto envFile = root / "env.png";
  const auto aFile = root / "A.png";
  const auto acFile = root / "AC.png";
  const auto iFile = root / "I.png";
  const auto icFile = root / "IC.png";
  const auto results = root / "results.h5";

  dcm::CellModelConfig config(configFile.string());
  const dcm::ViewI nucleus =
      dcm::matrixFromMask(cellFile.string(), cv::Vec3b(127, 127, 127));
  const dcm::ViewI cell =
      dcm::matrixFromMask(cellFile.string(), cv::Vec3b(255, 255, 255)) +
      nucleus;
  const dcm::ViewI env =
      dcm::matrixFromMask(envFile.string(), cv::Vec3b(255, 255, 255));
  const dcm::ViewI aInit =
      dcm::matrixFromMask(aFile.string(), cv::Vec3b(255, 255, 255));
  dcm::ViewD acInit =
      dcm::matrixFromMask(acFile.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();
  const dcm::ViewI iInit =
      dcm::matrixFromMask(iFile.string(), cv::Vec3b(255, 255, 255));
  dcm::ViewD icInit =
      dcm::matrixFromMask(icFile.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();

  // Preserve the metrics-runner initial polarization: inactive cytosolic
  // signal starts at 0.75, while active cytosolic signal occupies the front
  // (higher-row) half of the cell.
  icInit *= 0.75;
  int minRow = config.simRows;
  int maxRow = 0;
  for (int c = 0; c < config.simCols; ++c) {
    for (int r = 0; r < config.simRows; ++r) {
      if (cell(r, c) == 1) {
        minRow = std::min(minRow, r);
        maxRow = std::max(maxRow, r);
      }
    }
  }
  const int midRow = (minRow + maxRow) / 2;
  for (int c = 0; c < config.simCols; ++c) {
    for (int r = 0; r < config.simRows; ++r) {
      if (nucleus(r, c) == 1) {
        acInit(r, c) = 0;
        icInit(r, c) = 0;
      } else if (cell(r, c) == 1 && r > midRow) {
        acInit(r, c) = 0.75;
        icInit(r, c) = 0;
      }
    }
  }

  dcm::CellState state =
      dcm::initializeState(config, cell, nucleus, env.sparseView());
  state.A = aInit.cast<double>();
  state.AC = acInit;
  state.I = iInit.cast<double>();
  state.IC = icInit;
  dcm::setOutput(state, results.string());
  dcm::initializeAdhesions(state);
  dcm::saveState(state);

  metrics::ScopedTimer timer("iteration", false);
  std::vector<double> times;
  auto progress = tq::trange(config.numIters);
  for (int ignored : progress) {
    (void)ignored;
    timer.reset();
    dcm::stepDep(state);
    times.push_back(timer.elapsed().count());
    const Eigen::Map<dcm::Arr_d> values(times.data(), times.size());
    progress << values.mean() << " ms / it ";
  }

  const Eigen::Map<dcm::Arr_d> values(times.data(), times.size());
  const double mean = values.mean();
  const double stdev =
      times.size() > 1
          ? std::sqrt((values - mean).square().sum() / (times.size() - 1))
          : 0.0;
  std::cout << "\n----- Deprecated algorithm summary -----\n"
            << "Mean: " << mean << " ms / it\n"
            << "Stdev: " << stdev << " ms / it\n";
}
