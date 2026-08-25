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
  const auto results = root / "results_dep.h5";

  dcm::CellModelConfig config(configFile.string());
  const dcm::ViewI nucleus =
      dcm::matrixFromMask(cellFile.string(), cv::Vec3b(127, 127, 127));
  const dcm::ViewI cell =
      dcm::matrixFromMask(cellFile.string(), cv::Vec3b(255, 255, 255)) + nucleus;
  const dcm::ViewI env =
      dcm::matrixFromMask(envFile.string(), cv::Vec3b(255, 255, 255));

  dcm::CellState state = dcm::initializeState(config, cell, nucleus, env.sparseView());
  dcm::setOutput(state, results.string());
  dcm::initializeAdhesions(state);

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
  const double stdev = times.size() > 1
      ? std::sqrt((values - mean).square().sum() / (times.size() - 1)) : 0.0;
  std::cout << "\n----- Deprecated algorithm summary -----\n"
            << "Mean: " << mean << " ms / it\n"
            << "Stdev: " << stdev << " ms / it\n";
}
