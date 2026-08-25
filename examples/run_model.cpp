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
  std::filesystem::path config_file = root / "config.yaml";
  std::filesystem::path cell_file = root / "cell.png";
  std::filesystem::path env_file = root / "env.png";
  std::filesystem::path A_file = root / "A.png";
  std::filesystem::path AC_file = root / "AC.png";
  std::filesystem::path I_file = root / "I.png";
  std::filesystem::path IC_file = root / "IC.png";
  std::filesystem::path results = root / "results.h5";

  // read files
  dcm::CellModelConfig config(config_file.string());

  // parse masks
  dcm::Mat_i nucleus_mask =
      dcm::matrixFromMask(cell_file.string(), cv::Vec3b(127, 127, 127));
  dcm::Mat_i cell_mask =
      dcm::matrixFromMask(cell_file.string(), cv::Vec3b(255, 255, 255)) +
      nucleus_mask;
  dcm::Mat_i env_mask =
      dcm::matrixFromMask(env_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i A_init =
      dcm::matrixFromMask(A_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i AC_init =
      dcm::matrixFromMask(AC_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i I_init =
      dcm::matrixFromMask(I_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i IC_init =
      dcm::matrixFromMask(IC_file.string(), cv::Vec3b(255, 255, 255));

  dcm::CellState cell = dcm::initializeState(config, cell_mask, nucleus_mask,
                                             env_mask.sparseView());
  cell.A = A_init.cast<double>();
  cell.AC = AC_init.cast<double>();
  cell.I = I_init.cast<double>();
  cell.IC = IC_init.cast<double>();
  dcm::setOutput(cell, results.string());
  dcm::initializeAdhesions(cell);

  std::cout << "Setup done. (" << timer.elapsed().count() << " ms)"
            << std::endl;
  std::vector<double> iter_times;

  std::cout << "Running iterations: " << config.numIters << " iterations"
            << std::endl;
  auto A = tq::trange(config.numIters);
  for (int i : A) {
    timer.reset();
    dcm::step(cell);
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
