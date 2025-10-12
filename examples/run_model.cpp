#include <cmath>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

#include <opencv2/core/matx.hpp>

#include <dynein_cell_model/dynein_cell_model.hpp>
#include <metric_utils/metric_utils.hpp>
#include <tqdm.hpp>

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
  dcm::Mat_i nucleus_mask = dcm::matrix_from_mask(cell_file.string(), cv::Vec3b(127, 127, 127));
  dcm::Mat_i cell_mask = dcm::matrix_from_mask(cell_file.string(), cv::Vec3b(255, 255, 255)) + nucleus_mask;
  dcm::Mat_i env_mask = dcm::matrix_from_mask(env_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i A_init = dcm::matrix_from_mask(A_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i AC_init = dcm::matrix_from_mask(AC_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i I_init = dcm::matrix_from_mask(I_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_i IC_init = dcm::matrix_from_mask(IC_file.string(), cv::Vec3b(255, 255, 255));

  // create cell
  dcm::CellModel cell(config);
  cell.set_cell(cell_mask);
  cell.set_nuc(nucleus_mask);
  cell.set_env(env_mask.sparseView());
  cell.set_A(A_init.cast<double>());
  cell.set_AC(AC_init.cast<double>());
  cell.set_I(I_init.cast<double>());
  cell.set_IC(IC_init.cast<double>());
  cell.set_output(results.string());

  cell.init_adhesions();

  std::cout << "Setup done. (" << timer.elapsed().count() << " ms)" << std::endl;
  std::vector<double> iter_times;
  
  std::cout << "Running iterations: " << config.num_iters_ << " iterations" << std::endl;
  auto A = tq::trange(config.num_iters_);
  for (int i: A) {
    timer.reset();
    std::string out = cell.step();
    iter_times.push_back(timer.elapsed().count());

    Eigen::Map<dcm::Arr_d> iter_arr(iter_times.data(), iter_times.size());
    double mean = iter_arr.mean();
    A << mean << " ms / it " << out;
  }
  std::cout << std::endl;

  Eigen::Map<dcm::Arr_d> iter_arr(iter_times.data(), iter_times.size());
  double mean = iter_arr.mean();
  double stdev = sqrt((iter_arr - mean).square().sum() / (iter_times.size() - 1));
  std::cout << "----- Summary -----\n";
  std::cout << "Mean: " << mean << " ms / it\n";
  std::cout << "Stdev: " << stdev << " ms / it\n";
}
