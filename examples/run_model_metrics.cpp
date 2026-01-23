#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/matx.hpp>

#include <dynein_cell_model/dynein_cell_model.hpp>
#include <metric_utils/metric_utils.hpp>
#include <tqdm.hpp>

using json = nlohmann::json;

namespace dcm = dynein_cell_model;

/**
 * Run the cell model simulations with additional metric tracking.
 * Outputs metric data as csv instead of json.
 */
int main(int argc, char *argv[]) {
  metrics::ScopedTimer auto_timer("Total Elapsed Time");

  if (argc != 3) {
    std::cerr << "Expected 2 arguments, found " << argc - 1 << std::endl;
    return 1;
  }

  std::filesystem::path root{argv[1]};
  std::filesystem::path metrics_file_path{argv[2]};

  metrics::ScopedTimer timer{"sections", false};
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
      dcm::matrix_from_mask(cell_file.string(), cv::Vec3b(127, 127, 127));
  dcm::Mat_i cell_mask =
      dcm::matrix_from_mask(cell_file.string(), cv::Vec3b(255, 255, 255)) +
      nucleus_mask;
  dcm::Mat_i env_mask =
      dcm::matrix_from_mask(env_file.string(), cv::Vec3b(255, 255, 255));
  dcm::Mat_d A_init =
      dcm::matrix_from_mask(A_file.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();
  dcm::Mat_d AC_init =
      dcm::matrix_from_mask(AC_file.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();
  dcm::Mat_d I_init =
      dcm::matrix_from_mask(I_file.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();
  dcm::Mat_d IC_init =
      dcm::matrix_from_mask(IC_file.string(), cv::Vec3b(255, 255, 255))
          .cast<double>();

  // transform inputs
  IC_init *= 0.75;
  for (int j = 0; j < config.sim_cols_; ++j) {
    for (int i = 0; i < config.sim_rows_; ++i) {
      if (nucleus_mask(i, j) == 1) {
        AC_init(i, j) = 0;
        IC_init(i, j) = 0;
      } else if (cell_mask(i, j) == 1 && i > 500) {
        AC_init(i, j) = 0.75;
        IC_init(i, j) = 0;
      }
    }
  }

  // create cell
  dcm::CellModel cell(config);
  cell.set_cell(cell_mask);
  cell.set_nuc(nucleus_mask);
  cell.set_env(env_mask.sparseView());
  cell.set_A(A_init);
  cell.set_AC(AC_init);
  cell.set_I(I_init);
  cell.set_IC(IC_init);
  cell.set_output(results.string());

  cell.init_adhesions();
  cell.save_state(); // save initial state

  std::cout << "Setup done. (" << timer.elapsed().count() << " ms)"
            << std::endl;
  std::vector<double> iter_times;
  std::vector<std::vector<double>> times;

  std::cout << "Running iterations: " << config.num_iters_ << " iterations"
            << std::endl;
  auto A = tq::trange(config.num_iters_);
  for (int i : A) {
    timer.reset();
    times.push_back(cell.step_dep());
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

  // Recording metrics for further optimization
  std::ofstream metrics_file(metrics_file_path.string());
  size_t num_sections = times.empty() ? 0 : times[0].size();

  metrics_file << "rearrange_adhesions,update_frame,update_nuc,update_cell,"
                  "correct_conc,diffuse_k0,save_state,\n";

  for (std::vector<double> &row : times) {
    for (double &t : row) {
      metrics_file << t << ",";
    }
    metrics_file << "\n";
  }

  metrics_file.close();
}
