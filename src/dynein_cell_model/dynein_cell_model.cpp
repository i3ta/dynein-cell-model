#ifndef DYNEIN_CELL_MODEL_CPP
#define DYNEIN_CELL_MODEL_CPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5PropertyList.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include <dynein_cell_model/dynein_cell_model.hpp>
#include <metric_utils/metric_utils.hpp>
#include <test_utils/test_utils.hpp>

#ifdef LIB_DYNEIN_CELL_MODEL_DEBUG
#include <test_utils/test_utils.hpp>

// static test_utils::DebugRand<double> drand;
// #define prob_dist(rng) drand()

inline constexpr bool DYNEIN_CELL_MODEL_DEBUG_CPP = true;
#else
inline constexpr bool DYNEIN_CELL_MODEL_DEBUG_CPP = false;
#endif

#define TRACE_MSG(msg)                                                         \
  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP)                                   \
    std::cerr << "[ TRACE    ] [ DyneinCellModel ] " << msg << std::endl       \
              << std::flush;

#define TIME_AND_STORE(times, action)                                          \
  do {                                                                         \
    times.push_back(time_fn([&]() { action; }));                               \
  } while (0)

namespace dynein_cell_model {
namespace {
static constexpr size_t CHUNK_SIZE = 100;

struct pair_hash {
  std::size_t operator()(const std::pair<int, int> &p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

template <typename F> double time_fn(F &&f) {
  metrics::ScopedTimer timer("fn_timer", false);
  std::forward<F>(f)();
  return timer.elapsed().count();
}

/**
 * @brief Append a value (any row-major Eigen Matrix) to a dataset.
 *
 * @param file HighFive file to save dataset to
 * @param dataset Name of the dataset to save the data to
 * @param mat Data to append to dataset
 */
template <typename T>
void append_dataset(HighFive::File &file,
                    std::map<std::string, size_t> &next_index,
                    const std::string &dataset,
                    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor> &mat) {
  // Create dataset if it does not exist
  if (!file.exist(dataset)) {
    std::vector<size_t> dims = {CHUNK_SIZE, (size_t)mat.rows(),
                                (size_t)mat.cols()};
    std::vector<size_t> max_dims = {HighFive::DataSpace::UNLIMITED,
                                    (size_t)mat.rows(), (size_t)mat.cols()};
    HighFive::DataSpace dataspace(dims, max_dims);
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({1, (size_t)mat.rows(), (size_t)mat.cols()}));
    file.createDataSet<T>(dataset, dataspace, props);
    next_index[dataset] = 0;
  }

  // Get or initialize next index
  if (next_index.find(dataset) == next_index.end()) {
    std::vector<size_t> dims =
        file.getDataSet(dataset).getSpace().getDimensions();
    next_index[dataset] = dims[0];
  }

  HighFive::DataSet dset = file.getDataSet(dataset);
  size_t next_t = next_index[dataset];

  // Check if we need to extend
  std::vector<size_t> current_dims = dset.getSpace().getDimensions();
  if (next_t >= current_dims[0]) {
    // Extend by CHUNK_SIZE
    dset.resize(
        {current_dims[0] + CHUNK_SIZE, current_dims[1], current_dims[2]});
  }

  // Write data
  dset.select({next_t, 0, 0}, {1, (size_t)mat.rows(), (size_t)mat.cols()})
      .write_raw(mat.data());
  next_index[dataset]++;
}

void append_dataset(HighFive::File &file,
                    std::map<std::string, size_t> &next_index,
                    const std::string &dataset, const int v) {
  // Create dataset if it does not exist
  if (!file.exist(dataset)) {
    std::vector<size_t> dims = {0};
    std::vector<size_t> max_dims = {HighFive::DataSpace::UNLIMITED};

    HighFive::DataSpace dataspace(dims, max_dims);
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({1}));

    file.createDataSet<int>(dataset, dataspace, props);
  }

  // Open dataset
  HighFive::DataSet dset = file.getDataSet(dataset);

  // Get current dimensions
  std::vector<size_t> dims = dset.getSpace().getDimensions();
  if (dims.size() != 1) {
    throw std::runtime_error("Dataset is not 1D as expected.");
  }

  size_t next_t = dims[0];

  // Extend dataset by 1
  dset.resize({next_t + 1});
  dset.select({next_t}, {1}).write(&v);
}

void append_dataset(HighFive::File &file,
                    std::map<std::string, size_t> &next_index,
                    const std::string &dataset, const Mat_i &mat,
                    bool as_bool) {
  // Save as boolean for best compression
  if (!as_bool) {
    append_dataset<int>(file, next_index, dataset, mat);
    return;
  }

  // Create dataset if it does not exist
  if (!file.exist(dataset)) {
    std::vector<size_t> dims = {CHUNK_SIZE, (size_t)mat.rows(),
                                (size_t)mat.cols()};
    std::vector<size_t> max_dims = {HighFive::DataSpace::UNLIMITED,
                                    (size_t)mat.rows(), (size_t)mat.cols()};
    HighFive::DataSpace dataspace(dims, max_dims);
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({1, (size_t)mat.rows(), (size_t)mat.cols()}));
    file.createDataSet<bool>(dataset, dataspace, props); // Note: bool type
    next_index[dataset] = 0;
  }

  // Get or initialize next index
  if (next_index.find(dataset) == next_index.end()) {
    std::vector<size_t> dims =
        file.getDataSet(dataset).getSpace().getDimensions();
    next_index[dataset] = dims[0];
  }

  HighFive::DataSet dset = file.getDataSet(dataset);
  size_t next_t = next_index[dataset];

  // Check if we need to extend
  std::vector<size_t> current_dims = dset.getSpace().getDimensions();
  if (next_t >= current_dims[0]) {
    // Extend by CHUNK_SIZE
    dset.resize(
        {current_dims[0] + CHUNK_SIZE, current_dims[1], current_dims[2]});
  }

  // Convert to boolean buffer
  size_t total_size = mat.rows() * mat.cols();
  bool *bool_buffer = new bool[total_size];
  for (size_t i = 0; i < total_size; ++i) {
    bool_buffer[i] = mat.data()[i] != 0;
  }

  // Write data
  dset.select({next_t, 0, 0}, {1, (size_t)mat.rows(), (size_t)mat.cols()})
      .write_raw(bool_buffer);
  delete[] bool_buffer;
  next_index[dataset]++;
}

void append_dataset(HighFive::File &file,
                    std::map<std::string, size_t> &next_index,
                    const std::string &dataset, const SpMat_i &mat,
                    bool as_bool) {
  // Convert to dense and save
  const Mat_i dense = mat;
  append_dataset(file, next_index, dataset, dense, as_bool);
}

void append_dataset(HighFive::File &file,
                    std::map<std::string, size_t> &next_index,
                    const std::string &dataset, const Mat_d &mat) {
  // Convert to float
  append_dataset<float>(file, next_index, dataset, mat.cast<float>());
}

void finalize(HighFive::File &file, std::map<std::string, size_t> &next_index,
              const std::string &dataset) {
  if (next_index.find(dataset) != next_index.end()) {
    HighFive::DataSet dset = file.getDataSet(dataset);
    std::vector<size_t> dims = dset.getSpace().getDimensions();
    dset.resize({next_index[dataset], dims[1], dims[2]});
  }
}

inline std::vector<std::pair<int, int>> get_nonzero(const SpMat_i &mat) {
  std::vector<std::pair<int, int>> coords;
  for (int k = 0; k < mat.outerSize(); k++)
    for (SpMat_i::InnerIterator it(mat, k); it; ++it)
      coords.push_back({it.row(), it.col()});
  return coords;
}

/**
 * @brief Generate a random visit order for all of the nonzero pixels in the
 * SpMat_i.
 *
 * @param mat SpMat_i to randomize pixels of
 * @param rng std::mt19937 random number generator to use
 *
 * @return Vector of randomized order
 */
inline const std::vector<std::pair<int, int>>
randomize_nonzero(const SpMat_i &mat, std::mt19937 &rng) {
  auto coords = get_nonzero(mat);
  std::shuffle(coords.begin(), coords.end(), rng);
  return coords;
}

int outline_4(const SpMat_i &outline, const Mat_i &body, int sim_rows,
              int sim_cols) {
  const int DR[4] = {-1, 0, 1, 0};
  const int DC[4] = {0, -1, 0, 1};

  int perim = 0;

  for (int k = 0; k < outline.outerSize(); ++k) {
    for (SpMat_i::InnerIterator it(outline, k); it; ++it) {
      for (int i = 0; i < 4; i++) {
        const int nr = it.row() + DR[i];
        const int nc = it.col() + DC[i];
        if (nr < 0 || nr >= sim_rows || nc < 0 || nc >= sim_cols)
          continue;
        if (body(nr, nc) == 1) {
          perim++;
          break;
        }
      }
    }
  }

  return perim;
}

void gaussian_blur(Mat_d &mat, int row_start, int row_end, int col_start,
                   int col_end, double sigma) {
  const int block_rows = row_end - row_start + 1;
  const int block_cols = col_end - col_start + 1;
  const int radius = std::ceil(3.0 * sigma);
  const int size = radius * 2 + 1;
  Vec_d kernel(size);
  for (int i = 0; i < size; ++i) {
    float x = i - radius;
    kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
  }
  kernel /= kernel.sum();

  Mat_d padded = Mat_d::Zero(block_rows + 2 * radius, block_cols + 2 * radius);
  padded.block(radius, radius, block_rows, block_cols) =
      mat.block(row_start, col_start, block_rows, block_cols);

  Mat_d tmp = Mat_d::Zero(block_rows + 2 * radius, block_cols);
  for (int r = 0; r < padded.rows(); ++r) {
    for (int k = 0; k < size; ++k) {
      tmp.row(r) += kernel[k] * padded.block(r, k, 1, block_cols);
    }
  }

  auto target = mat.block(row_start, col_start, block_rows, block_cols);
  target.setZero();
  for (int c = 0; c < block_cols; ++c) {
    for (int k = 0; k < size; ++k) {
      target.col(c) += kernel[k] * tmp.block(k, c, block_rows, 1);
    }
  }
}
}; // namespace

Mat_i matrix_from_mask(std::string filepath, cv::Vec3b color) {
  cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Could not load the image at: " + filepath);
  }

  Mat_i mat = Mat_i::Zero(image.rows, image.cols);

  for (int c = 0; c < image.cols; c++) {
    for (int r = 0; r < image.rows; r++) {
      if (image.at<cv::Vec3b>(r, c) == color) {
        mat(r, c) = 1;
      }
    }
  }

  return mat;
}

CellModelConfig::CellModelConfig() {
  k0_ = 0.10;
  k0_scalar_ = 10;
  k_ = 1.6;
  T_ = 500;
  k_nuc_ = 4;
  T_nuc_ = 1;
  R_nuc_ = 1;
  R0_ = 20;
  dyn_basal_ = 0.9;
  dyn_sigma_ = 8.56;
  dyn_scale_ = 0.683;
  s1_ = 0.7;
  s2_ = 0.2;
  diff_t_ = 100;
  dt_ = 3.75e-4;
  dx_ = 7.0755e-3;
  act_slope_ = 0.03;
  adh_num_ = 50;
  adh_frac_ = 0.03;
  adh_sigma_ = 5;
  frame_padding_ = 20;
  save_t_ = 1000;
  adh_t_ = 200;
  fr_t_ = 50;
  adh_basal_ = 0.3;
  sim_rows_ = 1500;
  sim_cols_ = 600;
  seed_ = 0;
  num_iters_ = 5000000;
}

CellModelConfig::CellModelConfig(std::string config_file) {
  // Load file
  YAML::Node config = YAML::LoadFile(config_file);

  // Read config
  k_ = config["k"].as<double>();
  k_nuc_ = config["k_nuc"].as<double>();
  g_ = config["g"].as<double>();
  T_ = config["T"].as<double>();
  T_nuc_ = config["T_nuc"].as<double>();
  act_slope_ = config["act_slope"].as<double>();
  adh_sigma_ = config["adh_sigma"].as<double>();
  adh_basal_ = config["adh_basal"].as<double>();
  adh_frac_ = config["adh_frac"].as<double>();
  adh_num_ = config["adh_num"].as<int>();
  R0_ = config["R0"].as<int>();
  R_nuc_ = config["R_nuc"].as<double>();
  dyn_basal_ = config["dyn_basal"].as<double>();
  prop_factor_ = config["prop_factor"].as<double>();
  dyn_sigma_ = config["dyn_sigma"].as<double>();
  dyn_scale_ = config["dyn_scale"].as<double>();
  DA_ = config["DA"].as<double>();
  DI_ = config["DI"].as<double>();
  k0_ = config["k0"].as<double>();
  k0_min_ = config["k0_min"].as<double>();
  k0_scalar_ = config["k0_scalar"].as<double>();
  gamma_ = config["gamma"].as<double>();
  delta_ = config["delta"].as<double>();
  A0_ = config["A0"].as<double>();
  s1_ = config["s1"].as<double>();
  s2_ = config["s2"].as<double>();
  F0_ = config["F0"].as<double>();
  kn_ = config["kn"].as<double>();
  ks_ = config["ks"].as<double>();
  eps_ = config["eps"].as<double>();
  dt_ = config["dt"].as<double>();
  dx_ = config["dx"].as<double>();
  A_max_ = config["A_max"].as<double>();
  A_min_ = config["A_min"].as<double>();
  AC_max_ = config["AC_max"].as<double>();
  AC_min_ = config["AC_min"].as<double>();
  sim_rows_ = config["sim_rows"].as<int>();
  sim_cols_ = config["sim_cols"].as<int>();
  seed_ = config["seed"].as<int>();
  num_iters_ = config["num_iters"].as<int>();
  frame_padding_ = config["frame_padding"].as<int>();
  diff_t_ = config["diff_t"].as<int>();
  save_t_ = config["save_t"].as<int>();
  adh_t_ = config["adh_t"].as<int>();
  fr_t_ = config["fr_t"].as<int>();
}

void CellModelConfig::save_file(std::string dest_file) {
  YAML::Node config;
  config["k"] = k_;
  config["k_nuc"] = k_nuc_;
  config["g"] = g_;
  config["T"] = T_;
  config["T_nuc"] = T_nuc_;
  config["act_slope"] = act_slope_;
  config["adh_sigma"] = adh_sigma_;
  config["adh_basal"] = adh_basal_;
  config["adh_frac"] = adh_frac_;
  config["adh_num"] = adh_num_;
  config["R0"] = R0_;
  config["R_nuc"] = R_nuc_;
  config["dyn_basal"] = dyn_basal_;
  config["prop_factor"] = prop_factor_;
  config["dyn_sigma"] = dyn_sigma_;
  config["dyn_scale"] = dyn_scale_;
  config["DA"] = DA_;
  config["DI"] = DI_;
  config["k0"] = k0_;
  config["k0_min"] = k0_min_;
  config["k0_scalar"] = k0_scalar_;
  config["gamma"] = gamma_;
  config["delta"] = delta_;
  config["A0"] = A0_;
  config["s1"] = s1_;
  config["s2"] = s2_;
  config["F0"] = F0_;
  config["kn"] = kn_;
  config["ks"] = ks_;
  config["eps"] = eps_;
  config["dt"] = dt_;
  config["dx"] = dx_;
  config["A_max"] = A_max_;
  config["A_min"] = A_min_;
  config["AC_max"] = AC_max_;
  config["AC_min"] = AC_min_;
  config["sim_rows"] = sim_rows_;
  config["sim_cols"] = sim_cols_;
  config["seed"] = seed_;
  config["num_iters"] = num_iters_;

  std::ofstream fout(dest_file);
  fout << config;
}

CellModel::CellModel() {
  CellModelConfig config;
  update_config(config);
}

CellModel::CellModel(CellModelConfig config) { update_config(config); }

void CellModel::update_config(CellModelConfig config) {
  k_ = config.k_;
  k_nuc_ = config.k_nuc_;
  g_ = config.g_;
  T_ = config.T_;
  T_nuc_ = config.T_nuc_;
  act_slope_ = config.act_slope_;
  adh_sigma_ = config.adh_sigma_;
  adh_basal_ = config.adh_basal_;
  adh_frac_ = config.adh_frac_;
  adh_num_ = config.adh_num_;
  R0_ = config.R0_;
  R_nuc_ = config.R_nuc_;
  dyn_basal_ = config.dyn_basal_;
  prop_factor_ = config.prop_factor_;
  dyn_sigma_ = config.dyn_sigma_;
  dyn_scale_ = config.dyn_scale_;
  DA_ = config.DA_;
  DI_ = config.DI_;
  k0_ = config.k0_;
  k0_min_ = config.k0_min_;
  k0_scalar_ = config.k0_scalar_;
  gamma_ = config.gamma_;
  delta_ = config.delta_;
  A0_ = config.A0_;
  s1_ = config.s1_;
  s2_ = config.s2_;
  F0_ = config.F0_;
  kn_ = config.kn_;
  ks_ = config.ks_;
  eps_ = config.eps_;
  dt_ = config.dt_;
  dx_ = config.dx_;
  A_max_ = config.A_max_;
  A_min_ = config.A_min_;
  AC_max_ = config.AC_max_;
  AC_min_ = config.AC_min_;
  sim_rows_ = config.sim_rows_;
  sim_cols_ = config.sim_cols_;
  frame_padding_ = config.frame_padding_;
  diff_t_ = config.diff_t_;
  save_t_ = config.save_t_;
  adh_t_ = config.adh_t_;
  fr_t_ = config.fr_t_;

  // update variables that depend on config
  initialize_helpers();
}

CellModel::~CellModel() {
  // trim excess space for results
  finalize(*results_, next_index_, "cell");
  finalize(*results_, next_index_, "nuc");
  finalize(*results_, next_index_, "A");
  finalize(*results_, next_index_, "A");
  finalize(*results_, next_index_, "I");
  finalize(*results_, next_index_, "AC");
  finalize(*results_, next_index_, "IC");
  finalize(*results_, next_index_, "adh");
  finalize(*results_, next_index_, "env");
  finalize(*results_, next_index_, "F");
  finalize(*results_, next_index_, "k0_adh");
}

const CellModelConfig CellModel::get_config() {
  CellModelConfig config;
  config.k_ = k_;
  config.k_nuc_ = k_nuc_;
  config.g_ = g_;
  config.T_ = T_;
  config.T_nuc_ = T_nuc_;
  config.act_slope_ = act_slope_;
  config.adh_sigma_ = adh_sigma_;
  config.adh_basal_ = adh_basal_;
  config.adh_frac_ = adh_frac_;
  config.adh_num_ = adh_num_;
  config.R0_ = R0_;
  config.R_nuc_ = R_nuc_;
  config.dyn_basal_ = dyn_basal_;
  config.prop_factor_ = prop_factor_;
  config.dyn_sigma_ = dyn_sigma_;
  config.dyn_scale_ = dyn_scale_;
  config.DA_ = DA_;
  config.DI_ = DI_;
  config.k0_ = k0_;
  config.k0_min_ = k0_min_;
  config.k0_scalar_ = k0_scalar_;
  config.gamma_ = gamma_;
  config.A0_ = A0_;
  config.s1_ = s1_;
  config.s2_ = s2_;
  config.F0_ = F0_;
  config.kn_ = kn_;
  config.ks_ = ks_;
  config.dt_ = dt_;
  config.dx_ = dx_;
  config.A_max_ = A_max_;
  config.A_min_ = A_min_;
  config.AC_max_ = AC_max_;
  config.AC_min_ = AC_min_;
  config.sim_rows_ = sim_rows_;
  config.sim_cols_ = sim_cols_;

  return config;
}

void CellModel::simulate(double dt) {
  const int t_n = dt / dt_; // number of time steps to simulate
  simulate_steps(t_n);
}

void CellModel::simulate_steps(int n) {
  for (int i = 0; i < n; i++) {
    step();
  }
}

void CellModel::step() {
  if (t_ % adh_t_ == 0)
    rearrange_adhesions();

  if (t_ % fr_t_ == 0)
    update_frame();

  protrude_nuc();
  retract_nuc();

  protrude();
  retract();

  correct_concentrations();
  diffuse_k0_adh();
  update_dyn_nuc_field();

  if (++t_ % save_t_ == 0)
    save_state();
}

std::vector<double> CellModel::step_dep() {
  std::vector<double> times;
  times.reserve(6);

  TIME_AND_STORE(times, if (t_ % adh_t_ == 0) rearrange_adhesions(false));
  TIME_AND_STORE(times, if (t_ % fr_t_ == 0) update_frame());

  TIME_AND_STORE(times, protrude_nuc_dep(); retract_nuc_dep(););

  TIME_AND_STORE(times, protrude(); retract(););

  TIME_AND_STORE(times, correct_concentrations());
  TIME_AND_STORE(times, diffuse_k0_adh());

  TIME_AND_STORE(times, if (++t_ % save_t_ == 0) save_state());

  return times;
}

std::vector<double> CellModel::step_timed() {
  std::vector<double> times;
  times.reserve(6);

  TIME_AND_STORE(times, if (t_ % adh_t_ == 0) rearrange_adhesions(false));
  TIME_AND_STORE(times, if (t_ % fr_t_ == 0) update_frame());

  TIME_AND_STORE(times, update_dyn_nuc_field());
  TIME_AND_STORE(times, protrude_nuc(); retract_nuc(););

  TIME_AND_STORE(times, protrude(); retract(););

  TIME_AND_STORE(times, correct_concentrations());
  TIME_AND_STORE(times, diffuse_k0_adh());

  TIME_AND_STORE(times, if (++t_ % save_t_ == 0) save_state());

  return times;
}

void CellModel::set_output(const std::string filepath) {
  output_file_ = filepath;

  // Open file
  results_ = std::make_unique<HighFive::File>(
      output_file_, HighFive::File::ReadWrite | HighFive::File::Create |
                        HighFive::File::Truncate);
}

void CellModel::save_state() {
  if (output_file_ == "") {
    throw std::runtime_error(
        "Output file must be set (using cellModel.set_output()) before the "
        "state of the cell can be saved.");
  }

  // Append current time step (assuming all previous time steps have correct
  // data)
  append_dataset(*results_, next_index_, "t", t_);

  // Append data
  append_dataset(*results_, next_index_, "cell", cell_, true);
  append_dataset(*results_, next_index_, "nuc", nuc_, true);
  append_dataset(*results_, next_index_, "A", A_);
  append_dataset(*results_, next_index_, "I", I_);
  append_dataset(*results_, next_index_, "AC", AC_);
  append_dataset(*results_, next_index_, "IC", IC_);
  append_dataset(*results_, next_index_, "adh", adh_, true);
  append_dataset(*results_, next_index_, "F", F_);
  append_dataset(*results_, next_index_, "k0_adh", k0_adh_);
}

void CellModel::rearrange_adhesions(const bool bias, const bool rearrange_all) {
  /**
   * The logic behind this function is to move some number of adhesions to new
   * spots with polarization. To optimize this function, we perform as much
   * precomputation as possible. We generate a list of random indices of
   * adhesions to remove, and then generate an array to represent the CDF of A
   * within the frame. Then for each adhesion, we will generate a random
   * number and check if it is a valid target. If it is, we will use that as
   * the new adhesion, and if not we repeat the process.
   *
   * This optimizes the case in which all of the A values are somewhat
   * similar, making rejection sampling inefficient. Since each value should
   * have a very small difference in the CDF in this case, the chance that the
   * CDF has to be resampled is very low.
   */

  const int rearrange_adh = // number of adhesions to rearrange
      rearrange_all ? adh_num_ : int(adh_num_ * adh_frac_);
  const int rows = frame_row_end_ - frame_row_start_ + 1;
  const int cols = frame_col_end_ - frame_col_start_ + 1;
  const int frame_size = rows * cols;

  // generate indices to rearrange
  const std::vector<int> indices = generate_indices(rearrange_adh, 0, adh_num_);

  // precompute cumulative probability as array
  std::vector<double> A_lin(frame_size); // linearized version of A within frame
  std::vector<std::pair<int, int>> flat_pos(frame_size);
  double A_sum = 0; // total sum of A in frame
  if (bias) {
    for (int i = 0, r = frame_row_start_; r <= frame_row_end_; i++, r++) {
      for (int j = 0, c = frame_col_start_; c <= frame_col_end_; j++, c++) {
        if (env_.coeff(r, c) == 1 && cell_(r, c) == 1) {
          // if is valid attachment point, it is a valid place for new adhesion
          A_sum += A_(r, c);
        }
        A_lin[i * cols + j] = A_sum;
        flat_pos[i * cols + j] = {r, c};
      }
    }
  }

  if (!bias || A_sum == 0) {
    // if no A signal, assume uniform probability
    A_sum = 0;
    for (int i = 0, r = frame_row_start_; r <= frame_row_end_; i++, r++) {
      for (int j = 0, c = frame_col_start_; c <= frame_col_end_; j++, c++) {
        if (env_.coeff(r, c) == 1 && cell_(r, c) == 1) {
          A_sum += 1;
        }
        A_lin[i * cols + j] = A_sum;
        flat_pos[i * cols + j] = {r, c};
      }
    }
  }

  // iterate through adhesions
  for (int i = 0; i < rearrange_adh; i++) {
    // remove old adhesion
    const int idx = indices[i];
    adh_.coeffRef(adh_pos_(0, idx), adh_pos_(1, idx)) = 0;

    int r = -1, c = -1; // row and column for new adhesion to go to
    do {
      // generate random probability and find index in cumulative sum
      const double p = prob_dist(rng);
      const auto idx_it =
          std::lower_bound(A_lin.begin(), A_lin.end(), A_sum * p);
      const int idx = idx_it - A_lin.begin();

      // convert index to row and column
      const int cand_r = idx / cols + frame_row_start_;
      const int cand_c = idx % cols + frame_col_start_;

      // check new index valid
      if (adh_.coeff(cand_r, cand_c) != 1) {
        // env_ and cell_ should both always be satisfied, but to be safe
        r = cand_r;
        c = cand_c;
      }
    } while (r < 0 || c < 0);

    // add new adhesion
    adh_.coeffRef(r, c) = 1;
    adh_pos_(0, idx) = r;
    adh_pos_(1, idx) = c;
  }

  // smooth adhesion field
  update_adhesion_field();
}

void CellModel::init_adhesions() {
  /**
   * Initializes adhesions randomly within the cell boundary.
   * Sampling probability is weighted by the A-field, or uniform if A_sum == 0.
   */

  const int rows = frame_row_end_ - frame_row_start_ + 1;
  const int cols = frame_col_end_ - frame_col_start_ + 1;
  const int frame_size = rows * cols;

  // Reset adhesion matrix
  adh_.setZero();
  adh_pos_.resize(2, adh_num_);
  adh_pos_.setZero();

  // Precompute linearized A and CDF
  std::vector<double> A_lin(frame_size);
  std::vector<std::pair<int, int>> flat_pos(frame_size);
  double A_sum = 0;

  for (int i = 0, r = frame_row_start_; r <= frame_row_end_; ++r, ++i) {
    for (int j = 0, c = frame_col_start_; c <= frame_col_end_; ++c, ++j) {
      if (env_.coeff(r, c) == 1 && cell_(r, c) == 1) {
        A_sum += A_.coeff(r, c);
      }
      const int idx = i * cols + j;
      A_lin[idx] = A_sum;
      flat_pos[idx] = {r, c};
    }
  }

  // Handle uniform distribution if A_sum == 0
  if (A_sum == 0) {
    A_sum = 0;
    for (int i = 0, r = frame_row_start_; r <= frame_row_end_; ++r, ++i) {
      for (int j = 0, c = frame_col_start_; c <= frame_col_end_; ++c, ++j) {
        if (env_.coeff(r, c) == 1 && cell_(r, c) == 1) {
          A_sum += 1;
        }
        const int idx = i * cols + j;
        A_lin[idx] = A_sum;
        flat_pos[idx] = {r, c};
      }
    }
  }

  // Sample adhesions
  std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
  int placed = 0;

  while (placed < adh_num_) {
    const double p = prob_dist(rng);
    const auto idx_it = std::upper_bound(A_lin.begin(), A_lin.end(), A_sum * p);
    if (idx_it == A_lin.begin() || idx_it == A_lin.end())
      continue;

    const int idx = std::prev(idx_it) - A_lin.begin();
    const auto [r, c] = flat_pos[idx];

    if (adh_.coeff(r, c) == 1)
      continue; // already occupied

    adh_.coeffRef(r, c) = 1;
    adh_pos_(0, placed) = r;
    adh_pos_(1, placed) = c;
    placed++;
  }

  update_adhesion_field();
}

void CellModel::update_frame() {
  int min_row = std::numeric_limits<int>::max();
  int max_row = std::numeric_limits<int>::min();
  int min_col = std::numeric_limits<int>::max();
  int max_col = std::numeric_limits<int>::min();

  // only need to check inner outline of cell because those are the cell
  // boundaries
  for (int k = 0; k < inner_outline_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<int>::InnerIterator it(inner_outline_, k); it;
         ++it) {
      int i = it.row();
      int j = it.col();

      // Update bounding box
      if (cell_(i, j) != 0) {
        min_row = std::min(min_row, i);
        max_row = std::max(max_row, i);
        min_col = std::min(min_col, j);
        max_col = std::max(max_col, j);
      }
    }
  }

  frame_row_start_ = std::max(0, min_row - frame_padding_);
  frame_row_end_ = std::min(sim_rows_, max_row + frame_padding_);
  frame_col_start_ = std::max(0, min_col - frame_padding_);
  frame_col_end_ = std::min(sim_cols_, max_col + frame_padding_);
}

void CellModel::protrude_nuc() {
  /**
   * Protrude the nucleus. This function is split into 4 main sections:
   * - Calculate some coefficients for protrusion probabilities. We can
   *   calculate them early to prevent having to calculate again for each
   *   pixel.
   * - Get a random protrusion order.
   * - Calculate protrusion probabilities for each pixel and try protruding the
   *   nucleus in that direction.
   * - Update the nucleus outlines.
   */

  // precompute constants
  const double V_cor = 1.0 / (1 + std::exp((V_nuc_ - V0_nuc_) / T_nuc_));
  const double R = double(P_nuc_ * P_nuc_) / V_nuc_;
  const double R_cor = 1.0 / (1 + std::exp((R - R0_) / R_nuc_));
  const double n_diag = 1.0 / std::pow(M_SQRT2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // perform vectorized calculations over entire block
  const int r_start = nuc_min_r_ - 1, c_start = nuc_min_c_ - 1;
  const int rows = nuc_max_r_ - nuc_min_r_ + 3;
  const int cols = nuc_max_c_ - nuc_min_c_ + 3;

  // n_matrix holds the 'n' value for every pixel in the nucleus vicinity
  Eigen::MatrixXd n_matrix = Eigen::MatrixXd::Zero(rows, cols);
  auto n_acc = n_matrix.array();

  n_acc +=
      n_diag *
      (nuc_.block(r_start - 1, c_start - 1, rows, cols).array().cast<double>() +
       nuc_.block(r_start + 1, c_start - 1, rows, cols).array().cast<double>() +
       nuc_.block(r_start + 1, c_start + 1, rows, cols).array().cast<double>() +
       nuc_.block(r_start - 1, c_start + 1, rows, cols).array().cast<double>());
  n_acc +=
      (nuc_.block(r_start - 1, c_start, rows, cols).array().cast<double>() +
       nuc_.block(r_start, c_start - 1, rows, cols).array().cast<double>() +
       nuc_.block(r_start + 1, c_start, rows, cols).array().cast<double>() +
       nuc_.block(r_start, c_start + 1, rows, cols).array().cast<double>());

  // protrude logic
  std::vector<std::pair<int, int>> protrude_coords =
      randomize_nonzero(outline_nuc_, rng);

  for (auto &[r, c] : protrude_coords) {
    uint8_t config = encode_8(nuc_, r, c);
    if (outline_.coeff(r, c) == 1 || !protrude_conf_[config])
      continue;

    // lookup pre-calculated 'n'
    // NOTE: cells may change but n is still from initial state
    double n = n_matrix(r - r_start, c - c_start);

    const double w = std::pow(n / C, k_nuc_) * R_cor * V_cor *
                     (dyn_basal_ + (1 - dyn_basal_) * dyn_f_(r, c));

    if (prob_dist(rng) < w) {
      nuc_(r, c) = 1;

      AC_cor_sum_ -= AC_(r, c);
      AC_(r, c) = 0;
      IC_cor_sum_ -= IC_(r, c);
      IC_(r, c) = 0;
      FC_(r, c) = 0;
    }
  }

  update_nuc();
}

void CellModel::retract_nuc() {
  /**
   * The logic for this function is identical to protrude_nuc, but some of the
   * values are inverted for retraction.
   *
   * - The exponent for V_cor is negated
   * - counting the neighbors n we instead count the empty pixels
   * - dyn_f_ values are replaced with 1 - dyn_f_
   */

  // precompute constants
  const double V_cor = 1.0 / (1 + std::exp(-(V_nuc_ - V0_nuc_) / T_nuc_));
  const double R = double(P_nuc_ * P_nuc_) / V_nuc_;
  const double R_cor = 1.0 / (1 + std::exp((R - R0_) / R_nuc_));
  const double n_diag = 1.0 / std::pow(M_SQRT2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // perform vectorized calculations over entire block
  const int r_start = nuc_min_r_ - 1;
  const int c_start = nuc_min_c_ - 1;
  const int rows = nuc_max_r_ - nuc_min_r_ + 3;
  const int cols = nuc_max_c_ - nuc_min_c_ + 3;

  // n_matrix holds the 'n' value for every pixel in the nucleus vicinity
  Eigen::MatrixXd inv_nuc =
      1.0 - nuc_.block(r_start - 1, c_start - 1, rows + 2, cols + 2)
                .array()
                .cast<double>();

  Eigen::MatrixXd n_matrix = Eigen::MatrixXd::Zero(rows, cols);
  auto n_acc = n_matrix.array();

  // Diagonal neighbors (using the pre-inverted block)
  n_acc += n_diag * (inv_nuc.block(0, 0, rows, cols).array() + // top-left
                     inv_nuc.block(2, 0, rows, cols).array() + // bottom-left
                     inv_nuc.block(0, 2, rows, cols).array() + // top-right
                     inv_nuc.block(2, 2, rows, cols).array()   // bottom-right
                    );
  // Orthogonal neighbors
  n_acc += (inv_nuc.block(0, 1, rows, cols).array() + // top
            inv_nuc.block(1, 0, rows, cols).array() + // left
            inv_nuc.block(2, 1, rows, cols).array() + // bottom
            inv_nuc.block(1, 2, rows, cols).array()   // right
  );

  // retract logic
  std::vector<std::pair<int, int>> retract_coords =
      randomize_nonzero(inner_outline_nuc_, rng);

  for (auto &[r, c] : retract_coords) {
    uint8_t config = encode_8(nuc_, r, c);
    if (!retract_conf_[config])
      continue;

    // lookup pre-calculated 'n'
    // NOTE: cells may change but n is still from initial state
    double n = n_matrix(r - r_start, c - c_start);

    const double w =
        std::pow(n / C, k_nuc_) * R_cor * V_cor *
        (dyn_basal_ +
         (1 - dyn_basal_) * (1 - dyn_f_(r, c))); // inverted from protrusion

    if (prob_dist(rng) < w) {
      nuc_(r, c) = 0;

      // count number of neighbors and sum up values
      int n = 9 - nuc_.block<3, 3>(r - 1, c - 1)
                      .sum(); // number of cell pixels (non-nucleus)
      double AC = AC_.block<3, 3>(r - 1, c - 1).sum();
      double FC = FC_.block<3, 3>(r - 1, c - 1).sum();
      double IC = IC_.block<3, 3>(r - 1, c - 1).sum();

      AC_(r, c) = std::clamp(AC / n, AC_min_, AC_max_);
      AC_cor_sum_ += AC_(r, c);
      IC_(r, c) = IC / n;
      IC_cor_sum_ += IC_(r, c);
      FC_(r, c) = FC / n;
    }
  }

  update_nuc();
}

void CellModel::protrude_nuc_dep() {
  // calculate probability coefficients
  const double V_cor = 1.0 / (1 + std::exp((V_nuc_ - V0_nuc_) / T_nuc_));
  const double R = (P_nuc_ * P_nuc_) / V_nuc_; // WARN: Not casting to double
  const double R_cor = 1.0 / (1 + std::exp((R - R0_) / R_nuc_));
  const double n_diag = 1.0 / std::pow(M_SQRT2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // randomize protrude order
  std::vector<std::pair<int, int>> protrude_coords;

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    // NOTE: If debug, use non-random column-major order
    protrude_coords = get_nonzero(outline_nuc_);
  } else {
    protrude_coords = randomize_nonzero(outline_nuc_, rng);
  }

  // generate dynein field for protrusion probability
  generate_dyn_field(inner_outline_, outline_nuc_, false);

  // protrude
  for (int i = 0; i < protrude_coords.size(); i++) {
    auto [r, c] = protrude_coords[i];

    if (outline_.coeff(r, c) == 1 || !protrude_conf_[encode_8(nuc_, r, c)]

        ) // Check if protrusion would be valid
      continue;

    // get protrusion probability
    const double n = n_diag * (nuc_(r - 1, c - 1) + nuc_(r + 1, c - 1) +
                               nuc_(r + 1, c + 1) + nuc_(r - 1, c + 1)) +
                     nuc_(r - 1, c) + nuc_(r, c - 1) + nuc_(r + 1, c) +
                     nuc_(r, c + 1);
    const double w = std::pow(n / C, k_nuc_) * R_cor * V_cor *
                     (dyn_basal_ + (1 - dyn_basal_) * dyn_f_(r, c));

    // try protruding to this pixel
    const double p = prob_dist(rng);
    if (p < w) {
      nuc_(r, c) = 1;
      // Expand bounds incrementally
      nuc_min_r_ = std::min(nuc_min_r_, r);
      nuc_max_r_ = std::max(nuc_max_r_, r);
      nuc_min_c_ = std::min(nuc_min_c_, c);
      nuc_max_c_ = std::max(nuc_max_c_, c);
      AC_cor_sum_ -= AC_(r, c);
      AC_(r, c) = 0;
      IC_cor_sum_ -= IC_(r, c);
      IC_(r, c) = 0;
      FC_(r, c) = 0;
    }
  }

  // update nucleus outlines
  update_nuc();
}

void CellModel::retract_nuc_dep() {
  /**
   * The logic for this function is identical to protrude_nuc, but some of the
   * values are inverted for retraction.
   *
   * - The exponent for V_cor is negated
   * - counting the neighbors n we instead count the empty pixels
   * - dyn_f_ values are replaced with 1 - dyn_f_
   */
  // calculate probability coefficients
  const double V_cor = 1.0 / (1 + std::exp(-(V_nuc_ - V0_nuc_) / T_nuc_));
  const double R = (P_nuc_ * P_nuc_) / V_nuc_; // WARN: Not casting to double
  const double R_cor = 1.0 / (1 + std::exp((R - R0_) / R_nuc_));
  const double n_diag = 1.0 / std::pow(M_SQRT2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // randomize retract order
  std::vector<std::pair<int, int>> retract_coords;

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    // NOTE: If debug, use non-random column-major order
    retract_coords = get_nonzero(inner_outline_nuc_);
  } else {
    retract_coords = randomize_nonzero(inner_outline_nuc_, rng);
  }

  // generate dynein field for retraction probability
  generate_dyn_field(inner_outline_, inner_outline_nuc_, true);

  // retract
  bool recheck_bounds = false; // whether a retracted pixel was on the bounds
                               // and the nucleus bounds need to be rechecked
  for (int i = 0; i < retract_coords.size(); i++) {
    const auto [r, c] = retract_coords[i];

    if (!retract_conf_[encode_8(nuc_, r,
                                c)]) // Check if retraction would be valid
      continue;

    // get retraction probability
    const double n = n_diag * (!nuc_(r - 1, c - 1) + !nuc_(r + 1, c - 1) +
                               !nuc_(r + 1, c + 1) + !nuc_(r - 1, c + 1)) +
                     !nuc_(r - 1, c) + !nuc_(r, c - 1) + !nuc_(r + 1, c) +
                     !nuc_(r, c + 1);
    const double w = std::pow(n / C, k_nuc_) * R_cor * V_cor *
                     (dyn_basal_ + (1 - dyn_basal_) * dyn_f_(r, c));

    // try retracting this pixel
    const double p = prob_dist(rng);
    if (p < w) {
      nuc_(r, c) = 0;

      if (r == nuc_min_r_ || r == nuc_max_r_ || c == nuc_min_c_ ||
          c == nuc_max_c_)
        recheck_bounds = true;

      // count number of neighbors and sum up values
      int n = 8 - nuc_.block<3, 3>(r - 1, c - 1)
                      .sum(); // number of cell pixels (non-nucleus)
      double AC = AC_.block<3, 3>(r - 1, c - 1).sum() - AC_(r, c);
      double FC = FC_.block<3, 3>(r - 1, c - 1).sum() - FC_(r, c);
      double IC = IC_.block<3, 3>(r - 1, c - 1).sum() - IC_(r, c);

      AC_(r, c) = AC / n;
      AC_cor_sum_ += AC_(r, c);
      IC_(r, c) = IC / n;
      IC_cor_sum_ += IC_(r, c);
      FC_(r, c) = FC / n;
    }
  }

  // update nucleus outlines
  update_nuc(recheck_bounds);
}

void CellModel::generate_dyn_field(const SpMat_i &cell_outline,
                                   const SpMat_i &nuc_outline, bool retract) {
  dyn_f_.setZero();
  SpMat_i scaling{sim_rows_, sim_cols_};

  const int len = nuc_outline.nonZeros();
  int n = len / (retract ? 6 : 30);

  // Pre-compute nuc outline coordinates for faster iteration
  std::vector<std::pair<int, int>> nuc_coords;
  nuc_coords.reserve(nuc_outline.nonZeros());
  for (int j = 0; j < nuc_outline.outerSize(); j++) {
    for (SpMat_i::InnerIterator it_nuc(nuc_outline, j); it_nuc; ++it_nuc) {
      nuc_coords.push_back({it_nuc.row(), it_nuc.col()});
    }
  }

#ifdef USE_OPENMP
  // Collect cell outline coordinates for parallelization
  std::vector<std::pair<int, int>> cell_coords;
  cell_coords.reserve(cell_outline.nonZeros());
  for (int k = 0; k < cell_outline.outerSize(); k++) {
    for (SpMat_i::InnerIterator it(cell_outline, k); it; ++it) {
      cell_coords.push_back({it.row(), it.col()});
    }
  }

  // Thread-local accumulators for dyn_f and scaling
  const int num_threads = omp_get_max_threads();
  std::vector<Mat_d> dyn_f_local(num_threads,
                                 Mat_d::Zero(sim_rows_, sim_cols_));
  std::vector<SpMat_i> scaling_local(num_threads);
  for (int t = 0; t < num_threads; t++) {
    scaling_local[t] = SpMat_i(sim_rows_, sim_cols_);
  }

// Parallel loop over cell outline pixels
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();

#pragma omp for
    for (int idx = 0; idx < static_cast<int>(cell_coords.size()); idx++) {
      const int r = cell_coords[idx].first;
      const int c = cell_coords[idx].second;

      // Early exit if AC is too low
      const double ac_val = AC_(r, c);
      if (ac_val <= 0.1)
        continue;

      // get nucleus pixel closest to current pixel
      int min_dist2 = INT_MAX;
      int min_r = -1, min_c = -1;
      for (const auto &nc : nuc_coords) {
        const int dr = r - nc.first;
        const int dc = c - nc.second;
        const int dist2 = dr * dr + dc * dc;
        if (dist2 < min_dist2) {
          min_dist2 = dist2;
          min_r = nc.first;
          min_c = nc.second;
        }
      }

      const double dist_f = std::sqrt(min_dist2) * (ac_val - 0.1);
      const int r_start = std::max(min_r - n, 0);
      const int r_end = std::min(min_r + n, sim_rows_ - 1);
      const int c_start = std::max(min_c - n, 0);
      const int c_end = std::min(min_c + n, sim_cols_ - 1);

      for (int i = r_start; i <= r_end; ++i) {
        for (int j = c_start; j <= c_end; ++j) {
          if (nuc_outline.coeff(i, j) == 1) {
            dyn_f_local[thread_id](i, j) += dist_f;
            scaling_local[thread_id].coeffRef(i, j) += 1;
          }
        }
      }
    }
  }

  // Merge thread-local accumulators
  for (int t = 0; t < num_threads; t++) {
    dyn_f_ += dyn_f_local[t];
    scaling += scaling_local[t];
  }

#else
  // Single-threaded version
  for (int k = 0; k < cell_outline.outerSize(); k++) {
    for (SpMat_i::InnerIterator it(cell_outline, k); it; ++it) {
      const int r = it.row();
      const int c = it.col();

      // Early exit if AC is too low
      const double ac_val = AC_(r, c);
      if (ac_val <= 0.1)
        continue;

      // get nucleus pixel closest to current pixel
      int min_dist2 = INT_MAX;
      int min_r = -1, min_c = -1;
      for (const auto &nc : nuc_coords) {
        const int dr = r - nc.first;
        const int dc = c - nc.second;
        const int dist2 = dr * dr + dc * dc;
        if (dist2 < min_dist2) {
          min_dist2 = dist2;
          min_r = nc.first;
          min_c = nc.second;
        }
      }

      const double dist_f = std::sqrt(min_dist2) * (ac_val - 0.1);
      const int r_start = std::max(min_r - n, 0);
      const int r_end = std::min(min_r + n, sim_rows_ - 1);
      const int c_start = std::max(min_c - n, 0);
      const int c_end = std::min(min_c + n, sim_cols_ - 1);

      for (int i = r_start; i <= r_end; ++i) {
        for (int j = c_start; j <= c_end; ++j) {
          if (nuc_outline.coeff(i, j) == 1) {
            dyn_f_(i, j) += dist_f;
            scaling.coeffRef(i, j) += 1;
          }
        }
      }
    }
  }
#endif

  // normalize elements
  for (int k = 0; k < scaling.outerSize(); k++) {
    for (SpMat_i::InnerIterator it(scaling, k); it; ++it) {
      int r = it.row();
      int c = it.col();
      if (!retract) {
        dyn_f_(r, c) = std::min(dyn_f_(r, c) / scaling.coeff(r, c) / 60, 1.0);
      } else {
        dyn_f_(r, c) =
            std::max(1 - dyn_f_(r, c) / scaling.coeff(r, c) / 60, 0.0);
      }
    }
  }
}

void CellModel::protrude() {
  /**
   * This function attempts to protrude the cell. The logic of this function is
   * very similar to that of protrude_nuc, but the weight function is slightly
   * different and the values used are relative to the actin factor as opposed
   * to dynein factor.
   */

  // get probability coefficients
  const double V_cor = 1 / (1 + std::exp((V_ - V0_) / T_));
  const double A_max = A_.block(frame_row_start_, frame_col_start_,
                                frame_row_end_ - frame_row_start_ + 1,
                                frame_col_end_ - frame_col_start_ + 1)
                           .maxCoeff();
  const double AC_max = AC_.block(frame_row_start_, frame_col_start_,
                                  frame_row_end_ - frame_row_start_ + 1,
                                  frame_col_end_ - frame_col_start_ + 1)
                            .maxCoeff();
  const double n_diag = 1.0 / std::pow(M_SQRT2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // get random visiting order
  std::vector<std::pair<int, int>> protrude_coords;
  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    // NOTE: If debug, use non-random column-major order
    protrude_coords = get_nonzero(outline_);
  } else {
    protrude_coords = randomize_nonzero(outline_, rng);
  }

  // protrude
  for (int i = 0; i < protrude_coords.size(); i++) {
    auto &[r, c] = protrude_coords[i];

    if (!protrude_conf_[encode_8(cell_, r,
                                 c)]) // not valid protrude configuration
      continue;

    double w;
    if (outline_nuc_.coeff(r, c) == 1) {
      w = 1.0; // force push if nucleus is against edge of cell
    } else {
      double n = n_diag * (cell_(r - 1, c - 1) + cell_(r + 1, c - 1) +
                           cell_(r + 1, c + 1) + cell_(r - 1, c + 1)) +
                 cell_(r - 1, c) + cell_(r, c - 1) + cell_(r + 1, c) +
                 cell_(r, c + 1);
      int N = cell_.block<3, 3>(r - 1, c - 1).sum();
      double A_avg = (A_.block<3, 3>(r - 1, c - 1).sum() - A_(r, c)) / N;
      w = std::pow(n / C, k_) * V_cor *
          (1.0 - act_slope_ * (1.0 - A_avg / A_max)) *
          (adh_f_(r, c) * (adh_basal_ - 1.0) + 1.0);
    }

    // try protruding cell
    const double p = prob_dist(rng);
    if (p < w) {
      int N = cell_.block<3, 3>(r - 1, c - 1).sum();
      double A_avg = (A_.block<3, 3>(r - 1, c - 1).sum() - A_(r, c)) / N;
      double I_avg = (I_.block<3, 3>(r - 1, c - 1).sum() - I_(r, c)) / N;
      double F_avg = (F_.block<3, 3>(r - 1, c - 1).sum() - F_(r, c)) / N;
      double AC_avg = (AC_.block<3, 3>(r - 1, c - 1).sum() - AC_(r, c)) / N;
      double IC_avg = (IC_.block<3, 3>(r - 1, c - 1).sum() - IC_(r, c)) / N;
      double FC_avg = (FC_.block<3, 3>(r - 1, c - 1).sum() - FC_(r, c)) / N;

      cell_(r, c) = 1;
      A_(r, c) = A_avg;
      I_(r, c) = I_avg;
      F_(r, c) = F_avg;
      AC_(r, c) = AC_avg;
      IC_(r, c) = IC_avg;
      FC_(r, c) = FC_avg;

      // WARN: Make sure this sum is initialized properly
      A_cor_sum_ += A_(r, c);
      I_cor_sum_ += I_avg;
      AC_cor_sum_ += AC_(r, c);
      IC_cor_sum_ += IC_avg;
    }
  }

  // update cell
  update_cell();
}

void CellModel::retract() {
  /**
   * This function retracts pixels of the cell and is essentially the opposite
   * logic to the protrude() function. Note that higher adh_f_ results in
   */

  // get probability coefficients
  const double V_cor = 1 / (1 + std::exp(-(V_ - V0_) / T_));
  const double A_max = A_.block(frame_row_start_, frame_col_start_,
                                frame_row_end_ - frame_row_start_ + 1,
                                frame_col_end_ - frame_col_start_ + 1)
                           .maxCoeff();
  const double AC_max = AC_.block(frame_row_start_, frame_col_start_,
                                  frame_row_end_ - frame_row_start_ + 1,
                                  frame_col_end_ - frame_col_start_ + 1)
                            .maxCoeff();
  const double n_diag = 1.0 / std::pow(M_SQRT2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // get random visiting order
  std::vector<std::pair<int, int>> retract_coords;

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    retract_coords = get_nonzero(inner_outline_);
  } else {
    retract_coords = randomize_nonzero(inner_outline_, rng);
  }

  // retract
  for (int i = 0; i < retract_coords.size(); i++) {
    auto &[r, c] = retract_coords[i];

    if (!retract_conf_[encode_8(cell_, r,
                                c)]) // not valid retract configuration
      continue;
    if (nuc_(r, c) == 1) // can't retract nucleus
      continue;

    double n = n_diag * (!cell_(r - 1, c - 1) + !cell_(r + 1, c - 1) +
                         !cell_(r + 1, c + 1) + !cell_(r - 1, c + 1)) +
               !cell_(r - 1, c) + !cell_(r, c - 1) + !cell_(r + 1, c) +
               !cell_(r, c + 1);
    int N = cell_.block<3, 3>(r - 1, c - 1).sum() - 1;
    double A_avg = (A_.block<3, 3>(r - 1, c - 1).sum() - A_(r, c)) / N;
    double w = std::pow(n / C, k_) * V_cor *
               (1.0 - act_slope_ * A_avg / A_max) * adh_f_(r, c);

    // try retracting pixel
    const double p = prob_dist(rng);
    if (p < w) {
      cell_(r, c) = 0;
      // WARN: Make sure this sum is initialized properly
      A_cor_sum_ -= A_(r, c);
      I_cor_sum_ -= I_(r, c);
      AC_cor_sum_ -= AC_(r, c);
      IC_cor_sum_ -= IC_(r, c);

      A_(r, c) = 0;
      I_(r, c) = 0;
      F_(r, c) = 0;
      AC_(r, c) = 0;
      IC_(r, c) = 0;
      FC_(r, c) = 0;
    }
  }

  // update cell
  update_cell();
}

void CellModel::set_cell(const Mat_i cell) {
  cell_ = cell;
  update_cell(true);
  update_frame();

  V0_ = V_;
}

void CellModel::set_nuc(const Mat_i nuc) {
  nuc_ = nuc;
  update_nuc();
  V0_nuc_ = V_nuc_;
}

void CellModel::set_adh(const SpMat_i adh) {
  adh_ = adh;
  adh_num_ = adh.nonZeros();
  adh_pos_ = Mat_i(2, adh_num_);

  // Set adhesion coords
  int i = 0; // index in adh positions
  for (int k = 0; k < adh.outerSize(); k++) {
    for (SpMat_i::InnerIterator it(adh, k); it; ++it) {
      adh_pos_(0, i) = it.row();
      adh_pos_(1, i) = it.col();
      i++;
    }
  }
}

void CellModel::set_A(const Mat_d A) { A_ = A; }

void CellModel::set_AC(const Mat_d AC) { AC_ = AC; }

void CellModel::set_I(const Mat_d I) { I_ = I; }

void CellModel::set_IC(const Mat_d IC) { IC_ = IC; }

void CellModel::set_F(const Mat_d F) { F_ = F; }

void CellModel::set_FC(const Mat_d FC) { FC_ = FC; }

void CellModel::set_env(const SpMat_i env) { env_ = env; }

void CellModel::set_seed(const int seed) { seed_ = seed; }

void CellModel::initialize_helpers() {
  // Initialize correction variables
  A_cor_sum_ = 0;
  I_cor_sum_ = 0;
  AC_cor_sum_ = 0;
  IC_cor_sum_ = 0;

  // Initialize nucleus bounds (will be computed on first update_nuc call)
  nuc_min_r_ = sim_rows_;
  nuc_max_r_ = 0;
  nuc_min_c_ = sim_cols_;
  nuc_max_c_ = 0;

  // Get valid configurations
  update_valid_conf();

  // Initialize random
  rng = std::mt19937(seed_);
  prob_dist = std::uniform_real_distribution<>(0.0, 1.0);

  // Initialize matrices
  outline_ = SpMat_i(sim_rows_, sim_cols_);
  inner_outline_ = SpMat_i(sim_rows_, sim_cols_);
  outline_nuc_ = SpMat_i(sim_rows_, sim_cols_);
  inner_outline_nuc_ = SpMat_i(sim_rows_, sim_cols_);
  A_ = Mat_d(sim_rows_, sim_cols_);
  AC_ = Mat_d(sim_rows_, sim_cols_);
  I_ = Mat_d(sim_rows_, sim_cols_);
  IC_ = Mat_d(sim_rows_, sim_cols_);
  F_ = Mat_d(sim_rows_, sim_cols_);
  FC_ = Mat_d(sim_rows_, sim_cols_);
  env_ = SpMat_i(sim_rows_, sim_cols_);
  adh_ = SpMat_i(sim_rows_, sim_cols_);
  adh_pos_ = Mat_i(2, adh_num_);
  adh_f_ = Mat_d(sim_rows_, sim_cols_);
  dyn_f_ = Mat_d(sim_rows_, sim_cols_);
  k0_adh_ = Mat_d(sim_rows_, sim_cols_);
}

void CellModel::update_nuc(bool recheck_bounds) {
  /**
   * Iterate through nucleus pixels and add 4-neighbors that are not nucleus to
   * outer outline. Uses tracked bounds for optimized iteration.
   *
   * @param recheck_bounds if true, rescans to find new nucleus bounds after
   * retraction. If false, uses tracked bounds for iteration.
   */
  const int DR[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
  const int DC[8] = {0, 0, -1, 1, -1, 1, -1, 1};

  // Clear outlines
  outline_nuc_.setZero();
  inner_outline_nuc_.setZero();
  V_nuc_ = 0;

  // Initialize bounds on first call (when bounds are invalid)
  if (nuc_max_r_ == 0 && nuc_max_c_ == 0) {
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      for (int i = frame_row_start_; i <= frame_row_end_; i++) {
        if (nuc_(i, j) == 1) {
          nuc_min_r_ = std::min(nuc_min_r_, i);
          nuc_max_r_ = std::max(nuc_max_r_, i);
          nuc_min_c_ = std::min(nuc_min_c_, j);
          nuc_max_c_ = std::max(nuc_max_c_, j);
        }
      }
    }
  }

  // If recheck_bounds, scan within current bounds to find new bounds
  if (recheck_bounds) {
    int new_min_r = sim_rows_, new_max_r = 0;
    int new_min_c = sim_cols_, new_max_c = 0;
    for (int j = nuc_min_c_; j <= nuc_max_c_; j++) {
      for (int i = nuc_min_r_; i <= nuc_max_r_; i++) {
        if (nuc_(i, j) == 1) {
          new_min_r = std::min(new_min_r, i);
          new_max_r = std::max(new_max_r, i);
          new_min_c = std::min(new_min_c, j);
          new_max_c = std::max(new_max_c, j);
        }
      }
    }
    nuc_min_r_ = new_min_r;
    nuc_max_r_ = new_max_r;
    nuc_min_c_ = new_min_c;
    nuc_max_c_ = new_max_c;
  }

  // Iterate within current bounds (+1 margin for outline detection)
  const int row_start = std::max(frame_row_start_, nuc_min_r_ - 1);
  const int row_end = std::min(frame_row_end_, nuc_max_r_ + 1);
  const int col_start = std::max(frame_col_start_, nuc_min_c_ - 1);
  const int col_end = std::min(frame_col_end_, nuc_max_c_ + 1);

  // #pragma omp parallel
  {
    std::unordered_set<std::pair<int, int>, pair_hash> local_inner, local_outer;

    // Iterate through nucleus pixels within bounding box
    // #pragma omp for nowait
    for (int i = row_start; i <= row_end; i++) {
      for (int j = col_start; j <= col_end; j++) {
        if (nuc_(i, j) == 0)
          continue;
        V_nuc_++;

        bool is_inner = false;
        for (int k = 0; k < 8; k++) {
          const int nr = i + DR[k];
          const int nc = j + DC[k];
          if (nr < 0 || nr >= sim_rows_ || nc < 0 || nc >= sim_cols_)
            continue;
          if (nuc_(nr, nc) == 0) {
            is_inner = true;
            local_outer.insert({nr, nc});
            break;
          }
        }
        if (is_inner)
          local_inner.insert({i, j});
      }
    }

    // update outlines
    // #pragma omp critical
    {
      for (auto &[r, c] : local_inner) {
        inner_outline_nuc_.coeffRef(r, c) = 1;
      }
      for (auto &[r, c] : local_outer) {
        outline_nuc_.coeffRef(r, c) = 1;
      }
    }
  }

  // update nucleus volume and perimeter
  P_nuc_ = outline_4(outline_nuc_, nuc_, sim_rows_, sim_cols_);
}

void CellModel::update_cell() { update_cell(false); }

void CellModel::update_cell(const bool full) {
  /**
   * Iterate through cell pixels and add 4-neighbors that are not part of the
   * cell to outer outline.
   */
  const int DR[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
  const int DC[8] = {0, 0, -1, 1, -1, 1, -1, 1};

  // Clear outlines
  outline_.setZero();
  inner_outline_.setZero();

  // Set bounds
  int row_start = full ? 0 : frame_row_start_;
  int row_end = full ? sim_rows_ - 1 : frame_row_end_;
  int col_start = full ? 0 : frame_col_start_;
  int col_end = full ? sim_cols_ - 1 : frame_col_end_;

  // #pragma omp parallel
  {
    std::unordered_set<std::pair<int, int>, pair_hash> local_inner, local_outer;

    // Iterate through cell pixels
    // #pragma omp for nowait
    for (int i = row_start; i <= row_end; i++) {
      for (int j = col_start; j <= col_end; j++) {
        if (cell_(i, j) == 0)
          continue;
        for (int k = 0; k < 8; k++) {
          const int nr = i + DR[k];
          const int nc = j + DC[k];
          if (nr < 0 || nr >= sim_rows_ || nc < 0 || nc >= sim_cols_)
            continue;
          if (cell_(nr, nc) == 0) {
            local_inner.insert({i, j});
            local_outer.insert({nr, nc});
          }
        }
      }
    }

    // update outlines
    // #pragma omp critical
    {
      for (auto &[r, c] : local_inner) {
        inner_outline_.coeffRef(r, c) = 1;
      }
      for (auto &[r, c] : local_outer) {
        outline_.coeffRef(r, c) = 1;
      }
    }
  }

  // update cell volume and perimeter
  V_ = (cell_.array() != 0).count();
  P_ = outline_4(outline_, cell_, sim_rows_, sim_cols_);
}

void CellModel::correct_concentrations() {
  // Calculate amount of signal that needs to be distributed from all pixels
  const double A_dist = A_cor_sum_ / V_;
  const double I_dist = I_cor_sum_ / V_;
  const double AC_dist = AC_cor_sum_ / (V_ - V_nuc_);
  const double IC_dist = IC_cor_sum_ / (V_ - V_nuc_);

  // #pragma omp parallel for collapse(2)
  for (int j = frame_col_start_; j <= frame_col_end_; j++) {
    for (int i = frame_row_start_; i <= frame_row_end_; i++) {
      if (cell_(i, j) == 1) {
        A_(i, j) -= A_dist;
        I_(i, j) -= I_dist;
      }
      if (cell_(i, j) == 1 && nuc_(i, j) == 0) {
        AC_(i, j) -= AC_dist;
        IC_(i, j) -= IC_dist;
      }
    }
  }

  A_cor_sum_ = 0;
  I_cor_sum_ = 0;
  AC_cor_sum_ = 0;
  IC_cor_sum_ = 0;
}

void CellModel::diffuse_k0_adh() {
  // precompute coefficients
  const double inv_dx2 = 1.0 / (dx_ * dx_);
  const double dda = DA_ * inv_dx2;
  const double ddi = DI_ * inv_dx2;
  const double dt_dda = dt_ * dda;
  const double dt_ddi = dt_ * ddi;
  const double s2C = 0.05;
  const double A0_3 = A0_ * A0_ * A0_;

  // temporary variables for update
  Mat_d A_new(sim_rows_, sim_cols_);
  Mat_d I_new(sim_rows_, sim_cols_);
  Mat_d F_new(sim_rows_, sim_cols_);
  Mat_d AC_new(sim_rows_, sim_cols_);
  Mat_d IC_new(sim_rows_, sim_cols_);
  Mat_d FC_new(sim_rows_, sim_cols_);

  for (int k = 0; k < diff_t_; k++) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = frame_row_start_; i <= frame_row_end_; i++) {
      // get raw pointers for faster access
      const double *rA = &A_(i, 0);
      const double *rA_up = &A_(i - 1, 0);
      const double *rA_down = &A_(i + 1, 0);

      const double *rI = &I_(i, 0);
      const double *rI_up = &I_(i - 1, 0);
      const double *rI_down = &I_(i + 1, 0);

      const double *rAC = &AC_(i, 0);
      const double *rAC_up = &AC_(i - 1, 0);
      const double *rAC_down = &AC_(i + 1, 0);

      const double *rIC = &IC_(i, 0);
      const double *rIC_up = &IC_(i - 1, 0);
      const double *rIC_down = &IC_(i + 1, 0);

      const double *rF = &F_(i, 0);
      const double *rFC = &FC_(i, 0);
      const double *rK0 = &k0_adh_(i, 0);

      const int *rCell = &cell_(i, 0);
      const int *rCell_up = &cell_(i - 1, 0);
      const int *rCell_down = &cell_(i + 1, 0);

      const int *rNuc = &nuc_(i, 0);
      const int *rNuc_up = &nuc_(i - 1, 0);
      const int *rNuc_down = &nuc_(i + 1, 0);

      for (int j = frame_col_start_; j <= frame_col_end_; j++) {
        double cell_val = (double)rCell[j];
        if (cell_val == 0.0)
          continue; // skip empty space

        double nuc_val = (double)rNuc[j];
        double cyto_val = cell_val - nuc_val; // 1.0 if cytosol, 0.0 if nucleus

        double a = rA[j];
        double a3 = a * a * a;
        double reaction_a = (rK0[j] + gamma_ * a3 / (A0_3 + a3)) * rI[j] -
                            delta_ * (s1_ + s2_ * rF[j] / (F0_ + rF[j])) * a;

        // laplacian calculatiosn
        double lapA = (double)rCell_down[j] * (rA_down[j] - a) +
                      (double)rCell_up[j] * (rA_up[j] - a) +
                      (double)rCell[j + 1] * (rA[j + 1] - a) +
                      (double)rCell[j - 1] * (rA[j - 1] - a);

        double lapI = (double)rCell_down[j] * (rI_down[j] - rI[j]) +
                      (double)rCell_up[j] * (rI_up[j] - rI[j]) +
                      (double)rCell[j + 1] * (rI[j + 1] - rI[j]) +
                      (double)rCell[j - 1] * (rI[j - 1] - rI[j]);

        A_new(i, j) = a + dt_ * reaction_a + dt_dda * lapA;
        I_new(i, j) = rI[j] + dt_ * (-reaction_a) + dt_ddi * lapI;
        F_new(i, j) = rF[j] + dt_ * (eps_ * (kn_ * a - ks_ * rF[j]));

        if (cyto_val > 0.0) {
          double ac = rAC[j];
          double ac3 = ac * ac * ac;
          double reaction_ac =
              (k0_ + gamma_ * ac3 / (A0_3 + ac3)) * rIC[j] -
              delta_ * (s1_ + s2C * rFC[j] / (F0_ + rFC[j])) * ac;

          // helper function for nucleus laplacian
          auto get_c_boundary = [&](int row_offset, int col_offset) {
            if (row_offset == 1)
              return (double)(rCell_down[j] - rNuc_down[j]);
            if (row_offset == -1)
              return (double)(rCell_up[j] - rNuc_up[j]);
            if (col_offset == 1)
              return (double)(rCell[j + 1] - rNuc[j + 1]);
            return (double)(rCell[j - 1] - rNuc[j - 1]);
          };

          double lapAC = get_c_boundary(1, 0) * (rAC_down[j] - ac) +
                         get_c_boundary(-1, 0) * (rAC_up[j] - ac) +
                         get_c_boundary(0, 1) * (rAC[j + 1] - ac) +
                         get_c_boundary(0, -1) * (rAC[j - 1] - ac);

          double lapIC = get_c_boundary(1, 0) * (rIC_down[j] - rIC[j]) +
                         get_c_boundary(-1, 0) * (rIC_up[j] - rIC[j]) +
                         get_c_boundary(0, 1) * (rIC[j + 1] - rIC[j]) +
                         get_c_boundary(0, -1) * (rIC[j - 1] - rIC[j]);

          AC_new(i, j) = ac + dt_ * reaction_ac + dt_dda * lapAC;
          IC_new(i, j) = rIC[j] + dt_ * (-reaction_ac) + dt_ddi * lapIC;
          FC_new(i, j) = rFC[j] + dt_ * (eps_ * (kn_ * ac - ks_ * rFC[j]));
        } else {
          // keep old values
          AC_new(i, j) = rAC[j];
          IC_new(i, j) = rIC[j];
          FC_new(i, j) = rFC[j];
        }
      }
    }

    std::swap(A_, A_new);
    std::swap(I_, I_new);
    std::swap(F_, F_new);
    std::swap(AC_, AC_new);
    std::swap(IC_, IC_new);
    std::swap(FC_, FC_new);
  }
}

void CellModel::update_dyn_nuc_field() {
  /**
   * This function calculates a field of dynein factor influence from within
   * the cell by using BFS, then normalizes the values to be in the range (0,
   * 1) for protrusion/retraction probabilities.
   *
   * The logic works by using BFS to create a map of which pixel each other
   * pixel is propagated from. This produces the best performance and although
   * doesn't share signals between pixels equidistant to the nucleus,
   * shouldn't matter because the signals will later be smoothed using a
   * Gaussian kernel anyway (not done in this function because only a small
   * subset of values require smoothing).
   *
   * Normalization is achieved using a sigmoid function to ensure values are
   * in the range (0, 1) so that retraction probabilities can be 1 -
   * protrusion probabilities. The sigmoid is centered at the average so that
   * if all values are similar, it won't just result in all of the values
   * being high (close to 1).
   */

  // helper constants
  const int DR[4] = {1, -1, 0, 0};
  const int DC[4] = {0, 0, 1, -1};

  // generate random starting order
  std::vector<std::pair<int, int>> nuc_coords =
      randomize_nonzero(inner_outline_nuc_, rng);

  Mat_i parent_idx = Mat_i::Constant(sim_rows_, sim_cols_, -1);

  std::vector<std::pair<int, int>> traversal_order;
  traversal_order.reserve(sim_rows_ * sim_cols_ / 4);

  for (auto &p : nuc_coords) {
    parent_idx(p.first, p.second) = -2; // Mark as seed
    traversal_order.push_back(p);
  }

  Mat_i dist{sim_rows_, sim_cols_};
  Mat_i scaling{sim_rows_, sim_cols_};

  // clear previous signals
  dyn_f_.setZero();

  size_t head = 0;
  while (head < traversal_order.size()) {
    auto [r, c] = traversal_order[head++];
    if (inner_outline_.coeff(r, c) == 1) {
      dyn_f_(r, c) = AC_(r, c);
      scaling(r, c) = 1;
    }
    for (int i = 0; i < 4; i++) {
      int nr = r + DR[i];
      int nc = c + DC[i];
      std::pair<int, int> next(nr, nc);
      if (nr < frame_row_start_ || nr > frame_row_end_ ||
          nc < frame_col_start_ || nc > frame_col_end_ || cell_(nr, nc) == 0 ||
          nuc_.coeffRef(nr, nc) == 1)
        continue;
      if (parent_idx(nr, nc) == -1) {
        parent_idx(nr, nc) = r * sim_cols_ + c; // Store parent
        traversal_order.push_back({nr, nc});
      }
    }
  }

  // propagate signals towards nucleus
  for (auto it = traversal_order.rbegin(); it != traversal_order.rend(); ++it) {
    int r = it->first, c = it->second;
    int p_idx = parent_idx(r, c);
    if (p_idx >= 0) {
      int pr = p_idx / sim_cols_;
      int pc = p_idx % sim_cols_;
      dyn_f_(pr, pc) += dyn_f_(r, c);
      scaling(pr, pc) += scaling(r, c);
    }
  }

  // normalization
  const int rows = nuc_max_r_ - nuc_min_r_ + 1;
  const int cols = nuc_max_c_ - nuc_min_c_ + 1;
  auto f_block = dyn_f_.block(nuc_min_r_, nuc_min_c_, rows, cols);
  auto s_block = scaling.block(nuc_min_r_, nuc_min_c_, rows, cols);

  f_block.array() =
      (s_block.array() == 0)
          .select(0.0, (f_block.array() / s_block.array().cast<double>()) *
                           dyn_scale_);

  // blur dyn_f
  gaussian_blur(dyn_f_, nuc_min_r_, nuc_max_r_, nuc_min_c_, nuc_max_c_,
                dyn_sigma_);
}

void CellModel::update_adhesion_field() {
  /**
   * Update adhesion field logic. Works by computing normalization variables
   * at each adhesion, then applying a weighted Gaussian over every pixel from
   * the adhesions. Then inverts and normalizes adh_f, and used to calculate
   * k0_adh with IDW.
   */

  // Precompute constants
  const double sigma_2 = adh_sigma_ * adh_sigma_;
  const double ampl = 1 / (2 * M_PI * sigma_2);

  // Position of adhesions as doubles
  Mat_d adh_pos_d = adh_pos_.cast<double>();

  // Precompute adh_g at adhesion positions for IDW calculation
  Vec_d norm_sq_adh = adh_pos_d.colwise().squaredNorm();
  Mat_d dist2_mat_adh =
      ((-2 * adh_pos_d.transpose() * adh_pos_d).colwise() + norm_sq_adh)
          .rowwise() +
      norm_sq_adh.transpose();
  Vec_d adh_g_at_adhesions =
      ((-dist2_mat_adh.array() / (2.0 * sigma_2)).exp().rowwise().sum()) * ampl;

  // Calculate adh_g at all frame pixels and find the maximum value for
  // normalization
  double max_adh_g = 0;
  for (int i = frame_row_start_; i <= frame_row_end_; i++) {
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      if (adh_.coeff(i, j) == 1) {
        // At adhesion sites, set adh_g to max and k0_adh to k0
        k0_adh_(i, j) = k0_;
        adh_f_(i, j) = 0; // adh_f = 0 at adhesions
        continue;
      }

      // Calculate distance squared to all adhesions
      Arr_d dr = adh_pos_.row(0).cast<double>().array() - i;
      Arr_d dc = adh_pos_.row(1).cast<double>().array() - j;
      Arr_d dist2 = dr.square() + dc.square();
      dist2 = dist2.max(1e-12); // Avoid division by zero

      // Calculate local Gaussian intensity
      double local_gaussian = ampl * ((-dist2 / (2.0 * sigma_2)).exp().sum());
      adh_f_(i, j) = local_gaussian;

      // Track max for normalization
      if (local_gaussian > max_adh_g) {
        max_adh_g = local_gaussian;
      }
    }
  }

  // Normalize and invert to get adh_f, calculate k0_adh
  if (max_adh_g > 0) {
    for (int i = frame_row_start_; i <= frame_row_end_; i++) {
      for (int j = frame_col_start_; j <= frame_col_end_; j++) {
        if (adh_.coeff(i, j) == 1) {
          // Already set above
          continue;
        }

        double local_gaussian = adh_f_(i, j);
        double adh_g_normalized = local_gaussian / max_adh_g;
        adh_f_(i, j) = 1.0 - adh_g_normalized;

        Arr_d dr = adh_pos_.row(0).cast<double>().array() - i;
        Arr_d dc = adh_pos_.row(1).cast<double>().array() - j;
        Arr_d dist2 = dr.square() + dc.square();
        dist2 = dist2.max(1e-12);

        double norm_numer = (adh_g_at_adhesions.array() / dist2).sum();
        double norm_denom = (1.0 / dist2).sum();

        k0_adh_(i, j) = (k0_ - k0_min_) * k0_scalar_ *
                            (local_gaussian / norm_numer) * norm_denom +
                        k0_min_;
      }
    }
  }
}

const uint8_t CellModel::encode_8(Mat_i &mat, const int r, const int c) {
  const int *ptr = &mat(r, c);
  const int step = mat.outerStride();

  // using offsets based on Eigen's memory layout
  // r-1, c-1 is ptr - 1 - step
  // r+1, c+1 is ptr + 1 + step
  return (ptr[-1 - step] & 1) << 0 | (ptr[-1] & 1) << 1 |
         (ptr[-1 + step] & 1) << 2 | (ptr[step] & 1) << 3 |
         (ptr[1 + step] & 1) << 4 | (ptr[1] & 1) << 5 |
         (ptr[1 - step] & 1) << 6 | (ptr[-step] & 1) << 7;
}

const bool CellModel::is_valid_config_prot(uint8_t conf) {
  // Diagonal-only connections (L-shaped corners without edge support)
  bool diag1 =
      (conf & (1 << 0)) && !(conf & (1 << 1)) && !(conf & (1 << 7)); // top-left
  bool diag2 = (conf & (1 << 2)) && !(conf & (1 << 1)) &&
               !(conf & (1 << 3)); // top-right
  bool diag3 = (conf & (1 << 4)) && !(conf & (1 << 3)) &&
               !(conf & (1 << 5)); // bottom-right
  bool diag4 = (conf & (1 << 6)) && !(conf & (1 << 5)) &&
               !(conf & (1 << 7)); // bottom-left

  if (diag1 || diag2 || diag3 || diag4)
    return false;

  // Pinch cases
  bool vertical_pinch = (conf & (1 << 1)) && (conf & (1 << 5)) &&
                        !(conf & (1 << 3)) && !(conf & (1 << 7));
  bool horizontal_pinch = (conf & (1 << 3)) && (conf & (1 << 7)) &&
                          !(conf & (1 << 1)) && !(conf & (1 << 5));

  if (vertical_pinch || horizontal_pinch)
    return false;

  return true;
}

const bool CellModel::is_valid_config_retr(uint8_t conf) {
  // Diagonal-only connections (L-shaped corners without edge support)
  bool diag1 =
      !(conf & (1 << 0)) && (conf & (1 << 1)) && (conf & (1 << 7)); // top-left
  bool diag2 =
      !(conf & (1 << 2)) && (conf & (1 << 1)) && (conf & (1 << 3)); // top-right
  bool diag3 = !(conf & (1 << 4)) && (conf & (1 << 3)) &&
               (conf & (1 << 5)); // bottom-right
  bool diag4 = !(conf & (1 << 6)) && (conf & (1 << 5)) &&
               (conf & (1 << 7)); // bottom-left

  if (diag1 || diag2 || diag3 || diag4)
    return false;

  // Pinch cases
  bool vertical_pinch = (conf & (1 << 1)) && (conf & (1 << 5)) &&
                        !(conf & (1 << 3)) && !(conf & (1 << 7));
  bool horizontal_pinch = (conf & (1 << 3)) && (conf & (1 << 7)) &&
                          !(conf & (1 << 1)) && !(conf & (1 << 5));

  if (vertical_pinch || horizontal_pinch)
    return false;

  return true;
}

void CellModel::update_valid_conf() {
  // Create protrusion configurations
  for (int i = 0; i < (1 << 8); i++) {
    if (is_valid_config_prot(i)) {
      protrude_conf_[i] = true;
    }
  }

  // Create retraction configurations
  for (int i = 0; i < (1 << 8); i++) {
    if (is_valid_config_retr(i)) {
      retract_conf_[i] = false;
    }
  }
}

const std::vector<int> CellModel::generate_indices(const int n, const int lb,
                                                   const int ub) {
  if (ub - lb < n) {
    throw std::runtime_error("Bounds must be at least as large as the number "
                             "of indices to generate.");
  }

  std::vector<int> arr(ub - lb);
  for (int i = 0, v = lb; v < ub; i++, v++) {
    arr[i] = v;
  }

  std::shuffle(arr.begin(), arr.end(), rng);

  return std::vector<int>(arr.begin(), arr.begin() + n);
}

} // namespace dynein_cell_model

#endif
