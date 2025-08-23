#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include <dynein_cell_model/dynein_cell_model.hpp>

struct pair_hash {
  std::size_t operator()(const std::pair<int, int>& p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

namespace dynein_cell_model {
Mat_i matrix_from_mask(std::string filepath, cv::Vec3b color) {
  cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Could not load the image at: " + filepath);
  }

  Mat_i mat(image.rows, image.cols);

  for (int c = 0; c < image.cols; c++) {
    for (int r = 0; r < image.rows; r++) {
      if (image.at<cv::Vec3b>() == color) {
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
  dyn_norm_k_ = config["dyn_norm_k"].as<double>();
  dyn_sigma_ = config["dyn_sigma"].as<double>();
  dyn_kernel_size_ = config["dyn_kernel_size"].as<int>();
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
  config["dyn_norm_k"] = dyn_norm_k_;
  config["dyn_sigma"] = dyn_sigma_;
  config["dyn_kernel_size"] = dyn_kernel_size_;
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

  std::ofstream fout(dest_file);
  fout << config;
}

CellModel::CellModel() {
  CellModelConfig config;
  update_config(config);
  initialize_helpers();
}

CellModel::CellModel(CellModelConfig config) {
  update_config(config);
  initialize_helpers();
}

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
  dyn_norm_k_ = config.dyn_norm_k_;
  dyn_sigma_ = config.dyn_sigma_;
  dyn_kernel_size_ = config.dyn_kernel_size_;
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

  // update variables that depend on config
  initialize_helpers();
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
  config.dyn_norm_k_ = dyn_norm_k_;
  config.dyn_sigma_ = dyn_sigma_;
  config.dyn_kernel_size_ = dyn_kernel_size_;
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
  if (t_ % adh_t_ == 0) {
    rearrange_adhesions();
  }

  if (t_ % fr_t_ == 0) {
    update_frame();
  }

  protrude_nuc();
  retract_nuc();

  protrude();
  retract();

  correct_concentrations();
  diffuse_k0_adh();
  update_dyn_nuc_field();

  if (t_ % save_t_ == 0) {
    save_state(save_dir_);
  }

  t_++;
}

void CellModel::save_state(std::string dirname) {
  // TODO: Implement save_state
}

void CellModel::rearrange_adhesions() {
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

  const int rearrange_adh = int(adh_num_ * adh_frac_); // number of adhesions to rearrange
  const int rows = frame_row_end_ - frame_row_start_ + 1;
  const int cols = frame_col_end_ - frame_col_start_ + 1;
  const int frame_size = rows * cols;

  // generate indices to rearrange
  const std::vector<int> indices = generate_indices(rearrange_adh, 0, adh_num_);

  // precompute cumulative probability as array
  std::vector<double> A_lin(frame_size); // linearized version of A within frame
  std::vector<std::pair<int, int>> flat_pos(frame_size);
  double A_sum = 0; // total sum of A in frame
  for (int i = 0, r = frame_row_start_; r <= frame_row_end_; i++, r++) {
    for (int j = 0, c = frame_col_start_; c <= frame_col_end_; j++, c++) {
      if (env_.coeff(r, c) == 1) {
        // if is valid attachment point, it is a valid place for new adhesion
        A_sum += A_(r, c);
      }
      A_lin[i * cols + j] = A_sum;
      flat_pos[i * cols + j] = {r, c};
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
      const auto idx_it = prev(std::upper_bound(A_lin.begin(), A_lin.end(), A_sum * p));
      const int idx = idx_it - A_lin.begin();

      // convert index to row and column
      const int cand_r = idx / cols;
      const int cand_c = idx % cols;

      // check new index valid
      if (adh_.coeff(cand_r, cand_c) != 1 && env_.coeff(cand_r, cand_c) == 1 && cell_(cand_r, cand_c) == 1) {
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

void CellModel::update_frame() {
  int min_row = std::numeric_limits<int>::max();
  int max_row = std::numeric_limits<int>::min();
  int min_col = std::numeric_limits<int>::max();
  int max_col = std::numeric_limits<int>::min();

  // only need to check inner outline of cell because those are the cell boundaries
  for (int k = 0; k < inner_outline_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<int>::InnerIterator it(inner_outline_, k); it; ++it) {
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

  // calculate probability coefficients
  const double V_cor = 1.0 / (1 + std::exp((V_nuc_ - V0_nuc_) / T_nuc_));
  const double R = double(P_nuc_ * P_nuc_) / V_nuc_;
  const double R_cor = 1.0 / (1 + std::exp((R - R0_) / R_nuc_));
  const double n_diag = 1.0 / std::pow(M_SQRT1_2, g_);
  const double C = 4.0 * (1.0 + n_diag);
   
  // randomize protrude order
  std::vector<std::pair<int, int>> protrude_coords = randomize_nonzero(outline_nuc_);

  // protrude
  for (int i = 0; i < protrude_coords.size(); i++) {
    auto [r, c] = protrude_coords[i];
    
    if (outline_.coeff(r, c) == 1 || 
        protrude_conf_.count(encode_8(nuc_, r, c)) == 0) // Check if protrusion would be valid
      continue;

    // get protrusion probability
    const double n = 
      n_diag * (nuc_(r - 1, c - 1) + nuc_(r + 1, c - 1) + nuc_(r + 1, c + 1) + nuc_(r - 1, c + 1)) +
                nuc_(r - 1, c) + nuc_(r, c - 1) + nuc_(r + 1, c) + nuc_(r, c + 1);
    const double w = std::pow(n / C, k_nuc_) * R_cor * V_cor * 
      (dyn_basal_ + (1 - dyn_basal_) * get_smoothed_dyn_f(r, c));

    // try protruding to this pixel
    const double p = prob_dist(rng);
    if (p < w) {
      nuc_(r, c) = 1;
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

void CellModel::retract_nuc() {
  /**
   * The logic for this function is identical to protrude_nuc, but some of the values
   * are inverted for retraction.
   *
   * - The exponent for V_cor is negated
   * - counting the neighbors n we instead count the empty pixels
   * - dyn_f_ values are replaced with 1 - dyn_f_
   */
  // calculate probability coefficients
  const double V_cor = 1.0 / (1 + std::exp(-(V_nuc_ - V0_nuc_) / T_nuc_));
  const double R = double(P_nuc_ * P_nuc_) / V_nuc_;
  const double R_cor = 1.0 / (1 + std::exp((R - R0_) / R_nuc_));
  const double n_diag = 1.0 / std::pow(M_SQRT1_2, g_);
  const double C = 4.0 * (1.0 + n_diag);
   
  // randomize retract order
  std::vector<std::pair<int, int>> retract_coords = randomize_nonzero(inner_outline_nuc_);

  // protrude
  for (int i = 0; i < retract_coords.size(); i++) {
    auto [r, c] = retract_coords[i];
    
    if (inner_outline_.coeff(r, c) == 0 || 
        retract_conf_.count(encode_8(nuc_, r, c)) == 0) // Check if retraction would be valid
      continue;

    // get protrusion probability
    const double n = 
      n_diag * (!nuc_(r - 1, c - 1) + !nuc_(r + 1, c - 1) + !nuc_(r + 1, c + 1) + !nuc_(r - 1, c + 1)) +
                !nuc_(r - 1, c) + !nuc_(r, c - 1) + !nuc_(r + 1, c) + !nuc_(r, c + 1);
    const double w = std::pow(n / C, k_nuc_) * R_cor * V_cor * 
      (dyn_basal_ + (1 - dyn_basal_) * (1 - get_smoothed_dyn_f(r, c))); // Inverted from protrusion

    // try retracting this pixel
    const double p = prob_dist(rng);
    if (p < w) {
      nuc_(r, c) = 0;

      // count number of neighbors and sum up values
      int n = 9 - nuc_.block<3, 3>(r - 1, c - 1).sum(); // number of cell pixels (non-nucleus)
      double AC = AC_.block<3, 3>(r - 1, c - 1).sum();
      double FC = FC_.block<3, 3>(r - 1, c - 1).sum();
      double IC = IC_.block<3, 3>(r - 1, c - 1).sum();

      AC_(r, c) = AC / n;
      AC_cor_sum_ += AC_(r, c);
      IC_(r, c) = IC / n;
      IC_cor_sum_ += IC_(r, c);
      FC_(r, c) = FC / n;
    }
  }

  // update nucleus outlines
  update_nuc();
}

void CellModel::protrude() {
  /**
   * This function attempts to protrude the cell. The logic of this function is
   * very similar to that of protrude_nuc, but the weight function is slightly
   * different and the values used are relative to the actin factor as opposed to
   * dynein factor.
   */

  // get probability coefficients
  const double V_cor = 1 / (1 + std::exp((V_ - V0_) / T_));
  const double A_max = A_.block(frame_row_start_, frame_col_start_, 
                                frame_row_end_ - frame_row_start_ + 1, 
                                frame_col_end_ - frame_col_start_ + 1).maxCoeff();
  const double AC_max = AC_.block(frame_row_start_, frame_col_start_, 
                                  frame_row_end_ - frame_row_start_ + 1, 
                                  frame_col_end_ - frame_col_start_ + 1).maxCoeff();
  const double n_diag = 1.0 / std::pow(M_SQRT1_2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // get random visiting order
  std::vector<std::pair<int, int>> protrude_coords = randomize_nonzero(outline_);

  // protrude
  for (int i = 0; i < protrude_coords.size(); i++) {
    auto &[r, c] = protrude_coords[i];

    if (protrude_conf_.count(encode_8(cell_, r, c)) == 0) // not valid protrude configuration
      continue;

    double w;
    if (outline_nuc_.coeff(r, c) == 1) {
      w = 1.0; // force push if nucleus is against edge of cell
    } else {
      double n = 
        n_diag * (cell_(r - 1, c - 1) + cell_(r + 1, c - 1) + cell_(r + 1, c + 1) + cell_(r - 1, c + 1)) + 
                  cell_(r - 1, c) + cell_(r, c - 1) + cell_(r + 1, c) + cell_(r, c + 1);
      int N = cell_.block<3, 3>(r - 1, c - 1).sum();
      double A_avg = A_.block<3, 3>(r - 1, c - 1).sum() / N;
      w = std::pow(n / C, k_) * V_cor * 
       (1.0 - act_slope_ * (1.0 - A_avg / A_max)) *
       (adh_f_(r, c) * (adh_basal_ - 1.0) + 1.0);
    }

    // try protruding cell
    const double p = prob_dist(rng);
    if (p < w) {
      int N = cell_.block<3, 3>(r - 1, c - 1).sum();
      double A_avg = A_.block<3, 3>(r - 1, c - 1).sum() / N;
      double I_avg = I_.block<3, 3>(r - 1, c - 1).sum() / N;
      double F_avg = F_.block<3, 3>(r - 1, c - 1).sum() / N;
      double AC_avg = AC_.block<3, 3>(r - 1, c - 1).sum() / N;
      double IC_avg = IC_.block<3, 3>(r - 1, c - 1).sum() / N;
      double FC_avg = FC_.block<3, 3>(r - 1, c - 1).sum() / N;

      cell_(r, c) = 1;
      A_(r, c) = A_avg;
      I_(r, c) = I_avg;
      F_(r, c) = F_avg;
      AC_(r, c) = AC_avg;
      IC_(r, c) = IC_avg;
      FC_(r, c) = FC_avg;

      // WARN: Make sure this sum is initialized properly
      A_cor_sum_ += A_avg;
      I_cor_sum_ += I_avg;
      AC_cor_sum_ += AC_avg;
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
                                frame_col_end_ - frame_col_start_ + 1).maxCoeff();
  const double AC_max = AC_.block(frame_row_start_, frame_col_start_, 
                                  frame_row_end_ - frame_row_start_ + 1, 
                                  frame_col_end_ - frame_col_start_ + 1).maxCoeff();
  const double n_diag = 1.0 / std::pow(M_SQRT1_2, g_);
  const double C = 4.0 * (1.0 + n_diag);

  // get random visiting order
  std::vector<std::pair<int, int>> retract_coords = randomize_nonzero(inner_outline_);

  // retract
  for (int i = 0; i < retract_coords.size(); i++) {
    auto &[r, c] = retract_coords[i];

    if (retract_conf_.count(encode_8(cell_, r, c)) == 0) // not valid protrude configuration
      continue;

    double n = 
      n_diag * (!cell_(r - 1, c - 1) + !cell_(r + 1, c - 1) + !cell_(r + 1, c + 1) + !cell_(r - 1, c + 1)) + 
                !cell_(r - 1, c) + !cell_(r, c - 1) + !cell_(r + 1, c) + !cell_(r, c + 1);
    int N = cell_.block<3, 3>(r - 1, c - 1).sum();
    double A_avg = A_.block<3, 3>(r - 1, c - 1).sum() / N;
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

void CellModel::initialize_helpers() {
  // TODO: Implement initialize helpers
  
  // Initialize correction variables
  A_cor_sum_ = 0;
  I_cor_sum_ = 0;
  AC_cor_sum_ = 0;
  IC_cor_sum_ = 0;

  // Get values for Gaussian smoothing kernel
  update_smoothing_kernel();

  // Get valid configurations
  update_valid_conf();
}

void CellModel::update_nuc() {
  /**
   * Iterate through nucleus pixels and add 4-neighbors that are not nucleus to
   * outer outline.
   */
  const int DR[4] = {-1, 1, 0, 0};
  const int DC[4] = {0, 0, -1, 1};

  // Clear outlines
  outline_nuc_.setZero();
  inner_outline_nuc_.setZero();

  #pragma omp parallel
  {
    std::unordered_set<std::pair<int, int>, pair_hash> local_inner, local_outer;

    // Iterate through nucleus pixels
    #pragma omp for nowait
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      for (int i = frame_row_start_; i <= frame_row_end_; i++) {
        bool outline = false;
        for (int k = 0; k < 4; k++) {
          const int nr = i + DR[k];
          const int nc = j + DC[k];
          if (nr < 0 || nr >= sim_rows_ || nc < 0 || nc >= sim_cols_) continue;
          if (nuc_(nr, nc) == 0) {
            local_inner.insert({i, j});
            local_outer.insert({nr, nc});
          }
        }
      }
    }

    // update outlines
    #pragma omp critical
    {
      for (auto &[r, c]: local_inner) {
        inner_outline_nuc_.coeffRef(r, c) = 1;
      }
      for (auto &[r, c]: local_outer) {
        outline_nuc_.coeffRef(r, c) = 1;
      }
    }
  }

  // update nucleus volume and perimeter
  V_nuc_ = nuc_.nonZeros();
  P_nuc_ = inner_outline_nuc_.nonZeros();
}

void CellModel::update_cell() {
  /**
   * Iterate through cell pixels and add 4-neighbors that are not part of the
   * cell to outer outline.
   */
  const int DR[4] = {-1, 1, 0, 0};
  const int DC[4] = {0, 0, -1, 1};

  // Clear outlines
  outline_.setZero();
  inner_outline_.setZero();

  #pragma omp parallel
  {
    std::unordered_set<std::pair<int, int>, pair_hash> local_inner, local_outer;

    // Iterate through cell pixels
    #pragma omp for nowait
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      for (int i = frame_row_start_; i <= frame_row_end_; i++) {
        bool outline = false;
        for (int k = 0; k < 4; k++) {
          const int nr = i + DR[k];
          const int nc = j + DC[k];
          if (nr < 0 || nr >= sim_rows_ || nc < 0 || nc >= sim_cols_) continue;
          if (cell_(nr, nc) == 0) {
            local_inner.insert({i, j});
            local_outer.insert({nr, nc});
          }
        }
      }
    }

    // update outlines
    #pragma omp critical
    {
      for (auto &[r, c]: local_inner) {
        inner_outline_.coeffRef(r, c) = 1;
      }
      for (auto &[r, c]: local_outer) {
        outline_.coeffRef(r, c) = 1;
      }
    }
  }

  // update cell volume and perimeter
  V_ = cell_.nonZeros();
  P_ = inner_outline_.nonZeros();
}

void CellModel::correct_concentrations() {
  // Calculate amount of signal that needs to be distributed from all pixels
  const double A_dist = A_cor_sum_ / V_;
  const double I_dist = I_cor_sum_ / V_;
  const double AC_dist = AC_cor_sum_ / (V_ - V0_nuc_);
  const double IC_dist = IC_cor_sum_ / (V_ - V0_nuc_);

  #pragma omp parallel for collapse(2)
  for (int j = frame_col_start_; j <= frame_col_end_; j++) {
    for (int i = frame_row_start_; i <= frame_row_end_; i++) {
      if (cell_(i, j) == 1) {
        A_(i, j) -= A_dist;
        I_(i, j) -= I_dist;
      }
      if (cell_(i, j) == 1 && nuc_(i, j) == 0) {
        AC_(i, j) -= A_dist;
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
  double s2C = 0.05; // NOTE: what is this?
  double A0_3 = std::pow(A0_, 3);
  double dx_2 = dx_ * dx_;
  
  // temporary variables for update
  Mat_d A_new(sim_rows_, sim_cols_);
  Mat_d I_new(sim_rows_, sim_cols_);
  Mat_d F_new(sim_rows_, sim_cols_);
  Mat_d AC_new(sim_rows_, sim_cols_);
  Mat_d IC_new(sim_rows_, sim_cols_);
  Mat_d FC_new(sim_rows_, sim_cols_);

  for (int k = 0; k < diff_t_; k++) {
    #pragma omp parallel for collapse(2)
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      for (int i = frame_row_start_; i <= frame_col_end_; i++) {
        if (cell_(i, j)) {
          double A_3 = std::pow(A_(i, j), 3);
          double f = (k0_adh_(i, j) + gamma_ * A_3 / (A0_3 + A_3)) * I_(i, j)
            - delta_ * (s1_ + s2_ * F_(i, j) / (F0_ + F_(i, j))) * A_(i, j);
          double h = eps_ * (kn_ * A_(i, j) - ks_ * F_(i, j));

          A_new(i, j) = A_(i, j) + dt_ * (
            f + DA_ / dx_2 * (
              cell_(i + 1, j) * (A_(i + 1, j) - A_(i, j)) 
            - cell_(i - 1, j) * (A_(i, j) - A_(i - 1, j)) 
            + cell_(i, j + 1) * (A_(i, j + 1) - A_(i, j)) 
            - cell_(i, j - 1) * (A_(i, j) - A_(i, j - 1))
            )
          );
          I_new(i, j) = I_(i, j) + dt_ * (
            -f + DI_ / dx_2 * (
              cell_(i + 1, j) * (I_(i + 1, j) - I_(i, j)) 
            - cell_(i - 1, j) * (I_(i, j) - I_(i - 1, j)) 
            + cell_(i, j + 1) * (I_(i, j + 1) - I_(i, j)) 
            - cell_(i, j - 1) * (I_(i, j - 1) - I_(i, j))
            )
          );
          F_new(i, j) = F_(i, j) + h * dt_;
        }

        if (cell_(i, j) == 1 && nuc_(i, j) == 0) {
          double AC_3 = std::pow(AC_(i, j), 3);
          double fC = (k0_ + gamma_ * AC_3 / (A0_3 + AC_3)) * IC_(i, j)
            - delta_ * (s1_ + s2C * FC_(i, j) / (F0_ + FC_(i, j))) * AC_(i, j);
          double hC = eps_ * (kn_ * AC_(i, j) - ks_ * FC_(i, j));

          AC_new(i, j) = AC_(i, j) + dt_ * (
            fC + DA_ / dx_2 * (
              (cell_(i + 1, j) - nuc_(i + 1, j)) * (AC_(i + 1, j) - AC_(i, j)) 
            - (cell_(i - 1, j) - nuc_(i - 1, j)) * (AC_(i, j) - AC_(i - 1, j))
            + (cell_(i, j + 1) - nuc_(i, j + 1)) * (AC_(i, j + 1) - AC_(i, j))
            - (cell_(i, j - 1) - nuc_(i, j - 1)) * (AC_(i, j) - AC_(i, j - 1))
            )
          );
          IC_new(i, j) = IC_(i, j) + dt_ * (
            -fC + DI_ / dx_2 * (
              (cell_(i + 1, j) - nuc_(i + 1, j)) * (IC_(i + 1, j) - IC_(i, j)) 
            - (cell_(i - 1, j) - nuc_(i - 1, j)) * (IC_(i, j) - IC_(i - 1, j))
            + (cell_(i, j + 1) - nuc_(i, j + 1)) * (IC_(i, j + 1) - IC_(i, j))
            - (cell_(i, j - 1) - nuc_(i, j - 1)) * (IC_(i, j) - IC_(i, j - 1))
            )
          );
          FC_new(i, j) = FC_(i, j) + hC * dt_;
        }
      }
    }

    // replace elements
    A_ = A_new;
    I_ = I_new;
    F_ = F_new;
    AC_ = AC_new;
    IC_ = IC_new;
    FC_ = FC_new;
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
   * doesn't share signals between pixels equidistant to the nucleus, shouldn't
   * matter because the signals will later be smoothed using a Gaussian kernel
   * anyway (not done in this function because only a small subset of values
   * require smoothing).
   *
   * Normalization is achieved using a sigmoid function to ensure values are in
   * the range (0, 1) so that retraction probabilities can be 1 - protrusion
   * probabilities. The sigmoid is centered at the average so that if all
   * values are similar, it won't just result in all of the values being high
   * (close to 1).
   */

  // helper constants
  const int DR[4] = {1, -1, 0, 0};
  const int DC[4] = {0, 0, 1, -1};

  // generate random starting order
  std::vector<std::pair<int, int>> nuc_coords = randomize_nonzero(inner_outline_nuc_);

  // perform bfs and keep track of which pixel each pixel originated from,
  std::deque<std::pair<int, int>> q(nuc_coords.begin(), nuc_coords.end());
  std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash> from; // which pixel each pixel came from
  std::stack<std::pair<int, int>> rev; // order to revisit pixels (from out to in)
  while (!q.empty()) {
    auto [r, c] = q.front();
    q.pop_front();
    rev.push({r, c});
    for (int i = 0; i < 4; i++) {
      int nr = r + DR[i];
      int nc = c + DC[i];
      std::pair<int, int> next(nr, nc);
      if (nr < frame_row_start_ || nr > frame_row_end_ ||
          nc < frame_col_start_ || nc > frame_col_end_ ||
          cell_(nr, nc) == 0 ||
          nuc_.coeffRef(nr, nc) == 1 ||
          from.count(next) > 0)
        continue;
      from[next] = {r, c};
      q.push_back(next); 
    }
  }

  // clear previous signals
  dyn_f_.setZero();
  
  // propagate signals towards nucleus
  double dyn_f_sum = 0;
  const int n_cell = rev.size(); // number of pixels to propagate (for calculating avg)
  while (!rev.empty()) {
    std::pair<int, int> cur = rev.top();
    rev.pop();
    auto [r, c] = cur;
    dyn_f_(r, c) += AC_(r, c); // add signal at that pixel
    dyn_f_sum += dyn_f_(r, c);

    // propagate to closer pixel
    if (from.count(cur) > 0) {
      auto [fr, fc] = from[cur];
      dyn_f_(fr, fc) += dyn_f_(r, c);
    }
  }
  
  if (n_cell == 0) return;

  // sigmoid normalization
  const double dyn_f_avg = dyn_f_sum / n_cell;
  #pragma omp parallel for collapse(2)
  for (int i = frame_row_start_; i <= frame_row_end_; i++) {
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      if (dyn_f_(i, j) == 0) continue;
      const double diff = dyn_f_(i, j) - dyn_f_avg;
      dyn_f_(i, j) = 1.0 / (1.0 + std::exp(-dyn_norm_k_ * diff));
    }
  }
}

void CellModel::update_adhesion_field() {
  /**
    * This function updates the adhesion field based on the positions of the
    * adhesions. A Gaussian-smoothed adhesion field is first calculated for
    * each of the adhesion points, and then the points for the rest of the
    * pixels on the cell are calculated. The field is then normalized using IDW
    * normalization method, which produces a distribution similar to the
    * original code but runs faster. The values are then again normalized to be
    * between 0 and 1 and inverted to act as a protrusion probability.
    */

  const double ampl = 1 / (2 * M_PI * adh_sigma_);
  const double eps = 1e-10;

  // Calculate adh_g for adhesion points
  Mat_d K(adh_num_, adh_num_); // Gaussian kernel matrix
  Vec_d g_k(adh_num_); // adh_g for adhesions, for Eigen vectorization optimization
  #pragma omp parallel for
  for (int k = 0; k < adh_num_; k++) {
    for (int j = 0; j < adh_num_; j++) {
      double dr = adh_pos_(0, k) - adh_pos_(0, j);
      double dc = adh_pos_(1, k) - adh_pos_(1, j);
      K(k, j) = std::exp(-(dr * dr + dc * dc) / (2 * adh_sigma_));
    }
  }
  g_k = K.rowwise().sum();

  for (int k = 0; k < adh_num_; k++) {
    adh_g_(adh_pos_(0, k), adh_pos_(1, k)) = g_k(k);
  }

  // Calculate adh_g and adh_f
  #pragma omp parallel for collapse(2)
  for (int i = frame_row_start_; i <= frame_row_end_; i++) {
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      if (adh_.coeffRef(i, j) == 1) continue;

      // Get distances to adhesions
      Arr_d dr = adh_pos_.row(0).cast<double>().array() - i;
      Arr_d dc = adh_pos_.row(1).cast<double>().array() - j;
      Arr_d dist2 = dr.square() + dc.square();

      // Calculate adh_g
      Arr_d gaussian = (-dist2 / (2 * adh_sigma_)).exp();
      double g_val = gaussian.sum();
      adh_g_(i, j) = ampl * g_val;

      // Normalize using IDW normalization
      Arr_d inv = 1.0 / (dist2 + eps);
      Arr_d gaus_inv = gaussian * inv;
      adh_f_(i, j) = gaus_inv.sum() / inv.sum();
    }
  }

  // Invert for protrusion probability calculation
  double max_f = adh_f_.block(frame_row_start_, frame_col_start_, 
                              frame_row_end_ - frame_row_start_ + 1,
                              frame_col_end_ - frame_col_start_ + 1).maxCoeff();
  #pragma omp parallel for collapse(2)
  for (int i = frame_row_start_; i <= frame_row_end_; i++) {
    for (int j = frame_col_start_; j <= frame_col_end_; j++) {
      if (adh_.coeffRef(i, j) == 1) {
        adh_f_(i, j) = 0;
        k0_adh_(i, j) = k0_;
      } else {
        adh_f_(i, j) = 1 - (adh_f_(i, j) / max_f);
        k0_adh_(i, j) = (k0_ - k0_min_) * k0_scalar_ * (1 - adh_f_(i, j)) + k0_min_;
      }
    }
  }
}

const double CellModel::get_smoothed_dyn_f(const int r, const int c) {
  const int diff = dyn_kernel_size_ / 2;
  const int row_start = std::max(r - diff, 0);
  const int row_n = std::min(dyn_kernel_size_, sim_rows_ - r);
  const int col_start = std::max(c - diff, 0);
  const int col_n = std::min(dyn_kernel_size_, sim_cols_ - c);

  double smoothed = 0.0;
  #pragma omp parallel for collapse(2) reduction(+:smoothed)
  for (int i = 0; i < row_n; i++) {
    for (int j = 0; j < col_n; j++) {
      smoothed += dyn_f_(row_start + i, col_start + j) * g_dyn_f_(i, j);
    }
  }

  return smoothed;
}

void CellModel::update_smoothing_kernel() {
  const double sigma2 = dyn_sigma_ * dyn_sigma_;
  const int c = dyn_kernel_size_ / 2; // center of the kernel
  g_dyn_f_ = Mat_d(dyn_kernel_size_, dyn_kernel_size_);
  for (int i = 0; i < dyn_kernel_size_; i++) {
    for (int j = 0; j < dyn_kernel_size_; j++) {
      const double diff2 = (i - c) * (i - c) + (j - c) * (j - c);
      g_dyn_f_(i, j) = (1.0 / (2 * M_PI * sigma2)) * std::exp(-diff2 / (2 * sigma2));
    }
  }
}

const uint8_t CellModel::encode_8(Mat_i &mat, const int r, const int c) {
  int rows = mat.rows();
  int cols = mat.cols();
  
  // Helper lambda to safely get matrix value (returns 0 for out-of-bounds)
  auto safe_get = [&](int row, int col) -> int {
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      return mat(row, col);
    }
    return 0;
  };
  
  return (safe_get(r-1, c-1) << 0) |  // top-left
         (safe_get(r-1, c  ) << 1) |  // top
         (safe_get(r-1, c+1) << 2) |  // top-right
         (safe_get(r,   c+1) << 3) |  // right
         (safe_get(r+1, c+1) << 4) |  // bottom-right
         (safe_get(r+1, c  ) << 5) |  // bottom
         (safe_get(r+1, c-1) << 6) |  // bottom-left
         (safe_get(r,   c-1) << 7);   // left
}

const bool CellModel::is_valid_config_prot(uint8_t conf) {
  // Diagonal-only connections (L-shaped corners without edge support)
  bool diag1 = (conf & (1 << 0)) && !(conf & (1 << 1)) && !(conf & (1 << 7)); // top-left
  bool diag2 = (conf & (1 << 2)) && !(conf & (1 << 1)) && !(conf & (1 << 3)); // top-right
  bool diag3 = (conf & (1 << 4)) && !(conf & (1 << 3)) && !(conf & (1 << 5)); // bottom-right
  bool diag4 = (conf & (1 << 6)) && !(conf & (1 << 5)) && !(conf & (1 << 7)); // bottom-left

  if (diag1 || diag2 || diag3 || diag4) return false;

  // Pinch cases
  bool vertical_pinch = (conf & (1 << 1)) && (conf & (1 << 5)) && !(conf & (1 << 3)) && !(conf & (1 << 7));
  bool horizontal_pinch = (conf & (1 << 3)) && (conf & (1 << 7)) && !(conf & (1 << 1)) && !(conf & (1 << 5));

  if (vertical_pinch || horizontal_pinch) return false;

  return true;
}

const bool CellModel::is_valid_config_retr(uint8_t conf) {
  // Diagonal-only connections (L-shaped corners without edge support)
  bool diag1 = !(conf & (1 << 0)) && (conf & (1 << 1)) && (conf & (1 << 7)); // top-left
  bool diag2 = !(conf & (1 << 2)) && (conf & (1 << 1)) && (conf & (1 << 3)); // top-right
  bool diag3 = !(conf & (1 << 4)) && (conf & (1 << 3)) && (conf & (1 << 5)); // bottom-right
  bool diag4 = !(conf & (1 << 6)) && (conf & (1 << 5)) && (conf & (1 << 7)); // bottom-left

  if (diag1 || diag2 || diag3 || diag4) return false;

  // Pinch cases
  bool vertical_pinch = (conf & (1 << 1)) && (conf & (1 << 5)) && !(conf & (1 << 3)) && !(conf & (1 << 7));
  bool horizontal_pinch = (conf & (1 << 3)) && (conf & (1 << 7)) && !(conf & (1 << 1)) && !(conf & (1 << 5));

  if (vertical_pinch || horizontal_pinch) return false;

  return true;
}

void CellModel::update_valid_conf() {
  // Create protrusion configurations
  for (int i = 0; i < (1 << 8); i++) {
    if (is_valid_config_prot(i)) {
      protrude_conf_.insert(i);
    }
  }

  // Create retraction configurations
  for (int i = 0; i < (1 << 8); i++) {
    if (is_valid_config_retr(i)) {
      retract_conf_.insert(i);
    }
  }
}


const std::vector<int> CellModel::generate_indices(const int n, const int lb, const int ub) {
  if (ub - lb < n) {
    throw std::runtime_error("Bounds must be at least as large as the number of indices to generate.");
  }

  std::vector<int> arr(ub - lb);
  for (int i = 0, v = lb; v < ub; i++, v++) {
    arr[i] = v;
  }

  std::shuffle(arr.begin(), arr.end(), rng);

  return std::vector<int>(arr.begin(), arr.begin() + n);
}

const std::vector<std::pair<int, int>> CellModel::randomize_nonzero(const SpMat_i mat) {
  std::vector<std::pair<int, int>> coords;
  for (int k = 0; k < mat.outerSize(); k++) {
    for (SpMat_i::InnerIterator it(mat, k); it; ++it) {
      coords.push_back({it.row(), it.col()});
    }
  }
  std::shuffle(coords.begin(), coords.end(), rng);

  return coords;
}

} // dynein_cell_model namespace
