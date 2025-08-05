#include <cmath>
#include <deque>
#include <stack>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

#include <dynein_cell_model/dynein_cell_model.hpp>
#include <unordered_map>

namespace dynein_cell_model {
CellModel::CellModel() {
  // Initialize adhesions
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

  update_concentrations();
  diffuse_k0_adh();
  update_dyn_nuc_field();

  if (t_ % save_t_ == 0) {
    save_state(save_dir_);
  }

  t_++;
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
      if (adh_.coeff(cand_r, cand_c) != 1 && env_.coeff(cand_r, cand_c) == 1 && cell_.coeff(cand_r, cand_c) == 1) {
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

  for (int k = 0; k < cell_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<int>::InnerIterator it(cell_, k); it; ++it) {
      int i = it.row();
      int j = it.col();

      // Update bounding box
      if (cell_.coeff(i, j) != 0) {
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
  std::vector<std::pair<int, int>> nuc_coords; // coordinates of the inner pixels of the nucleus outline
  for (int k = 0; k < inner_outline_nuc_.outerSize(); k++) {
    for (SpMat_i::InnerIterator it(inner_outline_nuc_, k); it; ++it) {
      nuc_coords.push_back({it.row(), it.col()});
    }
  }
  std::shuffle(nuc_coords.begin(), nuc_coords.end(), rng); // random visit order

  // perform bfs and keep track of which pixel each pixel originated from,
  std::deque<std::pair<int, int>> q(nuc_coords.begin(), nuc_coords.end());
  std::unordered_map<std::pair<int, int>, std::pair<int, int>> from; // which pixel each pixel came from
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
          cell_.coeffRef(nr, nc) == 0 ||
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

} // dynein_cell_model namespace
