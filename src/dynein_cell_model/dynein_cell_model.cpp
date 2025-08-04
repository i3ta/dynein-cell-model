#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

#include <dynein_cell_model/dynein_cell_model.hpp>

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
    update_k0_adh();
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

void CellModel::update_adhesion_field() {
  /**
    * This function updates the adhesion field based on the positions of the
    * adhesions. A Gaussian-smoothed adhesion field is first calculated for
    * each of the adhesion points, and then the points for the rest of the
    * pixels on the cell are calculated. The field is then normalized using a
    * hybrid Gaussian-IDW normalization method, which produces a distribution
    * similar to the original code but runs faster. The values are then again
    * normalized to be between 0 and 1 and inverted to act as a protrusion
    * probability.
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

      // Normalize using hybrid Gaussian + IDW normalization
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
      } else {
        adh_f_(i, j) = 1 - (adh_f_(i, j) / max_f);
      }
    }
  }
}
} // dynein_cell_model namespace
