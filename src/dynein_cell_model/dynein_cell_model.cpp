#include <dynein_cell_model/dynein_cell_model.hpp>
#include <stdexcept>
#include <algorithm>

namespace dynein_cell_model {
CellModel::CellModel() {
  // Initialize adhesions
}

void CellModel::step() {
  this->protrude_nuc();
  this->retract_nuc();

  this->protrude();
  this->retract();

  this->update_concentrations();
  this->diffuse_k0_adh();
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
} // dynein_cell_model namespace
