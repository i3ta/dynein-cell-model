#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>

namespace test_utils {
namespace dcm = dynein_cell_model;

template <typename T> class DebugRand {
public:
  DebugRand() : current_index_(0) {}

  static constexpr T min() { return 0; }
  static constexpr T max() { return std::numeric_limits<T>::max(); }

  void set_outputs(std::vector<T> &&outputs) {
    outputs_ = std::move(outputs);
    current_index_ = 0;
  }

  T operator()() {
    if (outputs_.empty())
      return 0;
    return outputs_[current_index_++ % outputs_.size()];
  }

private:
  static std::vector<T> outputs_;
  size_t current_index_;
};

// This line must stay in the header to tell the compiler
// how to allocate the static vector for any T used.
template <typename T> std::vector<T> DebugRand<T>::outputs_;

class CellModelTest {
private:
  dcm::CellModel model;

public:
  CellModelTest(const dcm::CellModelConfig &conf);

  void init(const dcm::CellModelConfig &conf);

  void protrude_nuc_dep();

  void retract_nuc_dep();

  void generate_dyn_field(const dcm::SpMat_i &cell_outline,
                          const dcm::SpMat_i &nuc_outline, bool retract);

  void set_cell(const dcm::Mat_i cell);

  void set_nuc(const dcm::Mat_i nuc);

  void set_adh(const dcm::SpMat_i adh);

  void set_A(const dcm::Mat_d A);

  void set_AC(const dcm::Mat_d AC);

  void set_I(const dcm::Mat_d I);

  void set_IC(const dcm::Mat_d IC);

  void set_F(const dcm::Mat_d F);

  void set_FC(const dcm::Mat_d FC);

  void set_env(const dcm::SpMat_i env);

  const dcm::Mat_i &get_cell();

  const dcm::Mat_i &get_nuc();

  const dcm::SpMat_i &get_adh();

  const dcm::Mat_d &get_A();

  const dcm::Mat_d &get_AC();

  const dcm::Mat_d &get_I();

  const dcm::Mat_d &get_IC();

  const dcm::Mat_d &get_F();

  const dcm::Mat_d &get_FC();

  const dcm::SpMat_i &get_env();

  const dcm::SpMat_i &get_outline();

  const dcm::SpMat_i &get_outline_nuc();

  const dcm::SpMat_i &get_inner_outline();

  const dcm::SpMat_i &get_inner_outline_nuc();

  const dcm::Mat_d &get_dyn_f();

  const double get_AC_cor_sum();

  const double get_IC_cor_sum();

  const int get_V_nuc();

  const int get_P_nuc();
}; // class CellModelTest

class ModelTestBase : public ::testing::Test {
protected:
  int rows = 200;
  int cols = 200;
  dcm::CellModelConfig config;
  std::vector<double **> legacy_pointers;

  void SetUp() override {
    config.sim_rows_ = rows;
    config.sim_cols_ = cols;
  }

  void TearDown() override {
    for (double **p : legacy_pointers) {
      free(p);
    }
    legacy_pointers.clear();
  }

  // Helper to cleanup raw pointers in the format of the legacy model
  void free_legacy(double **m) {
    if (!m)
      return;
    delete[] m[0];
    delete[] m;
  }

  // Helper to convert Eigen matrices to double matrices
  double **eigen_to_raw(const dcm::Mat_d &mat) {
    int r_num = mat.rows();
    int c_num = mat.cols();

    double **raw = new double *[r_num];

    raw[0] = new double[r_num * c_num];

    for (int i = 1; i < r_num; ++i) {
      raw[i] = raw[i - 1] + c_num;
    }

    for (int i = 0; i < r_num; ++i) {
      for (int j = 0; j < c_num; ++j) {
        raw[i][j] = mat(i, j);
      }
    }

    legacy_pointers.push_back(raw);

    return raw;
  }

  void fill_circle(dcm::Mat_i &mat, int center_r, int center_c, int radius) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if (std::sqrt(std::pow(i - center_r, 2) + std::pow(j - center_c, 2)) <=
            radius) {
          mat(i, j) = 1;
        }
      }
    }
  }
};

} // namespace test_utils

#endif
