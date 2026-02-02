#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <cell_nuc/cell_nuc.hpp>
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

  void rearrange_adhesions(const bool bias = false,
                           const bool rearrange_all = false);

  void protrude();

  void protrude_nuc_dep();

  void retract_nuc_dep();

  void generate_dyn_field(const dcm::SpMat_i &cell_outline,
                          const dcm::SpMat_i &nuc_outline, bool retract);

  void diffuse_k0_adh();

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

  void set_k0_adh(const dcm::Mat_d k0_adh);

  const dcm::Mat_i &get_cell();

  const dcm::Mat_i &get_nuc();

  const dcm::SpMat_i &get_adh();

  const dcm::Mat_i &get_adh_pos();

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

  const dcm::Mat_d &get_k0_adh();

  const double get_AC_cor_sum();

  const double get_IC_cor_sum();

  const int get_V();

  const int get_V_nuc();

  const int get_P_nuc();
}; // class CellModelTest

class ModelTestBase : public ::testing::Test {
protected:
  int rows = 200;
  int cols = 200;
  dcm::CellModelConfig config;
  std::vector<double **> legacy_pointers;
  std::vector<int **> legacy_pointers_int;

  void SetUp() override {
    config.sim_rows_ = rows;
    config.sim_cols_ = cols;
  }

  void TearDown() override {
    for (double **p : legacy_pointers) {
      free(p);
    }
    for (int **p : legacy_pointers_int) {
      free(p);
    }
    legacy_pointers.clear();
    legacy_pointers_int.clear();
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

  int **eigen_to_int(const dcm::Mat_i &mat) {
    int r_num = mat.rows();
    int c_num = mat.cols();

    int **raw = new int *[r_num];
    raw[0] = new int[r_num * c_num];

    for (int i = 1; i < r_num; ++i) {
      raw[i] = raw[i - 1] + c_num;
    }

    for (int i = 0; i < r_num; ++i) {
      for (int j = 0; j < c_num; ++j) {
        raw[i][j] = mat(i, j);
      }
    }

    legacy_pointers_int.push_back(raw);

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

  void sync_params(Cell &legacy, const dcm::CellModelConfig &config) {
    legacy.k = config.k_;
    legacy.k_nuc = config.k_nuc_;
    legacy.g = config.g_;
    legacy.T = config.T_;
    legacy.T_nuc = config.T_nuc_;
    legacy.act_slope = config.act_slope_;
    legacy.R0 = config.R0_;
    legacy.R_nuc = config.R_nuc_;
    legacy.prop_factor = config.prop_factor_;
    legacy.d_basal = config.dyn_basal_;

    legacy.DA = config.DA_;
    legacy.DI = config.DI_;
    legacy.k0 = config.k0_;
    legacy.k0_min = config.k0_min_;
    legacy.scalar = config.k0_scalar_;
    legacy.gamma = config.gamma_;
    legacy.delta = config.delta_;
    legacy.A0 = config.A0_;
    legacy.s1 = config.s1_;
    legacy.s2 = config.s2_;
    legacy.F0 = config.F0_;
    legacy.kn = config.kn_;
    legacy.ks = config.ks_;
    legacy.eps = config.eps_;
    legacy.dt = config.dt_;
    legacy.dx = config.dx_;

    legacy.A_max = config.A_max_;
    legacy.A_min = config.A_min_;
    legacy.AC_max = config.AC_max_;
    legacy.AC_min = config.AC_min_;

    legacy.adh_num = config.adh_num_;
    legacy.adh_frac = config.adh_frac_;
    legacy.adh_sigma = config.adh_sigma_;
    legacy.adh_basal_prot = config.adh_basal_;

    legacy.diff_t = config.diff_t_;
    legacy.fr_dist = config.frame_padding_;

    legacy.env_rows_num = config.sim_rows_;
    legacy.env_cols_num = config.sim_cols_;

    legacy.fr_rows_pos = 1;
    legacy.fr_cols_pos = 1;

    legacy.fr_rows_num = config.sim_rows_ - 2;
    legacy.fr_cols_num = config.sim_cols_ - 2;
  }

  void test_mat(double **legacy, const dcm::Mat_i &modern,
                const std::string &test_name) {
    int mat_mismatches = 0;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if ((int)legacy[i][j] != modern(i, j)) {
          if (mat_mismatches < 10) { // Limit logging to avoid wall of text
            std::cout << test_name << " Mismatch at (" << i << "," << j
                      << ") - "
                      << "Legacy: " << (int)legacy[i][j]
                      << ", Modern: " << modern(i, j) << std::endl;
          }
          mat_mismatches++;
        }
      }
    }
    EXPECT_EQ(mat_mismatches, 0)
        << test_name
        << " masks do not match. Fix this before checking outlines.";
  }

  void test_outline(double **legacy, const dcm::SpMat_i &modern,
                    const std::string &test_name) {
    int outline_mismatches = 0;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int leg_val = (int)legacy[i][j];
        int mod_val = (int)modern.coeff(i, j);
        if (leg_val != mod_val) {
          if (outline_mismatches < 10) {
            std::cout << test_name << " Mismatch at (" << i << "," << j
                      << ") - "
                      << "Legacy: " << leg_val << ", Modern: " << mod_val
                      << std::endl;
          }
          outline_mismatches++;
        }
      }
    }
    EXPECT_EQ(outline_mismatches, 0) << "Outer outlines do not match. Check "
                                        "connectivity (4 vs 8 neighbors).";
  }

  void test_mat_near(double **legacy, const dcm::Mat_d &modern,
                     const std::string &test_name, const double tol = 1e-8) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        EXPECT_NEAR(legacy[i][j], modern(i, j), tol)
            << test_name << " mismatch at (" << i << "," << j << ")";
      }
    }
  }
};

} // namespace test_utils

#endif
