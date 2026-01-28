#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils/test_utils.hpp>

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class CompModelTest : public ::testing::Test {
protected:
  int rows = 200;
  int cols = 200;
  dcm::CellModelConfig config;

  void SetUp() override {
    config.sim_rows_ = rows;
    config.sim_cols_ = cols;
  }

  // Helper to cleanup raw pointers
  void free_raw(double **ptr, int r) {
    if (!ptr)
      return;
    for (int i = 0; i < r; ++i)
      delete[] ptr[i];
    delete[] ptr;
  }

  // Helper to convert Eigen matrices to double matrices
  double **eigen_to_raw(const dcm::Mat_d &mat) {
    double **raw = new double *[rows];
    for (int i = 0; i < rows; ++i) {
      raw[i] = new double[cols];
      for (int j = 0; j < cols; ++j)
        raw[i][j] = mat(i, j);
    }
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

TEST_F(CompModelTest, ProtrudeNucConsistency) {
  TRACE_MSG("Initializing RNG...");
  test_utils::DebugRand<double> drand;
  std::vector<double> mock_probs;
  mock_probs.reserve(1000);

  std::mt19937 temp_engine(42); // Fixed seed
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < 1000; ++i) {
    mock_probs.push_back(dist(temp_engine));
  }

  drand.set_outputs(std::move(mock_probs));

  TRACE_MSG("Initializing Config...");
  config.k_nuc_ = 1.0;
  config.T_nuc_ = 100.0;
  config.R_nuc_ = 100.0;
  config.g_ = 1.0;
  config.R0_ = 10;
  config.dyn_basal_ = 0.1;
  config.sim_rows_ = rows;
  config.sim_cols_ = cols;

  TRACE_MSG("Instantiating Modern and Legacy models...");
  test_utils::CellModelTest modern{config};
  Cell legacy;

  TRACE_MSG("Creating masks...");
  dcm::Mat_i nuc_mask = dcm::Mat_i::Zero(rows, cols);
  dcm::Mat_i cell_mask = dcm::Mat_i::Zero(rows, cols);
  for (int i = 40; i < 60; ++i)
    for (int j = 40; j < 60; ++j)
      nuc_mask(i, j) = 1;

  for (int i = 20; i < 80; ++i)
    for (int j = 20; j < 80; ++j)
      cell_mask(i, j) = 1;

  TRACE_MSG("Setting Modern values...");
  modern.set_cell(cell_mask);
  modern.set_nuc(nuc_mask);
  modern.set_AC(dcm::Mat_d::Constant(rows, cols, 1.0));
  modern.set_IC(dcm::Mat_d::Constant(rows, cols, 1.0));

  TRACE_MSG("Converting Eigen to Legacy Raw Pointers...");
  legacy.Im_nuc = eigen_to_raw(nuc_mask.cast<double>());
  legacy.Im = eigen_to_raw(cell_mask.cast<double>());
  legacy.AC = eigen_to_raw(modern.get_AC());
  legacy.IC = eigen_to_raw(modern.get_IC());
  legacy.FC = eigen_to_raw(dcm::Mat_d::Zero(rows, cols));
  legacy.outline = eigen_to_raw(modern.get_outline().cast<double>());
  legacy.inner_outline =
      eigen_to_raw(modern.get_inner_outline().cast<double>());
  legacy.outline_nuc = eigen_to_raw(modern.get_outline_nuc().cast<double>());
  legacy.inner_outline_nuc =
      eigen_to_raw(modern.get_inner_outline_nuc().cast<double>());

  TRACE_MSG("Syncing Legacy parameters...");
  legacy.V0_nuc = nuc_mask.sum();
  legacy.V_nuc = legacy.V0_nuc;
  legacy.T_nuc = config.T_nuc_;
  legacy.R0 = config.R0_;
  legacy.R_nuc = config.R_nuc_;
  legacy.g = config.g_;
  legacy.k_nuc = config.k_nuc_;
  legacy.d_basal = config.dyn_basal_;
  legacy.fr_rows_num = rows - 2;
  legacy.fr_cols_num = cols - 2;
  legacy.env_rows_num = rows;
  legacy.env_cols_num = cols;
  legacy.fr_rows_pos = 1;
  legacy.fr_cols_pos = 1;

  TRACE_MSG("Executing Modern: protrude_nuc_dep()...");
  modern.protrude_nuc_dep();

  TRACE_MSG("Executing Legacy: protrude_nuc()...");
  legacy.protrude_nuc();

  TRACE_MSG("Starting Comparison Loop...");
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      ASSERT_EQ(modern.get_nuc()(i, j), (int)legacy.Im_nuc[i][j])
          << "Crash/Mismatch at indices: " << i << ", " << j;
    }
  }
  TRACE_MSG("Test Finished Successfully.");
}
