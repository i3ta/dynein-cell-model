#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils/test_utils.hpp>

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class CompModelTest : public test_utils::ModelTestBase {};

TEST_F(CompModelTest, DyneinFieldProtConsistency) {
  dcm::Mat_i nuc_mask = dcm::Mat_i::Zero(rows, cols);
  dcm::Mat_i cell_mask = dcm::Mat_i::Zero(rows, cols);
  fill_circle(nuc_mask, 100, 100, 20);  // Center nucleus
  fill_circle(cell_mask, 100, 100, 50); // Larger cell

  test_utils::CellModelTest modern{config};
  modern.set_cell(cell_mask);
  modern.set_nuc(nuc_mask);
  modern.set_AC(dcm::Mat_d::Constant(rows, cols, 1.0));

  double **raw_cell = eigen_to_raw(cell_mask.cast<double>());
  double **raw_nuc = eigen_to_raw(nuc_mask.cast<double>());
  double **raw_AC = eigen_to_raw(modern.get_AC());
  double **raw_inner_outline =
      eigen_to_raw(modern.get_inner_outline().cast<double>());
  double **raw_outline_nuc =
      eigen_to_raw(modern.get_outline_nuc().cast<double>());

  TRACE_MSG("Generating Legacy Dynein Field...");
  double **legacy_dyn_f_prot = generate_dyn_field_protr(
      raw_cell, raw_nuc, raw_inner_outline, raw_outline_nuc, rows - 2, cols - 2,
      1, 1, rows, cols, raw_AC);

  TRACE_MSG("Generating New Dynein Field...");
  modern.generate_dyn_field(false);
  const auto &modern_dyn_f_prot = modern.get_dyn_f();

  TRACE_MSG("Comparing Dynein Field Outputs...");
  for (int i = 1; i < rows - 1; ++i) {
    for (int j = 1; j < cols - 1; ++j) {
      EXPECT_NEAR(legacy_dyn_f_prot[i][j], modern_dyn_f_prot(i, j), 1e-8);
    }
  }
}

TEST_F(CompModelTest, DyneinFieldRetrConsistency) {
  dcm::Mat_i nuc_mask = dcm::Mat_i::Zero(rows, cols);
  dcm::Mat_i cell_mask = dcm::Mat_i::Zero(rows, cols);
  fill_circle(nuc_mask, 100, 100, 20);  // Center nucleus
  fill_circle(cell_mask, 100, 100, 50); // Larger cell

  test_utils::CellModelTest modern{config};
  modern.set_cell(cell_mask);
  modern.set_nuc(nuc_mask);
  modern.set_AC(dcm::Mat_d::Constant(rows, cols, 1.0));

  double **raw_cell = eigen_to_raw(cell_mask.cast<double>());
  double **raw_nuc = eigen_to_raw(nuc_mask.cast<double>());
  double **raw_AC = eigen_to_raw(modern.get_AC());
  double **raw_inner_outline =
      eigen_to_raw(modern.get_inner_outline().cast<double>());
  double **raw_outline_nuc =
      eigen_to_raw(modern.get_outline_nuc().cast<double>());

  TRACE_MSG("Generating Legacy Dynein Field...");
  double **legacy_dyn_f_ret = generate_dyn_field_retr(
      raw_cell, raw_nuc, raw_inner_outline, raw_outline_nuc, rows - 2, cols - 2,
      1, 1, rows, cols, raw_AC);

  TRACE_MSG("Generating New Dynein Field...");
  modern.generate_dyn_field(true);
  const auto &modern_dyn_f_ret = modern.get_dyn_f();

  TRACE_MSG("Comparing Dynein Field Outputs...");
  for (int i = 1; i < rows - 1; ++i) {
    for (int j = 1; j < cols - 1; ++j) {
      EXPECT_NEAR(legacy_dyn_f_ret[i][j], modern_dyn_f_ret(i, j), 1e-8);
    }
  }
}
