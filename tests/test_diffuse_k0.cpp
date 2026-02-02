#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils/test_utils.hpp>

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class CompModelTest : public test_utils::ModelTestBase {};

TEST_F(CompModelTest, DiffuseK0AdhPrecisionTest) {
  int rows = 200;
  int cols = 200;

  TRACE_MSG("Setting Specific Simulation Parameters...");
  config.dt_ = 3.75e-4;
  config.dx_ = 7.0755e-3;
  config.diff_t_ = 100;
  config.DA_ = 0.0003333333;
  config.DI_ = 0.0333333333;
  config.k0_ = 0.10;
  config.gamma_ = 1.0;
  config.delta_ = 1.0;
  config.A0_ = 0.4;
  config.s1_ = 0.7;
  config.s2_ = 0.05;
  config.F0_ = 0.5;
  config.kn_ = 1.0;
  config.ks_ = 0.25;
  config.eps_ = 0.1;
  config.sim_cols_ = cols;
  config.sim_rows_ = rows;
  double A_max = 0.67;
  double A_min = 0.044;

  TRACE_MSG("Instantiating Models...");
  test_utils::CellModelTest modern{config};
  Cell legacy;

  dcm::Mat_i nuc_mask = dcm::Mat_i::Zero(rows, cols);
  dcm::Mat_i cell_mask = dcm::Mat_i::Zero(rows, cols);
  fill_circle(nuc_mask, 100, 100, 15);
  fill_circle(cell_mask, 100, 100, 40);

  modern.set_cell(cell_mask);
  modern.set_nuc(nuc_mask);

  dcm::Mat_d init_fields =
      dcm::Mat_d::Constant(rows, cols, (A_max + A_min) / 2.0);
  modern.set_A(init_fields);
  modern.set_I(init_fields);
  modern.set_F(init_fields);
  modern.set_AC(init_fields);
  modern.set_IC(init_fields);
  modern.set_FC(init_fields);
  modern.set_k0_adh(dcm::Mat_d::Constant(rows, cols, config.k0_));

  TRACE_MSG("Syncing Legacy Parameters...");
  sync_params(legacy, config);

  legacy.Im_nuc = eigen_to_raw(nuc_mask.cast<double>());
  legacy.Im = eigen_to_raw(cell_mask.cast<double>());
  legacy.A = eigen_to_raw(modern.get_A());
  legacy.I = eigen_to_raw(modern.get_I());
  legacy.F = eigen_to_raw(modern.get_F());
  legacy.AC = eigen_to_raw(modern.get_AC());
  legacy.IC = eigen_to_raw(modern.get_IC());
  legacy.FC = eigen_to_raw(modern.get_FC());
  legacy.A_new = create_array2d(rows, cols);
  legacy.I_new = create_array2d(rows, cols);
  legacy.F_new = create_array2d(rows, cols);
  legacy.AC_new = create_array2d(rows, cols);
  legacy.IC_new = create_array2d(rows, cols);
  legacy.FC_new = create_array2d(rows, cols);
  legacy.k0_adh = eigen_to_raw(modern.get_k0_adh());

  TRACE_MSG("Executing modern diffuse_k0_adh...");
  modern.diffuse_k0_adh();

  TRACE_MSG("Executing legacy diffuse_k0_adh...");
  legacy.diffuse_k0_adh();

  TRACE_MSG("Verifying numerical parity...");

  double tolerance = 1e-12;

  test_mat_near(legacy.A, modern.get_A(), "A", tolerance);
  test_mat_near(legacy.AC, modern.get_AC(), "AC", tolerance);
  test_mat_near(legacy.F, modern.get_F(), "F", tolerance);

  TRACE_MSG("Freeing new arrays...");
  free_legacy(legacy.A_new);
  free_legacy(legacy.I_new);
  free_legacy(legacy.F_new);
  free_legacy(legacy.AC_new);
  free_legacy(legacy.IC_new);
  free_legacy(legacy.FC_new);

  TRACE_MSG("Test Finished Successfully.");
}
