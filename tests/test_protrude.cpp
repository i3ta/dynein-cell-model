#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils/test_utils.hpp>

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class CompModelTest : public test_utils::ModelTestBase {};

TEST_F(CompModelTest, ProtrudeAdhNucPushConsistency) {
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

  TRACE_MSG("Setting Simulation Parameters...");
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
  dcm::SpMat_i env_mask = fill_env(rows, cols);

  fill_circle(nuc_mask, 100, 100, 15);
  fill_circle(cell_mask, 99, 100, 16);

  modern.set_cell(cell_mask);
  modern.set_nuc(nuc_mask);
  modern.set_env(env_mask);

  modern.set_AC(dcm::Mat_d::Constant(rows, cols, 0.5));
  modern.set_IC(dcm::Mat_d::Constant(rows, cols, 0.3));
  modern.set_FC(dcm::Mat_d::Constant(rows, cols, 0.8));
  modern.set_A(dcm::Mat_d::Constant(rows, cols, 0.5));
  modern.set_I(dcm::Mat_d::Constant(rows, cols, 0.3));
  modern.set_F(dcm::Mat_d::Constant(rows, cols, 0.8));

  modern.rearrange_adhesions(false, true); // initialize adhesions

  TRACE_MSG("Syncing Legacy parameters...");
  sync_params(legacy, config);
  legacy.V = modern.get_V();
  legacy.V0 = modern.get_V();

  legacy.Im_nuc = eigen_to_raw(nuc_mask.cast<double>());
  legacy.Im = eigen_to_raw(cell_mask.cast<double>());
  legacy.outline = eigen_to_raw(modern.get_outline().cast<double>());
  legacy.inner_outline =
      eigen_to_raw(modern.get_inner_outline().cast<double>());
  legacy.outline_nuc = eigen_to_raw(modern.get_outline_nuc().cast<double>());
  legacy.inner_outline_nuc =
      eigen_to_raw(modern.get_inner_outline_nuc().cast<double>());
  legacy.k0_adh = eigen_to_raw(modern.get_k0_adh());
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
  legacy.adh_f = eigen_to_raw(modern.get_adh_f());

  drand.reset_all_instances();

  TRACE_MSG("Executing modern protrude()...");
  modern.protrude();

  TRACE_MSG("Executing legacy protrude_adh_nuc_push()...");
  legacy.protrude_adh_nuc_push();

  TRACE_MSG("Comparing Results...");
  EXPECT_NEAR(legacy.V, modern.get_V(), 1e-7) << "Final volume mismatch";

  TRACE_MSG("Loop 1/9: Checking Initial Cell Mask...");
  test_mat(legacy.Im, modern.get_cell(), "Cell");

  TRACE_MSG("Loop 2/9: Checking Outer Outline...");
  test_outline(legacy.outline, modern.get_outline(), "Outline");

  TRACE_MSG("Loop 3/9: Checking Inner Outline (inner_outline)...");
  test_outline(legacy.inner_outline, modern.get_inner_outline(),
               "Inner Outline");

  TRACE_MSG("Loop 4/9: Checking A...");
  test_mat_near(legacy.A, modern.get_A(), "A");

  TRACE_MSG("Loop 5/9: Checking I...");
  test_mat_near(legacy.I, modern.get_I(), "I");

  TRACE_MSG("Loop 6/9: Checking F...");
  test_mat_near(legacy.F, modern.get_F(), "F");

  TRACE_MSG("Loop 7/9: Checking AC...");
  test_mat_near(legacy.AC, modern.get_AC(), "AC");

  TRACE_MSG("Loop 8/9: Checking IC...");
  test_mat_near(legacy.IC, modern.get_IC(), "IC");

  TRACE_MSG("Loop 9/9: Checking FC...");
  test_mat_near(legacy.FC, modern.get_FC(), "FC");

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy.AC_cor_sum, modern.get_AC_cor_sum(), 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy.IC_cor_sum, modern.get_IC_cor_sum(), 1e-8)
      << "IC_cor_sum mismatch";

  TRACE_MSG("Freeing new arrays...");
  free_legacy(legacy.A_new);
  free_legacy(legacy.I_new);
  free_legacy(legacy.F_new);
  free_legacy(legacy.AC_new);
  free_legacy(legacy.IC_new);
  free_legacy(legacy.FC_new);

  TRACE_MSG("Test Finished.");
}
