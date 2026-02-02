#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils/test_utils.hpp>

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class CompModelTest : public test_utils::ModelTestBase {};

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
  config.k_nuc_ = 4.0;
  config.T_nuc_ = 1.0;
  config.R_nuc_ = 1.0;
  config.g_ = 2.0;
  config.R0_ = 20;
  config.dyn_basal_ = 0.9;
  config.sim_rows_ = rows;
  config.sim_cols_ = cols;

  TRACE_MSG("Instantiating Modern and Legacy models...");
  test_utils::CellModelTest modern{config};
  Cell legacy;

  TRACE_MSG("Creating masks...");
  dcm::Mat_i nuc_mask = dcm::Mat_i::Zero(rows, cols);
  dcm::Mat_i cell_mask = dcm::Mat_i::Zero(rows, cols);
  fill_circle(nuc_mask, 100, 100, 20);  // Center nucleus
  fill_circle(cell_mask, 100, 100, 50); // Larger cell

  TRACE_MSG("Setting Modern values...");
  modern.set_cell(cell_mask);
  modern.set_nuc(nuc_mask);
  modern.set_AC(dcm::Mat_d::Constant(rows, cols, 1.0));
  modern.set_IC(dcm::Mat_d::Constant(rows, cols, 1.0));
  modern.set_FC(dcm::Mat_d::Constant(rows, cols, 1.0));

  TRACE_MSG("Converting Eigen to Legacy Raw Pointers...");
  legacy.Im_nuc = eigen_to_raw(nuc_mask.cast<double>());
  legacy.Im = eigen_to_raw(cell_mask.cast<double>());
  legacy.AC = eigen_to_raw(modern.get_AC());
  legacy.IC = eigen_to_raw(modern.get_IC());
  legacy.FC = eigen_to_raw(modern.get_FC());
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

  TRACE_MSG("Checking outline calculations equal...");
  int legacy_perim = outline_4(
      legacy.Im_nuc, legacy.fr_rows_num, legacy.fr_cols_num, legacy.fr_rows_pos,
      legacy.fr_cols_pos, legacy.env_rows_num, legacy.env_cols_num);
  ASSERT_EQ(legacy_perim, modern.get_P_nuc()) << "Nucleus perimeter mismatch";

  TRACE_MSG("Executing Modern: retract_nuc_dep()...");
  modern.retract_nuc_dep();

  TRACE_MSG("Executing Legacy: retract_nuc()...");
  legacy.retract_nuc();

  TRACE_MSG("Checking Global State Parity...");
  EXPECT_EQ(legacy.V_nuc, modern.get_V_nuc()) << "Nucleus volume mismatch";

  TRACE_MSG("Loop 1/6: Checking Initial Nucleus Mask (Im_nuc)...");
  int nuc_mismatches = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if ((int)legacy.Im_nuc[i][j] != modern.get_nuc()(i, j)) {
        if (nuc_mismatches < 10) { // Limit logging to avoid wall of text
          std::cout << "  Im_nuc Mismatch at (" << i << "," << j << ") - "
                    << "Legacy: " << (int)legacy.Im_nuc[i][j]
                    << ", Modern: " << modern.get_nuc()(i, j) << std::endl;
        }
        nuc_mismatches++;
      }
    }
  }
  EXPECT_EQ(nuc_mismatches, 0)
      << "Nucleus masks do not match. Fix this before checking outlines.";

  TRACE_MSG("Loop 2/6: Checking Outer Outline (outline_nuc)...");
  int outer_mismatches = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int leg_val = (int)legacy.outline_nuc[i][j];
      int mod_val = (int)modern.get_outline_nuc().coeff(i, j);
      if (leg_val != mod_val) {
        if (outer_mismatches < 10) {
          std::cout << "  Outline Mismatch at (" << i << "," << j << ") - "
                    << "Legacy: " << leg_val << ", Modern: " << mod_val
                    << std::endl;
        }
        outer_mismatches++;
      }
    }
  }
  EXPECT_EQ(outer_mismatches, 0)
      << "Outer outlines do not match. Check connectivity (4 vs 8 neighbors).";

  TRACE_MSG("Loop 3/6: Checking Inner Outline (inner_outline_nuc)...");
  int inner_mismatches = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int leg_val = (int)legacy.inner_outline_nuc[i][j];
      int mod_val = (int)modern.get_inner_outline_nuc().coeff(i, j);
      if (leg_val != mod_val) {
        if (inner_mismatches < 10) {
          std::cout << "  Inner Outline Mismatch at (" << i << "," << j
                    << ") - "
                    << "Legacy: " << leg_val << ", Modern: " << mod_val
                    << std::endl;
        }
        inner_mismatches++;
      }
    }
  }
  EXPECT_EQ(inner_mismatches, 0) << "Inner outlines do not match.";

  TRACE_MSG("Loop 4/6: Checking AC (Actin Concentration)...");
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      EXPECT_NEAR(legacy.AC[i][j], modern.get_AC()(i, j), 1e-8)
          << "AC mismatch at (" << i << "," << j << ")";
    }
  }

  TRACE_MSG("Loop 5/6: Checking IC (Inhibitor Concentration)...");
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      EXPECT_NEAR(legacy.IC[i][j], modern.get_IC()(i, j), 1e-8)
          << "IC mismatch at (" << i << "," << j << ")";
    }
  }

  TRACE_MSG("Loop 6/6: Checking FC (Force Correlation)...");
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      EXPECT_NEAR(legacy.FC[i][j], modern.get_FC()(i, j), 1e-8)
          << "FC mismatch at (" << i << "," << j << ")";
    }
  }

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy.AC_cor_sum, modern.get_AC_cor_sum(), 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy.IC_cor_sum, modern.get_IC_cor_sum(), 1e-8)
      << "IC_cor_sum mismatch";

  TRACE_MSG("Comparing Dynein Field Outputs...");
  for (int i = 1; i < rows - 1; ++i) {
    for (int j = 1; j < cols - 1; ++j) {
      EXPECT_NEAR(legacy.test_dyn_f[i][j], modern.get_dyn_f()(i, j), 1e-8);
    }
  }

  TRACE_MSG("Test Finished Successfully.");
}
