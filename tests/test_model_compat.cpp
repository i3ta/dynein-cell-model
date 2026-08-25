#include <gtest/gtest.h>
#include <iostream>

#include "cell_nuc/cell_nuc.hpp"
#include "dynein_cell_model/diffusion.h"
#include "dynein_cell_model/morphology.h"
#include "dynein_cell_model/state.h"
#include "test_utils/test_utils.h"

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class ModelCompatTest : public ::testing::Test {
protected:
  const int rows = 200;
  const int cols = 200;

  std::unique_ptr<dcm::CellModelConfig> config;
  std::unique_ptr<dcm::CellState> modern;
  std::unique_ptr<Cell> legacy;

  std::vector<double **> legacyPointers;
  std::vector<int **> legacyPointersInt;

  void SetUp() override {
    test_utils::DebugRand<double> drand;
    std::vector<double> mockProbs;
    mockProbs.reserve(1000);

    std::mt19937 tempEngine(124);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 1000; ++i) {
      mockProbs.push_back(dist(tempEngine));
    }
    drand.setOutputs(std::move(mockProbs));
    drand.resetAllInstances();

    // set config variables
    config = std::make_unique<dcm::CellModelConfig>();
    config->dt = 3.75e-4;
    config->dx = 7.0755e-3;
    config->diffT = 100;
    config->DA = 0.0003333333;
    config->DI = 0.0333333333;
    config->k0 = 0.10;
    config->gamma = 1.0;
    config->delta = 1.0;
    config->A0 = 0.4;
    config->s1 = 0.7;
    config->s2 = 0.05;
    config->F0 = 0.5;
    config->kn = 1.0;
    config->ks = 0.25;
    config->eps = 0.1;
    config->simCols = cols;
    config->simRows = rows;

    // set up cell and environment
    dcm::Mat_i nucMask = dcm::Mat_i::Zero(rows, cols);
    dcm::Mat_i cellMask = dcm::Mat_i::Zero(rows, cols);
    dcm::SpMat_i envMask = fill_env(rows, cols);

    fill_circle(nucMask, 100, 100, 10);
    fill_circle(cellMask, 100, 100, 20);

    modern = std::make_unique<dcm::CellState>(*config);
    modern->cell = cellMask;
    modern->nuc = nucMask;
    modern->env = envMask;

    // set up cell dynamics values
    modern->AC = dcm::Mat_d::Constant(rows, cols, 0.5);
    modern->IC = dcm::Mat_d::Constant(rows, cols, 0.3);
    modern->FC = dcm::Mat_d::Constant(rows, cols, 0.8);
    modern->A = dcm::Mat_d::Constant(rows, cols, 0.5);
    modern->I = dcm::Mat_d::Constant(rows, cols, 0.3);
    modern->F = dcm::Mat_d::Constant(rows, cols, 0.8);

    // initialize adhesions to realistic values
    dcm::rearrangeAdhesions(*modern, false, true);

    // sync legacy model with modern
    legacy = std::make_unique<Cell>();
    legacy->V = modern->V;
    legacy->V0 = modern->V;
    legacy->V_nuc = modern->VNuc;
    legacy->V0_nuc = modern->VNuc;
    legacy->A_cor_sum = 0;
    legacy->I_cor_sum = 0;
    legacy->AC_cor_sum = 0;
    legacy->IC_cor_sum = 0;

    sync_params(*legacy, *config);
    legacy->Im_nuc = eigen_to_raw(nucMask.cast<double>());
    legacy->Im = eigen_to_raw(cellMask.cast<double>());
    legacy->outline = eigen_to_raw(modern->outline.mask().cast<double>());
    legacy->inner_outline = eigen_to_raw(modern->innerOutline.mask().cast<double>());
    legacy->outline_nuc = eigen_to_raw(modern->outlineNuc.mask().cast<double>());
    legacy->inner_outline_nuc =
        eigen_to_raw(modern->innerOutlineNuc.mask().cast<double>());
    legacy->k0_adh = eigen_to_raw(modern->k0Adh);
    legacy->A = eigen_to_raw(modern->A);
    legacy->I = eigen_to_raw(modern->I);
    legacy->F = eigen_to_raw(modern->F);
    legacy->AC = eigen_to_raw(modern->AC);
    legacy->IC = eigen_to_raw(modern->IC);
    legacy->FC = eigen_to_raw(modern->FC);
    legacy->A_new = create_array2d(rows, cols);
    legacy->I_new = create_array2d(rows, cols);
    legacy->F_new = create_array2d(rows, cols);
    legacy->AC_new = create_array2d(rows, cols);
    legacy->IC_new = create_array2d(rows, cols);
    legacy->FC_new = create_array2d(rows, cols);
    legacy->adh_f = eigen_to_raw(modern->adhF);
    legacy->adh = eigen_to_raw(dcm::Mat_d(modern->adh.cast<double>()));
    legacy->env = eigen_to_raw(dcm::Mat_d(modern->env.cast<double>()));
    legacy->adh_g = create_array2d(rows, cols);
    legacy->CoM_track = create_array2d(rows, cols);
    legacy->adh_r_pos = new int[config->adhNum];
    legacy->adh_c_pos = new int[config->adhNum];
    for (int i = 0; i < config->adhNum; ++i) {
      legacy->adh_r_pos[i] = modern->adhPos(0, i);
      legacy->adh_c_pos[i] = modern->adhPos(1, i);
    }

    // reset rng
    drand.resetAllInstances();
  }

  void TearDown() override {
    // for (double **p : legacyPointers) {
    //   free(p);
    // }
    // for (int **p : legacyPointersInt) {
    //   free(p);
    // }
    // legacyPointers.clear();
    // legacyPointersInt.clear();
    //
    // free_legacy(legacy->A_new);
    // free_legacy(legacy->I_new);
    // free_legacy(legacy->F_new);
    // free_legacy(legacy->AC_new);
    // free_legacy(legacy->IC_new);
    // free_legacy(legacy->FC_new);
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

    legacyPointers.push_back(raw);

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

    legacyPointersInt.push_back(raw);

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

  dcm::SpMat_i fill_env(int rows, int cols) {
    dcm::SpMat_i env_mask{rows, cols};
    std::vector<Eigen::Triplet<int>> t;
    int cr = rows / 2, cc = cols / 2;

    for (int i = 0; i < std::max(rows, cols); ++i) {
      for (int w = -2; w <= 2; ++w) {
        if (i < cols && (cr + w) >= 0 && (cr + w) < rows)
          t.push_back({cr + w, i, 1});
        if (i < rows && (cc + w) >= 0 && (cc + w) < cols)
          t.push_back({i, cc + w, 1});
      }
    }

    env_mask.setFromTriplets(t.begin(), t.end());
    for (int i = 0; i < env_mask.nonZeros(); ++i)
      if (env_mask.valuePtr()[i] > 1)
        env_mask.valuePtr()[i] = 1;

    return env_mask;
  }

  void sync_params(Cell &legacy, const dcm::CellModelConfig &config) {
    legacy.k = config.k;
    legacy.k_nuc = config.kNuc;
    legacy.g = config.g;
    legacy.T = config.T;
    legacy.T_nuc = config.TNuc;
    legacy.act_slope = config.actSlope;
    legacy.R0 = config.R0;
    legacy.R_nuc = config.RNuc;
    legacy.prop_factor = config.propFactor;
    legacy.d_basal = config.dynBasal;

    legacy.DA = config.DA;
    legacy.DI = config.DI;
    legacy.k0 = config.k0;
    legacy.k0_min = config.k0Min;
    legacy.scalar = config.k0Scalar;
    legacy.gamma = config.gamma;
    legacy.delta = config.delta;
    legacy.A0 = config.A0;
    legacy.s1 = config.s1;
    legacy.s2 = config.s2;
    legacy.F0 = config.F0;
    legacy.kn = config.kn;
    legacy.ks = config.ks;
    legacy.eps = config.eps;
    legacy.dt = config.dt;
    legacy.dx = config.dx;

    legacy.A_max = config.AMax;
    legacy.A_min = config.AMin;
    legacy.AC_max = config.ACMax;
    legacy.AC_min = config.ACMin;

    legacy.adh_num = config.adhNum;
    legacy.adh_frac = config.adhFrac;
    legacy.adh_sigma = config.adhSigma;
    legacy.adh_basal_prot = config.adhBasal;

    legacy.diff_t = config.diffT;
    legacy.fr_dist = config.framePadding;

    legacy.env_rows_num = config.simRows;
    legacy.env_cols_num = config.simCols;

    legacy.fr_rows_pos = 1;
    legacy.fr_cols_pos = 1;

    legacy.fr_rows_num = config.simRows - 2;
    legacy.fr_cols_num = config.simCols - 2;
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
    ASSERT_EQ(mat_mismatches, 0)
        << test_name
        << " masks do not match. Fix this before checking outlines.";
  }

  void test_outline(double **legacy, const dcm::OutlineMask &modern,
                    const std::string &test_name) {
    int outline_mismatches = 0;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int leg_val = static_cast<int>(legacy[i][j]);
        int mod_val = static_cast<int>(modern.contains(i, j));
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
    ASSERT_EQ(outline_mismatches, 0) << "Outer outlines do not match. Check "
                                        "connectivity (4 vs 8 neighbors).";
  }

  void test_mat_near(double **legacy, const dcm::Mat_d &modern,
                     const std::string &test_name, const double tol = 1e-8) {
    int near_mismatches = 0;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        double leg_val = legacy[i][j];
        double mod_val = modern(i, j);
        // Check if absolute difference exceeds tolerance
        if (std::abs(leg_val - mod_val) > tol) {
          if (near_mismatches < 10) {
            std::cout << test_name << " Near Mismatch at (" << i << "," << j
                      << ") - "
                      << "Legacy: " << leg_val << ", Modern: " << mod_val
                      << ", Diff: " << std::abs(leg_val - mod_val) << std::endl;
          }
          near_mismatches++;
        }
      }
    }
    ASSERT_EQ(near_mismatches, 0)
        << test_name << " values outside tolerance (" << tol << ").";
  }
};

TEST_F(ModelCompatTest, DiffuseK0AdhCompatability) {
  TRACE_MSG("Executing modern diffuseK0Adh...");
  dcm::diffuseK0Adh(*modern);

  TRACE_MSG("Executing legacy diffuse_k0_adh...");
  legacy->diffuse_k0_adh();

  TRACE_MSG("Verifying numerical parity...");
  double tolerance = 1e-12;

  test_mat_near(legacy->A, modern->A, "A", tolerance);
  test_mat_near(legacy->AC, modern->AC, "AC", tolerance);
  test_mat_near(legacy->F, modern->F, "F", tolerance);
  TRACE_MSG("Test Finished Successfully.");
}

TEST_F(ModelCompatTest, DyneinFieldProtrCompatability) {
  TRACE_MSG("Generating Legacy Dynein Field...");
  double **legacy_dyn_f_prot = generate_dyn_field_protr(
      legacy->Im, legacy->Im_nuc, legacy->inner_outline, legacy->outline_nuc,
      legacy->fr_rows_num, legacy->fr_cols_num, legacy->fr_rows_pos,
      legacy->fr_cols_pos, legacy->env_rows_num, legacy->env_cols_num,
      legacy->AC);

  TRACE_MSG("Generating New Dynein Field...");
  dcm::generateDynField(*modern, modern->innerOutline, modern->outlineNuc,
                        false);
  const auto &modern_dynFProt = modern->dynF;

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacy_dyn_f_prot, modern_dynFProt, "dyn_f");

  free_legacy(legacy_dyn_f_prot);
}

TEST_F(ModelCompatTest, DyneinFieldRetrCompatability) {
  TRACE_MSG("Generating Legacy Dynein Field...");
  double **legacy_dyn_f_retr = generate_dyn_field_retr(
      legacy->Im, legacy->Im_nuc, legacy->inner_outline, legacy->outline_nuc,
      legacy->fr_rows_num, legacy->fr_cols_num, legacy->fr_rows_pos,
      legacy->fr_cols_pos, legacy->env_rows_num, legacy->env_cols_num,
      legacy->AC);

  TRACE_MSG("Generating New Dynein Field...");
  dcm::generateDynField(*modern, modern->innerOutline, modern->outlineNuc,
                        true);
  const auto &modern_dynFRetr = modern->dynF;

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacy_dyn_f_retr, modern_dynFRetr, "dyn_f");

  free_legacy(legacy_dyn_f_retr);
}

TEST_F(ModelCompatTest, ProtrudeCompatability) {
  TRACE_MSG("Executing modern protrude()...");
  dcm::protrudeCell(*modern);

  TRACE_MSG("Executing legacy protrude_adh_nuc_push()...");
  legacy->protrude_adh_nuc_push();

  TRACE_MSG("Comparing Results...");
  EXPECT_NEAR(legacy->V, modern->V, 1e-7) << "Final volume mismatch";

  TRACE_MSG("Loop 1/9: Checking Initial Cell Mask...");
  test_mat(legacy->Im, modern->cell, "Cell");

  TRACE_MSG("Loop 2/9: Checking Outer Outline...");
  test_outline(legacy->outline, modern->outline, "Outline");

  TRACE_MSG("Loop 3/9: Checking Inner Outline (inner_outline)...");
  test_outline(legacy->inner_outline, modern->innerOutline, "Inner Outline");

  TRACE_MSG("Loop 4/9: Checking A...");
  test_mat_near(legacy->A, modern->A, "A");

  TRACE_MSG("Loop 5/9: Checking I...");
  test_mat_near(legacy->I, modern->I, "I");

  TRACE_MSG("Loop 6/9: Checking F...");
  test_mat_near(legacy->F, modern->F, "F");

  TRACE_MSG("Loop 7/9: Checking AC...");
  test_mat_near(legacy->AC, modern->AC, "AC");

  TRACE_MSG("Loop 8/9: Checking IC...");
  test_mat_near(legacy->IC, modern->IC, "IC");

  TRACE_MSG("Loop 9/9: Checking FC...");
  test_mat_near(legacy->FC, modern->FC, "FC");

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy->AC_cor_sum, modern->ACCorSum, 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy->IC_cor_sum, modern->ICCorSum, 1e-8)
      << "IC_cor_sum mismatch";
}

TEST_F(ModelCompatTest, ProtrudeNucCompatability) {
  TRACE_MSG("Executing Modern: protrude_nuc_dep()...");
  dcm::protrudeNucDep(*modern);

  TRACE_MSG("Executing Legacy: protrude_nuc()...");
  legacy->protrude_nuc();

  TRACE_MSG("Checking Global State Parity...");
  EXPECT_EQ(legacy->V_nuc, modern->VNuc) << "Nucleus volume mismatch";

  TRACE_MSG("Loop 1/6: Checking Initial Nucleus Mask...");
  test_mat(legacy->Im_nuc, modern->nuc, "Nucleus");

  TRACE_MSG("Loop 2/6: Checking Outer Outline...");
  test_outline(legacy->outline_nuc, modern->outlineNuc, "Nucleus Outline");

  TRACE_MSG("Loop 3/6: Checking Inner Outline...");
  test_outline(legacy->inner_outline_nuc, modern->innerOutlineNuc,
               "Nucleus Inner Outline");

  TRACE_MSG("Loop 4/6: Checking AC...");
  test_mat_near(legacy->AC, modern->AC, "AC");

  TRACE_MSG("Loop 5/6: Checking IC...");
  test_mat_near(legacy->IC, modern->IC, "IC");

  TRACE_MSG("Loop 6/6: Checking FC...");
  test_mat_near(legacy->FC, modern->FC, "FC");

  TRACE_MSG("Comparing Dynein Field Outputs...");
  double **legacyDynF = generate_dyn_field_protr(
      legacy->Im, legacy->Im_nuc, legacy->inner_outline, legacy->outline_nuc,
      legacy->fr_rows_num, legacy->fr_cols_num, legacy->fr_rows_pos,
      legacy->fr_cols_pos, legacy->env_rows_num, legacy->env_cols_num,
      legacy->AC);
  test_mat_near(legacyDynF, modern->dynF, "dyn_f");
  free_array2d(legacyDynF);

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy->AC_cor_sum, modern->ACCorSum, 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy->IC_cor_sum, modern->ICCorSum, 1e-8)
      << "IC_cor_sum mismatch";
}

TEST_F(ModelCompatTest, RetractCompatability) {
  TRACE_MSG("Executing modern retract()...");
  retractCell(*modern);

  TRACE_MSG("Executing legacy retract()...");
  legacy->retract();

  TRACE_MSG("Comparing Results...");
  EXPECT_NEAR(legacy->V, modern->V, 1e-7) << "Final volume mismatch";

  TRACE_MSG("Loop 1/9: Checking Initial Cell Mask...");
  test_mat(legacy->Im, modern->cell, "Cell");

  TRACE_MSG("Loop 2/9: Checking Outer Outline...");
  test_outline(legacy->outline, modern->outline, "Outline");

  TRACE_MSG("Loop 3/9: Checking Inner Outline (inner_outline)...");
  test_outline(legacy->inner_outline, modern->innerOutline, "Inner Outline");

  TRACE_MSG("Loop 4/9: Checking A...");
  test_mat_near(legacy->A, modern->A, "A");

  TRACE_MSG("Loop 5/9: Checking I...");
  test_mat_near(legacy->I, modern->I, "I");

  TRACE_MSG("Loop 6/9: Checking F...");
  test_mat_near(legacy->F, modern->F, "F");

  TRACE_MSG("Loop 7/9: Checking AC...");
  test_mat_near(legacy->AC, modern->AC, "AC");

  TRACE_MSG("Loop 8/9: Checking IC...");
  test_mat_near(legacy->IC, modern->IC, "IC");

  TRACE_MSG("Loop 9/9: Checking FC...");
  test_mat_near(legacy->FC, modern->FC, "FC");

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy->AC_cor_sum, modern->ACCorSum, 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy->IC_cor_sum, modern->ICCorSum, 1e-8)
      << "IC_cor_sum mismatch";
}

TEST_F(ModelCompatTest, RetractNucCompatability) {
  TRACE_MSG("Executing Modern: retract_nuc_dep()...");
  dcm::retractNucDep(*modern);

  TRACE_MSG("Executing Legacy: retract_nuc()...");
  // retract_nuc computes its dynein field from the pre-retraction state, so
  // snapshot the field before the nucleus changes.
  double **legacyDynF = generate_dyn_field_retr(
      legacy->Im, legacy->Im_nuc, legacy->inner_outline,
      legacy->inner_outline_nuc, legacy->fr_rows_num, legacy->fr_cols_num,
      legacy->fr_rows_pos, legacy->fr_cols_pos, legacy->env_rows_num,
      legacy->env_cols_num, legacy->AC);
  legacy->retract_nuc();

  TRACE_MSG("Checking Global State Parity...");
  EXPECT_EQ(legacy->V_nuc, modern->VNuc) << "Nucleus volume mismatch";

  TRACE_MSG("Loop 1/6: Checking Initial Nucleus Mask...");
  test_mat(legacy->Im_nuc, modern->nuc, "Nucleus");

  TRACE_MSG("Loop 2/6: Checking Outer Outline...");
  test_outline(legacy->outline_nuc, modern->outlineNuc, "Nucleus Outline");

  TRACE_MSG("Loop 3/6: Checking Inner Outline...");
  test_outline(legacy->inner_outline_nuc, modern->innerOutlineNuc,
               "Nucleus Inner Outline");

  TRACE_MSG("Loop 4/6: Checking AC...");
  test_mat_near(legacy->AC, modern->AC, "AC");

  TRACE_MSG("Loop 5/6: Checking IC...");
  test_mat_near(legacy->IC, modern->IC, "IC");

  TRACE_MSG("Loop 6/6: Checking FC...");
  test_mat_near(legacy->FC, modern->FC, "FC");

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacyDynF, modern->dynF, "dyn_f");
  free_array2d(legacyDynF);

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy->AC_cor_sum, modern->ACCorSum, 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy->IC_cor_sum, modern->ICCorSum, 1e-8)
      << "IC_cor_sum mismatch";
}

TEST_F(ModelCompatTest, CorrectConcentrationCompatability) {
  TRACE_MSG("Executing protrude() to put cell in a state with corrections...");
  // NOTE: The protrude tests must pass for this test to work properly
  dcm::protrudeCell(*modern);
  legacy->protrude_adh_nuc_push();

  TRACE_MSG("Testing modern correctConcentrations()...")
  dcm::correctConcentrations(*modern);

  TRACE_MSG("Testing legacy correct_concentrations()...")
  legacy->correct_concentrations();

  TRACE_MSG("Loop 1/4: Checking A...");
  test_mat_near(legacy->A, modern->A, "A");

  TRACE_MSG("Loop 2/4: Checking I...");
  test_mat_near(legacy->I, modern->I, "I");

  TRACE_MSG("Loop 3/4: Checking AC...");
  test_mat_near(legacy->AC, modern->AC, "AC");

  TRACE_MSG("Loop 4/4: Checking IC...");
  test_mat_near(legacy->IC, modern->IC, "IC");
}
