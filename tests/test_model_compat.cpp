#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils/test_utils.hpp>

#define TRACE_MSG(msg)                                                         \
  std::cerr << "[ TRACE    ] " << msg << std::endl << std::flush;

namespace dcm = dynein_cell_model;

class ModelCompatTest : public ::testing::Test {
protected:
  int rows = 200;
  int cols = 200;

  dcm::CellModelConfig config;
  test_utils::CellModelTest modern;
  Cell legacy;

  std::vector<double **> legacy_pointers;
  std::vector<int **> legacy_pointers_int;

  void SetUp() override {
    test_utils::DebugRand<double> drand;
    std::vector<double> mock_probs;
    mock_probs.reserve(1000);

    std::mt19937 temp_engine(124);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 1000; ++i) {
      mock_probs.push_back(dist(temp_engine));
    }
    drand.set_outputs(std::move(mock_probs));
    drand.reset_all_instances();

    // set config variables
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

    // init modern model
    modern.init(config);

    // set up cell and environment
    dcm::Mat_i nuc_mask = dcm::Mat_i::Zero(rows, cols);
    dcm::Mat_i cell_mask = dcm::Mat_i::Zero(rows, cols);
    dcm::SpMat_i env_mask = fill_env(rows, cols);

    fill_circle(nuc_mask, 100, 100, 10);
    fill_circle(cell_mask, 100, 100, 20);

    modern.set_cell(cell_mask);
    modern.set_nuc(nuc_mask);
    modern.set_env(env_mask);

    // set up cell dynamics values
    modern.set_AC(dcm::Mat_d::Constant(rows, cols, 0.5));
    modern.set_IC(dcm::Mat_d::Constant(rows, cols, 0.3));
    modern.set_FC(dcm::Mat_d::Constant(rows, cols, 0.8));
    modern.set_A(dcm::Mat_d::Constant(rows, cols, 0.5));
    modern.set_I(dcm::Mat_d::Constant(rows, cols, 0.3));
    modern.set_F(dcm::Mat_d::Constant(rows, cols, 0.8));

    // initialize adhesions to realistic values
    modern.rearrange_adhesions(false, true);

    // sync legacy model with modern
    legacy.V = modern.get_V();
    legacy.V0 = modern.get_V();
    legacy.V_nuc = modern.get_V_nuc();
    legacy.V0_nuc = modern.get_V_nuc();

    sync_params(legacy, config);
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

    // reset rng
    drand.reset_all_instances();
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

    free_legacy(legacy.A_new);
    free_legacy(legacy.I_new);
    free_legacy(legacy.F_new);
    free_legacy(legacy.AC_new);
    free_legacy(legacy.IC_new);
    free_legacy(legacy.FC_new);
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
    ASSERT_EQ(mat_mismatches, 0)
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
  TRACE_MSG("Executing modern diffuse_k0_adh...");
  modern.diffuse_k0_adh();

  TRACE_MSG("Executing legacy diffuse_k0_adh...");
  legacy.diffuse_k0_adh();

  TRACE_MSG("Verifying numerical parity...");
  double tolerance = 1e-12;

  test_mat_near(legacy.A, modern.get_A(), "A", tolerance);
  test_mat_near(legacy.AC, modern.get_AC(), "AC", tolerance);
  test_mat_near(legacy.F, modern.get_F(), "F", tolerance);
  TRACE_MSG("Test Finished Successfully.");
}

TEST_F(ModelCompatTest, DyneinFieldProtrCompatability) {
  TRACE_MSG("Generating Legacy Dynein Field...");
  double **legacy_dyn_f_prot = generate_dyn_field_protr(
      legacy.Im, legacy.Im_nuc, legacy.inner_outline, legacy.outline_nuc,
      legacy.fr_rows_num, legacy.fr_cols_num, legacy.fr_rows_pos,
      legacy.fr_cols_pos, legacy.env_rows_num, legacy.env_cols_num, legacy.AC);

  TRACE_MSG("Generating New Dynein Field...");
  modern.generate_dyn_field(modern.get_inner_outline(),
                            modern.get_outline_nuc(), false);
  const auto &modern_dyn_f_prot = modern.get_dyn_f();

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacy_dyn_f_prot, modern_dyn_f_prot, "dyn_f");

  free_legacy(legacy_dyn_f_prot);
}

TEST_F(ModelCompatTest, DyneinFieldRetrCompatability) {
  TRACE_MSG("Generating Legacy Dynein Field...");
  double **legacy_dyn_f_retr = generate_dyn_field_retr(
      legacy.Im, legacy.Im_nuc, legacy.inner_outline, legacy.outline_nuc,
      legacy.fr_rows_num, legacy.fr_cols_num, legacy.fr_rows_pos,
      legacy.fr_cols_pos, legacy.env_rows_num, legacy.env_cols_num, legacy.AC);

  TRACE_MSG("Generating New Dynein Field...");
  modern.generate_dyn_field(modern.get_inner_outline(),
                            modern.get_outline_nuc(), true);
  const auto &modern_dyn_f_retr = modern.get_dyn_f();

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacy_dyn_f_retr, modern_dyn_f_retr, "dyn_f");

  free_legacy(legacy_dyn_f_retr);
}

TEST_F(ModelCompatTest, ProtrudeCompatability) {
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
}

TEST_F(ModelCompatTest, ProtrudeNucCompatability) {
  TRACE_MSG("Executing Modern: protrude_nuc_dep()...");
  modern.protrude_nuc_dep();

  TRACE_MSG("Executing Legacy: protrude_nuc()...");
  legacy.protrude_nuc();

  TRACE_MSG("Checking Global State Parity...");
  EXPECT_EQ(legacy.V_nuc, modern.get_V_nuc()) << "Nucleus volume mismatch";

  TRACE_MSG("Loop 1/6: Checking Initial Nucleus Mask...");
  test_mat(legacy.Im_nuc, modern.get_nuc(), "Nucleus");

  TRACE_MSG("Loop 2/6: Checking Outer Outline...");
  test_outline(legacy.outline_nuc, modern.get_outline_nuc(), "Nucleus Outline");

  TRACE_MSG("Loop 3/6: Checking Inner Outline...");
  test_outline(legacy.inner_outline_nuc, modern.get_inner_outline_nuc(),
               "Nucleus Inner Outline");

  TRACE_MSG("Loop 4/6: Checking AC...");
  test_mat_near(legacy.AC, modern.get_AC(), "AC");

  TRACE_MSG("Loop 5/6: Checking IC...");
  test_mat_near(legacy.IC, modern.get_IC(), "IC");

  TRACE_MSG("Loop 6/6: Checking FC...");
  test_mat_near(legacy.FC, modern.get_FC(), "FC");

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacy.test_dyn_f, modern.get_dyn_f(), "dyn_f");

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy.AC_cor_sum, modern.get_AC_cor_sum(), 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy.IC_cor_sum, modern.get_IC_cor_sum(), 1e-8)
      << "IC_cor_sum mismatch";
}

TEST_F(ModelCompatTest, RetractCompatability) {
  TRACE_MSG("Executing modern retract()...");
  modern.retract();

  TRACE_MSG("Executing legacy retract()...");
  legacy.retract();

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
}

TEST_F(ModelCompatTest, RetractNucCompatability) {
  TRACE_MSG("Executing Modern: retract_nuc_dep()...");
  modern.retract_nuc_dep();

  TRACE_MSG("Executing Legacy: retract_nuc()...");
  legacy.retract_nuc();

  TRACE_MSG("Checking Global State Parity...");
  EXPECT_EQ(legacy.V_nuc, modern.get_V_nuc()) << "Nucleus volume mismatch";

  TRACE_MSG("Loop 1/6: Checking Initial Nucleus Mask...");
  test_mat(legacy.Im_nuc, modern.get_nuc(), "Nucleus");

  TRACE_MSG("Loop 2/6: Checking Outer Outline...");
  test_outline(legacy.outline_nuc, modern.get_outline_nuc(), "Nucleus Outline");

  TRACE_MSG("Loop 3/6: Checking Inner Outline...");
  test_outline(legacy.inner_outline_nuc, modern.get_inner_outline_nuc(),
               "Nucleus Inner Outline");

  TRACE_MSG("Loop 4/6: Checking AC...");
  test_mat_near(legacy.AC, modern.get_AC(), "AC");

  TRACE_MSG("Loop 5/6: Checking IC...");
  test_mat_near(legacy.IC, modern.get_IC(), "IC");

  TRACE_MSG("Loop 6/6: Checking FC...");
  test_mat_near(legacy.FC, modern.get_FC(), "FC");

  TRACE_MSG("Comparing Dynein Field Outputs...");
  test_mat_near(legacy.test_dyn_f, modern.get_dyn_f(), "dyn_f");

  TRACE_MSG("Checking Coordination Sum Parity...");
  EXPECT_NEAR(legacy.AC_cor_sum, modern.get_AC_cor_sum(), 1e-8)
      << "AC_cor_sum mismatch";
  EXPECT_NEAR(legacy.IC_cor_sum, modern.get_IC_cor_sum(), 1e-8)
      << "IC_cor_sum mismatch";
}
