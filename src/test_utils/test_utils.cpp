#include <dynein_cell_model/dynein_cell_model.hpp>
#include <test_utils/test_utils.hpp>

namespace test_utils {
CellModelTest::CellModelTest(const dcm::CellModelConfig &conf) { init(conf); }

void CellModelTest::init(const dcm::CellModelConfig &conf) {
  model.update_config(conf);
}

void CellModelTest::rearrange_adhesions(const bool bias,
                                        const bool rearrange_all) {
  model.rearrange_adhesions(bias, rearrange_all);
}

void CellModelTest::protrude() { model.protrude(); }

void CellModelTest::protrude_nuc_dep() { model.protrude_nuc_dep(); }

void CellModelTest::retract_nuc_dep() { model.retract_nuc_dep(); }

void CellModelTest::generate_dyn_field(const dcm::SpMat_i &cell_outline,
                                       const dcm::SpMat_i &nuc_outline,
                                       bool retract) {
  model.generate_dyn_field(cell_outline, nuc_outline, retract);
}

void CellModelTest::diffuse_k0_adh() { model.diffuse_k0_adh(); }

void CellModelTest::set_cell(const dcm::Mat_i cell) { model.set_cell(cell); }

void CellModelTest::set_nuc(const dcm::Mat_i nuc) { model.set_nuc(nuc); }

void CellModelTest::set_adh(const dcm::SpMat_i adh) { model.set_adh(adh); }

void CellModelTest::set_A(const dcm::Mat_d A) { model.set_A(A); }

void CellModelTest::set_AC(const dcm::Mat_d AC) { model.set_AC(AC); }

void CellModelTest::set_I(const dcm::Mat_d I) { model.set_I(I); }

void CellModelTest::set_IC(const dcm::Mat_d IC) { model.set_IC(IC); }

void CellModelTest::set_F(const dcm::Mat_d F) { model.set_F(F); }

void CellModelTest::set_FC(const dcm::Mat_d FC) { model.set_FC(FC); }

void CellModelTest::set_env(const dcm::SpMat_i env) { model.set_env(env); }

void CellModelTest::set_k0_adh(const dcm::Mat_d k0_adh) {
  model.k0_adh_ = k0_adh;
}

const dcm::Mat_i &CellModelTest::get_cell() { return model.cell_; }

const dcm::Mat_i &CellModelTest::get_nuc() { return model.nuc_; }

const dcm::SpMat_i &CellModelTest::get_adh() { return model.adh_; }

const dcm::Mat_i &CellModelTest::get_adh_pos() { return model.adh_pos_; }

const dcm::Mat_d &CellModelTest::get_A() { return model.A_; }

const dcm::Mat_d &CellModelTest::get_AC() { return model.AC_; }

const dcm::Mat_d &CellModelTest::get_I() { return model.I_; }

const dcm::Mat_d &CellModelTest::get_IC() { return model.IC_; }

const dcm::Mat_d &CellModelTest::get_F() { return model.F_; }

const dcm::Mat_d &CellModelTest::get_FC() { return model.FC_; }

const dcm::Mat_d &CellModelTest::get_k0_adh() { return model.k0_adh_; }

const dcm::SpMat_i &CellModelTest::get_env() { return model.env_; }

const dcm::SpMat_i &CellModelTest::get_outline() { return model.outline_; }

const dcm::SpMat_i &CellModelTest::get_outline_nuc() {
  return model.outline_nuc_;
}

const dcm::SpMat_i &CellModelTest::get_inner_outline() {
  return model.inner_outline_;
}

const dcm::SpMat_i &CellModelTest::get_inner_outline_nuc() {
  return model.inner_outline_nuc_;
}

const dcm::Mat_d &CellModelTest::get_dyn_f() { return model.dyn_f_; }

const double CellModelTest::get_AC_cor_sum() { return model.AC_cor_sum_; }

const double CellModelTest::get_IC_cor_sum() { return model.IC_cor_sum_; }

const int CellModelTest::get_V() { return model.V_; }

const int CellModelTest::get_V_nuc() { return model.V_nuc_; }

const int CellModelTest::get_P_nuc() { return model.P_nuc_; }
}; // namespace test_utils
