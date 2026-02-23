#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstdlib>
#include <limits>
#include <set>
#include <vector>

#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.hpp>

namespace test_utils {
namespace dcm = dynein_cell_model;

template <typename T> class DebugRand {
private:
  static std::set<DebugRand *> instances_;

public:
  DebugRand() : current_index_(0) { instances_.insert(this); }
  ~DebugRand() { instances_.erase(this); }

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

  static void reset_all_instances(size_t index = 0) {
    for (auto *inst : instances_)
      inst->current_index_ = index;
  }

private:
  static std::vector<T> outputs_;
  size_t current_index_;
};

template <typename T> std::set<DebugRand<T> *> DebugRand<T>::instances_;

// This line ust stay in the header to tell the compiler
// how to allocate the static vector for any T used.
template <typename T> std::vector<T> DebugRand<T>::outputs_;

class CellModelTest {
private:
  dcm::CellModel model;

public:
  CellModelTest() = default;

  CellModelTest(const dcm::CellModelConfig &conf);

  void init(const dcm::CellModelConfig &conf);

  void rearrange_adhesions(const bool bias = false,
                           const bool rearrange_all = false);

  void protrude();

  void retract();

  void protrude_nuc_dep();

  void retract_nuc_dep();

  void generate_dyn_field(const dcm::SpMat_i &cell_outline,
                          const dcm::SpMat_i &nuc_outline, bool retract);

  void correct_concentrations();

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

  const dcm::Mat_d &get_adh_f();

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

} // namespace test_utils

#endif
