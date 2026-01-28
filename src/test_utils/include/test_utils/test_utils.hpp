#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <limits>
#include <vector>

#include <dynein_cell_model/dynein_cell_model.hpp>

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

  void protrude_nuc_dep();

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

  const dcm::Mat_i &get_cell();

  const dcm::Mat_i &get_nuc();

  const dcm::SpMat_i &get_adh();

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

  const int get_AC_cor_sum();

  const int get_IC_cor_sum();
}; // class CellModelTest

} // namespace test_utils

#endif
