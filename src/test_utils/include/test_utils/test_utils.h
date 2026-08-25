#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstdlib>
#include <limits>
#include <set>
#include <vector>

#include <cell_nuc/cell_nuc.hpp>
#include <dynein_cell_model/dynein_cell_model.h>

namespace test_utils {

template <typename T> class DebugRand {
private:
  static std::set<DebugRand *> instances;

public:
  DebugRand() : currentIndex(0) { instances.insert(this); }
  ~DebugRand() { instances.erase(this); }

  static constexpr T min() { return 0; }
  static constexpr T max() { return std::numeric_limits<T>::max(); }

  void setOutputs(std::vector<T> &&outputs) {
    this->outputs = std::move(outputs);
    currentIndex = 0;
  }

  T operator()() {
    if (outputs.empty())
      return 0;
    return outputs[currentIndex++ % outputs.size()];
  }

  static void resetAllInstances(size_t index = 0) {
    for (auto *inst : instances)
      inst->currentIndex = index;
  }

private:
  static std::vector<T> outputs;
  size_t currentIndex;
};

template <typename T> std::set<DebugRand<T> *> DebugRand<T>::instances;

// This line just stay in the header to tell the compiler
// how to allocate the static vector for any T used.
template <typename T> std::vector<T> DebugRand<T>::outputs;

} // namespace test_utils

#endif
