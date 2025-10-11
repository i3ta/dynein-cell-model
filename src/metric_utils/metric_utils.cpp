#include <chrono>
#include <iostream>
#include <string>

#include <metric_utils/metric_utils.hpp>

namespace metrics {
ScopedTimer::ScopedTimer(const std::string name)
  : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

ScopedTimer::~ScopedTimer() {
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
  std::cout << "[Timer] " << name_ << ": " << elapsed << " ms\n";
}
}; // namespace metrics

