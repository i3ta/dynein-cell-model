#include <chrono>
#include <iostream>
#include <string>

#include <metric_utils/metric_utils.hpp>

namespace metrics {
ScopedTimer::ScopedTimer(const std::string name)
  : name_(name), start_(std::chrono::high_resolution_clock::now()),
    auto_time_(true) {}

ScopedTimer::ScopedTimer(const std::string name, bool auto_time)
  : name_(name), start_(std::chrono::high_resolution_clock::now()),
    auto_time_(auto_time) {}

ScopedTimer::~ScopedTimer() {
  if (!auto_time_) return;

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
  std::cout << "[Timer] " << name_ << ": " << elapsed << " ms\n";
}

std::chrono::milliseconds ScopedTimer::elapsed() {
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
  return elapsed;
}

void ScopedTimer::reset() {
  start_ = std::chrono::high_resolution_clock::now();
}
}; // namespace metrics

