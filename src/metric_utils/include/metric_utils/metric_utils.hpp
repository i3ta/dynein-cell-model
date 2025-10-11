#ifndef METRIC_UTILS_HPP
#define METRIC_UTILS_HPP

#include <chrono>
#include <string>

#define TIME_BLOCK(name) ScopedTimer timer_##__LINE__(name);

namespace metrics {
class ScopedTimer {
public:
  ScopedTimer(const std::string name);

  ~ScopedTimer();

private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};
}; // namespace metrics

#endif
