#ifndef METRIC_UTILS_HPP
#define METRIC_UTILS_HPP

#include <chrono>
#include <string>

#define TIME_BLOCK(name) ScopedTimer timer_##__LINE__(name);

namespace metrics {
class ScopedTimer {
public:
  ScopedTimer(const std::string name);

  ScopedTimer(const std::string name, bool auto_time);

  ~ScopedTimer();

  std::chrono::milliseconds elapsed();

  void reset();

private:
  bool auto_time_;
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};
}; // namespace metrics

#endif
