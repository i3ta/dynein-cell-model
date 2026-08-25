#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace dynein_cell_model {

/**
 * Dense outline-membership mask with a coordinate list for O(n) boundary
 * traversal. The coordinate list is the authoritative iteration interface;
 * mutating the dense storage directly is intentionally not supported.
 */
class OutlineMask {
public:
  using DenseMask = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>;
  using Coord = std::pair<int, int>;
  using value_type = Coord;
  using const_iterator = std::vector<Coord>::const_iterator;

  OutlineMask() = default;
  OutlineMask(int rows, int cols) : mask_(DenseMask::Zero(rows, cols)) {}

  [[nodiscard]] int rows() const noexcept { return mask_.rows(); }
  [[nodiscard]] int cols() const noexcept { return mask_.cols(); }
  [[nodiscard]] std::size_t size() const noexcept { return coords_.size(); }
  [[nodiscard]] bool empty() const noexcept { return coords_.empty(); }

  /** True when (row, col) belongs to this outline. */
  [[nodiscard]] bool contains(int row, int col) const noexcept {
    return mask_(row, col) != 0;
  }

  /** Equivalent to contains(), convenient at existing mask-style call sites. */
  [[nodiscard]] bool operator()(int row, int col) const noexcept {
    return contains(row, col);
  }

  /** Add a coordinate if absent, preserving insertion order. */
  void set(int row, int col) {
    if (!contains(row, col)) {
      mask_(row, col) = 1;
      coords_.emplace_back(row, col);
    }
  }

  /** Remove a coordinate. This is O(n) and intended for infrequent updates. */
  void unset(int row, int col) {
    if (!contains(row, col)) return;
    mask_(row, col) = 0;
    const Coord target{row, col};
    coords_.erase(std::find(coords_.begin(), coords_.end(), target));
  }

  void clear() noexcept {
    mask_.setZero();
    coords_.clear();
  }

  /**
   * Recreate coordinates in Eigen sparse's previous column-major traversal
   * order. Use this after bulk construction through a future dense importer.
   */
  void rebuildCoordinatesColumnMajor() {
    coords_.clear();
    for (int col = 0; col < cols(); ++col)
      for (int row = 0; row < rows(); ++row)
        if (contains(row, col)) coords_.emplace_back(row, col);
  }

  [[nodiscard]] const DenseMask &mask() const noexcept { return mask_; }
  [[nodiscard]] const std::vector<Coord> &coords() const noexcept {
    return coords_;
  }

  const_iterator begin() const noexcept { return coords_.begin(); }
  const_iterator end() const noexcept { return coords_.end(); }
  const_iterator cbegin() const noexcept { return coords_.cbegin(); }
  const_iterator cend() const noexcept { return coords_.cend(); }

  /**
   * Return a randomly ordered copy without mutating stable outline order.
   * This consumes RNG exactly through std::shuffle.
   */
  [[nodiscard]] std::vector<Coord> shuffled(std::mt19937 &rng) const {
    auto result = coords_;
    std::shuffle(result.begin(), result.end(), rng);
    return result;
  }

private:
  DenseMask mask_;
  std::vector<Coord> coords_;
};

} // namespace dynein_cell_model
