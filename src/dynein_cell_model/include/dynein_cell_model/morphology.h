#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "dynein_cell_model/state.h"
#include "dynein_cell_model/types.h"

namespace dynein_cell_model {

using AdjSet = uint8_t;
using AdjArr = std::array<std::array<bool, 3>, 3>;

AdjSet encodeAdj(const ViewI &mask, const int r, const int c);

/**
 * @brief Protrude the cell boundary, taking into account adhesions and the
 * nucleus.
 */
void protrudeCell(CellState &state);

/**
 * @brief Retract the cell boundary.
 */
void retractCell(CellState &state);

/**
 * @brief Protrude the nucleus pixels.
 */
void protrudeNuc(CellState &state);

/**
 * @brief Retract the nucleus pixels.
 */
void retractNuc(CellState &state);

/**
 * @brief Protrude the nucleus pixels (old algorithm).
 */
[[deprecated]]
void protrudeNucDep(CellState &state);

/**
 * @brief Retract the nucleus pixels (old algorithm).
 */
[[deprecated]]
void retractNucDep(CellState &state);

/**
 * @brief Rearrange the adhesion points around the cell to simulate evolution
 * of cell adhesions. Randomly picks adhFrac of the adhesions and finds other
 * valid positions to move them cell adhesions.
 *
 * @param bias Whether polarization should be used to determine rearrangement
 * distribution
 */
void rearrangeAdhesions(CellState &state, const bool bias = false,
                        const bool rearrangeAll = false);

/**
 * @brief Generate the dynein field using old logic.
 */
[[deprecated]]
void generateDynField(CellState &state, const OutlineMask &cellOutline,
                      const OutlineMask &nucOutline, const bool retract);

/**
 * @brief Update the cell processing boundaries to be where the cell is.
 */
void updateFrame(CellState &state);

/** Rebuild cell/nucleus outlines and derived volumes after manual mask setup. */
void updateGeometry(CellState &state);

namespace {

constexpr std::array<std::pair<int, int>, 8> adjOffsets = {{
    {-1, -1}, // top-left
    {-1, 0},  // top
    {-1, 1},  // top-right
    {0, 1},   // right
    {1, 1},   // bottom-right
    {1, 0},   // bottom
    {1, -1},  // bottom-left
    {0, -1},  // left
}};

constexpr AdjSet encodeAdj(const AdjArr &grid) {
  uint8_t conf = 0;
  for (int bit = 0; bit < 8; ++bit) {
    const auto [dr, dc] = adjOffsets[bit];
    if (grid[1 + dr][1 + dc]) {
      conf |= (1 << bit);
    }
  }
  return conf;
}

constexpr bool hasBit(const AdjSet conf, int bit) {
  return (conf & (1u << bit)) != 0;
}

// Preserve the pre-refactor local topology rules.  These reject unsupported
// diagonal connections and four-neighbor pinches, but do not impose the newer
// requirement that both values in the full 3x3 neighborhood are connected.
constexpr bool isValidProtrudeConfig(const AdjSet conf) {
  const bool unsupportedDiagonal =
      (hasBit(conf, 0) && !hasBit(conf, 1) && !hasBit(conf, 7)) ||
      (hasBit(conf, 2) && !hasBit(conf, 1) && !hasBit(conf, 3)) ||
      (hasBit(conf, 4) && !hasBit(conf, 3) && !hasBit(conf, 5)) ||
      (hasBit(conf, 6) && !hasBit(conf, 5) && !hasBit(conf, 7));
  const bool pinch =
      (hasBit(conf, 1) && hasBit(conf, 5) && !hasBit(conf, 3) &&
       !hasBit(conf, 7)) ||
      (hasBit(conf, 3) && hasBit(conf, 7) && !hasBit(conf, 1) &&
       !hasBit(conf, 5));
  return !unsupportedDiagonal && !pinch;
}

constexpr bool isValidRetractConfig(const AdjSet conf) {
  const bool unsupportedDiagonalGap =
      (!hasBit(conf, 0) && hasBit(conf, 1) && hasBit(conf, 7)) ||
      (!hasBit(conf, 2) && hasBit(conf, 1) && hasBit(conf, 3)) ||
      (!hasBit(conf, 4) && hasBit(conf, 3) && hasBit(conf, 5)) ||
      (!hasBit(conf, 6) && hasBit(conf, 5) && hasBit(conf, 7));
  const bool pinch =
      (hasBit(conf, 1) && hasBit(conf, 5) && !hasBit(conf, 3) &&
       !hasBit(conf, 7)) ||
      (hasBit(conf, 3) && hasBit(conf, 7) && !hasBit(conf, 1) &&
       !hasBit(conf, 5));
  return !unsupportedDiagonalGap && !pinch;
}

constexpr std::array<bool, 256> generateProtrudeConf() {
  std::array<bool, 256> confs{};
  for (int i = 0; i < 256; ++i) {
    confs[i] = isValidProtrudeConfig(static_cast<uint8_t>(i));
  }
  return confs;
}

constexpr std::array<bool, 256> generateRetractConf() {
  std::array<bool, 256> confs{};
  for (int i = 0; i < 256; ++i) {
    confs[i] = isValidRetractConfig(static_cast<uint8_t>(i));
  }
  return confs;
}

} // namespace

constexpr std::array<bool, 256> protrudeConf = generateProtrudeConf();
constexpr std::array<bool, 256> retractConf = generateRetractConf();

} // namespace dynein_cell_model
