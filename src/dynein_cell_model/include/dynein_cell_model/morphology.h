#pragma once

#include <array>
#include <bitset>
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
void generateDynField(CellState &state, const ViewMask &cellOutline,
                      const ViewMask &nucOutline, const bool retract);

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

constexpr AdjArr decodeAdj(const AdjSet conf) {
  AdjArr grid{};
  for (int bit = 0; bit < 8; ++bit) {
    const auto [dr, dc] = adjOffsets[bit];
    grid[1 + dr][1 + dc] = (conf & (1 << bit)) != 0;
  }
  return grid;
}

constexpr bool connected(const AdjArr &grid, bool val) {
  constexpr int kDR[4] = {-1, 0, 1, 0};
  constexpr int kDC[4] = {0, -1, 0, 1};
  constexpr auto inBounds = [](int r, int c) {
    return r >= 0 && r < 3 && c >= 0 && c < 3;
  };

  int total = 0, sr = -1, sc = -1;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (grid[r][c] == val) {
        ++total;
        if (sr < 0) {
          sr = r;
          sc = c;
        }
      }
    }
  }
  if (total == 0) {
    return true;
  }

  std::array<std::pair<int, int>, 9> stack;
  int top = 0;
  stack[top++] = {sr, sc};
  AdjArr visitedGrid{};
  int visited = 0;
  while (top > 0) {
    const auto [r, c] = stack[--top];
    if (visitedGrid[r][c])
      continue;
    visitedGrid[r][c] = true;
    ++visited;
    for (int i = 0; i < 4; ++i) {
      const int nr = r + kDR[i], nc = c + kDC[i];
      if (inBounds(nr, nc) && grid[nr][nc] == val && !visitedGrid[nr][nc]) {
        stack[top++] = {nr, nc};
      }
    }
  }
  return visited == total;
}

constexpr bool isValidTransition(const uint8_t conf, bool centerAfter) {
  AdjArr grid = decodeAdj(conf);
  grid[1][1] = centerAfter;
  return connected(grid, false) && connected(grid, true);
}

constexpr std::bitset<256> generateProtrudeConf() {
  std::bitset<256> confs;
  for (int i = 0; i < 256; ++i) {
    confs[i] = isValidTransition(static_cast<uint8_t>(i), /*centerAfter=*/true);
  }
  return confs;
}

constexpr std::bitset<256> generateRetractConf() {
  std::bitset<256> confs;
  for (int i = 0; i < 256; ++i) {
    confs[i] =
        isValidTransition(static_cast<uint8_t>(i), /*centerAfter=*/false);
  }
  return confs;
}

} // namespace

constexpr std::bitset<256> protrudeConf = generateProtrudeConf();
constexpr std::bitset<256> retractConf = generateRetractConf();

} // namespace dynein_cell_model
