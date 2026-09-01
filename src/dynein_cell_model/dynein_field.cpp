#include "dynein_cell_model/state.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dynein_cell_model {
namespace {
void boxBlur(ViewD &field, int r0, int r1, int c0, int c1, int radius) {
  if (radius <= 0 || r0 > r1 || c0 > c1)
    return;
  const int rows = r1 - r0 + 1;
  const int cols = c1 - c0 + 1;

  ViewD horizontal(rows, cols);
  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      const int cBegin = std::max(c0, c - radius);
      const int cEnd = std::min(c1, c + radius);
      double sum = 0;
      for (int cc = cBegin; cc <= cEnd; ++cc)
        sum += field(r, cc);
      horizontal(r - r0, c - c0) = sum / (cEnd - cBegin + 1);
    }
  }

  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      const int rBegin = std::max(r0, r - radius);
      const int rEnd = std::min(r1, r + radius);
      double sum = 0;
      for (int rr = rBegin; rr <= rEnd; ++rr)
        sum += horizontal(rr - r0, c - c0);
      field(r, c) = sum / (rEnd - rBegin + 1);
    }
  }
}
} // namespace

// Kept implementation-private: it is an orchestration detail of step().
void updateDynNucField(CellState &state, bool retract) {
  const auto &config = state.config;
  const int patchRadius = static_cast<int>(
      (retract ? state.innerOutlineNuc.size() : state.outlineNuc.size()) /
      (retract ? 6 : 30));
  // Nucleus protrusion samples dynF on outlineNuc, which is one pixel outside
  // the current nucleus bounds.  Build the field on that halo as well, so a
  // force is available for every candidate rather than only for candidates
  // that happen to lie inside the current bounding box.
  const int candidateMinR = std::max(0, state.nucMinR - 1);
  const int candidateMaxR = std::min(config.simRows - 1, state.nucMaxR + 1);
  const int candidateMinC = std::max(0, state.nucMinC - 1);
  const int candidateMaxC = std::min(config.simCols - 1, state.nucMaxC + 1);

  // Include the entire old-model square footprint around every candidate.
  const int blurMinR = std::max(0, candidateMinR - patchRadius);
  const int blurMaxR =
      std::min(config.simRows - 1, candidateMaxR + patchRadius);
  const int blurMinC = std::max(0, candidateMinC - patchRadius);
  const int blurMaxC =
      std::min(config.simCols - 1, candidateMaxC + patchRadius);

  state.dynF.setZero();
  ViewI parent = ViewI::Constant(config.simRows, config.simCols, -1);
  ViewI scaling = ViewI::Zero(config.simRows, config.simCols);
  std::vector<std::pair<int, int>> order;
  std::vector<int> distance;
  for (const auto &[row, col] : state.innerOutlineNuc) {
    parent(row, col) = -2;
    order.push_back({row, col});
    distance.push_back(0);
  }
  const int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};
  for (size_t head = 0; head < order.size(); ++head) {
    const auto [r, c] = order[head];
    if (state.innerOutline.contains(r, c)) {
      state.dynF(r, c) = distance[head] * std::max(state.AC(r, c) - 0.1, 0.0);
      scaling(r, c) = 1;
    }
    for (int n = 0; n < 4; ++n) {
      const int nr = r + dr[n], nc = c + dc[n];
      if (nr < state.frameRowStart || nr > state.frameRowEnd ||
          nc < state.frameColStart || nc > state.frameColEnd ||
          state.cell(nr, nc) == 0 || state.nuc(nr, nc) != 0 ||
          parent(nr, nc) != -1)
        continue;
      parent(nr, nc) = r * config.simCols + c;
      order.push_back({nr, nc});
      distance.push_back(distance[head] + 1);
    }
  }
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const int p = parent(it->first, it->second);
    if (p >= 0) {
      const int r = p / config.simCols, c = p % config.simCols;
      state.dynF(r, c) += state.dynF(it->first, it->second);
      scaling(r, c) += scaling(it->first, it->second);
    }
  }
  for (int r = blurMinR; r <= blurMaxR; ++r)
    for (int c = blurMinC; c <= blurMaxC; ++c)
      state.dynF(r, c) =
          scaling(r, c) ? state.dynF(r, c) / scaling(r, c) * config.dynScale
                        : 0;
  boxBlur(state.dynF, blurMinR, blurMaxR, blurMinC, blurMaxC, patchRadius);

  for (int r = blurMinR; r <= blurMaxR; ++r)
    for (int c = blurMinC; c <= blurMaxC; ++c)
      state.dynF(r, c) = std::clamp(state.dynF(r, c), 0.0, 1.0);
}
} // namespace dynein_cell_model
