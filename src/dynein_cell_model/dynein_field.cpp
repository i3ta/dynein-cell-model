#include "dynein_cell_model/state.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dynein_cell_model {
namespace {
void gaussianBlur(ViewD &field, int r0, int r1, int c0, int c1, double sigma) {
  if (sigma <= 0 || r0 > r1 || c0 > c1)
    return;
  const int radius = std::max(1, static_cast<int>(std::ceil(3 * sigma)));
  const int rows = r1 - r0 + 1;
  const int cols = c1 - c0 + 1;

  std::vector<double> kernel(radius + 1);
  const double denominator = 2 * sigma * sigma;
  for (int d = 0; d <= radius; ++d)
    kernel[d] = std::exp(-double(d * d) / denominator);

  ViewD horizontal(rows, cols);
  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      const int cBegin = std::max(c0, c - radius);
      const int cEnd = std::min(c1, c + radius);
      double weighted = 0;
      double norm = 0;
      for (int cc = cBegin; cc <= cEnd; ++cc) {
        const double w = kernel[std::abs(c - cc)];
        weighted += w * field(r, cc);
        norm += w;
      }
      horizontal(r - r0, c - c0) = norm == 0 ? 0 : weighted / norm;
    }
  }

  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      const int rBegin = std::max(r0, r - radius);
      const int rEnd = std::min(r1, r + radius);
      double weighted = 0;
      double norm = 0;
      for (int rr = rBegin; rr <= rEnd; ++rr) {
        const double w = kernel[std::abs(r - rr)];
        weighted += w * horizontal(rr - r0, c - c0);
        norm += w;
      }
      field(r, c) = norm == 0 ? 0 : weighted / norm;
    }
  }
}
} // namespace

// Kept implementation-private: it is an orchestration detail of step().
void updateDynNucField(CellState &state) {
  const auto &config = state.config;
  // Nucleus protrusion samples dynF on outlineNuc, which is one pixel outside
  // the current nucleus bounds.  Build the field on that halo as well, so a
  // force is available for every candidate rather than only for candidates
  // that happen to lie inside the current bounding box.
  const int candidateMinR = std::max(0, state.nucMinR - 1);
  const int candidateMaxR = std::min(config.simRows - 1, state.nucMaxR + 1);
  const int candidateMinC = std::max(0, state.nucMinC - 1);
  const int candidateMaxC = std::min(config.simCols - 1, state.nucMaxC + 1);

  // A one-pixel candidate halo is enough for morphology, but not for a
  // Gaussian with this radius.  Extend the blur workspace by its full support
  // so that rectangular workspace edges cannot shape the force at candidates.
  const int blurRadius =
      std::max(1, static_cast<int>(std::ceil(3 * config.dynSigma)));
  const int blurMinR = std::max(0, candidateMinR - blurRadius);
  const int blurMaxR = std::min(config.simRows - 1, candidateMaxR + blurRadius);
  const int blurMinC = std::max(0, candidateMinC - blurRadius);
  const int blurMaxC = std::min(config.simCols - 1, candidateMaxC + blurRadius);

  state.dynF.setZero();
  ViewI parent = ViewI::Constant(config.simRows, config.simCols, -1);
  ViewI scaling = ViewI::Zero(config.simRows, config.simCols);
  std::vector<std::pair<int, int>> order;
  for (const auto &[row, col] : state.innerOutlineNuc) {
    parent(row, col) = -2;
    order.push_back({row, col});
  }
  const int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};
  for (size_t head = 0; head < order.size(); ++head) {
    const auto [r, c] = order[head];
    if (state.innerOutline.contains(r, c)) {
      state.dynF(r, c) = state.AC(r, c);
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
  gaussianBlur(state.dynF, blurMinR, blurMaxR, blurMinC, blurMaxC,
               config.dynSigma);
}
} // namespace dynein_cell_model
