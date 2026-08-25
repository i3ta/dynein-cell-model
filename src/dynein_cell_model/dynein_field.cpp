#include "dynein_cell_model/state.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace dynein_cell_model {
namespace {
void gaussianBlur(ViewD &field, int r0, int r1, int c0, int c1, double sigma) {
  if (sigma <= 0 || r0 > r1 || c0 > c1) return;
  const int radius = std::max(1, static_cast<int>(std::ceil(3 * sigma)));
  ViewD copy = field;
  for (int r = r0; r <= r1; ++r) for (int c = c0; c <= c1; ++c) {
    double weighted = 0, norm = 0;
    for (int rr = std::max(r0, r - radius); rr <= std::min(r1, r + radius); ++rr)
      for (int cc = std::max(c0, c - radius); cc <= std::min(c1, c + radius); ++cc) {
        const double dr = r - rr, dc = c - cc, w = std::exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
        weighted += w * copy(rr, cc); norm += w;
      }
    field(r, c) = norm == 0 ? 0 : weighted / norm;
  }
}
} // namespace

// Kept implementation-private: it is an orchestration detail of step().
void updateDynNucField(CellState &state) {
  const auto &config = state.config;
  state.dynF.setZero();
  ViewI parent = ViewI::Constant(config.simRows, config.simCols, -1);
  ViewI scaling = ViewI::Zero(config.simRows, config.simCols);
  std::vector<std::pair<int, int>> order;
  for (int k = 0; k < state.innerOutlineNuc.outerSize(); ++k)
    for (ViewMask::InnerIterator it(state.innerOutlineNuc, k); it; ++it) {
      parent(it.row(), it.col()) = -2; order.push_back({it.row(), it.col()});
    }
  const int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};
  for (size_t head = 0; head < order.size(); ++head) {
    const auto [r, c] = order[head];
    if (state.innerOutline.coeff(r, c)) { state.dynF(r, c) = state.AC(r, c); scaling(r, c) = 1; }
    for (int n = 0; n < 4; ++n) {
      const int nr = r + dr[n], nc = c + dc[n];
      if (nr < state.frameRowStart || nr > state.frameRowEnd || nc < state.frameColStart || nc > state.frameColEnd ||
          state.cell(nr, nc) == 0 || state.nuc(nr, nc) != 0 || parent(nr, nc) != -1) continue;
      parent(nr, nc) = r * config.simCols + c; order.push_back({nr, nc});
    }
  }
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const int p = parent(it->first, it->second);
    if (p >= 0) { const int r = p / config.simCols, c = p % config.simCols; state.dynF(r, c) += state.dynF(it->first, it->second); scaling(r, c) += scaling(it->first, it->second); }
  }
  for (int r = state.nucMinR; r <= state.nucMaxR; ++r) for (int c = state.nucMinC; c <= state.nucMaxC; ++c)
    state.dynF(r, c) = scaling(r, c) ? state.dynF(r, c) / scaling(r, c) * config.dynScale : 0;
  gaussianBlur(state.dynF, state.nucMinR, state.nucMaxR, state.nucMinC, state.nucMaxC, config.dynSigma);
}
} // namespace dynein_cell_model
