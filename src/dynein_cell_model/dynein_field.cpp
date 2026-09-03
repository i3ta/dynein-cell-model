#include "dynein_cell_model/state.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dynein_cell_model {
namespace {
void gaussianBlurNucCandidates(CellState &state,
                               const OutlineMask &targets,
                               double sigma) {
  if (sigma <= 0 || targets.empty())
    return;
  const int radius = static_cast<int>(std::ceil(3 * sigma));

  std::vector<double> kernel(radius + 1);
  const double denominator = 2 * sigma * sigma;
  for (int d = 0; d <= radius; ++d)
    kernel[d] = std::exp(-double(d * d) / denominator);

  // dynF has samples only along the one-pixel-wide nuclear boundary.  Divide
  // by the 1-D kernel sum, rather than its square, to approximate a Gaussian
  // average along a locally straight boundary without diluting the signal
  // into the unsampled 2-D interior/background.
  const double kernelSum = kernel[0] +
                           2 * std::accumulate(kernel.begin() + 1,
                                               kernel.end(), 0.0);
  const double normalization = kernelSum;
  std::vector<double> sourceValues;
  sourceValues.reserve(state.innerOutlineNuc.size());
  for (const auto &[r, c] : state.innerOutlineNuc)
    sourceValues.push_back(state.dynF(r, c));

  state.dynF.setZero();
  for (const auto &[targetR, targetC] : targets) {
    double weighted = 0;
    for (size_t i = 0; i < state.innerOutlineNuc.size(); ++i) {
      const auto &[sourceR, sourceC] = state.innerOutlineNuc.coords()[i];
      const int dr = std::abs(targetR - sourceR);
      const int dc = std::abs(targetC - sourceC);
      if (dr <= radius && dc <= radius)
        weighted += kernel[dr] * kernel[dc] * sourceValues[i];
    }
    state.dynF(targetR, targetC) = weighted / normalization;
  }
}
} // namespace

// Kept implementation-private: it is an orchestration detail of step().
void updateDynNucField(CellState &state, bool retract) {
  const auto &config = state.config;
  const int patchRadius = static_cast<int>(
      (retract ? state.innerOutlineNuc.size() : state.outlineNuc.size()) /
      (retract ? 6 : 30));
  // A discrete square patch over [-n, n] has variance n(n + 1) / 3 along
  // either axis. Use its Gaussian equivalent, scaled by the fitted multiplier.
  const double baseSigma =
      std::sqrt(double(patchRadius) * (patchRadius + 1) / 3.0);
  const double sigma = std::max(0.0, config.dynSigma) * baseSigma;
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
    // Preserve the deprecated field's threshold behavior: low-AC cell
    // boundary pixels do not contribute force or a scaling count.
    if (state.innerOutline.contains(r, c) && state.AC(r, c) > 0.1) {
      state.dynF(r, c) = distance[head] * (state.AC(r, c) - 0.1);
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
  for (const auto &[r, c] : state.innerOutlineNuc)
    state.dynF(r, c) =
        scaling(r, c) ? state.dynF(r, c) / scaling(r, c) * config.dynScale
                      : 0;
  gaussianBlurNucCandidates(state,
                            retract ? state.innerOutlineNuc : state.outlineNuc,
                            sigma);
  for (const auto &[r, c] : retract ? state.innerOutlineNuc
                                     : state.outlineNuc)
    state.dynF(r, c) = std::clamp(state.dynF(r, c), 0.0, 1.0);
}
} // namespace dynein_cell_model
