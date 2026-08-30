#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dynein_cell_model/diffusion.h"
#include "dynein_cell_model/macros.h"
#include "dynein_cell_model/morphology.h"
#include "dynein_cell_model/types.h"

namespace dynein_cell_model {

namespace {

struct pair_hash {
  std::size_t operator()(const std::pair<int, int> &p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

/**
 * @brief Helper function to generate random distinct indices.
 *
 * @param n Number of indices to generate
 * @param lb Lower bound of indices (inclusive)
 * @param ub Upper bound of indices (exclusive)
 * @param rng Random number generator to use
 *
 * @return Vector of random indices
 */
std::vector<int> generateIndices(const int n, const int lb, const int ub,
                                 std::mt19937 &rng) {
  if (ub - lb < n) {
    throw std::runtime_error("Bounds must be at least as large as the number "
                             "of indices to generate.");
  }

  std::vector<int> arr(ub - lb);
  for (int i = 0, v = lb; v < ub; i++, v++) {
    arr[i] = v;
  }

  std::shuffle(arr.begin(), arr.end(), rng);

  return std::vector<int>(arr.begin(), arr.begin() + n);
}

std::vector<OutlineMask::Coord> stableCoords(const OutlineMask &outline) {
  return {outline.begin(), outline.end()};
}

int outline4(const OutlineMask &outline, const ViewI &body, int simRows,
             int sim_cols) {
  const int DR[4] = {-1, 0, 1, 0};
  const int DC[4] = {0, -1, 0, 1};

  int perim = 0;

  for (const auto &[row, col] : outline) {
    for (int i = 0; i < 4; i++) {
      const int nr = row + DR[i];
      const int nc = col + DC[i];
      if (nr < 0 || nr >= simRows || nc < 0 || nc >= sim_cols)
        continue;
      if (body(nr, nc) == 1) {
        perim++;
        break;
      }
    }
  }

  return perim;
}

void updateCell(CellState &state, const bool full = false) {
  /**
   * Iterate through cell pixels and add 4-neighbors that are not part of the
   * cell to outer outline.
   */
  const auto &conf = state.config;
  const int DR[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
  const int DC[8] = {0, 0, -1, 1, -1, 1, -1, 1};

  // Clear outlines
  state.outline.clear();
  state.innerOutline.clear();

  // Set bounds
  int rowStart = full ? 0 : state.frameRowStart;
  int rowEnd = full ? conf.simRows - 1 : state.frameRowEnd;
  int colStart = full ? 0 : state.frameColStart;
  int colEnd = full ? conf.simCols - 1 : state.frameColEnd;

  // #pragma omp parallel
  {
    std::unordered_set<std::pair<int, int>, pair_hash> localInner, localOuter;

    // Iterate through cell pixels
    // #pragma omp for nowait
    for (int i = rowStart; i <= rowEnd; i++) {
      for (int j = colStart; j <= colEnd; j++) {
        if (state.cell(i, j) == 0)
          continue;
        for (int k = 0; k < 8; k++) {
          const int nr = i + DR[k];
          const int nc = j + DC[k];
          if (nr < 0 || nr >= conf.simRows || nc < 0 || nc >= conf.simCols)
            continue;
          if (state.cell(nr, nc) == 0) {
            localInner.insert({i, j});
            localOuter.insert({nr, nc});
          }
        }
      }
    }

    // update outlines
    // #pragma omp critical
    {
      for (auto &[r, c] : localInner) {
        state.innerOutline.set(r, c);
      }
      for (auto &[r, c] : localOuter) {
        state.outline.set(r, c);
      }
    }
  }

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    state.innerOutline.rebuildCoordinatesColumnMajor();
    state.outline.rebuildCoordinatesColumnMajor();
  }

  // update cell volume and perimeter
  state.V = (state.cell.array() != 0).count();
  state.P = outline4(state.outline, state.cell, conf.simRows, conf.simCols);
}

void updateNuc(CellState &state, const bool recheckBounds = false) {
  /**
   * Iterate through nucleus pixels and add 4-neighbors that are not nucleus to
   * outer outline. Uses tracked bounds for optimized iteration.
   *
   * @param recheckBounds if true, rescans to find new nucleus bounds after
   * retraction. If false, uses tracked bounds for iteration.
   */
  const auto &conf = state.config;
  const int DR[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
  const int DC[8] = {0, 0, -1, 1, -1, 1, -1, 1};

  // Clear outlines
  state.outlineNuc.clear();
  state.innerOutlineNuc.clear();
  state.VNuc = 0;

  // Initialize bounds on first call (when bounds are invalid)
  if (state.nucMaxR == 0 && state.nucMaxC == 0) {
    for (int j = state.frameColStart; j <= state.frameColEnd; j++) {
      for (int i = state.frameRowStart; i <= state.frameRowEnd; i++) {
        if (state.nuc(i, j) == 1) {
          state.nucMinR = std::min(state.nucMinR, i);
          state.nucMaxR = std::max(state.nucMaxR, i);
          state.nucMinC = std::min(state.nucMinC, j);
          state.nucMaxC = std::max(state.nucMaxC, j);
        }
      }
    }
  }

  // If recheckBounds, scan within current bounds to find new bounds
  if (recheckBounds) {
    int newMinR = conf.simRows, newMaxR = 0;
    int newMinC = conf.simCols, newMaxC = 0;
    for (int j = state.nucMinC; j <= state.nucMaxC; j++) {
      for (int i = state.nucMinR; i <= state.nucMaxR; i++) {
        if (state.nuc(i, j) == 1) {
          newMinR = std::min(newMinR, i);
          newMaxR = std::max(newMaxR, i);
          newMinC = std::min(newMinC, j);
          newMaxC = std::max(newMaxC, j);
        }
      }
    }
    state.nucMinR = newMinR;
    state.nucMaxR = newMaxR;
    state.nucMinC = newMinC;
    state.nucMaxC = newMaxC;
  }

  // Iterate within current bounds (+1 margin for outline detection)
  const int rowStart = std::max(state.frameRowStart, state.nucMinR - 1);
  const int rowEnd = std::min(state.frameRowEnd, state.nucMaxR + 1);
  const int colStart = std::max(state.frameColStart, state.nucMinC - 1);
  const int colEnd = std::min(state.frameColEnd, state.nucMaxC + 1);

  // #pragma omp parallel
  {
    std::unordered_set<std::pair<int, int>, pair_hash> localInner, localOuter;

    // Iterate through nucleus pixels within bounding box
    // #pragma omp for nowait
    for (int i = rowStart; i <= rowEnd; i++) {
      for (int j = colStart; j <= colEnd; j++) {
        if (state.nuc(i, j) == 0)
          continue;
        state.VNuc++;

        bool isInner = false;
        for (int k = 0; k < 8; k++) {
          const int nr = i + DR[k];
          const int nc = j + DC[k];
          if (nr < 0 || nr >= conf.simRows || nc < 0 || nc >= conf.simCols)
            continue;
          if (state.nuc(nr, nc) == 0) {
            isInner = true;
            localOuter.insert({nr, nc});
          }
        }
        if (isInner)
          localInner.insert({i, j});
      }
    }

    // update outlines
    // #pragma omp critical
    {
      for (auto &[r, c] : localInner) {
        state.innerOutlineNuc.set(r, c);
      }
      for (auto &[r, c] : localOuter) {
        state.outlineNuc.set(r, c);
      }
    }
  }

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    state.innerOutlineNuc.rebuildCoordinatesColumnMajor();
    state.outlineNuc.rebuildCoordinatesColumnMajor();
  }

  // update nucleus volume and perimeter
  state.PNuc =
      outline4(state.outlineNuc, state.nuc, conf.simRows, conf.simCols);
}

} // namespace

AdjSet encodeAdj(const ViewI &mask, const int r, const int c) {
  const int rows = mask.rows();
  const int cols = mask.cols();
  const auto occupied = [&](int rr, int cc) {
    return rr >= 0 && rr < rows && cc >= 0 && cc < cols && mask(rr, cc) != 0;
  };

  AdjSet conf = 0;
  for (int bit = 0; bit < 8; ++bit) {
    const auto [dr, dc] = adjOffsets[bit];
    if (occupied(r + dr, c + dc)) {
      conf |= (1 << bit);
    }
  }

  return conf;
}

void protrudeCell(CellState &state) {
  /**
   * This function attempts to protrude the cell. The logic of this function is
   * very similar to that of protrudeNuc, but the weight function is slightly
   * different and the values used are relative to the actin factor as opposed
   * to dynein factor.
   */
  const auto &config = state.config;

  // get probability coefficients
  const double vCor = 1 / (1 + std::exp((state.V - state.V0) / config.T));
  const double aMax = state.A
                          .block(state.frameRowStart, state.frameColStart,
                                 state.frameRowEnd - state.frameRowStart + 1,
                                 state.frameColEnd - state.frameColStart + 1)
                          .maxCoeff();
  const double acMax = state.AC
                           .block(state.frameRowStart, state.frameColStart,
                                  state.frameRowEnd - state.frameRowStart + 1,
                                  state.frameColEnd - state.frameColStart + 1)
                           .maxCoeff();
  const double nDiag = 1.0 / std::pow(M_SQRT2, config.g);
  const double C = 4.0 * (1.0 + nDiag);

  // get random visiting order
  std::vector<std::pair<int, int>> protrudeCoords;
  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    // NOTE: If debug, use non-random column-major order
    protrudeCoords = stableCoords(state.outline);
  } else {
    protrudeCoords = state.outline.shuffled(state.rng);
  }

  // protrude
  for (int i = 0; i < protrudeCoords.size(); i++) {
    auto &[r, c] = protrudeCoords[i];

    if (!protrudeConf[encodeAdj(state.cell, r,
                                c)]) // not valid protrude configuration
      continue;

    double w;
    if (state.outlineNuc.contains(r, c)) {
      w = 1.0; // force push if nucleus is against edge of cell
    } else {
      double n = nDiag * (state.cell(r - 1, c - 1) + state.cell(r + 1, c - 1) +
                          state.cell(r + 1, c + 1) + state.cell(r - 1, c + 1)) +
                 state.cell(r - 1, c) + state.cell(r, c - 1) +
                 state.cell(r + 1, c) + state.cell(r, c + 1);
      int N = state.cell.block<3, 3>(r - 1, c - 1).sum();
      double aAvg =
          (state.A.block<3, 3>(r - 1, c - 1).sum() - state.A(r, c)) / N;
      w = std::pow(n / C, config.k) * vCor *
          (1.0 - config.actSlope * (1.0 - aAvg / aMax)) *
          (state.adhF(r, c) * (config.adhBasal - 1.0) + 1.0);
    }

    // try protruding cell
    const double p = state.probDist(state.rng);
    if (p < w) {
      int N = state.cell.block<3, 3>(r - 1, c - 1).sum();
      double aAvg =
          (state.A.block<3, 3>(r - 1, c - 1).sum() - state.A(r, c)) / N;
      double iAvg =
          (state.I.block<3, 3>(r - 1, c - 1).sum() - state.I(r, c)) / N;
      double fAvg =
          (state.F.block<3, 3>(r - 1, c - 1).sum() - state.F(r, c)) / N;
      double acAvg =
          (state.AC.block<3, 3>(r - 1, c - 1).sum() - state.AC(r, c)) / N;
      double icAvg =
          (state.IC.block<3, 3>(r - 1, c - 1).sum() - state.IC(r, c)) / N;
      double fcAvg =
          (state.FC.block<3, 3>(r - 1, c - 1).sum() - state.FC(r, c)) / N;

      state.cell.coeffRef(r, c) = 1;
      state.A(r, c) = aAvg;
      state.I(r, c) = iAvg;
      state.F(r, c) = fAvg;
      state.AC(r, c) = acAvg;
      state.IC(r, c) = icAvg;
      state.FC(r, c) = fcAvg;

      // WARN: Make sure this sum is initialized properly
      state.ACorSum += state.A(r, c);
      state.ICorSum += iAvg;
      state.ACCorSum += state.AC(r, c);
      state.ICCorSum += icAvg;
    }
  }

  // update cell
  updateCell(state);
}

void retractCell(CellState &state) {
  /**
   * This function retracts pixels of the cell and is essentially the opposite
   * logic to the protrudeCell() function. Note that higher adhF results in
   * lower probability of retraction.
   */
  const auto &config = state.config;

  // get probability coefficients
  const double vCor = 1 / (1 + std::exp(-(state.V - state.V0) / config.T));
  const double aMax = state.A
                          .block(state.frameRowStart, state.frameColStart,
                                 state.frameRowEnd - state.frameRowStart + 1,
                                 state.frameColEnd - state.frameColStart + 1)
                          .maxCoeff();
  const double acMax = state.AC
                           .block(state.frameRowStart, state.frameColStart,
                                  state.frameRowEnd - state.frameRowStart + 1,
                                  state.frameColEnd - state.frameColStart + 1)
                           .maxCoeff();
  const double nDiag = 1.0 / std::pow(M_SQRT2, config.g);
  const double C = 4.0 * (1.0 + nDiag);

  // get random visiting order
  std::vector<std::pair<int, int>> retractCoords;

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    retractCoords = stableCoords(state.innerOutline);
  } else {
    retractCoords = state.innerOutline.shuffled(state.rng);
  }

  // retract
  for (int i = 0; i < retractCoords.size(); i++) {
    auto &[r, c] = retractCoords[i];

    if (!retractConf[encodeAdj(state.cell, r,
                               c)]) // not valid retract configuration
      continue;
    if (state.nuc(r, c) == 1) // can't retract nucleus
      continue;

    double n = nDiag * (!state.cell(r - 1, c - 1) + !state.cell(r + 1, c - 1) +
                        !state.cell(r + 1, c + 1) + !state.cell(r - 1, c + 1)) +
               !state.cell(r - 1, c) + !state.cell(r, c - 1) +
               !state.cell(r + 1, c) + !state.cell(r, c + 1);
    int N = state.cell.block<3, 3>(r - 1, c - 1).sum() - 1;
    double aAvg = (state.A.block<3, 3>(r - 1, c - 1).sum() - state.A(r, c)) / N;
    double w = std::pow(n / C, config.k) * vCor *
               (1.0 - config.actSlope * aAvg / aMax) * state.adhF(r, c);

    // try retracting pixel
    const double p = state.probDist(state.rng);
    if (p < w) {
      state.cell(r, c) = 0;
      // WARN: Make sure this sum is initialized properly
      state.ACorSum -= state.A(r, c);
      state.ICorSum -= state.I(r, c);
      state.ACCorSum -= state.AC(r, c);
      state.ICCorSum -= state.IC(r, c);

      state.A(r, c) = 0;
      state.I(r, c) = 0;
      state.F(r, c) = 0;
      state.AC(r, c) = 0;
      state.IC(r, c) = 0;
      state.FC(r, c) = 0;
    }
  }

  // update cell
  updateCell(state);
}

void protrudeNuc(CellState &state) {
  /**
   * Protrude the nucleus. This function is split into 4 main sections:
   * - Calculate some coefficients for protrusion probabilities. We can
   *   calculate them early to prevent having to calculate again for each
   *   pixel.
   * - Get a random protrusion order.
   * - Calculate protrusion probabilities for each pixel and try protruding the
   *   nucleus in that direction.
   * - Update the nucleus outlines.
   */
  const auto &config = state.config;

  // precompute constants
  const double vCor =
      1.0 / (1 + std::exp((state.VNuc - state.V0Nuc) / config.TNuc));
  const double R = double(state.PNuc * state.PNuc) / state.VNuc;
  const double rCor = 1.0 / (1 + std::exp((R - config.R0) / config.RNuc));
  const double nDiag = 1.0 / std::pow(M_SQRT2, config.g);
  const double C = 4.0 * (1.0 + nDiag);

  // perform vectorized calculations over entire block
  const int rStart = state.nucMinR - 1, cStart = state.nucMinC - 1;
  const int rows = state.nucMaxR - state.nucMinR + 3;
  const int cols = state.nucMaxC - state.nucMinC + 3;

  // nMatrix holds the 'n' value for every pixel in the nucleus vicinity
  Eigen::MatrixXd nMatrix = Eigen::MatrixXd::Zero(rows, cols);
  auto nAcc = nMatrix.array();

  nAcc += nDiag * (state.nuc.block(rStart - 1, cStart - 1, rows, cols)
                       .array()
                       .cast<double>() +
                   state.nuc.block(rStart + 1, cStart - 1, rows, cols)
                       .array()
                       .cast<double>() +
                   state.nuc.block(rStart + 1, cStart + 1, rows, cols)
                       .array()
                       .cast<double>() +
                   state.nuc.block(rStart - 1, cStart + 1, rows, cols)
                       .array()
                       .cast<double>());
  nAcc +=
      (state.nuc.block(rStart - 1, cStart, rows, cols).array().cast<double>() +
       state.nuc.block(rStart, cStart - 1, rows, cols).array().cast<double>() +
       state.nuc.block(rStart + 1, cStart, rows, cols).array().cast<double>() +
       state.nuc.block(rStart, cStart + 1, rows, cols).array().cast<double>());

  // protrude logic
  std::vector<std::pair<int, int>> protrudeCoords =
      state.outlineNuc.shuffled(state.rng);

  for (auto &[r, c] : protrudeCoords) {
    uint8_t conf = encodeAdj(state.nuc, r, c);
    if (state.outline.contains(r, c) || !protrudeConf[conf])
      continue;

    // lookup pre-calculated 'n'
    // NOTE: cells may change but n is still from initial state
    double n = nMatrix(r - rStart, c - cStart);

    const double w =
        std::pow(n / C, config.kNuc) * rCor * vCor *
        (config.dynBasal + (1 - config.dynBasal) * state.dynF(r, c));

    if (state.probDist(state.rng) < w) {
      state.nuc(r, c) = 1;

      // Keep the tracked bounds in step with outward growth. updateNuc()
      // scans only these bounds plus one pixel when rebuilding candidates.
      state.nucMinR = std::min(state.nucMinR, r);
      state.nucMaxR = std::max(state.nucMaxR, r);
      state.nucMinC = std::min(state.nucMinC, c);
      state.nucMaxC = std::max(state.nucMaxC, c);

      state.ACCorSum -= state.AC(r, c);
      state.AC(r, c) = 0;
      state.ICCorSum -= state.IC(r, c);
      state.IC(r, c) = 0;
      state.FC(r, c) = 0;
    }
  }

  updateNuc(state);
}

void retractNuc(CellState &state) {
  /**
   * The logic for this function is identical to protrudeNuc, but some of the
   * values are inverted for retraction.
   *
   * - The exponent for vCor is negated
   * - counting the neighbors n we instead count the empty pixels
   * - dynF values are replaced with 1 - dynF
   */
  const auto &config = state.config;

  // precompute constants
  const double vCor =
      1.0 / (1 + std::exp(-(state.VNuc - state.V0Nuc) / config.TNuc));
  const double R = double(state.PNuc * state.PNuc) / state.VNuc;
  const double rCor = 1.0 / (1 + std::exp((R - config.R0) / config.RNuc));
  const double nDiag = 1.0 / std::pow(M_SQRT2, config.g);
  const double C = 4.0 * (1.0 + nDiag);

  // perform vectorized calculations over entire block
  const int rStart = state.nucMinR - 1;
  const int cStart = state.nucMinC - 1;
  const int rows = state.nucMaxR - state.nucMinR + 3;
  const int cols = state.nucMaxC - state.nucMinC + 3;

  // nMatrix holds the 'n' value for every pixel in the nucleus vicinity
  Eigen::MatrixXd invNuc =
      1.0 - state.nuc.block(rStart - 1, cStart - 1, rows + 2, cols + 2)
                .array()
                .cast<double>();

  Eigen::MatrixXd nMatrix = Eigen::MatrixXd::Zero(rows, cols);
  auto nAcc = nMatrix.array();

  // Diagonal neighbors (using the pre-inverted block)
  nAcc += nDiag * (invNuc.block(0, 0, rows, cols).array() + // top-left
                   invNuc.block(2, 0, rows, cols).array() + // bottom-left
                   invNuc.block(0, 2, rows, cols).array() + // top-right
                   invNuc.block(2, 2, rows, cols).array()   // bottom-right
                  );
  // Orthogonal neighbors
  nAcc += (invNuc.block(0, 1, rows, cols).array() + // top
           invNuc.block(1, 0, rows, cols).array() + // left
           invNuc.block(2, 1, rows, cols).array() + // bottom
           invNuc.block(1, 2, rows, cols).array()   // right
  );

  // retract logic
  std::vector<std::pair<int, int>> retractCoords =
      state.innerOutlineNuc.shuffled(state.rng);

  bool recheckBounds = false;
  for (auto &[r, c] : retractCoords) {
    uint8_t conf = encodeAdj(state.nuc, r, c);
    if (!retractConf[conf])
      continue;

    // lookup pre-calculated 'n'
    // NOTE: cells may change but n is still from initial state
    double n = nMatrix(r - rStart, c - cStart);

    const double w = std::pow(n / C, config.kNuc) * rCor * vCor *
                     (config.dynBasal +
                      (1 - config.dynBasal) *
                          (1 - state.dynF(r, c))); // inverted from protrusion

    if (state.probDist(state.rng) < w) {
      state.nuc(r, c) = 0;

      if (r == state.nucMinR || r == state.nucMaxR || c == state.nucMinC ||
          c == state.nucMaxC)
        recheckBounds = true;

      // count number of neighbors and sum up values
      int n = 9 - state.nuc.block<3, 3>(r - 1, c - 1)
                      .sum(); // number of cell pixels (non-nucleus)
      double AC = state.AC.block<3, 3>(r - 1, c - 1).sum();
      double FC = state.FC.block<3, 3>(r - 1, c - 1).sum();
      double IC = state.IC.block<3, 3>(r - 1, c - 1).sum();

      state.AC(r, c) = std::clamp(AC / n, config.ACMin, config.ACMax);
      state.ACCorSum += state.AC(r, c);
      state.IC(r, c) = IC / n;
      state.ICCorSum += state.IC(r, c);
      state.FC(r, c) = FC / n;
    }
  }

  updateNuc(state, recheckBounds);
}

[[deprecated]]
void protrudeNucDep(CellState &state) {
  // calculate probability coefficients
  const auto &config = state.config;
  const double vCor =
      1.0 / (1 + std::exp((state.VNuc - state.V0Nuc) / config.TNuc));
  const double R = (state.PNuc * state.PNuc) / state.VNuc; // WARN: Not casting
                                                           // to double
  const double rCor = 1.0 / (1 + std::exp((R - config.R0) / config.RNuc));
  const double nDiag = 1.0 / std::pow(M_SQRT2, config.g);
  const double C = 4.0 * (1.0 + nDiag);

  // randomize protrude order
  std::vector<std::pair<int, int>> protrudeCoords;

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    // NOTE: If debug, use non-random column-major order
    protrudeCoords = stableCoords(state.outlineNuc);
  } else {
    protrudeCoords = state.outlineNuc.shuffled(state.rng);
  }

  // generate dynein field for protrusion probability
  generateDynField(state, state.innerOutline, state.outlineNuc, false);

  // protrude
  for (int i = 0; i < protrudeCoords.size(); i++) {
    auto [r, c] = protrudeCoords[i];

    if (state.outline.contains(r, c) ||
        !protrudeConf[encodeAdj(state.nuc, r, c)]) // Check if protrusion would
                                                   // be valid
      continue;

    // get protrusion probability
    const double n =
        nDiag * (state.nuc(r - 1, c - 1) + state.nuc(r + 1, c - 1) +
                 state.nuc(r + 1, c + 1) + state.nuc(r - 1, c + 1)) +
        state.nuc(r - 1, c) + state.nuc(r, c - 1) + state.nuc(r + 1, c) +
        state.nuc(r, c + 1);
    const double w =
        std::pow(n / C, config.kNuc) * rCor * vCor *
        (config.dynBasal + (1 - config.dynBasal) * state.dynF(r, c));

    // try protruding to this pixel
    const double p = state.probDist(state.rng);
    if (p < w) {
      state.nuc(r, c) = 1;
      // Expand bounds incrementally
      state.nucMinR = std::min(state.nucMinR, r);
      state.nucMaxR = std::max(state.nucMaxR, r);
      state.nucMinC = std::min(state.nucMinC, c);
      state.nucMaxC = std::max(state.nucMaxC, c);
      state.ACCorSum -= state.AC(r, c);
      state.AC(r, c) = 0;
      state.ICCorSum -= state.IC(r, c);
      state.IC(r, c) = 0;
      state.FC(r, c) = 0;
    }
  }

  // update nucleus outlines
  updateNuc(state);
}

[[deprecated]]
void retractNucDep(CellState &state) {
  /**
   * The logic for this function is identical to protrudeNuc, but some of the
   * values are inverted for retraction.
   *
   * - The exponent for vCor is negated
   * - counting the neighbors n we instead count the empty pixels
   * - dynF values are replaced with 1 - dynF
   */
  const auto &config = state.config;

  // calculate probability coefficients
  const double vCor =
      1.0 / (1 + std::exp(-(state.VNuc - state.V0Nuc) / config.TNuc));
  const double R = (state.PNuc * state.PNuc) / state.VNuc; // WARN: Not casting
                                                           // to double
  const double rCor = 1.0 / (1 + std::exp((R - config.R0) / config.RNuc));
  const double nDiag = 1.0 / std::pow(M_SQRT2, config.g);
  const double C = 4.0 * (1.0 + nDiag);

  // randomize retract order
  std::vector<std::pair<int, int>> retractCoords;

  if constexpr (DYNEIN_CELL_MODEL_DEBUG_CPP) {
    // NOTE: If debug, use non-random column-major order
    retractCoords = stableCoords(state.innerOutlineNuc);
  } else {
    retractCoords = state.innerOutlineNuc.shuffled(state.rng);
  }

  // generate dynein field for retraction probability
  generateDynField(state, state.innerOutline, state.innerOutlineNuc, true);

  // retract
  bool recheckBounds = false; // whether a retracted pixel was on the bounds
                              // and the nucleus bounds need to be rechecked
  for (int i = 0; i < retractCoords.size(); i++) {
    const auto [r, c] = retractCoords[i];

    if (!retractConf[encodeAdj(state.nuc, r,
                               c)]) // Check if retraction would be valid
      continue;

    // get retraction probability
    const double n =
        nDiag * (!state.nuc(r - 1, c - 1) + !state.nuc(r + 1, c - 1) +
                 !state.nuc(r + 1, c + 1) + !state.nuc(r - 1, c + 1)) +
        !state.nuc(r - 1, c) + !state.nuc(r, c - 1) + !state.nuc(r + 1, c) +
        !state.nuc(r, c + 1);
    const double w =
        std::pow(n / C, config.kNuc) * rCor * vCor *
        (config.dynBasal + (1 - config.dynBasal) * state.dynF(r, c));

    // try retracting this pixel
    const double p = state.probDist(state.rng);
    if (p < w) {
      state.nuc(r, c) = 0;

      if (r == state.nucMinR || r == state.nucMaxR || c == state.nucMinC ||
          c == state.nucMaxC)
        recheckBounds = true;

      // count number of neighbors and sum up values
      int n = 8 - state.nuc.block<3, 3>(r - 1, c - 1)
                      .sum(); // number of cell pixels (non-nucleus)
      double AC = state.AC.block<3, 3>(r - 1, c - 1).sum() - state.AC(r, c);
      double FC = state.FC.block<3, 3>(r - 1, c - 1).sum() - state.FC(r, c);
      double IC = state.IC.block<3, 3>(r - 1, c - 1).sum() - state.IC(r, c);

      state.AC(r, c) = AC / n;
      state.ACCorSum += state.AC(r, c);
      state.IC(r, c) = IC / n;
      state.ICCorSum += state.IC(r, c);
      state.FC(r, c) = FC / n;
    }
  }

  // update nucleus outlines
  updateNuc(state, recheckBounds);
}

void rearrangeAdhesions(CellState &state, const bool bias,
                        const bool rearrangeAll) {
  /**
   * The logic behind this function is to move some number of adhesions to new
   * spots with polarization. To optimize this function, we perform as much
   * precomputation as possible. We generate a list of random indices of
   * adhesions to remove, and then generate an array to represent the CDF of A
   * within the frame. Then for each adhesion, we will generate a random
   * number and check if it is a valid target. If it is, we will use that as
   * the new adhesion, and if not we repeat the process.
   *
   * This optimizes the case in which all of the A values are somewhat
   * similar, making rejection sampling inefficient. Since each value should
   * have a very small difference in the CDF in this case, the chance that the
   * CDF has to be resampled is very low.
   */
  const auto &config = state.config;

  const int rearrangeAdh = // number of adhesions to rearrange
      rearrangeAll ? config.adhNum : int(config.adhNum * config.adhFrac);
  const int rows = state.frameRowEnd - state.frameRowStart + 1;
  const int cols = state.frameColEnd - state.frameColStart + 1;
  const int frameSize = rows * cols;

  // generate indices to rearrange
  const std::vector<int> indices =
      generateIndices(rearrangeAdh, 0, config.adhNum, state.rng);

  // precompute cumulative probability as array
  std::vector<double> aLin(frameSize); // linearized version of A within frame
  std::vector<std::pair<int, int>> flatPos(frameSize);
  double aSum = 0; // total sum of A in frame
  if (bias) {
    for (int i = 0, r = state.frameRowStart; r <= state.frameRowEnd; i++, r++) {
      for (int j = 0, c = state.frameColStart; c <= state.frameColEnd;
           j++, c++) {
        if (state.env.coeff(r, c) == 1 && state.cell(r, c) == 1) {
          // if is valid attachment point, it is a valid place for new adhesion
          aSum += state.A(r, c);
        }
        aLin[i * cols + j] = aSum;
        flatPos[i * cols + j] = {r, c};
      }
    }
  }

  if (!bias || aSum == 0) {
    // if no A signal, assume uniform probability
    aSum = 0;
    for (int i = 0, r = state.frameRowStart; r <= state.frameRowEnd; i++, r++) {
      for (int j = 0, c = state.frameColStart; c <= state.frameColEnd;
           j++, c++) {
        if (state.env.coeff(r, c) == 1 && state.cell(r, c) == 1) {
          aSum += 1;
        }
        aLin[i * cols + j] = aSum;
        flatPos[i * cols + j] = {r, c};
      }
    }
  }

  // iterate through adhesions
  for (int i = 0; i < rearrangeAdh; i++) {
    // remove old adhesion
    const int selIdx = indices[i];
    state.adh.coeffRef(state.adhPos(0, selIdx), state.adhPos(1, selIdx)) = 0;

    int r = -1, c = -1; // row and column for new adhesion to go to
    do {
      // generate random probability and find index in cumulative sum
      const double p = state.probDist(state.rng);
      const auto idxIt = std::lower_bound(aLin.begin(), aLin.end(), aSum * p);
      const int flatIdx = idxIt - aLin.begin();

      // convert index to row and column
      const int candR = flatIdx / cols + state.frameRowStart;
      const int candC = flatIdx % cols + state.frameColStart;

      // check new index valid
      if (state.adh.coeff(candR, candC) != 1) {
        // env and cell should both always be satisfied, but to be safe
        r = candR;
        c = candC;
      }
    } while (r < 0 || c < 0);

    // add new adhesion
    state.adh.coeffRef(r, c) = 1;
    state.adhPos(0, selIdx) = r;
    state.adhPos(1, selIdx) = c;
  }

  // smooth adhesion field
  updateAdhesionField(state);
}

void generateDynField(CellState &state, const OutlineMask &cellOutline,
                      const OutlineMask &nucOutline, const bool retract) {
  auto &config = state.config;
  state.dynF.setZero();
  ViewI scaling = ViewI::Zero(config.simRows, config.simCols);

  const int len = static_cast<int>(nucOutline.size());
  int n = len / (retract ? 6 : 30);

  // Pre-compute nuc outline coordinates for faster iteration
  std::vector<std::pair<int, int>> nucCoords;
  nucCoords.reserve(nucOutline.size());
  for (const auto &coord : nucOutline)
    nucCoords.push_back(coord);

#ifdef USE_OPENMP
  // Collect cell outline coordinates for parallelization
  std::vector<std::pair<int, int>> cellCoords;
  cellCoords.reserve(cellOutline.size());
  for (const auto &coord : cellOutline)
    cellCoords.push_back(coord);

  // Thread-local accumulators for dynF and scaling
  const int numThreads = omp_get_max_threads();
  std::vector<ViewD> dynFLocal(numThreads,
                               ViewD::Zero(config.simRows, config.simCols));
  std::vector<ViewI> scalingLocal(numThreads);
  for (int t = 0; t < numThreads; t++) {
    scalingLocal[t] = ViewI::Zero(config.simRows, config.simCols);
  }

// Parallel loop over cell outline pixels
#pragma omp parallel
  {
    const int threadId = omp_get_thread_num();

#pragma omp for
    for (int idx = 0; idx < static_cast<int>(cellCoords.size()); idx++) {
      const int r = cellCoords[idx].first;
      const int c = cellCoords[idx].second;

      // Early exit if AC is too low
      const double acVal = state.AC(r, c);
      if (acVal <= 0.1)
        continue;

      // get nucleus pixel closest to current pixel
      int minDist2 = INT_MAX;
      int minR = -1, minC = -1;
      for (const auto &nc : nucCoords) {
        const int dr = r - nc.first;
        const int dc = c - nc.second;
        const int dist2 = dr * dr + dc * dc;
        if (dist2 < minDist2) {
          minDist2 = dist2;
          minR = nc.first;
          minC = nc.second;
        }
      }

      const double distF = std::sqrt(minDist2) * (acVal - 0.1);
      const int rStart = std::max(minR - n, 0);
      const int rEnd = std::min(minR + n, config.simRows - 1);
      const int cStart = std::max(minC - n, 0);
      const int cEnd = std::min(minC + n, config.simCols - 1);

      for (int i = rStart; i <= rEnd; ++i) {
        for (int j = cStart; j <= cEnd; ++j) {
          if (nucOutline.contains(i, j)) {
            dynFLocal[threadId](i, j) += distF;
            scalingLocal[threadId](i, j) += 1;
          }
        }
      }
    }
  }

  // Merge thread-local accumulators
  for (int t = 0; t < numThreads; t++) {
    state.dynF += dynFLocal[t];
    scaling += scalingLocal[t];
  }

#else
  // Single-threaded version
  for (const auto &[r, c] : cellOutline) {

    // Early exit if AC is too low
    const double acVal = state.AC(r, c);
    if (acVal <= 0.1)
      continue;

    // get nucleus pixel closest to current pixel
    int minDist2 = INT_MAX;
    int minR = -1, minC = -1;
    for (const auto &nc : nucCoords) {
      const int dr = r - nc.first;
      const int dc = c - nc.second;
      const int dist2 = dr * dr + dc * dc;
      if (dist2 < minDist2) {
        minDist2 = dist2;
        minR = nc.first;
        minC = nc.second;
      }
    }

    const double distF = std::sqrt(minDist2) * (acVal - 0.1);
    const int rStart = std::max(minR - n, 0);
    const int rEnd = std::min(minR + n, config.simRows - 1);
    const int cStart = std::max(minC - n, 0);
    const int cEnd = std::min(minC + n, config.simCols - 1);

    for (int i = rStart; i <= rEnd; ++i) {
      for (int j = cStart; j <= cEnd; ++j) {
        if (nucOutline.contains(i, j)) {
          state.dynF(i, j) += distF;
          scaling(i, j) += 1;
        }
      }
    }
  }
#endif

  // normalize elements
  for (const auto &[r, c] : nucOutline) {
    if (scaling(r, c) == 0)
      continue;
    if (!retract) {
      state.dynF(r, c) = std::min(state.dynF(r, c) / scaling(r, c) / 60, 1.0);
    } else {
      state.dynF(r, c) =
          std::max(1 - state.dynF(r, c) / scaling(r, c) / 60, 0.0);
    }
  }
}

void updateFrame(CellState &state) {
  const auto &conf = state.config;
  int minRow = std::numeric_limits<int>::max();
  int maxRow = std::numeric_limits<int>::min();
  int minCol = std::numeric_limits<int>::max();
  int maxCol = std::numeric_limits<int>::min();

  // only need to check inner outline of cell because those are the cell
  // boundaries
  for (const auto &[i, j] : state.innerOutline) {

    // Update bounding box
    if (state.cell(i, j) != 0) {
      minRow = std::min(minRow, i);
      maxRow = std::max(maxRow, i);
      minCol = std::min(minCol, j);
      maxCol = std::max(maxCol, j);
    }
  }

  if (minRow == std::numeric_limits<int>::max()) {
    state.frameRowStart = 0;
    state.frameRowEnd = conf.simRows - 1;
    state.frameColStart = 0;
    state.frameColEnd = conf.simCols - 1;
    return;
  }
  state.frameRowStart = std::max(0, minRow - conf.framePadding);
  state.frameRowEnd = std::min(conf.simRows - 1, maxRow + conf.framePadding);
  state.frameColStart = std::max(0, minCol - conf.framePadding);
  state.frameColEnd = std::min(conf.simCols - 1, maxCol + conf.framePadding);
}

void updateGeometry(CellState &state) {
  updateCell(state, true);
  updateNuc(state, true);
}

} // namespace dynein_cell_model
