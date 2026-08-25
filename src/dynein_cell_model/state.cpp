#include "dynein_cell_model/state.h"
#include "dynein_cell_model/config.h"
#include "dynein_cell_model/morphology.h"
#include "dynein_cell_model/io.h"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace dynein_cell_model {

CellState::CellState() : CellState(CellModelConfig{}) {}

CellState::CellState(const CellModelConfig &conf)
    : config{conf}, params{conf.getDiffusionParams()} {
  // Frame starts as the whole simulation area; narrowed once a cell mask
  // is available.
  frameRowStart = 0;
  frameRowEnd = config.simRows - 1;
  frameColStart = 0;
  frameColEnd = config.simCols - 1;

  t = 0;

  // Nucleus bounds, recomputed on the first update_nuc call.
  nucMinR = config.simRows;
  nucMaxR = 0;
  nucMinC = config.simCols;
  nucMaxC = 0;

  V0 = 0;
  V = 0;
  P = 0;
  V0Nuc = 0;
  VNuc = 0;
  PNuc = 0;

  ACorSum = 0;
  ICorSum = 0;
  ACCorSum = 0;
  ICCorSum = 0;

  rng = std::mt19937(config.seed);
  probDist = std::uniform_real_distribution<>(0.0, 1.0);

  const int rows = config.simRows;
  const int cols = config.simCols;
  outline = OutlineMask(rows, cols);
  innerOutline = OutlineMask(rows, cols);
  outlineNuc = OutlineMask(rows, cols);
  innerOutlineNuc = OutlineMask(rows, cols);
  env = ViewMask(rows, cols);
  adh = ViewMask(rows, cols);
  adhPos = ViewI::Zero(2, config.adhNum);
  k0Adh = ViewD::Zero(rows, cols);
  A = ViewD::Zero(rows, cols);
  I = ViewD::Zero(rows, cols);
  F = ViewD::Zero(rows, cols);
  AC = ViewD::Zero(rows, cols);
  IC = ViewD::Zero(rows, cols);
  FC = ViewD::Zero(rows, cols);
  adhF = ViewD::Zero(rows, cols);
  dynF = ViewD::Zero(rows, cols);
}

CellState::~CellState() = default;
CellState::CellState(CellState &&) noexcept = default;
CellState &CellState::operator=(CellState &&) noexcept = default;

namespace {
void validateMask(const char *name, const ViewI &mask, int rows, int cols) {
  if (mask.rows() != rows || mask.cols() != cols)
    throw std::invalid_argument(std::string(name) + " mask dimensions must match config");
}
void validateMask(const char *name, const ViewMask &mask, int rows, int cols) {
  if (mask.rows() != rows || mask.cols() != cols)
    throw std::invalid_argument(std::string(name) + " mask dimensions must match config");
}
} // namespace

void initializeState(CellState &state, const ViewI &cell, const ViewI &nuc,
                     const ViewMask &env) {
  const auto &config = state.config;
  validateMask("cell", cell, config.simRows, config.simCols);
  validateMask("nucleus", nuc, config.simRows, config.simCols);
  validateMask("environment", env, config.simRows, config.simCols);
  if (((nuc.array() != 0) && (cell.array() == 0)).any())
    throw std::invalid_argument("nucleus mask must be contained in the cell mask");
  state.cell = cell;
  state.nuc = nuc;
  state.env = env;
  state.rng.seed(config.seed);
  state.t = 0;
  updateGeometry(state);
  state.V0 = state.V;
  state.V0Nuc = state.VNuc;
  updateFrame(state);
}

CellState initializeState(const CellModelConfig &config, const ViewI &cell,
                          const ViewI &nuc, const ViewMask &env) {
  CellState state(config);
  initializeState(state, cell, nuc, env);
  return state;
}

void initializeAdhesions(CellState &state, bool bias) {
  rearrangeAdhesions(state, bias, true);
}

ViewI matrixFromMask(const std::string &filepath, cv::Vec3b color) {
  cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::invalid_argument("Could not load image: " + filepath);
  ViewI mask = ViewI::Zero(image.rows, image.cols);
  for (int r = 0; r < image.rows; ++r)
    for (int c = 0; c < image.cols; ++c)
      mask(r, c) = image.at<cv::Vec3b>(r, c) == color;
  return mask;
}

} // namespace dynein_cell_model
