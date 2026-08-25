// Umbrella header for the function-based API. CellModel was intentionally
// removed as part of the breaking procedural refactor.
#pragma once

#include <opencv2/core.hpp>

#include "dynein_cell_model/config.h"
#include "dynein_cell_model/state.h"

namespace dynein_cell_model {

ViewI matrixFromMask(const std::string &filepath, cv::Vec3b color);

CellState initializeState(const CellModelConfig &config, const ViewI &cell,
                          const ViewI &nuc, const ViewMask &env);

void initializeState(CellState &state, const ViewI &cell, const ViewI &nuc,
                     const ViewMask &env);

void initializeAdhesions(CellState &state, bool bias = false);

} // namespace dynein_cell_model
