#pragma once

#include "dynein_cell_model/state.h"

namespace dynein_cell_model {

/**
 * @brief Perform one time step of cell simulations.
 */
void step(CellState &state);

void simulateSteps(CellState &state, int steps);
void simulate(CellState &state, double duration);

/**
 * @brief Perform one time step of cell simulations using old logic, with
 * timing metrics.
 */
[[deprecated]]
void step_dep(CellState &state);

[[deprecated]]
void stepDep(CellState &state);

} // namespace dynein_cell_model
