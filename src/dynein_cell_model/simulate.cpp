#include "dynein_cell_model/simulate.h"
#include "dynein_cell_model/diffusion.h"
#include "dynein_cell_model/io.h"
#include "dynein_cell_model/morphology.h"
#include "dynein_cell_model/state.h"

namespace dynein_cell_model {

void updateDynNucField(CellState &state);

void step(CellState &state) {
  auto &conf = state.config;
  if (state.t % conf.adhT == 0)
    rearrangeAdhesions(state);

  if (state.t % conf.frT == 0)
    updateFrame(state);

  // Use a field derived from the current nucleus for protrusion, then refresh
  // it after the boundary changes so retraction does not use stale geometry.
  updateDynNucField(state);
  protrudeNuc(state);
  updateDynNucField(state);
  retractNuc(state);

  protrudeCell(state);
  retractCell(state);

  correctConcentrations(state);
  diffuseK0Adh(state);

  if (++state.t % conf.saveT == 0 && state.output)
    saveState(state);
}

void simulateSteps(CellState &state, int steps) {
  if (steps < 0)
    throw std::invalid_argument("steps must be non-negative");
  for (int i = 0; i < steps; ++i)
    step(state);
}

void simulate(CellState &state, double duration) {
  if (duration < 0)
    throw std::invalid_argument("duration must be non-negative");
  simulateSteps(state, static_cast<int>(duration / state.config.dt));
}

[[deprecated]] void stepDep(CellState &state) {
  auto &conf = state.config;
  if (state.t % conf.adhT == 0) rearrangeAdhesions(state);
  if (state.t % conf.frT == 0) updateFrame(state);
  protrudeNucDep(state); retractNucDep(state);
  protrudeCell(state); retractCell(state);
  correctConcentrations(state); diffuseK0Adh(state);
  if (++state.t % conf.saveT == 0 && state.output) saveState(state);
}

[[deprecated]] void step_dep(CellState &state) { stepDep(state); }

} // namespace dynein_cell_model
