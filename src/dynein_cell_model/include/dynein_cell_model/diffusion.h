#pragma once

#include "dynein_cell_model/state.h"

namespace dynein_cell_model {

/**
 * @brief Correct A, I, AC, IC concentrations after protrusion/retraction.
 */
void correctConcentrations(CellState &state);

/**
 * @brief Run the reaction-diffusion update on A/I/F/AC/IC/FC within the frame,
 * driven by the k0Adh field.
 */
void diffuseK0Adh(CellState &state);

/** Reference Eigen implementation, retained for numerical parity checks. */
void diffuseK0AdhEigen(CellState &state);

/** Kokkos implementation of the reaction-diffusion update. */
void diffuseK0AdhKokkos(CellState &state);

/**
 * @brief Update and smooth the adhesion field
 */
void updateAdhesionField(CellState &state);

} // namespace dynein_cell_model
