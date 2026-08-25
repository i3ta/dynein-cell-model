#pragma once

#include <memory>
#include <random>

#include "dynein_cell_model/config.h"
#include "dynein_cell_model/types.h"

namespace dynein_cell_model {

class OutputWriter;

struct CellState {
  CellState();
  explicit CellState(const CellModelConfig &conf);
  ~CellState();
  CellState(CellState &&) noexcept;
  CellState &operator=(CellState &&) noexcept;
  CellState(const CellState &) = delete;
  CellState &operator=(const CellState &) = delete;

  // Fixed config params for the simulation
  CellModelConfig config;

  // Reaction-diffusion equation params
  DiffusionParams params;

  // Simulation parameters
  int t; ///< current time step

  // Frame variables
  int frameRowStart; ///< first row in the frame (inclusive)
  int frameRowEnd;   ///< last row in the frame (inclusive)
  int frameColStart; ///< first column in the frame (inclusive)
  int frameColEnd;   ///< last column in the frame (inclusive)

  // Cell state variables
  int V0;    ///< initial volume of the cell, set up when initial Im is read
  int V;     ///< volume of the cell on the current step
  int P;     ///<< perimeter of the cell
  int V0Nuc; ///< initial volume of the cell, set up when initial Im is read
  int VNuc;  ///< volume of the cell on the current step
  int PNuc;  ///< perimeter of the nucleus

  // Nucleus bounds for optimized iteration
  int nucMinR; ///< minimum row of nucleus
  int nucMaxR; ///< maximum row of nucleus
  int nucMinC; ///< minimum column of nucleus
  int nucMaxC; ///< maximum column of nucleus

  double ACorSum;  ///< Correct A values after retraction and protrusion
  double ICorSum;  ///< Correct I values after retraction and protrusion
  double ACCorSum; ///< Correct AC values after retraction and protrusion
  double ICCorSum; ///< Correct IC values after retraction and protrusion

  ViewI cell;               ///< cell mask
  ViewI nuc;                ///< nucleus mask
  ViewMask outline;         ///< cell outline
  ViewMask innerOutline;    ///< cell outline inner pixel
  ViewMask outlineNuc;      ///< nucleus outline
  ViewMask innerOutlineNuc; ///< nucleus outline inner pixel
  ViewMask
      env; ///< environment the cell is in defining pixels the cell can sense
  ViewMask adh; ///< cell adhesions
  ViewI adhPos; ///< cell adhesion coordinates

  ViewD k0Adh; ///< distribution of k0
  ViewD A;     ///< values of A
  ViewD I;     ///< values of I
  ViewD F;     ///< values of F
  ViewD AC;    ///< values of AC
  ViewD IC;    ///< values of IC
  ViewD FC;    ///< values of FC
  ViewD adhF;  ///< field of adhesion influence
  ViewD dynF;  ///< dynein field force

  // random number generation helpers
  std::mt19937 rng;
  std::uniform_real_distribution<> probDist;

  // Optional output sink.  HDF5 implementation details intentionally remain
  // outside the simulation state.
  std::unique_ptr<OutputWriter> output;
};

} // namespace dynein_cell_model
