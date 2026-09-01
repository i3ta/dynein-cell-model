#pragma once

#include "dynein_cell_model/types.h"
#include <string>

namespace dynein_cell_model {

class CellModelConfig {
  friend class CellModelTest;

public:
  /**
   * @brief Set up config with default values
   */
  CellModelConfig();

  /**
   * @brief Set up config with values from config file
   *
   * @param configFile config file to read
   */
  CellModelConfig(std::string configFile);

  /**
   * @brief Save config to file
   *
   * @param destFile destination file
   */
  void saveFile(std::string destFile) const;

  /**
   * @brief get the DiffusionParams from this config
   */
  DiffusionParams getDiffusionParams() const;

  double k;    ///< Relative contribution of geometry factor to cell
               ///< protrusion/retraction probability
  double kNuc; ///< controls degree of geometry constraint
  double g;    ///< Sensitivity of geometry factor to local membrane curvature
  double T;    ///< Parameter controlling steepness of volume factor function
               ///< (sensitivity to changes in cell volume)
  double TNuc; ///< controls sharpness of volume constraint
  double actSlope; ///< Slope of actin factor function
  double adhSigma; ///< Sigma value for gaussian smoothing of adhesion field
  double adhBasal; ///< Basal value for adhesion factor protrusion probability
  double adhFrac;  ///< Fraction of adhesions that are rearranged at each adhT
                   ///< time step
  int adhNum;      ///< number of adhesions in the cell
  int R0;          ///< Roundness (perimeter^2/area) of a 4-connected circle
  double RNuc;     ///< controls sharpness of roundness constraint
  double dynBasal; ///< basal weight for protrusion probability of dynein factor
  double propFactor; ///< number in range [0, 1] to multiply protrusions and
                     ///< retraction weights to study effect of scaling
  double dynSigma;   ///< multiplier for perimeter-derived dynein Gaussian width
  double dynScale;   ///< factor to scale dynF values by

  // Reaction-diffusion parameters
  double DA;       ///< Diffusion coefficient of active GTPase
  double DI;       ///< Diffusion coefficient of inactive GTPase
  double k0;       ///< Activation rate of GTPase
  double k0Min;    ///< Minimum basal activation rate of GTPase
  double k0Scalar; ///< Effect of adhesion field on GTPase activation
  double gamma;    ///< Rate constant of autocatalytic activation of GTPase
  double delta;    ///<
  double A0;       ///< Sensitivity of positive feedback of GTPase to the
                   ///< concentration of active GTPase
  double s1;       ///< Basal deactivation rate of GTPase
  double s2;  ///< Rate constant of negative feedback from F-actin on GTPase
  double F0;  ///< Sensitivity of negative feedback of GTPase to the
              ///< concentration of F-actin
  double kn;  ///< Rate constant of F-actin polymerization
  double ks;  ///< Rate constant of F-actin depolymerization
  double eps; ///<
  double dt;  ///< Temporal step of finite difference scheme
  double dx;  ///< Spatial step of finite difference scheme

  // Concentration limit parameters
  double AMax;  ///< maximal value of A
  double AMin;  ///< minimal value of A
  double ACMax; ///< maximal value of AC
  double ACMin; ///< minimal value of AC

  // Simulation size
  int simRows; ///< Total number of rows for the simulation
  int simCols; ///< Total number of columns for the simualation

  // Simulation parameters
  int t;               ///< current time step
  int adhT;            ///< number of time steps per adhesion rearrangement
  int frT;             ///< number of time steps per frame update
  int saveT;           ///< number of time steps per save
  int diffT;           ///< time of diffusion
  int framePadding;    ///< distance from the cell border to the edge of the
                       ///< frame
  std::string saveDir; ///< directory to save snapshots to;
  int seed;
  int numIters;
  std::string diffusionBackend; ///< "eigen" (default) or "kokkos"
};

} // namespace dynein_cell_model
