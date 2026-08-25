#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dynein_cell_model/diffusion.h"
#include "dynein_cell_model/types.h"

namespace dynein_cell_model {

void correctConcentrations(CellState &state) {
  // Calculate amount of signal that needs to be distributed from all pixels
  const double aDist = state.ACorSum / state.V;
  const double iDist = state.ICorSum / state.V;
  const double acDist = state.ACCorSum / (state.V - state.VNuc);
  const double icDist = state.ICCorSum / (state.V - state.VNuc);

  // #pragma omp parallel for collapse(2)
  for (int j = state.frameColStart; j <= state.frameColEnd; j++) {
    for (int i = state.frameRowStart; i <= state.frameRowEnd; i++) {
      if (state.cell(i, j) == 1) {
        state.A(i, j) -= aDist;
        state.I(i, j) -= iDist;
      }
      if (state.cell(i, j) == 1 && state.nuc(i, j) == 0) {
        state.AC(i, j) -= acDist;
        state.IC(i, j) -= icDist;
      }
    }
  }

  state.ACorSum = 0;
  state.ICorSum = 0;
  state.ACCorSum = 0;
  state.ICCorSum = 0;
}

void diffuseK0Adh(CellState &state) {
  const auto &config = state.config;

  double s2C = 0.05;
  double a0Cubed = std::pow(config.A0, 3);
  double dx2 = config.dx * config.dx;

  // temporary variables for update
  ViewD aNew(config.simRows, config.simCols);
  ViewD iNew(config.simRows, config.simCols);
  ViewD fNew(config.simRows, config.simCols);
  ViewD acNew(config.simRows, config.simCols);
  ViewD icNew(config.simRows, config.simCols);
  ViewD fcNew(config.simRows, config.simCols);

  aNew = state.A.eval();
  iNew = state.I.eval();
  fNew = state.F.eval();
  acNew = state.AC.eval();
  icNew = state.IC.eval();
  fcNew = state.FC.eval();

  for (int k = 0; k < config.diffT; k++) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int i = state.frameRowStart; i <= state.frameRowEnd; i++) {
      for (int j = state.frameColStart; j <= state.frameColEnd; j++) {
        if (state.cell(i, j) == 1) {
          double a3 = std::pow(state.A(i, j), 3);
          double f =
              (state.k0Adh(i, j) + config.gamma * a3 / (a0Cubed + a3)) *
                  state.I(i, j) -
              config.delta *
                  (config.s1 +
                   config.s2 * state.F(i, j) /
                       (config.F0 + state.F(i, j))) *
                  state.A(i, j);
          double h =
              config.eps * (config.kn * state.A(i, j) - config.ks * state.F(i, j));

          aNew(i, j) =
              state.A(i, j) +
              config.dt *
                  (f +
                   config.DA / dx2 *
                       (double)(state.cell(i + 1, j) *
                                    (state.A(i + 1, j) - state.A(i, j)) -
                                state.cell(i - 1, j) *
                                    (state.A(i, j) - state.A(i - 1, j)) +
                                state.cell(i, j + 1) *
                                    (state.A(i, j + 1) - state.A(i, j)) -
                                state.cell(i, j - 1) *
                                    (state.A(i, j) - state.A(i, j - 1))));
          iNew(i, j) =
              state.I(i, j) +
              config.dt *
                  (-f +
                   config.DI / dx2 *
                       (double)(state.cell(i + 1, j) *
                                    (state.I(i + 1, j) - state.I(i, j)) -
                                state.cell(i - 1, j) *
                                    (state.I(i, j) - state.I(i - 1, j)) +
                                state.cell(i, j + 1) *
                                    (state.I(i, j + 1) - state.I(i, j)) -
                                state.cell(i, j - 1) *
                                    (state.I(i, j) - state.I(i, j - 1))));
          fNew(i, j) = state.F(i, j) + h * config.dt;

          if (state.nuc(i, j) == 0) {
            double ac3 = std::pow(state.AC(i, j), 3);
            double fC =
                (config.k0 + config.gamma * ac3 / (a0Cubed + ac3)) *
                    state.IC(i, j) -
                config.delta *
                    (config.s1 +
                     s2C * state.FC(i, j) /
                         (config.F0 + state.FC(i, j))) *
                    state.AC(i, j);
            double hC =
                config.eps * (config.kn * state.AC(i, j) - config.ks * state.FC(i, j));

            acNew(i, j) =
                state.AC(i, j) +
                config.dt *
                    (fC +
                     config.DA / dx2 *
                         (double)((state.cell(i + 1, j) - state.nuc(i + 1, j)) *
                                      (state.AC(i + 1, j) - state.AC(i, j)) -
                                  (state.cell(i - 1, j) - state.nuc(i - 1, j)) *
                                      (state.AC(i, j) - state.AC(i - 1, j)) +
                                  (state.cell(i, j + 1) - state.nuc(i, j + 1)) *
                                      (state.AC(i, j + 1) - state.AC(i, j)) -
                                  (state.cell(i, j - 1) - state.nuc(i, j - 1)) *
                                      (state.AC(i, j) - state.AC(i, j - 1))));
            icNew(i, j) =
                state.IC(i, j) +
                config.dt *
                    (-fC +
                     config.DI / dx2 *
                         (double)((state.cell(i + 1, j) - state.nuc(i + 1, j)) *
                                      (state.IC(i + 1, j) - state.IC(i, j)) -
                                  (state.cell(i - 1, j) - state.nuc(i - 1, j)) *
                                      (state.IC(i, j) - state.IC(i - 1, j)) +
                                  (state.cell(i, j + 1) - state.nuc(i, j + 1)) *
                                      (state.IC(i, j + 1) - state.IC(i, j)) -
                                  (state.cell(i, j - 1) - state.nuc(i, j - 1)) *
                                      (state.IC(i, j) - state.IC(i, j - 1))));
            fcNew(i, j) = state.FC(i, j) + hC * config.dt;
          }
        }
      }
    }

    // replace elements
    state.A.swap(aNew);
    state.I.swap(iNew);
    state.F.swap(fNew);
    state.AC.swap(acNew);
    state.IC.swap(icNew);
    state.FC.swap(fcNew);
  }
}

void updateAdhesionField(CellState &state) {
  /**
   * Update adhesion field logic. Works by computing normalization variables
   * at each adhesion, then applying a weighted Gaussian over every pixel from
   * the adhesions. Then inverts and normalizes adhF, and used to calculate
   * k0Adh with IDW.
   */
  const auto &config = state.config;

  // Precompute constants
  const double sigma2 = config.adhSigma * config.adhSigma;
  const double ampl = 1 / (2 * M_PI * sigma2);

  // Position of adhesions as doubles
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      adhPosD = state.adhPos.cast<double>();

  // Precompute adhG at adhesion positions for IDW calculation
  Eigen::VectorXd normSqAdh = adhPosD.colwise().squaredNorm();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dist2MatAdh =
          ((-2 * adhPosD.transpose() * adhPosD).colwise() + normSqAdh).rowwise() +
          normSqAdh.transpose();
  Eigen::ArrayXd adhGAtAdhesions =
      ((-dist2MatAdh.array() / (2.0 * sigma2)).exp().rowwise().sum()) * ampl;

  // Calculate adhG at all frame pixels and find the maximum value for
  // normalization
  double maxAdhG = 0;
  for (int i = state.frameRowStart; i <= state.frameRowEnd; i++) {
    for (int j = state.frameColStart; j <= state.frameColEnd; j++) {
      if (state.adh.coeff(i, j) == 1) {
        // At adhesion sites, set adhG to max and k0Adh to k0
        state.k0Adh(i, j) = config.k0;
        state.adhF(i, j) = 0; // adhF = 0 at adhesions
        continue;
      }

      // Calculate distance squared to all adhesions
      Eigen::ArrayXd dr = state.adhPos.row(0).cast<double>().array() - i;
      Eigen::ArrayXd dc = state.adhPos.row(1).cast<double>().array() - j;
      Eigen::ArrayXd dist2 = dr.square() + dc.square();
      dist2 = dist2.max(1e-12); // Avoid division by zero

      // Calculate local Gaussian intensity
      double localGaussian = ampl * ((-dist2 / (2.0 * sigma2)).exp().sum());
      state.adhF(i, j) = localGaussian;

      // Track max for normalization
      if (localGaussian > maxAdhG) {
        maxAdhG = localGaussian;
      }
    }
  }

  // Normalize and invert to get adhF, calculate k0Adh
  if (maxAdhG > 0) {
    for (int i = state.frameRowStart; i <= state.frameRowEnd; i++) {
      for (int j = state.frameColStart; j <= state.frameColEnd; j++) {
        if (state.adh.coeff(i, j) == 1) {
          // Already set above
          continue;
        }

        double localGaussian = state.adhF(i, j);
        double adhGNormalized = localGaussian / maxAdhG;
        state.adhF(i, j) = 1.0 - adhGNormalized;

        Eigen::ArrayXd dr = state.adhPos.row(0).cast<double>().array() - i;
        Eigen::ArrayXd dc = state.adhPos.row(1).cast<double>().array() - j;
        Eigen::ArrayXd dist2 = dr.square() + dc.square();
        dist2 = dist2.max(1e-12);

        double normNumer = (adhGAtAdhesions / dist2).sum();
        double normDenom = (1.0 / dist2).sum();

        state.k0Adh(i, j) =
            (config.k0 - config.k0Min) * config.k0Scalar *
                (localGaussian / normNumer) * normDenom +
            config.k0Min;
      }
    }
  }
}

} // namespace dynein_cell_model
