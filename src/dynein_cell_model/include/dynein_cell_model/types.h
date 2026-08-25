#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace dynein_cell_model {

using ViewD =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ViewI =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ViewMask = Eigen::SparseMatrix<int>;
// Compatibility names for matrix value types; these do not restore CellModel.
using Mat_d = ViewD;
using Mat_i = ViewI;
using SpMat_i = ViewMask;
using Vec_d = Eigen::VectorXd;
using Arr_d = Eigen::ArrayXd;
using Arr_i = Eigen::ArrayXi;

// Diffusion parameters, passed into kernels by value.
struct DiffusionParams {
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
};

} // namespace dynein_cell_model
