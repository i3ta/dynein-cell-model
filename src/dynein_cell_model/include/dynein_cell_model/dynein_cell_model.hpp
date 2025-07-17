#ifndef DYNEIN_CELL_MODEL_HPP
#define DYNEIN_CELL_MODEL_HPP

#include <string>
#include <vector>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace dynein_cell_model {

typedef Eigen::MatrixXd Mat_d;
typedef Eigen::MatrixXi Mat_i;
typedef Eigen::SparseMatrix<int> SpMat_i;

class CellModel {
public:
  /**
   * @brief Initialize the cell model with default parameters.
   */
  CellModel();

  /**
   * @brief Initialize the cell model with a config file.
   *
   * @param filename Name of the config file to read parameters from.
   */
  CellModel(std::string filename);

  /**
   * @brief Perform one time step of cell simulations.
   */
  void step();

  /**
    * @brief Save the current state of the cell to a file.
    *
    * @param dirname Name of the folder to save the state of the cell to.
    */
  void save_state(std::string dirname);

  /**
    * @brief Rearrange the adhesion points around the cell to simulate evolution
    * of cell adhesions. Randomly picks adh_frac of the adhesions and finds other
    * valid positions to move them cell adhesions.
    */
  void rearrange_adhesions();

  /**
    * @brief Update the values of k_adh for all pixels on the cell.
    */
  void update_k0_adh();

  /**
   * @brief Update the cell processing boundaries to be where the cell is.
   */
  void update_frame();

private:
  /**
    * @brief Protrude the nucleus pixels.
    */
  void protrude_nuc(); 

  /**
    * @brief Retract the nucleus pixels.
    */
  void retract_nuc();

  /**
    * @brief Protrude the cell boundary, taking into account adhesions and the
    * nucleus.
    */
  void protrude();

  /**
    * @brief Retract the cell boundary.
    */
  void retract();

  /**
    * @brief Update the concentrations of cell signals for each pixel.
    */
  void update_concentrations();

  /**
    * @brief Diffuse k0_adh over the cell.
    */
  void diffuse_k0_adh();

  /**
   * @brief Update and smooth the adhesion field
   */
  void update_adhesion_field();

  /**
    * @brief Helper function to generate random distinct indices.
    *
    * @param n Number of indices to generate
    * @param lb Lower bound of indices (inclusive)
    * @param ub Upper bound of indices (exclusive)
    *
    * @return Vector of random indices
    */
  const std::vector<int> generate_indices(const int n, const int lb, const int ub);

  // Protrusion and retraction parameters
  double k_; ///< Relative contribution of geometry factor to cell protrusion/retraction probability
  double k_nuc_; ///< controls degree of geometry constraint
  double g_; ///< Sensitivity of geometry factor to local membrane curvature
  double T_; ///< Parameter controlling steepness of volume factor function (sensitivity to changes in cell volume)
  double T_nuc_; ///< controls sharpness of volume constraint
  double act_slope_; ///< Slope of actin factor function 
  double adh_sigma_; ///< Sigma value for gaussian smoothing of adhesion field
  double adh_basal_; ///< Basal value for adhesion factor protrusion probability
  double adh_frac_; ///< Fraction of adhesions that are rearranged at each adh_T  time step
  int adh_num_; ///< number of adhesions in the cell
  int R0_; ///< Roundness (perimeter^2/area) of a 4-connected circle
  double R_nuc_; ///< controls sharpness of roundness constraint
  double dyn_basal_; ///< basal weight for protrusion probability of dynein factor
  double prop_factor; ///< number in range [0 1] to multiply protrusions and retraction weights to study effect of scaling

  // Reaction-diffusion parameters
  double DA_; ///< Diffusion coefficient of active GTPase
  double DI_; ///< Diffusion coefficient of inactive GTPase
  double k0_; ///< Activation rate of GTPase
  double k0_min_; ///< Minimum basal activation rate of GTPase
  double scalar_; ///< Effect of adhesion field on GTPase activation
  double gamma_; ///< Rate constant of autocatalytic activation of GTPase
  double A0_; ///< Sensitivity of positive feedback of GTPase to the concentration of active GTPase
  double s1_; ///< Basal deactivation rate of GTPase
  double s2_; ///< Rate constant of negative feedback from F-actin on GTPase
  double F0_; ///< Sensitivity of negative feedback of GTPase to the concentration of F-actin
  double kn_; ///< Rate constant of F-actin polymerization
  double ks_; ///< Rate constant of F-actin depolymerization
  double dt_; ///< Temporal step of finite difference scheme
  double dx_; ///< Spatial step of finite difference scheme

  // Concentration limit parameters
  double A_max_; // maximal value of A
  double A_min_; // minimal value of A
  double AC_max_; // maximal value of AC
  double AC_min_; // minimal value of AC

  // Simulation parameters
  int diff_t_; ///< time of diffusion
  int frame_padding_; //< distance from the cell border to the edge of the frame

  // Frame variables
  int frame_row_start_; ///< first row in the frame (inclusive)
  int frame_row_end_; ///< last row in the frame (inclusive)
  int frame_col_start_; ///< first column in the frame (inclusive)
  int frame_col_end_; ///< last column in the frame (inclusive)

  // Cell state variables
  int V0_; ///< initial volume of the cell, set up when initial Im is read
  int V_; ///< volume of the cell on the current step 
  int V0_nuc_; ///< initial volume of the cell, set up when initial Im is read
  int V_nuc_; ///< volume of the cell on the current step 

  double A_cor_sum_; ///< Correct A values after retraction and protrusion
  double I_cor_sum_; ///< Correct I values after retraction and protrusion
  double AC_cor_sum_; ///< Correct AC values after retraction and protrusion
  double IC_cor_sum_; ///< Correct IC values after retraction and protrusion

  SpMat_i cell_; ///< cell mask
  SpMat_i nuc_; ///< nucleus mask
  SpMat_i outline_; ///< cell outline
  SpMat_i inner_outline_; ///< cell outline inner pixel
  SpMat_i outline_nuc_; ///< nucleus outline
  SpMat_i inner_outline_nuc_; ///< nucleus outline inner pixel
  SpMat_i env_; ///< environment the cell is in defining pixels the cell can sense
  SpMat_i adh_; ///< cell adhesions
  Mat_i adh_pos_; ///< cell adhesion coordinates

  Mat_d k0_adh_; ///< distribution of k0
  Mat_d A_; ///< values of A
  Mat_d I_; ///< values of I
  Mat_d F_; ///< values of F
  Mat_d AC_; ///< values of AC
  Mat_d IC_; ///< values of IC
  Mat_d FC_; ///< values of FC
  Mat_d adh_g_; ///< smoothed adhesion points
  Mat_d adh_f_; ///< field of adhesion influence
  
  // random number generation helpers
  std::mt19937 rng;
  std::uniform_real_distribution<> prob_dist = std::uniform_real_distribution<>(0.0, 1.0);
  
}; // CellModel class

}; // namespace dynein_cell_model

#endif // DYNEIN_CELL_MODEL_HPP
