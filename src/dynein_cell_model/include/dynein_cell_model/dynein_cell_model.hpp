#ifndef DYNEIN_CELL_MODEL_HPP
#define DYNEIN_CELL_MODEL_HPP

#include <string>
#include <unordered_set>
#include <vector>
#include <random>

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <highfive/H5File.hpp>

namespace dynein_cell_model {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat_d;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat_i;
typedef Eigen::SparseMatrix<int> SpMat_i;
typedef Eigen::VectorXd Vec_d;
typedef Eigen::ArrayXd Arr_d;


/**
 * @brief Create a matrix mask from a png image
 *
 * @param filepath path of image to get matrix mask from
 * @param color color to mask as 1, all other colors are 0
 *
 * @return boolean matrix marking where the image has the specified color
 */
Mat_i matrix_from_mask(std::string filepath, cv::Vec3b color);


class CellModelConfig {
public:
  /**
   * @brief Set up config with default values
   */
  CellModelConfig();

  /**
   * @brief Set up config with values from config file
   *
   * @param config_file config file to read
   */
  CellModelConfig(std::string config_file);

  /**
   * @brief Save config to file
   *
   * @param dest_file destination file
   */
  void save_file(std::string dest_file);

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
  double prop_factor_; ///< number in range [0, 1] to multiply protrusions and retraction weights to study effect of scaling
  double dyn_norm_k_; ///< "steepness" value for smoothing dynein force values using sigmoid
  double dyn_sigma_; ///< sigma value for gaussian smoothing of dynein force
  int dyn_kernel_size_; ///< size of the gaussian smoothing kernel (should be odd)

  // Reaction-diffusion parameters
  double DA_; ///< Diffusion coefficient of active GTPase
  double DI_; ///< Diffusion coefficient of inactive GTPase
  double k0_; ///< Activation rate of GTPase
  double k0_min_; ///< Minimum basal activation rate of GTPase
  double k0_scalar_; ///< Effect of adhesion field on GTPase activation
  double gamma_; ///< Rate constant of autocatalytic activation of GTPase
  double delta_; ///<
  double A0_; ///< Sensitivity of positive feedback of GTPase to the concentration of active GTPase
  double s1_; ///< Basal deactivation rate of GTPase
  double s2_; ///< Rate constant of negative feedback from F-actin on GTPase
  double F0_; ///< Sensitivity of negative feedback of GTPase to the concentration of F-actin
  double kn_; ///< Rate constant of F-actin polymerization
  double ks_; ///< Rate constant of F-actin depolymerization
  double eps_; ///<
  double dt_; ///< Temporal step of finite difference scheme
  double dx_; ///< Spatial step of finite difference scheme

  // Concentration limit parameters
  double A_max_; ///< maximal value of A
  double A_min_; ///< minimal value of A
  double AC_max_; ///< maximal value of AC
  double AC_min_; ///< minimal value of AC
  
  // Simulation size
  int sim_rows_; ///< Total number of rows for the simulation
  int sim_cols_; ///< Total number of columns for the simualation

  // Simulation parameters
  int t_; ///< current time step
  int adh_t_; ///< number of time steps per adhesion rearrangement
  int fr_t_; ///< number of time steps per frame update
  int save_t_; ///< number of time steps per save
  int diff_t_; ///< time of diffusion
  int frame_padding_; ///< distance from the cell border to the edge of the frame
  std::string save_dir_; ///< directory to save snapshots to;
  int seed_;
  int num_iters_;
};


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
  CellModel(CellModelConfig config);

  /**
   * @brief Update all config parameters of the model
   *
   * @param config CellModelConfig to use
   */
  void update_config(CellModelConfig config);

  /**
   * @brief Get all system parameters as CellModelConfig
   *
   * @return config object
   */
  const CellModelConfig get_config();

  /**
   * @brief Simulate the cell for dt amount of time.
   *
   * @param dt Length of the simulation to run
   */
  void simulate(double dt);

  /**
   * @brief Simulate the cell for n time steps.
   *
   * @param n Number of time steps to simulate
   */
  void simulate_steps(int n);

  /**
   * @brief Perform one time step of cell simulations.
   */
  void step();

  /**
   * @brief Set the file to save outputs to.
   *
   * @param filepath Path of the file to save the state of the cell to.
   */
  void set_output(const std::string filepath);

  /**
   * @brief Save the current state of the cell to a file.
   */
  void save_state();

  /**
   * @brief Rearrange the adhesion points around the cell to simulate evolution
   * of cell adhesions. Randomly picks adh_frac of the adhesions and finds other
   * valid positions to move them cell adhesions.
   */
  void rearrange_adhesions();

  /**
   * @brief Initialize the adhesion points around the cell to simulate evolution
   * of cell adhesions. Same logic as rearrange_adhesions but does all adhesions
   */
  void init_adhesions();

  /**
   * @brief Update the cell processing boundaries to be where the cell is.
   */
  void update_frame();

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

  void set_cell(const Mat_i cell);

  void set_nuc(const Mat_i nuc);

  void set_adh(const SpMat_i adh);

  void set_A(const Mat_d A);

  void set_AC(const Mat_d AC);

  void set_I(const Mat_d I);

  void set_IC(const Mat_d IC);

  void set_F(const Mat_d F);

  void set_FC(const Mat_d FC);

  void set_env(const SpMat_i env);

  void set_seed(const int seed);

private:
  /**
   * @brief Initialize the helper variables.
   */
  void initialize_helpers();

  /**
   * @brief Update the nucleus outlines and values.
   */
  void update_nuc();

  /**
   * @brief Update the cell outlines and values.
   *
   * @param full whether to use the full simulation space (for initialization)
   */
  void update_cell(const bool full);

  /**
   * @brief Update the cell outlines and values.
   *
   * @param full whether to use the full simulation space (for initialization)
   */
  void update_cell();

  /**
   * @brief Correct the concentrations of cell signals for each pixel so the total A and I remains constant.
   */
  void correct_concentrations();

  /**
   * @brief Diffuse k0_adh over the cell.
   */
  void diffuse_k0_adh();

  /**
   * @brief Update the forces that dynein is putting on different parts of the cell.
   */
  void update_dyn_nuc_field();

  /**
   * @brief Update and smooth the adhesion field
   */
  void update_adhesion_field();

  /**
   * @brief Get the dyn_f value at (r, c) smoothed with a Gaussian smoothing kernel.
   *
   * @param r row
   * @param c column
   *
   * @return smoothed dyn_f
   */
  const double get_smoothed_dyn_f(const int r, const int c);

  /**
   * @brief Update and save the values of the Gaussian smoothing kernel.
   */
  void update_smoothing_kernel();

  /**
   * @brief Get the integer encoding of the 8-neighbors of a specific pixel in a matrix.
   *
   * @param mat matrix
   * @param r row
   * @param c column
   *
   * @return 8-neighbors of the specific pixel encoded into an integer.
   */
  const uint8_t encode_8(Mat_i &mat, const int r, const int c);

  /**
   * @brief Determine if a specific integer configuration is valid.
   *
   * @param conf integer configuration from encode_8
   *
   * @return whether the configuration is valid
   */
  const bool is_valid_config_prot(uint8_t conf);

  /**
   * @brief Determine if a specific integer configuration is valid.
   *
   * @param conf integer configuration from encode_8
   *
   * @return whether the configuration is valid
   */
  const bool is_valid_config_retr(uint8_t conf);

  /**
   * @brief Update and save the valid pixel configurations for protrusion and retraction.
   */
  void update_valid_conf();

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

  /**
   * @brief Generate a random visit order for all of the nonzero pixels in the SpMat_i.
   *
   * @param mat SpMat_i to randomize pixels of 
   *
   * @return Vector of randomized order
   */
  const std::vector<std::pair<int, int>> randomize_nonzero(const SpMat_i mat);

  /**
   * @brief Append a value (int or SpMat_d) to a dataset.
   *
   * @param file HighFive file to save dataset to
   * @param dataset Name of the dataset to save the data to
   * @param mat Data to append to dataset
   */
  void append_dataset(HighFive::File &file, const std::string &dataset, int v);
  void append_dataset(HighFive::File &file, const std::string &dataset, SpMat_i &mat);

  /**
   * @brief Append a value (any row-major Eigen Matrix) to a dataset.
   *
   * @param file HighFive file to save dataset to
   * @param dataset Name of the dataset to save the data to
   * @param mat Data to append to dataset
   */
  template <typename T>
  void append_dataset(HighFive::File &file, const std::string& dataset, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat) {
    // Create dataset if it does not exist
    if (!file.exist(dataset)) {
      std::vector<size_t> dims = {0, (size_t)mat.rows(), (size_t)mat.cols()};
      std::vector<size_t> max_dims = {HighFive::DataSpace::UNLIMITED, (size_t)mat.rows(), (size_t)mat.cols()};

      HighFive::DataSpace dataspace(dims, max_dims);
      HighFive::DataSetCreateProps props;
      props.add(HighFive::Chunking({1, (size_t)mat.rows(), (size_t)mat.cols()}));

      file.createDataSet<T>(dataset, dataspace, props);
    }

    // Open dataset
    HighFive::DataSet dset = file.getDataSet(dataset);

    // Get current dimensions
    std::vector<size_t> dims = dset.getSpace().getDimensions();
    if (dims.size() != 3 || dims[1] != (size_t)mat.rows() || dims[2] != (size_t)mat.cols()) {
      throw std::runtime_error("Dataset does not have expected dimensions.");
    }

    size_t next_t = dims[0];

    // Extend dataset by 1
    dset.resize({next_t + 1, dims[1], dims[2]});
    dset.select({next_t, 0, 0}, {1, dims[1], dims[2]}).write_raw(mat.data());
  }

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
  double prop_factor_; ///< number in range [0, 1] to multiply protrusions and retraction weights to study effect of scaling
  double dyn_norm_k_; ///< "steepness" value for smoothing dynein force values using sigmoid
  double dyn_sigma_; ///< sigma value for gaussian smoothing of dynein force
  int dyn_kernel_size_; ///< size of the gaussian smoothing kernel (should be odd)

  // Reaction-diffusion parameters
  double DA_; ///< Diffusion coefficient of active GTPase
  double DI_; ///< Diffusion coefficient of inactive GTPase
  double k0_; ///< Activation rate of GTPase
  double k0_min_; ///< Minimum basal activation rate of GTPase
  double k0_scalar_; ///< Effect of adhesion field on GTPase activation
  double gamma_; ///< Rate constant of autocatalytic activation of GTPase
  double delta_; ///<
  double A0_; ///< Sensitivity of positive feedback of GTPase to the concentration of active GTPase
  double s1_; ///< Basal deactivation rate of GTPase
  double s2_; ///< Rate constant of negative feedback from F-actin on GTPase
  double F0_; ///< Sensitivity of negative feedback of GTPase to the concentration of F-actin
  double kn_; ///< Rate constant of F-actin polymerization
  double ks_; ///< Rate constant of F-actin depolymerization
  double eps_; ///<
  double dt_; ///< Temporal step of finite difference scheme
  double dx_; ///< Spatial step of finite difference scheme

  // Concentration limit parameters
  double A_max_; ///< maximal value of A
  double A_min_; ///< minimal value of A
  double AC_max_; ///< maximal value of AC
  double AC_min_; ///< minimal value of AC
  
  // Simulation size
  int sim_rows_; ///< Total number of rows for the simulation
  int sim_cols_; ///< Total number of columns for the simualation

  // Simulation parameters
  int t_; ///< current time step
  int adh_t_; ///< number of time steps per adhesion rearrangement
  int fr_t_; ///< number of time steps per frame update
  int save_t_; ///< number of time steps per save
  int diff_t_; ///< time of diffusion
  int frame_padding_; ///< distance from the cell border to the edge of the frame
  int seed_ = 0;

  // Frame variables
  int frame_row_start_; ///< first row in the frame (inclusive)
  int frame_row_end_; ///< last row in the frame (inclusive)
  int frame_col_start_; ///< first column in the frame (inclusive)
  int frame_col_end_; ///< last column in the frame (inclusive)

  // Cell state variables
  int V0_; ///< initial volume of the cell, set up when initial Im is read
  int V_; ///< volume of the cell on the current step 
  int P_; ///<< perimeter of the cell
  int V0_nuc_; ///< initial volume of the cell, set up when initial Im is read
  int V_nuc_; ///< volume of the cell on the current step 
  int P_nuc_; ///< perimeter of the nucleus

  double A_cor_sum_; ///< Correct A values after retraction and protrusion
  double I_cor_sum_; ///< Correct I values after retraction and protrusion
  double AC_cor_sum_; ///< Correct AC values after retraction and protrusion
  double IC_cor_sum_; ///< Correct IC values after retraction and protrusion

  Mat_i cell_; ///< cell mask
  Mat_i nuc_; ///< nucleus mask
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
  Mat_d adh_f_; ///< field of adhesion influence
  Mat_d dyn_f_; ///< dynein field force

  Mat_d g_dyn_f_; ///< saved kernel for gaussian smoothing of dyn_f
  std::unordered_set<int> protrude_conf_; ///< numerical encoding of allowed protrusion configurations
  std::unordered_set<int> retract_conf_; ///< numerical encoding of allowed retraction configurations
  
  std::string output_file_; ///< file to save cell states to
  
  // random number generation helpers
  std::mt19937 rng;
  std::uniform_real_distribution<> prob_dist;
  
}; // CellModel class

}; // namespace dynein_cell_model

#endif // DYNEIN_CELL_MODEL_HPP
