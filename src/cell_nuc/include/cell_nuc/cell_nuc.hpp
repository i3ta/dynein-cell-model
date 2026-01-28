#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

void print_matrix(double **m, int rows_num, int cols_num);

void print_frame(double **m, int fr_rows_num, int fr_cols_num, int fr_rows_pos,
                 int fr_cols_pos);

double max_element(double **m, int rows_num, int cols_num);

double max_element_in_frame(double **m, int fr_rows_num, int fr_cols_num,
                            int fr_rows_pos, int fr_cols_pos);

double **create_array2d(size_t rows_num, size_t cols_num);

double **copy_array2d(double **m, size_t rows_num, size_t cols_num);

void swap_array2d(double **a, double **b, size_t rows_num, size_t cols_num);

void free_array2d(double **m);

void initiate_matrix_with_zeros(double **m, int rows_num, int cols_num);

void initiate_frame_with_zeros(double **m, int fr_rows_num, int fr_cols_num,
                               int fr_rows_pos, int fr_cols_pos);

void initiate_matrix_with_ones(double **m, int rows_num, int cols_num);

double **text_to_matrix(string file, int &rows_num, int &cols_num);

void matrix_to_text(double **m, string file, int rows_num, int cols_num);

double **text_frame_to_matrix_frame(string file, int &fr_rows_num,
                                    int &fr_cols_num, int &fr_rows_pos,
                                    int &fr_cols_pos, int &env_rows_num,
                                    int &env_cols_num);

void matrix_frame_to_text_frame(double **m, string file, int fr_rows_num,
                                int fr_cols_num, int fr_rows_pos,
                                int fr_cols_pos, int env_rows_num,
                                int env_cols_num);

int cell_volume(double **m, int rows_num, int cols_num);

double **cell_outline(double **m, int rows_num, int cols_num);

int outline_4(double **m, int fr_rows_num, int fr_cols_num, int fr_rows_pos,
              int fr_cols_pos, int env_rows_num, int env_cols_num);

void swap(int *a, int *b);

void randomize(int *arr, int n);

string get_config_field(string config_file, string field);

string trim(string str);

void max_min_position_in_frame(double **m, int fr_rows_num, int fr_cols_num,
                               int fr_rows_pos, int fr_cols_pos,
                               int env_rows_num, int env_cols_num,
                               int &fr_r_min, int &fr_r_max, int &fr_c_min,
                               int &fr_c_max);

double **generate_adhesion_distribution(double **Im, double **env,
                                        size_t rows_num, size_t cols_num,
                                        int adh_num, int *adh_r_pos,
                                        int *adh_c_pos);

double **calculate_adhesion_field(double **Im, double **adh, int rows_num,
                                  int cols_num, int adh_num, double adh_sigma,
                                  int *adh_r_pos, int *adh_c_pos);

void adhesions_rearrange(double **Im, double **env, double **adh,
                         size_t rows_num, size_t cols_num, int *adh_r_pos,
                         int *adh_c_pos, double adh_frac, int adh_num,
                         double adh_sigma);

double **place_matrix_on_big_canvas(double **m, int rows_num, int cols_num,
                                    size_t env_rows_num, size_t env_cols_num,
                                    size_t frame_rows_pos,
                                    size_t frame_cols_pos);

double **generate_dyn_field_protr(double **cell, double **nuc,
                                  double **cell_outline, double **nuc_outline,
                                  int cell_rows, int cell_cols, int fr_rows_pos,
                                  int fr_cols_pos, int env_rows_num,
                                  int env_cols_num);

double **generate_dyn_field_retr(double **cell, double **nuc,
                                 double **cell_outline, double **nuc_outline,
                                 int cell_rows, int cell_cols, int fr_rows_pos,
                                 int fr_cols_pos, int env_rows_num,
                                 int env_cols_num);

double **generate_dyn_field_protr_polarized(double **cell, double **nuc,
                                            double **cell_outline,
                                            double **nuc_outline, int V_nuc,
                                            int cell_rows, int cell_cols,
                                            int fr_rows_pos, int fr_cols_pos,
                                            int env_rows_num, int env_cols_num);

double **generate_dyn_field_retr_polarized(double **cell, double **nuc,
                                           double **cell_outline,
                                           double **nuc_outline, int V_nuc,
                                           int cell_rows, int cell_cols,
                                           int fr_rows_pos, int fr_cols_pos,
                                           int env_rows_num, int env_cols_num);

struct Cell {
  Cell();
  Cell(string config_file);
  ~Cell();

  // methods to describe cell behaviour
  void update_volume();
  void update_outline();
  void update_inner_outline();
  void protrude();
  void protrude_adh(); // protrusion weights that depend on adhesion field
  void
  protrude_adh_nuc_push();  // protrusion weights that depend on adhesion field,
                            // AND cell membrane can be pushed by nucleus
  void protrude_no_actin(); // protrusion, when actin weights are not taken into
                            // account
  void protrude_no_actin_prop_factor(); // differs from the above method by
                                        // inclusion of prop_factor to study
                                        // effect of scaling factor
  void protrude_k_heter(); // protrusion for heterogeneous values of k
  void protrude_A_prop();  // protrusion rate linearly depends on concentration
                           // of A (normalized by A_max), no adhesion influence
  void retract();
  void retract_no_actin(); // retraction, when actin weights are not taken into
                           // account
  void retract_no_actin_prop_factor(); // differs from the above method by
                                       // inclusion of prop_factor to study
                                       // effect of scaling factor
  void retract_k_heter(); // retraction for heterogeneous values of k
  void retract_A_prop(); // retraction rate linearly depends on concentration of
                         // A (normalized by A_max)
  void diffuse();
  void diffuse_noise();  // reaction-diffusion equations include noise
  void diffuse_k0_adh(); // same as diffuse except that the activation rate of
                         // k0 of the GTPase is influenced by adhesion location
  void
  correct_concentrations(); // correction is based on A_cor_sum and I_cor_sum
  void generate_adhesion_distribution();
  void
  generate_adhesion_distribution_polarized(); // adhesion distribution follow A
                                              // concentration, the higher A,
                                              // the higher is probability to
                                              // form adhesion
  void update_adhesion_field();
  void update_adhesion_field_normalized();
  void rearrange_adhesions();
  void
  rearrange_adhesions_polarized(); // adhesion distribution follow A
                                   // concentration, the higher A, the higher is
                                   // probability to form adhesion
  void
  update_adh_positions(); // read positions of adhesions from adh matrix and
                          // write this data to arrays adh_r_pos and adh_c_pos
  void excite_random_points(); // function sets specific value of A (A_excite)
                               // in A matrix in specified (excite_num) number
                               // of random points. Additional corrections are
                               // made to satisfy mass concervation law
  void adjust_frame();
  void update_k0_adh();
  void update_k0_adh_new(
      double scalar); // same as update_k0_adh except that a scalar value
                      // controls the degree to which adhesions affect local k0
                      // values (higher scalar, higher k0 near adhesions)
  void add_CoM_to_track(int value);
  void save_parameters_configuration(string file_name);

  // methods to describe nucleus behavior
  void update_volume_nuc();
  void update_outline_nuc();
  void update_inner_outline_nuc();
  void protrude_nuc(); // protrusion for cell nucleus where dynein forces &
                       // roundness taken into account
  void retract_nuc();  // retraction for cell nucleus where dynein forces &
                       // roundness taken into account
  void protrude_nuc_no_geo(); // same as protrude_nuc but no roundness (global
                              // geometry) factor
  void retract_nuc_no_geo();  // same as retract_nuc but no roundness factor
  void protrude_nuc_polarized(); // protrusion is polarized based on polarized
                                 // dynein field (assuming some cell protrusions
                                 // have strong dynein pulling forces)
  void retract_nuc_polarized();  // retraction is polarized based on polariez
                                 // dynein field (assuming some cell protrusions
                                 // have strong dynein pulling forces)

  int th_num;          // number of threads to use for parallel computing
  unsigned int *seeds; // array of seeds to use rand_r() in parallel loops

  int diff_t; // time of diffusion - number of difusion steps
  // may be I should remove this parameter and add it as external parameter for
  // diffusion() function

  int fr_rows_num;  // number of rows in the frame, containing cell
  int fr_cols_num;  // number of columns in the frame, containing cell
  int env_rows_num; // number of rows in the whole environment ot the cell
  int env_cols_num; // number of columns in the whole envorinment of the cell

  // position of the frame (row and columns coordinates of the upper left
  // corner)
  int fr_rows_pos; // row coordinate
  int fr_cols_pos; // column coordinate

  int fr_dist; // distance from the cell border to the edge of the frame, set
               // when frame is rearranged

  int V0; // initial volume of the cell, set up when initial Im is read
  int V;  // volume of the cell on the current step
  double T;
  double k;
  // for polarized fuctuation in the cell I introduce the following parameters:
  double k_front; // at the front fluctuations are more intense and k must be
                  // smaller (notless then 1)
  double k_back;  // at the back fluctuations are less pronounced and k must be
                  // bigger
  double g;
  double act_slope;

  // params for nucleus
  double d_basal; // basal weight for protrusion probability of dynein factor
  int V0_nuc;     // initial volume of the cell, set up when initial Im is read
  int V_nuc;      // volume of the cell on the current step
  double T_nuc;   // controls sharpness of volume constraint
  double k_nuc;   // controls degree of geometry constraint
  double R_nuc;   // controls sharpness of roundness constraint
  int R0;

  // parameters for RD equations
  double DA;
  double DI;
  double k0;
  double **k0_adh; // nonhomogeneous distribution of k0 parameter, defined by
                   // adhesions, describes interaction of adhesions and RD
                   // system through GEF activation
  double
      k0_min; /*minimal value of k0 parameter, used for calculation of k0_adh,
this value will be set in points, distant from all adhesions (otherwise we will
have k0 ~= 0 -> A ~= 0 and after correction of concentrations after
protrusion/retraction step we can get negative concentrations)*/
  double scalar;      // for multiplying adh factor in k0_adh
  double noise_value; // noise value, in case of sumulation diffusion with noise
  double s1;
  double s2;
  double A0;
  double F0;
  double gamma;
  double delta;
  double kn;
  double ks;
  double eps;
  // steps for numerical solution
  double dt;
  double dx;

  // range of A concentration
  double A_max; // maximal value of A
  double A_min; // minimal value of A
  // DTmod start
  // range of AC concentration
  double AC_max; // maximal value of A
  double AC_min; // minimal value of A
  // DTmod end

  // adhesion parameters
  int adh_num;     // number of adhesions in the cell
  double adh_frac; // fraction of adhesions that are rearranged on every step of
                   // fluctuation
  double adh_sigma; // parameter of gaussian smoothing of adhesions, responsible
                    // for radius of influence of adhesion,
  double adh_basal_prot; // basal weight for protrusion in points distant from
                         // adhesions

  int excite_num;  // number of random points to excite (set specified
                   // concentration of A in this points)
  double A_excite; // concentration of A to set in excited points
  // DTmod start
  double AC_excite; // concentration of A to set in excited points
  // DTmod end

  // this are additional parameters to correct concentrations after retraction
  // and protrusion to keep sum(A) + sum(I) = const, we need to correct A and I
  // values after protrusion and retraction
  double A_cor_sum;
  double I_cor_sum;
  // DTmod start
  double AC_cor_sum;
  double IC_cor_sum;
  // DTmod end
  // there is no F_cor as F is not conserved in our system, notice, that in
  // MATLAB there was a correction for F

  double
      prop_factor; // just a number in range [0 1] to multiply protrusions and
                   // retraction weights, purpose is to study effect of scaling

  // supplementary variables and matrices
  double **Im;     // 2d array with cell mask
  double **Im_nuc; // 2d array with nucleus
  double **outline;
  double **inner_outline;
  double **outline_nuc;
  double **inner_outline_nuc;
  double **A; // 2d array with values of A
  double **I; // 2d array with values of I
  double **F; // 2d array with values of F
  double **A_new;
  double **I_new;
  double **F_new;
  // DTmod start
  double **AC; // 2d array with values of AC
  double **IC; // 2d array with values of IC
  double **FC; // 2d array with values of FC
  double **AC_new;
  double **IC_new;
  double **FC_new;
  // DTmod end
  double **env; // matrix, that defines environment of the cell, 1 - something,
                // that the cell can sence, 0 - if not
  double **adh; // adhesions, 1 - adhesion,  0 - no adhesion
  int *adh_r_pos; // array, containing row coordinates of adhesions
  int *adh_c_pos; // array, containing column coordinates of adhesions
  double **adh_g; // gaussian smoothing of discrete adhesion points
  double *
      *adh_f; // field of adhesion influence, gaussian smoothing of discrete
              // points and processed to represent correction weights correctly
  // adhesions can form only in points, where env == 1
  // normalized adhesion field. normalization is done in such way to obtain
  // peaks of the same height after Gaussian smoothing
  double **CoM_track; /*matrix to track center of mass of the cell, contains
 integer numbers in points of CoM, user must specify the number (which should be
 the number of iteration to enumerate points of the track)*/
};
