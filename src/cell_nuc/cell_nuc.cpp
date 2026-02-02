#include <algorithm>
#include <cell_nuc/cell_nuc.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <test_utils/test_utils.hpp>
#include <vector>

#ifdef LIB_CELL_NUC_DEBUG
static test_utils::DebugRand<double> drand;

#undef rand
#define rand() (static_cast<int>(drand() * RAND_MAX))
inline constexpr bool LIB_CELL_NUC_DEBUG_CPP = true;
#else
#include <cstdlib>
#include <random>
inline constexpr bool LIB_CELL_NUC_DEBUG_CPP = false;
#endif

using namespace std;

#define TRACE_MSG(msg)                                                         \
  if constexpr (LIB_CELL_NUC_DEBUG_CPP)                                        \
    std::cerr << "[ TRACE    ] [ LegacyModel ] " << msg << std::endl           \
              << std::flush;

void print_matrix(double **m, int rows_num, int cols_num) {
  for (int i = 0; i < rows_num; i++) {
    for (int j = 0; j < cols_num; j++) {
      cout << m[i][j] << ' ';
    }
    cout << endl;
  }
}

void print_frame(double **m, int fr_rows_num, int fr_cols_num, int fr_rows_pos,
                 int fr_cols_pos) {
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      cout << m[i][j] << ' ';
    }
    cout << endl;
  }
}

double max_element(double **m, int rows_num, int cols_num) {
  double max = m[0][0];
  for (int i = 0; i != rows_num; i++)
    for (int j = 0; j != cols_num; j++)
      if (m[i][j] > max)
        max = m[i][j];
  return max;
}

double max_element_in_frame(double **m, int fr_rows_num, int fr_cols_num,
                            int fr_rows_pos, int fr_cols_pos) {
  double max = m[fr_rows_pos][fr_cols_pos];
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (m[i][j] > max)
        max = m[i][j];
  return max;
}

double **create_array2d(size_t rows_num, size_t cols_num) {
  double **m = new double *[rows_num];
  m[0] = new double[rows_num * cols_num];
  for (size_t i = 1; i != rows_num; i++)
    m[i] = m[i - 1] + cols_num;
  return m;
}

double **copy_array2d(double **m, size_t rows_num, size_t cols_num) {
  double **mm = create_array2d(rows_num, cols_num);
  for (size_t i = 0; i != rows_num; i++)
    for (size_t j = 0; j != cols_num; j++)
      mm[i][j] = m[i][j];
  return mm;
}

void swap_array2d(double **a, double **b, size_t rows_num, size_t cols_num) {
  double *temp = 0;
  for (size_t i = 0; i != rows_num; i++) {
    temp = a[i];
    a[i] = b[i];
    b[i] = temp;
  }
}

void free_array2d(double **m) {
  delete[] m[0];
  delete[] m;
}

void initiate_matrix_with_zeros(double **m, int rows_num, int cols_num) {
  for (int i = 0; i < rows_num; i++)
    for (int j = 0; j < cols_num; j++)
      m[i][j] = 0;
}

void initiate_frame_with_zeros(double **m, int fr_rows_num, int fr_cols_num,
                               int fr_rows_pos, int fr_cols_pos) {
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      m[i][j] = 0;
}

void initiate_matrix_with_ones(double **m, int rows_num, int cols_num) {
  for (int i = 0; i < rows_num; i++)
    for (int j = 0; j < cols_num; j++)
      m[i][j] = 1;
}

double **text_to_matrix(string file, int &rows_num, int &cols_num) {
  // read 2d matrix with double values from file
  // file specification:
  // contains numbers separated by spaces
  // first number: number of rows in matrix
  // second number:number of columns in matrix
  // all other numbers: consequence of numbers in matrix, row by row

  double d = 0;
  int c = 0, r = 0;
  double **m;

  ifstream m_file(file.c_str());
  if (m_file.is_open()) {
    m_file >> rows_num;
    m_file >> cols_num;
    m = create_array2d(rows_num, cols_num);
    while (m_file >> d) {
      m[r][c] = d;
      c++;
      if (c == cols_num) {
        c = 0;
        r++;
      }
    }
    m_file.close();
  } else {
    cout << "text_to_matrix: Can't open file!" << endl;
  }
  return m;
}

void matrix_to_text(double **m, string file, int rows_num, int cols_num) {
  // write 2d matrix with double values to file
  // file specification:
  // contains numbers separated by spaces
  // first number: number of rows in matrix
  // second number:number of columns in matrix
  // all other numbers: consequence of numbers in matrix, row by row

  ofstream m_file(file.c_str());
  m_file << rows_num << ' ';
  m_file << cols_num << ' ';
  for (int i = 0; i != rows_num; i++)
    for (int j = 0; j != cols_num; j++) {
      m_file << m[i][j] << ' ';
    }
  m_file.close();
}

double **text_frame_to_matrix_frame(string file, int &fr_rows_num,
                                    int &fr_cols_num, int &fr_rows_pos,
                                    int &fr_cols_pos, int &env_rows_num,
                                    int &env_cols_num) {
  // read 2d matrix frame with double values from file
  // create matrix of the size of environment and write frame in the propper
  // position in this matrix file specification: contains numbers separated by
  // spaces first number: number of frame rows second number: number of frame
  // columns (position of the frame is described by coordinates of the upper
  // left corner) third number: frame rows position fourth number: frame columns
  // porition fifth number: number of rows in environment sixth number: number
  // of columns in environment all other numbers: consequence of numbers in the
  // frame, row by row

  double d = 0;
  int c = 0, r = 0;
  double **m;

  ifstream m_file(file.c_str());
  if (m_file.is_open()) {
    m_file >> fr_rows_num;
    m_file >> fr_cols_num;
    m_file >> fr_rows_pos;
    m_file >> fr_cols_pos;
    m_file >> env_rows_num;
    m_file >> env_cols_num;
    m = create_array2d(env_rows_num, env_cols_num);
    initiate_matrix_with_zeros(m, env_rows_num, env_cols_num);
    while (m_file >> d) {
      m[fr_rows_pos + r][fr_cols_pos + c] = d;
      c++;
      if (c == fr_cols_num) {
        c = 0;
        r++;
      }
    }
    m_file.close();
  } else {
    cout << "text_to_matrix: Can't open file!" << endl;
  }
  return m;
}

void matrix_frame_to_text_frame(double **m, string file, int fr_rows_num,
                                int fr_cols_num, int fr_rows_pos,
                                int fr_cols_pos, int env_rows_num,
                                int env_cols_num) {
  // write 2d frame of the matrix with double values to file
  // file specification:
  // contains numbers separated by spaces
  // first number: number of frame rows
  // second number: number of frame columns
  //(position of the frame is described by coordinates of the upper left corner)
  // third number: frame rows position
  // fourth number: frame columns porition
  // fifth number: number of rows in environment
  // sixth number: number of columns in environment
  // all other numbers: consequence of numbers in the frame, row by row

  ofstream m_file(file.c_str());
  m_file << fr_rows_num << ' ';
  m_file << fr_cols_num << ' ';
  m_file << fr_rows_pos << ' ';
  m_file << fr_cols_pos << ' ';
  m_file << env_rows_num << ' ';
  m_file << env_cols_num << ' ';
  m_file << "\n";

  double mx = 0;
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      m_file << m[i][j] << ' ';
      mx = max(mx, m[i][j]);
    }
    m_file << "\n";
  }
  m_file.close();

  cout << file << " " << mx << "\n";
}

int cell_volume(double **m, int rows_num, int cols_num) {
  int V = 0;
  for (int i = 0; i != rows_num; i++)
    for (int j = 0; j != cols_num; j++)
      if (m[i][j] > 0)
        V++;
  return V;
}

double **cell_outline(double **m, int rows_num, int cols_num) {
  double **outline = create_array2d(rows_num, cols_num);
  initiate_matrix_with_zeros(outline, rows_num, cols_num);
  for (int i = 1; i != (rows_num - 1); i++)
    for (int j = 1; j != (cols_num - 1); j++)
      outline[i][j] = m[i - 1][j] + m[i + 1][j] + m[i][j - 1] + m[i][j + 1];
  for (int i = 0; i != rows_num; i++)
    for (int j = 0; j != cols_num; j++)
      outline[i][j] -= 4 * m[i][j];
  for (int i = 0; i != rows_num; i++)
    for (int j = 0; j != cols_num; j++)
      if (outline[i][j] > 0)
        outline[i][j] = 1;
      else
        outline[i][j] = 0;
  return outline;
}

int outline_4(double **m, int fr_rows_num, int fr_cols_num, int fr_rows_pos,
              int fr_cols_pos, int env_rows_num, int env_cols_num) {
  int perim = 0;
  double **outline = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(outline, env_rows_num, env_cols_num);
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      outline[i][j] = m[i - 1][j] + m[i + 1][j] + m[i][j - 1] + m[i][j + 1];
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      outline[i][j] -= 4 * m[i][j];
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      if (outline[i][j] > 0) {
        outline[i][j] = 1;
        perim++;
      } else
        outline[i][j] = 0;
  free_array2d(outline);
  return perim;
}

void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

void randomize(int arr[], int n) {
  srand(0);
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    swap(&arr[i], &arr[j]);
  }
}

string get_config_field(string config_file, string field) {
  // the function looks for a propper field in config file and retirns the value
  // of this field as string
  ifstream conf_file(config_file.c_str());
  if (conf_file.is_open()) {
    string pattern = field + ": ";
    string line;
    string value = "None";
    while (getline(conf_file, line)) {
      // if (line.find(pattern) != string::npos) {
      if (line.find(pattern) == 0) {

        value = line.substr(pattern.length(), line.length());
        value = trim(value);
        break;
      }
    }
    conf_file.close();
    return value;
  } else {
    cout << "get_config_field: Can't open file!" << endl;
    return "error";
  }
  return "0";
}

string trim(string str) {
  while (str[0] == ' ') {
    str = str.substr(1, str.length() - 1);
  }
  while (str[str.length() - 1] == ' ') {
    str = str.substr(0, str.length() - 1);
  }
  while ((int)str[str.length() - 1] == 13) {
    str = str.substr(0, str.length() - 1);
  }
  return str;
}

void max_min_position_in_frame(double **m, int fr_rows_num, int fr_cols_num,
                               int fr_rows_pos, int fr_cols_pos,
                               int env_rows_num, int env_cols_num,
                               int &fr_r_min, int &fr_r_max, int &fr_c_min,
                               int &fr_c_max) {

  // function analyses position of the cell in frame and returns
  // fr_r_min - number of first row in the frame, where you meet the pixel,
  // occupied by the cell fr_r_max - number of last row in the frame, where you
  // meet the pixel, occupied by the cell fr_c_min - number of first column in
  // the frame, where you meet the pixel, occupied by the cell fr_c_max - number
  // of last column in the frame, where you meet the pixel, occupied by the cell

  bool check = false;
  fr_r_min = -1;
  fr_r_max = fr_rows_num;
  fr_c_min = -1;
  fr_c_max = fr_cols_num;
  while (not check) {
    fr_r_min += 1;
    for (int i = 0; i < fr_cols_num; i++)
      if (m[fr_rows_pos + fr_r_min][fr_cols_pos + i] > 0)
        check = true;
  }

  check = false;
  while (not check) {
    fr_r_max -= 1;
    for (int i = (fr_cols_num - 1); i > -1; i--)
      if (m[fr_rows_pos + fr_r_max][fr_cols_pos + i] > 0)
        check = true;
  }

  check = false;
  while (not check) {
    fr_c_min += 1;
    for (int i = 0; i < fr_rows_num; i++)
      if (m[fr_rows_pos + i][fr_cols_pos + fr_c_min] > 0)
        check = true;
  }

  check = false;
  while (not check) {
    fr_c_max -= 1;
    for (int i = (fr_rows_num - 1); i > -1; i--)
      if (m[fr_rows_pos + i][fr_cols_pos + fr_c_max] > 0)
        check = true;
  }
}

double **generate_adhesion_distribution(double **Im, double **env,
                                        size_t rows_num, size_t cols_num,
                                        int adh_num, int *adh_r_pos,
                                        int *adh_c_pos) {

  double **adh = create_array2d(rows_num, cols_num);
  initiate_matrix_with_zeros(adh, rows_num, cols_num);
  int cur_num = 0;
  int i = 0;  // row index of the random point
  int j = 0;  // column index of the random point
  int rn = 0; // random number
  while (cur_num < adh_num) {
    rn = (int)((double)rand() / RAND_MAX * rows_num * cols_num);
    // random integer number in the range [0, rows_num*cols_num]
    // convert it to two indexes of the matrix
    i = rn / cols_num;
    j = rn % cols_num;
    if (Im[i][j] == 1 and env[i][j] == 1) {
      adh[i][j] = 1;
      adh_r_pos[cur_num] = i;
      adh_c_pos[cur_num] = j;
      cur_num += 1;
    }
  }
  return adh;
}

double **calculate_adhesion_field(double **Im, double **adh, int rows_num,
                                  int cols_num, int adh_num, double adh_sigma,
                                  int *adh_r_pos, int *adh_c_pos) {

  double **adh_f = create_array2d(rows_num, cols_num);
  double f_value = 0;
  double sigma_sq = pow(adh_sigma, 2);
#pragma omp parallel for collapse(2) private(f_value)
  for (int i = 0; i < rows_num; i++) {
    for (int j = 0; j < cols_num; j++) {
      if (Im[i][j] == 1) {
        f_value = 0;
        for (int k = 0; k < adh_num; k++) {
          f_value +=
              1 / (2 * M_PI * sigma_sq) *
              exp(-(pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2)) /
                  (2 * sigma_sq));
        }
        adh_f[i][j] = f_value;
      } else {
        adh_f[i][j] = 0;
      }
    }
  }
  double max_f = max_element(adh_f, rows_num, cols_num);
#pragma omp parallel for collapse(2) private(f_value)
  for (int i = 0; i < rows_num; i++) {
    for (int j = 0; j < cols_num; j++) {
      adh_f[i][j] = adh_f[i][j] / max_f;
    }
  }

  return adh_f;
}

void adhesions_rearrange(double **Im, double **env, double **adh,
                         size_t rows_num, size_t cols_num, int *adh_r_pos,
                         int *adh_c_pos, double adh_frac, int adh_num,
                         double adh_sigma) {

  int N = round(adh_num * adh_frac); // number of adhesions to rearrange
  int ind = 0; // indext of adhesion in adh_r_num and adh_c_num to rearrange

  bool check_set = false; // check if adhesion was set on propper place (inside
                          // cell and good environment)
  int rn = 0;
  int i = 0;
  int j = 0;
  // logic of permutation is the following:
  // to avoid selecting the same index and do it efficiently,
  // I chose a random integer number in the range [0, adh_num-1]
  // then the last not updated adhesion in array (with index adh_num-1-k)
  // changes it index to ind the new adhesion goes to position with index
  // (adh_num-1-k)
  for (int k = 0; k < N; k++) {
    ind = round((double)rand() / RAND_MAX * (adh_num - 1 - k));
    // cout << (adh_num-1-k) << endl;

    check_set = false;
    while (not check_set) {
      rn = (int)((double)rand() / RAND_MAX * rows_num * cols_num);
      // random integer number in the range [0, rows_num*cols_num]
      // convert it to two indexes of the matrix
      i = rn / cols_num;
      j = rn % cols_num;
      if (Im[i][j] == 1 and env[i][j] == 1 and adh[i][j] != 1) {
        adh[adh_r_pos[ind]][adh_c_pos[ind]] = 0;
        adh[i][j] = 1;
        adh_r_pos[ind] = adh_r_pos[adh_num - 1 - k];
        adh_c_pos[ind] = adh_c_pos[adh_num - 1 - k];
        adh_r_pos[adh_num - 1 - k] = i;
        adh_c_pos[adh_num - 1 - k] = j;
        check_set = true;
      }
    }
  }
}

double **place_matrix_on_big_canvas(double **m, int rows_num, int cols_num,
                                    size_t env_rows_num, size_t env_cols_num,
                                    size_t frame_rows_pos,
                                    size_t frame_cols_pos) {
  double **m_env = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(m_env, env_rows_num, env_cols_num);
  for (int i = 0; i < rows_num; i++) {
    for (int j = 0; j < cols_num; j++) {
      m_env[i + frame_rows_pos][j + frame_cols_pos] = m[i][j];
    }
  }
  return m_env;
}

double **generate_dyn_field_protr(double **cell, double **nuc,
                                  double **cell_outline, double **nuc_outline,
                                  int cell_rows, int cell_cols, int fr_rows_pos,
                                  int fr_cols_pos, int env_rows_num,
                                  int env_cols_num, double **AC) {
  // Summary: this function creates a dynein field, which is a 2d array of
  // doubles with values between 0 and 1 representing protrusion probabilities.
  // The probability values are at the nuc outline, with higher probability of
  // protrusion at areas of the nucelus that are in the direction of long cell
  // mask protrusions. In order to get high probability values in the direction
  // of long protrusions, I calculate the distance a given point on the cell
  // outline to the nearest point on the nucleus outline, representing cell
  // protrusion length. this distance value gets projected onto the nucleus
  // outline, so that longer distances (longer protrusions) result in higher
  // values at the surrounding location on the nucleus outline. Values are then
  // normalized between 0 and 1 to convert to probability.

  // Create vector containing indices of all pixels on nucleus outline
  vector<vector<int>> nuc_inds;
  for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
    for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
      if (nuc_outline[i][j] == 1) {
        nuc_inds.push_back({i, j});
      }
    }
  }

  // Calculate min distance from a given point on the cell edge to the nuc edge,
  // and write/project that distance value on the nucleus edge
  int len = nuc_inds.size();
  double **projected_nuc = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(projected_nuc, env_rows_num, env_cols_num);
  // DTmod start
  double **scaling_nuc = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(scaling_nuc, env_rows_num, env_cols_num);
  // DTmod end

  for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
    for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
      if (cell_outline[i][j] == 1) {
        // create vector with distance from cell edge pixel to each nuc edge
        // pixel
        vector<double> distances(len, 0);
        for (int ii = 0; ii < len; ++ii) {
          distances[ii] =
              sqrt(pow(i - nuc_inds[ii][0], 2) + pow(j - nuc_inds[ii][1], 2));
        }

        // find location of min_dist pixel (the pixel on nuc edge closest to the
        // current cell edge pixel)
        auto minDist = min_element(distances.begin(), distances.end());
        int idx = distance(distances.begin(), minDist);

        // Add distance value to the pixels within radius n of min_dist pixel
        // (size of n determines amount of smoothing)
        //  DTmod start
        // double dist_val = distances[idx];
        // double dist_val = AC[i][j]*distances[idx];
        double tmpv = AC[i][j] - 0.1;
        if (tmpv < 0) {
          tmpv = 0;
        }
        double dist_val = tmpv * distances[idx];

        // double dist_val = AC[i][j];
        //  DTmode end
        int new_r = nuc_inds[idx][0];
        int new_c = nuc_inds[idx][1];

        int n = len / 30; // len = 2*pi*r, so radius = len/(2*pi) so len/30 is
                          // about one fifth the radius
        for (int c = (new_c - n); c < (new_c + n); ++c) {
          if (c < 0 || c >= env_cols_num)
            continue;
          for (int r = (new_r - n); r < (new_r + n); ++r) {
            if (r < 0 || r >= env_rows_num)
              continue;
            if (nuc_outline[r][c] == 1) {
              projected_nuc[r][c] += dist_val;
              // DTmod start
              scaling_nuc[r][c] += 1;
              // DTmod end
            }
          }
        }
      }
    }
  }

  // Find the minimum and maximum values in the array
  double minVal = projected_nuc[0][0];
  double maxVal = projected_nuc[0][0];
  for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
      // DTmod start
      if (scaling_nuc[i][j] > 0) {
        projected_nuc[i][j] = projected_nuc[i][j] / scaling_nuc[i][j];
      }
      // DTmod start
      minVal = min(minVal, projected_nuc[i][j]);
      maxVal = max(maxVal, projected_nuc[i][j]);
    }
  }

  // Normalize each element between 0 and 1 so they are now probability
  // values
  for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
      // projected_nuc[i][j] = (projected_nuc[i][j] - minVal) / (maxVal -
      // minVal);
      projected_nuc[i][j] = projected_nuc[i][j] / 60;
      if (projected_nuc[i][j] > 1) {
        projected_nuc[i][j] = 1;
      }
    }
  }

  // DTmod start
  free_array2d(scaling_nuc);
  // DTmod end

  TRACE_MSG("Returning...")
  return projected_nuc;
}

double **generate_dyn_field_retr(double **cell, double **nuc,
                                 double **cell_outline, double **nuc_outline,
                                 int cell_rows, int cell_cols, int fr_rows_pos,
                                 int fr_cols_pos, int env_rows_num,
                                 int env_cols_num, double **AC) {
  // Summary: this function creates a dynein field, which is a 2d array of
  // doubles with values between 0 and 1 representing
  //  retraction probabilities. The probability values are at the nuc outline,
  //  with lower probability of retraction at areas of the nucelus that are in
  //  the direction of long cell mask protrusions. This is essential the inverse
  //  of the dyn_field protrusion probabilities, except for one difference which
  //  is that for retraction a larger smoothing value (n) is used.

  // Get indices of all pixels on nucleus outline
  vector<vector<int>> nuc_inds;
  for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
      if (nuc_outline[i][j] == 1) {
        nuc_inds.push_back({i, j});
      }
    }
  }

  // Calculate min distance from cell edge to nuc edge and write the distance
  // value on the nucleus edge
  int len = nuc_inds.size();
  double **projected_nuc = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(projected_nuc, env_rows_num, env_cols_num);
  // DTmod start
  double **scaling_nuc = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(scaling_nuc, env_rows_num, env_cols_num);
  // DTmod end
  for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
    for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
      if (cell_outline[i][j] == 1) {
        // create vector with distance from cell edge pixel to each nuc edge
        // pixel
        vector<double> distances(len, 0);
        for (int ii = 0; ii < len; ++ii) {
          distances[ii] =
              sqrt(pow(i - nuc_inds[ii][0], 2) + pow(j - nuc_inds[ii][1], 2));
        }

        // find location of min_dist pixel (the pixel on nuc edge closest to the
        // current cell edge pixel)
        auto minDist = min_element(distances.begin(), distances.end());
        int idx = distance(distances.begin(), minDist);

        // Add distance value to the pixels within radius n of min_dist pixel (n
        // determines the degree of smoothing)
        //  DTmod start
        // double dist_val = distances[idx];
        // double dist_val = AC[i][j]*distances[idx];
        double tmpv = AC[i][j] - 0.1;
        if (tmpv < 0) {
          tmpv = 0;
        }
        double dist_val = tmpv * distances[idx];

        // double dist_val = AC[i][j];
        //  DTmode end
        int new_r = nuc_inds[idx][0];
        int new_c = nuc_inds[idx][1];
        int n = len / 6; // len = 2*pi*r, so radius = len/(2*pi) so len/6 is
                         // about half the radius
        for (int r = (new_r - n); r < (new_r + n); ++r) {
          if (r < 0 || r >= env_rows_num)
            continue;
          for (int c = (new_c - n); c < (new_c + n); ++c) {
            if (c < 0 || c >= env_cols_num)
              continue;
            if (nuc_outline[r][c] == 1) {
              projected_nuc[r][c] += dist_val;
              // DTmod start
              scaling_nuc[r][c] += 1;
              // DTmod end
            }
          }
        }
      }
    }
  }

  // Find the minimum and maximum values in the array
  double minVal = projected_nuc[0][0];
  double maxVal = projected_nuc[0][0];
  for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
      // DTmod start
      if (scaling_nuc[i][j] > 0) {
        projected_nuc[i][j] = projected_nuc[i][j] / scaling_nuc[i][j];
      }
      // DTmod start
      minVal = min(minVal, projected_nuc[i][j]);
      maxVal = max(maxVal, projected_nuc[i][j]);
    }
  }

  // Normalize each element between 0 and 1 so they are now probability values
  for (int i = fr_rows_pos; i < (fr_rows_pos + cell_rows); ++i) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + cell_cols); ++j) {
      if (nuc_outline[i][j] == 1) {
        projected_nuc[i][j] = 1 - projected_nuc[i][j] / 60;
        if (projected_nuc[i][j] < 0) {
          projected_nuc[i][j] = 0;
        }
      }
    }
  }
  // DTmod start
  free_array2d(scaling_nuc);
  // DTmod end

  return projected_nuc;
}

Cell::Cell() {}

Cell::Cell(string config_file) {

  srand(0);
  // the constructor sets the values of fields in CELL object from config file
  // if values are not provided in config, they are have defould values
  // format of config file:
  //[name of the field] [:] [soace] [value]

  // default values of parameters
  // will be rewritten if Im is read from outside file

  string value = "";

  value = get_config_field(config_file, "threads_number");
  if (value != "None")
    th_num = strtod(value.c_str(), 0);
  else
    th_num = 4;
  seeds = new unsigned int[th_num];
  seeds[0] = time(0);
  for (int i = 1; i < th_num; i++)
    seeds[i] = seeds[i - 1] + 1;

  value = get_config_field(config_file, "diff_t");
  if (value != "None")
    diff_t = strtod(value.c_str(), 0);
  else
    diff_t = 100;

  value = get_config_field(config_file, "T");
  if (value != "None")
    T = strtod(value.c_str(), 0);
  else
    T = 1;

  value = get_config_field(config_file, "k");
  if (value != "None")
    k = strtod(value.c_str(), 0);
  else
    k = 3;

  value = get_config_field(config_file, "d_basal");
  if (value != "None")
    d_basal = strtod(value.c_str(), 0);
  else
    d_basal = 0.5;

  value = get_config_field(config_file, "T_nuc");
  if (value != "None")
    T_nuc = strtod(value.c_str(), 0);
  else
    T_nuc = 1;

  value = get_config_field(config_file, "R_nuc");
  if (value != "None")
    R_nuc = strtod(value.c_str(), 0);
  else
    R_nuc = 1;

  value = get_config_field(config_file, "R0");
  if (value != "None")
    R0 = strtod(value.c_str(), 0);
  else
    R0 = 10;

  value = get_config_field(config_file, "k_nuc");
  if (value != "None")
    k_nuc = strtod(value.c_str(), 0);
  else
    k_nuc = 3;

  value = get_config_field(config_file, "k_front");
  if (value != "None")
    k_front = strtod(value.c_str(), 0);
  else
    k_front = 2;

  value = get_config_field(config_file, "k_back");
  if (value != "None")
    k_back = strtod(value.c_str(), 0);
  else
    k_back = 5;

  value = get_config_field(config_file, "g");
  if (value != "None")
    g = strtod(value.c_str(), 0);
  else
    g = 2;

  value = get_config_field(config_file, "act_slope");
  if (value != "None")
    act_slope = strtod(value.c_str(), 0);
  else
    act_slope = 0.03;

  value = get_config_field(config_file, "DA");
  if (value != "None")
    DA = strtod(value.c_str(), 0);
  else
    DA = 0.001 / 3;

  value = get_config_field(config_file, "DI");
  if (value != "None")
    DI = strtod(value.c_str(), 0);
  else
    DI = 0.1 / 3;

  value = get_config_field(config_file, "k0");
  if (value != "None")
    k0 = strtod(value.c_str(), 0);
  else
    k0 = 0.05;

  value = get_config_field(config_file, "k0_min");
  if (value != "None")
    k0_min = strtod(value.c_str(), 0);
  else
    k0_min = 0.01;

  value = get_config_field(config_file, "scalar");
  if (value != "None")
    scalar = strtod(value.c_str(), 0);
  else
    scalar = 1.0;

  value = get_config_field(config_file, "noise_value");
  if (value != "None")
    noise_value = strtod(value.c_str(), 0);
  else
    noise_value = 0.1;

  value = get_config_field(config_file, "s1");
  if (value != "None")
    s1 = strtod(value.c_str(), 0);
  else
    s1 = 0.7;

  value = get_config_field(config_file, "s2");
  if (value != "None")
    s2 = strtod(value.c_str(), 0);
  else
    s2 = 0.2;

  value = get_config_field(config_file, "A0");
  if (value != "None")
    A0 = strtod(value.c_str(), 0);
  else
    A0 = 0.4;

  value = get_config_field(config_file, "F0");
  if (value != "None")
    F0 = strtod(value.c_str(), 0);
  else
    F0 = 0.5;

  value = get_config_field(config_file, "gamma");
  if (value != "None")
    gamma = strtod(value.c_str(), 0);
  else
    gamma = 1;

  value = get_config_field(config_file, "delta");
  if (value != "None")
    delta = strtod(value.c_str(), 0);
  else
    delta = 1;

  value = get_config_field(config_file, "kn");
  if (value != "None")
    kn = strtod(value.c_str(), 0);
  else
    kn = 1;

  value = get_config_field(config_file, "ks");
  if (value != "None")
    ks = strtod(value.c_str(), 0);
  else
    ks = 0.25;

  value = get_config_field(config_file, "eps");
  if (value != "None")
    eps = strtod(value.c_str(), 0);
  else
    eps = 0.1;

  value = get_config_field(config_file, "dt");
  if (value != "None")
    dt = strtod(value.c_str(), 0);
  else
    dt = 0.000375;

  value = get_config_field(config_file, "dx");
  if (value != "None")
    dx = strtod(value.c_str(), 0);
  else
    dx = 1.5 / 212;

  value = get_config_field(config_file, "adh_num");
  if (value != "None")
    adh_num = strtod(value.c_str(), 0);
  else
    adh_num = 50;

  value = get_config_field(config_file, "A_max");
  if (value != "None")
    A_max = strtod(value.c_str(), 0);
  else
    A_max = 0.67;

  value = get_config_field(config_file, "A_min");
  if (value != "None")
    A_min = strtod(value.c_str(), 0);
  else
    A_min = 0.044;

  // DTmod start
  value = get_config_field(config_file, "AC_max");
  if (value != "None")
    AC_max = strtod(value.c_str(), 0);
  else
    AC_max = 0.67;

  value = get_config_field(config_file, "AC_min");
  if (value != "None")
    AC_min = strtod(value.c_str(), 0);
  else
    AC_min = 0.044;
  // DTmod end

  adh_r_pos = new int[adh_num];
  adh_c_pos = new int[adh_num];

  value = get_config_field(config_file, "adh_frac");
  if (value != "None")
    adh_frac = strtod(value.c_str(), 0);
  else
    adh_frac = 0.05;

  value = get_config_field(config_file, "adh_sigma");
  if (value != "None")
    adh_sigma = strtod(value.c_str(), 0);
  else
    adh_sigma = 5;

  value = get_config_field(config_file, "adh_basal_prot");
  if (value != "None")
    adh_basal_prot = strtod(value.c_str(), 0);
  else
    adh_basal_prot = 0.5;

  value = get_config_field(config_file, "excite_num");
  if (value != "None")
    excite_num = strtod(value.c_str(), 0);
  else
    excite_num = 1;

  value = get_config_field(config_file, "A_excite");
  if (value != "None")
    A_excite = strtod(value.c_str(), 0);
  else
    A_excite = 100;

  // DTmod start
  value = get_config_field(config_file, "AC_excite");
  if (value != "None")
    AC_excite = strtod(value.c_str(), 0);
  else
    AC_excite = 100;
  // DTmod end

  value = get_config_field(config_file, "prop_factor");
  if (value != "None")
    prop_factor = strtod(value.c_str(), 0);
  else
    prop_factor = 1;

  value = get_config_field(config_file, "fr_dist");
  if (value != "None")
    fr_dist = strtod(value.c_str(), 0);
  else
    fr_dist = 20;

  value = get_config_field(config_file, "fr_rows_pos");
  if (value != "None")
    fr_rows_pos = strtod(value.c_str(), 0);
  else
    fr_rows_pos = 0;

  value = get_config_field(config_file, "fr_cols_pos");
  if (value != "None")
    fr_cols_pos = strtod(value.c_str(), 0);
  else
    fr_cols_pos = 0;

  value = get_config_field(config_file, "fr_rows_pos");
  if (value != "None")
    fr_rows_pos = strtod(value.c_str(), 0);
  else
    fr_rows_pos = 0;

  value = get_config_field(config_file, "fr_cols_pos");
  if (value != "None")
    fr_cols_pos = strtod(value.c_str(), 0);
  else
    fr_cols_pos = 0;

  value = get_config_field(config_file, "Im");
  Im = text_to_matrix(value, fr_rows_num, fr_cols_num);
  for (int i = 0; i < fr_rows_num; ++i) {
    for (int j = 0; j < fr_cols_num; ++j) {
      if (Im[i][j] == 2) { // 1s and 2s make up cell body
        Im[i][j] = 1;
      }
    }
  }
  Im_nuc = text_to_matrix(value, fr_rows_num, fr_cols_num);
  for (int i = 0; i < fr_rows_num; ++i) {
    for (int j = 0; j < fr_cols_num; ++j) {
      if (Im_nuc[i][j] == 2) { // 2 is nucleus only
        Im_nuc[i][j] = 1;
      } else {
        Im_nuc[i][j] = 0;
      }
    }
  }

  value = get_config_field(config_file, "A");
  if (value != "None")
    A = text_to_matrix(value, fr_rows_num, fr_cols_num);
  else {
    A = create_array2d(fr_rows_num, fr_cols_num);
    initiate_matrix_with_zeros(A, fr_rows_num, fr_cols_num);
  }

  value = get_config_field(config_file, "I");
  if (value != "None")
    I = text_to_matrix(value, fr_rows_num, fr_cols_num);
  else {
    I = create_array2d(fr_rows_num, fr_cols_num);
    initiate_matrix_with_zeros(I, fr_rows_num, fr_cols_num);
  }

  value = get_config_field(config_file, "F");
  if (value != "None")
    F = text_to_matrix(value, fr_rows_num, fr_cols_num);
  else {
    F = create_array2d(fr_rows_num, fr_cols_num);
    initiate_matrix_with_zeros(F, fr_rows_num, fr_cols_num);
  }

  // DTmod start
  value = get_config_field(config_file, "AC");
  if (value != "None")
    AC = text_to_matrix(value, fr_rows_num, fr_cols_num);
  else {
    AC = create_array2d(fr_rows_num, fr_cols_num);
    initiate_matrix_with_zeros(AC, fr_rows_num, fr_cols_num);
  }

  value = get_config_field(config_file, "FC");
  if (value != "None")
    FC = text_to_matrix(value, fr_rows_num, fr_cols_num);
  else {
    FC = create_array2d(fr_rows_num, fr_cols_num);
    initiate_matrix_with_zeros(FC, fr_rows_num, fr_cols_num);
  }

  value = get_config_field(config_file, "IC");
  if (value != "None")
    IC = text_to_matrix(value, fr_rows_num, fr_cols_num);
  else {
    IC = create_array2d(fr_rows_num, fr_cols_num);
    initiate_matrix_with_zeros(IC, fr_rows_num, fr_cols_num);
  }
  for (int i = 0; i < fr_rows_num; ++i) {
    for (int j = 0; j < fr_cols_num; ++j) {
      IC[i][j] = 0.75 * IC[i][j];
      if (Im[i][j] == 1 && i > 500) {
        AC[i][j] = 0.75;
        IC[i][j] = 0;
      }
      if (Im_nuc[i][j] == 1) { // exclude nucleus
        AC[i][j] = 0;
        IC[i][j] = 0;
        FC[i][j] = 0;
      }
    }
  }

  // DTmod end

  value = get_config_field(config_file, "env");
  if (value != "None") {
    env = text_to_matrix(value, env_rows_num, env_cols_num);
  } else {
    env_rows_num = fr_rows_num;
    env_cols_num = fr_cols_num;
    env = create_array2d(env_rows_num, env_cols_num);
    initiate_matrix_with_ones(env, env_rows_num, env_cols_num);
  }

  // print dimensions
  cout << "env_r: " << env_rows_num << endl;
  cout << "env_c: " << env_cols_num << endl;
  cout << "cell_r: " << fr_rows_num << endl;
  cout << "cell_c: " << fr_cols_num << endl;

  if ((env_rows_num > fr_rows_num) or (env_cols_num > fr_cols_num)) {
    double **Im_env =
        place_matrix_on_big_canvas(Im, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(Im);
    Im = Im_env;

    double **Im_nuc_env = place_matrix_on_big_canvas(
        Im_nuc, fr_rows_num, fr_cols_num, env_rows_num, env_cols_num,
        fr_rows_pos, fr_cols_pos);
    free_array2d(Im_nuc);
    Im_nuc = Im_nuc_env;

    double **A_env =
        place_matrix_on_big_canvas(A, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(A);
    A = A_env;

    double **I_env =
        place_matrix_on_big_canvas(I, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(I);
    I = I_env;

    double **F_env =
        place_matrix_on_big_canvas(F, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(F);
    F = F_env;

    // DTmod start
    double **AC_env =
        place_matrix_on_big_canvas(AC, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(AC);
    AC = AC_env;

    double **IC_env =
        place_matrix_on_big_canvas(IC, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(IC);
    IC = IC_env;

    double **FC_env =
        place_matrix_on_big_canvas(FC, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(FC);
    FC = FC_env;
    // DTmod end
  }

  A_new = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(A_new, env_rows_num, env_cols_num);

  I_new = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(I_new, env_rows_num, env_cols_num);

  F_new = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(F_new, env_rows_num, env_cols_num);

  // DTmod start
  AC_new = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(AC_new, env_rows_num, env_cols_num);

  IC_new = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(IC_new, env_rows_num, env_cols_num);

  FC_new = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(FC_new, env_rows_num, env_cols_num);
  // DTmod end

  outline = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(outline, env_rows_num, env_cols_num);
  update_outline();

  inner_outline = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(inner_outline, env_rows_num, env_cols_num);
  update_inner_outline();

  outline_nuc = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(outline_nuc, env_rows_num, env_cols_num);
  update_outline_nuc();

  inner_outline_nuc = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(inner_outline_nuc, env_rows_num, env_cols_num);
  update_inner_outline_nuc();

  value = get_config_field(config_file, "adh");
  if (value != "None") {
    adh = text_to_matrix(value, fr_rows_num, fr_cols_num);
    double **adh_env =
        place_matrix_on_big_canvas(adh, fr_rows_num, fr_cols_num, env_rows_num,
                                   env_cols_num, fr_rows_pos, fr_cols_pos);
    free_array2d(adh);
    adh = adh_env;
    update_adh_positions();
  } else {
    adh = create_array2d(env_rows_num, env_cols_num);
    initiate_matrix_with_zeros(adh, env_rows_num, env_cols_num);
    generate_adhesion_distribution(); // NOTE: Can get stuck in a loop if the
                                      // cell mask does not overlap with any
                                      // fibers in its initial position
  }

  adh_g = create_array2d(env_rows_num, env_cols_num);
  adh_f = create_array2d(env_rows_num, env_cols_num);
  update_adhesion_field_normalized();

  value = get_config_field(config_file, "adh_mode");
  if (value == "no_adhesions")
    initiate_matrix_with_ones(adh_f, env_rows_num, env_cols_num);

  update_volume();
  V0 = V;
  update_volume_nuc();
  V0_nuc = V_nuc;

  A_cor_sum = 0;
  I_cor_sum = 0;
  // DTmod start
  AC_cor_sum = 0;
  IC_cor_sum = 0;
  // DTmod end

  k0_adh = create_array2d(env_rows_num, env_cols_num);
  update_k0_adh();

  CoM_track = create_array2d(env_rows_num, env_cols_num);
  initiate_matrix_with_zeros(CoM_track, env_rows_num, env_cols_num);
}

Cell::~Cell() {
  if constexpr (!LIB_CELL_NUC_DEBUG_CPP) {
    free_array2d(Im);
    free_array2d(Im_nuc);
    free_array2d(A);
    free_array2d(I);
    free_array2d(F);
    free_array2d(A_new);
    free_array2d(I_new);
    free_array2d(F_new);
    // DTmod start
    free_array2d(AC);
    free_array2d(IC);
    free_array2d(FC);
    free_array2d(AC_new);
    free_array2d(IC_new);
    free_array2d(FC_new);
    // DTmod end
    free_array2d(outline);
    free_array2d(inner_outline);
    free_array2d(outline_nuc);
    free_array2d(inner_outline_nuc);
    free_array2d(adh);
    free_array2d(adh_g);
    free_array2d(adh_f);
    free_array2d(env);
    free_array2d(k0_adh);
    free_array2d(CoM_track);
    delete[] adh_r_pos;
    delete[] adh_c_pos;
  }
}

void Cell::update_volume() {
  V = 0;
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (Im[i][j] > 0)
        V++;
}

void Cell::update_outline() {
  initiate_frame_with_zeros(outline, fr_rows_num, fr_cols_num, fr_rows_pos,
                            fr_cols_pos);
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      outline[i][j] = Im[i - 1][j - 1] + Im[i - 1][j] + Im[i - 1][j + 1] +
                      Im[i][j + 1] + Im[i + 1][j + 1] + Im[i + 1][j] +
                      Im[i + 1][j - 1] + Im[i][j - 1];
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      outline[i][j] -= 8 * Im[i][j];
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (outline[i][j] > 0)
        outline[i][j] = 1;
      else
        outline[i][j] = 0;
}

void Cell::update_inner_outline() {
  initiate_frame_with_zeros(inner_outline, fr_rows_num, fr_cols_num,
                            fr_rows_pos, fr_cols_pos);
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      inner_outline[i][j] = not Im[i - 1][j - 1] + not Im[i - 1][j] +
                            not Im[i - 1][j + 1] + not Im[i][j + 1] +
                            not Im[i + 1][j + 1] + not Im[i + 1][j] +
                            not Im[i + 1][j - 1] + not Im[i][j - 1];
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      inner_outline[i][j] -= 8 * (not Im[i][j]);
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (inner_outline[i][j] > 0)
        inner_outline[i][j] = 1;
      else
        inner_outline[i][j] = 0;
}

void Cell::update_volume_nuc() {
  V_nuc = 0;
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (Im_nuc[i][j] > 0)
        V_nuc++;
}

void Cell::update_outline_nuc() {
  // perim_nuc = 0;
  initiate_frame_with_zeros(outline_nuc, fr_rows_num, fr_cols_num, fr_rows_pos,
                            fr_cols_pos);
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      outline_nuc[i][j] = Im_nuc[i - 1][j - 1] + Im_nuc[i - 1][j] +
                          Im_nuc[i - 1][j + 1] + Im_nuc[i][j + 1] +
                          Im_nuc[i + 1][j + 1] + Im_nuc[i + 1][j] +
                          Im_nuc[i + 1][j - 1] + Im_nuc[i][j - 1];
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      outline_nuc[i][j] -= 8 * Im_nuc[i][j];
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (outline_nuc[i][j] > 0) {
        // perim_nuc ++;
        outline_nuc[i][j] = 1;
      } else
        outline_nuc[i][j] = 0;
}

void Cell::update_inner_outline_nuc() {
  initiate_frame_with_zeros(inner_outline_nuc, fr_rows_num, fr_cols_num,
                            fr_rows_pos, fr_cols_pos);
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      inner_outline_nuc[i][j] =
          not Im_nuc[i - 1][j - 1] + not Im_nuc[i - 1][j] +
          not Im_nuc[i - 1][j + 1] + not Im_nuc[i][j + 1] +
          not Im_nuc[i + 1][j + 1] + not Im_nuc[i + 1][j] +
          not Im_nuc[i + 1][j - 1] + not Im_nuc[i][j - 1];
  for (int i = (1 + fr_rows_pos); i < (fr_rows_pos + fr_rows_num - 1); i++)
    for (int j = (1 + fr_cols_pos); j < (fr_cols_pos + fr_cols_num - 1); j++)
      inner_outline_nuc[i][j] -= 8 * (not Im_nuc[i][j]);
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++)
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++)
      if (inner_outline_nuc[i][j] > 0)
        inner_outline_nuc[i][j] = 1;
      else
        inner_outline_nuc[i][j] = 0;
}

void Cell::protrude_adh_nuc_push() {
  int rows_rand_idx[fr_rows_num - 2];
  int cols_rand_idx[fr_cols_num - 2];
  for (int i = 1; i != fr_rows_num - 1; i++)
    rows_rand_idx[i - 1] = i + fr_rows_pos;
  randomize(rows_rand_idx, fr_rows_num - 2);
  for (int i = 1; i != fr_cols_num - 1; i++)
    cols_rand_idx[i - 1] = i + fr_cols_pos;
  randomize(cols_rand_idx, fr_cols_num - 2);

  int i = 0;
  int j = 0;
  double V_cor = 1 / (1 + exp((V - V0) / T));
  // cout << V_cor << endl;
  // cout << V0 << ' ' << V << ' ' << T << endl;
  double A_max = max_element_in_frame(A, fr_rows_num, fr_cols_num, fr_rows_pos,
                                      fr_cols_pos);
  // DTmod start
  double AC_max = max_element_in_frame(AC, fr_rows_num, fr_cols_num,
                                       fr_rows_pos, fr_cols_pos);
  // DTmod end
  double w = 0;
  double A_average = 0;
  double I_average = 0;
  double F_average = 0;
  // DTmod start
  double AC_average = 0;
  double IC_average = 0;
  double FC_average = 0;
  // DTmod end

  double Im_local_sum = 0;
  // cout << "protrude" << endl;
  for (int ii = 0; ii < (fr_rows_num - 2); ii++) {
    i = rows_rand_idx[ii];
    for (int jj = 0; jj < (fr_cols_num - 2); jj++) {
      j = cols_rand_idx[jj];
      if (outline[i][j] == 1 and

          not((Im[i - 1][j - 1] == 1 and Im[i][j - 1] == 0 and
               Im[i - 1][j] == 0) or
              (Im[i - 1][j + 1] == 1 and Im[i - 1][j] == 0 and
               Im[i][j + 1] == 0) or
              (Im[i + 1][j + 1] == 1 and Im[i][j + 1] == 0 and
               Im[i + 1][j] == 0) or
              (Im[i + 1][j - 1] == 1 and Im[i + 1][j] == 0 and
               Im[i][j - 1] == 0)) and

          not((Im[i - 1][j] == 1 and Im[i + 1][j] == 1 and Im[i][j - 1] == 0 and
               Im[i][j + 1] == 0) or
              (Im[i - 1][j] == 0 and Im[i + 1][j] == 0 and Im[i][j - 1] == 1 and
               Im[i][j + 1] == 1))) {

        Im_local_sum = Im[i - 1][j - 1] + Im[i - 1][j] + Im[i - 1][j + 1] +
                       Im[i][j + 1] + Im[i + 1][j + 1] + Im[i + 1][j] +
                       Im[i + 1][j - 1] + Im[i][j - 1];

        A_average =
            (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j + 1] +
             A[i + 1][j + 1] + A[i + 1][j] + A[i + 1][j - 1] + A[i][j - 1]) /
            Im_local_sum;
        // DTmod start
        AC_average = (AC[i - 1][j - 1] + AC[i - 1][j] + AC[i - 1][j + 1] +
                      AC[i][j + 1] + AC[i + 1][j + 1] + AC[i + 1][j] +
                      AC[i + 1][j - 1] + AC[i][j - 1]) /
                     Im_local_sum;
        // DTmod end

        w = pow((Im[i - 1][j] + Im[i][j + 1] + Im[i + 1][j] + Im[i][j - 1] +
                 (Im[i - 1][j - 1] + Im[i - 1][j + 1] + Im[i + 1][j + 1] +
                  Im[i + 1][j - 1]) /
                     pow(sqrt(2), g)) /
                    (4 + 4 / pow(sqrt(2), g)),
                k) *
            V_cor * (1 - act_slope + act_slope * A_average / A_max) *
            (adh_basal_prot + (1 - adh_f[i][j]) * (1 - adh_basal_prot));
        // cout << i << ' ' << j << ' ' << ' ' << w << endl;
        // cout << "V_cor: " << V_cor << endl;
        // cout << "geom: " << pow(
        //( Im[i-1][j] + Im[i][j+1] + Im[i+1][j] + Im[i][j-1] +
        //( Im[i-1][j-1] + Im[i-1][j+1] + Im[i+1][j+1] + Im[i+1][j-1] ) /
        // pow(sqrt(2),g) ) / ( 4 + 4 / pow(sqrt(2),g) ), k) << endl;

        // Implement nucleus push:
        if (outline_nuc[i][j] == 1) {
          // for points where the nucleus is against the edge of cell (nuc
          // outline overlaps with cell outline), we guarantee protrusion
          // Rationale: a moving nucleus would push cytosol, so we force the
          // cell to protrude Note: this could mess up volume conservation since
          // I am bypassing all the protrusion factors
          w = 1;
        }

        Im[i][j] =
            (double)rand() / RAND_MAX <
            w; // more likely to protrude if w is higher (Im = 1 if w > rand)

        if (Im[i][j] == 1) {
          // cout << "ok" << endl;
          I_average =
              (I[i - 1][j - 1] + I[i - 1][j] + I[i - 1][j + 1] + I[i][j + 1] +
               I[i + 1][j + 1] + I[i + 1][j] + I[i + 1][j - 1] + I[i][j - 1]) /
              Im_local_sum;
          F_average =
              (F[i - 1][j - 1] + F[i - 1][j] + F[i - 1][j + 1] + F[i][j + 1] +
               F[i + 1][j + 1] + F[i + 1][j] + F[i + 1][j - 1] + F[i][j - 1]) /
              Im_local_sum;
          A_cor_sum += A_average;
          I_cor_sum += I_average;
          // DTmod start
          IC_average = (IC[i - 1][j - 1] + IC[i - 1][j] + IC[i - 1][j + 1] +
                        IC[i][j + 1] + IC[i + 1][j + 1] + IC[i + 1][j] +
                        IC[i + 1][j - 1] + IC[i][j - 1]) /
                       Im_local_sum;
          FC_average = (FC[i - 1][j - 1] + FC[i - 1][j] + FC[i - 1][j + 1] +
                        FC[i][j + 1] + FC[i + 1][j + 1] + FC[i + 1][j] +
                        FC[i + 1][j - 1] + FC[i][j - 1]) /
                       Im_local_sum;
          AC_cor_sum += AC_average;
          IC_cor_sum += IC_average;
          // DTmod end

          A[i][j] = A_average;
          I[i][j] = I_average;
          F[i][j] = F_average;
          // DTmod start
          AC[i][j] = AC_average;
          IC[i][j] = IC_average;
          FC[i][j] = FC_average;
          // DTmod end
        }
      }
    }
  }
  update_volume();
  update_outline();
  update_inner_outline();
}

void Cell::protrude_nuc() {
  int rows_rand_idx[fr_rows_num - 2];
  int cols_rand_idx[fr_cols_num - 2];
  for (int i = 1; i != fr_rows_num - 1; i++)
    rows_rand_idx[i - 1] = i + fr_rows_pos;
  for (int i = 1; i != fr_cols_num - 1; i++)
    cols_rand_idx[i - 1] = i + fr_cols_pos;
  if constexpr (!LIB_CELL_NUC_DEBUG_CPP) {
    randomize(rows_rand_idx, fr_rows_num - 2);
    randomize(cols_rand_idx, fr_cols_num - 2);
  } else {
    TRACE_MSG("Initialized random values.")
  }

  TRACE_MSG("Initializing constants...")
  int i = 0;
  int j = 0;
  double V_cor = 1 / (1 + exp((V_nuc - V0_nuc) / T_nuc));

  TRACE_MSG("Calculating nucleus perimeter...")
  // In addition to vol constraint add a roundness constraint. Calculated that
  // 4-neighbor outline of circle gives roundness of 10, should be minimum R0.
  int perim_nuc = outline_4(Im_nuc, fr_rows_num, fr_cols_num, fr_rows_pos,
                            fr_cols_pos, env_rows_num, env_cols_num);
  double R = (perim_nuc * perim_nuc) / V_nuc;
  double R_cor =
      1 / (1 + exp((R - R0) /
                   R_nuc)); // R_cor is lower when R is large (non-circular),
                            // less protrusion when less circular
  auto C = 4 + 4 / pow(sqrt(2), g);

  // DTmod start
  // double ** dyn_f = generate_dyn_field_protr(Im,Im_nuc,outline,outline_nuc,
  //                                                    fr_rows_num,
  //                                                    fr_cols_num,
  //                                                    fr_rows_pos,
  //                                                    fr_cols_pos,
  //                                                    env_rows_num,
  //                                                    env_cols_num,AC);

  TRACE_MSG("Generating dynein field...")
  double **dyn_f = generate_dyn_field_protr(
      Im, Im_nuc, inner_outline, outline_nuc, fr_rows_num, fr_cols_num,
      fr_rows_pos, fr_cols_pos, env_rows_num, env_cols_num, AC);
  if constexpr (LIB_CELL_NUC_DEBUG_CPP) {
    test_dyn_f = dyn_f;
  }
  // DTmod end

  // cout << "protrude: made dmap" << endl;

  TRACE_MSG("Attempting to protrude nucleus pixels...")
  double w = 0;
  for (int jj = 0; jj < (fr_cols_num - 2); jj++) {
    j = cols_rand_idx[jj];
    for (int ii = 0; ii < (fr_rows_num - 2); ii++) {
      i = rows_rand_idx[ii];

      if constexpr (LIB_CELL_NUC_DEBUG_CPP) {
        i = ii + fr_rows_pos;
        j = jj + fr_cols_pos;
      }

      if (outline_nuc[i][j] == 1 and

          not(outline[i][j] ==
              1) and // nucleus can't protrude past edge of cell

          not((Im_nuc[i - 1][j - 1] == 1 and Im_nuc[i][j - 1] == 0 and
               Im_nuc[i - 1][j] == 0) or
              (Im_nuc[i - 1][j + 1] == 1 and Im_nuc[i - 1][j] == 0 and
               Im_nuc[i][j + 1] == 0) or
              (Im_nuc[i + 1][j + 1] == 1 and Im_nuc[i][j + 1] == 0 and
               Im_nuc[i + 1][j] == 0) or
              (Im_nuc[i + 1][j - 1] == 1 and Im_nuc[i + 1][j] == 0 and
               Im_nuc[i][j - 1] == 0)) and

          not((Im_nuc[i - 1][j] == 1 and Im_nuc[i + 1][j] == 1 and
               Im_nuc[i][j - 1] == 0 and Im_nuc[i][j + 1] == 0) or
              (Im_nuc[i - 1][j] == 0 and Im_nuc[i + 1][j] == 0 and
               Im_nuc[i][j - 1] == 1 and Im_nuc[i][j + 1] == 1))) {

        const double n = (Im_nuc[i - 1][j] + Im_nuc[i][j + 1] +
                          Im_nuc[i + 1][j] + Im_nuc[i][j - 1] +
                          (Im_nuc[i - 1][j - 1] + Im_nuc[i - 1][j + 1] +
                           Im_nuc[i + 1][j + 1] + Im_nuc[i + 1][j - 1]) /
                              pow(sqrt(2), g));
        w = pow(n / (4 + 4 / pow(sqrt(2), g)), k_nuc) * R_cor * V_cor *
            (d_basal + (1 - d_basal) * dyn_f[i][j]);

        // reference from protrusion_adh()
        //  ( 1 - act_slope + act_slope * A_average / A_max ) : when A_avg is
        //  higher, w (prob of prot) is higher

        Im_nuc[i][j] = (double)rand() / RAND_MAX < w;

        // DTmod start: removes concentration values due to nucleus protrusion
        if (Im_nuc[i][j] == 1) {
          AC_cor_sum -= AC[i][j];
          AC[i][j] = 0;
          IC_cor_sum -= IC[i][j];
          IC[i][j] = 0;
          FC[i][j] = 0;
        }
        // DTmod end
      }
    }
  }

  if constexpr (!LIB_CELL_NUC_DEBUG_CPP) {
    free_array2d(dyn_f);
  }
  // cout << "protrusion done" <<endl;

  TRACE_MSG("Updating nucleus volume...")
  update_volume_nuc();
  TRACE_MSG("Updating nucleus outline...")
  update_outline_nuc(); // updates outline and calculates number of pixels as
                        // perimeter
  TRACE_MSG("Updating nucleus inner outline...")
  update_inner_outline_nuc();
}

void Cell::retract() {
  int rows_rand_idx[fr_rows_num - 2];
  int cols_rand_idx[fr_cols_num - 2];
  for (int i = 1; i != fr_rows_num - 1; i++)
    rows_rand_idx[i - 1] = i + fr_rows_pos;
  randomize(rows_rand_idx, fr_rows_num - 2);
  for (int i = 1; i != fr_cols_num - 1; i++)
    cols_rand_idx[i - 1] = i + fr_cols_pos;
  randomize(cols_rand_idx, fr_cols_num - 2);

  int i = 0;
  int j = 0;
  double V_cor = 1 / (1 + exp(-(V - V0) / T));
  double A_max = max_element_in_frame(A, fr_rows_num, fr_cols_num, fr_rows_pos,
                                      fr_cols_pos);
  // DTmod start
  double AC_max = max_element_in_frame(AC, fr_rows_num, fr_cols_num,
                                       fr_rows_pos, fr_cols_pos);
  // DTmod end
  double w = 0;
  double A_average = 0;
  // DTmod start
  double AC_average = 0;
  // DTmod end
  // cout << "retract" << endl;
  for (int ii = 0; ii < (fr_rows_num - 2); ii++) {
    i = rows_rand_idx[ii];
    for (int jj = 0; jj < (fr_cols_num - 2); jj++) {
      j = cols_rand_idx[jj];
      if (inner_outline[i][j] == 1 and

          not(Im_nuc[i][j] ==
              1) and // Im_nuc is mask of nucleus (area of no retraction)

          not((Im[i - 1][j - 1] == 0 and Im[i][j - 1] == 1 and
               Im[i - 1][j] == 1) or
              (Im[i - 1][j + 1] == 0 and Im[i - 1][j] == 1 and
               Im[i][j + 1] == 1) or
              (Im[i + 1][j + 1] == 0 and Im[i][j + 1] == 1 and
               Im[i + 1][j] == 1) or
              (Im[i + 1][j - 1] == 0 and Im[i + 1][j] == 1 and
               Im[i][j - 1] == 1)) and

          not((Im[i - 1][j] == 0 and Im[i + 1][j] == 0 and Im[i][j - 1] == 1 and
               Im[i][j + 1] == 1) or
              (Im[i - 1][j] == 1 and Im[i + 1][j] == 1 and Im[i][j - 1] == 0 and
               Im[i][j + 1] == 0))) {

        A_average =
            (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j + 1] +
             A[i + 1][j + 1] + A[i + 1][j] + A[i + 1][j - 1] + A[i][j - 1]) /
            (Im[i - 1][j - 1] + Im[i - 1][j] + Im[i - 1][j + 1] + Im[i][j + 1] +
             Im[i + 1][j + 1] + Im[i + 1][j] + Im[i + 1][j - 1] + Im[i][j - 1]);
        // DTmod start
        AC_average =
            (AC[i - 1][j - 1] + AC[i - 1][j] + AC[i - 1][j + 1] + AC[i][j + 1] +
             AC[i + 1][j + 1] + AC[i + 1][j] + AC[i + 1][j - 1] +
             AC[i][j - 1]) /
            (Im[i - 1][j - 1] + Im[i - 1][j] + Im[i - 1][j + 1] + Im[i][j + 1] +
             Im[i + 1][j + 1] + Im[i + 1][j] + Im[i + 1][j - 1] + Im[i][j - 1]);
        // DTmod end
        w = pow((not Im[i - 1][j] + not Im[i][j + 1] + not Im[i + 1][j] +
                 not Im[i][j - 1] +
                 (not Im[i - 1][j - 1] + not Im[i - 1][j + 1] +
                  not Im[i + 1][j + 1] + not Im[i + 1][j - 1]) /
                     pow(sqrt(2), g)) /
                    (4 + 4 / pow(sqrt(2), g)),
                k) *
            V_cor * (1 - act_slope * A_average / A_max) * adh_f[i][j];
        // cout << adh_f[i][j] << endl;
        // cout << A_max << endl;
        // cout << i << ' ' << j << ' ' << ' ' << w << endl;
        // cout << A_max << ' ' << A_average << ' ' << A_average/A_max << endl;

        /*
        cout << "retract: " <<  ( 1 - act_slope * A_average / A_max ) << ' ';
        cout << "A_average: " << A_average << ' ';
        cout << "A_max: " << A_max << ' ';
        cout << "(i,j): " << i << ' ' << j << ' ';
        cout << "Im_local_sum: " << Im[i-1][j-1] + Im[i-1][j] + Im[i-1][j+1] +
        Im[i][j+1] + Im[i+1][j+1] + Im[i+1][j] + Im[i+1][j-1] + Im[i][j-1] << '
        '; cout << "A_local_sum: " <<  A[i-1][j-1] + A[i-1][j] + A[i-1][j+1] +
        A[i][j+1] + A[i+1][j+1] + A[i+1][j] + A[i+1][j-1] + A[i][j-1] << endl;
        cout << " Im[i-1][j-1]: " << Im[i-1][j-1] << " Im[i-1][j]: " <<
        Im[i-1][j]
             << " Im[i-1][j+1]: " << Im[i-1][j+1] << " Im[i][j+1]: " <<
        Im[i][j+1]
             << " Im[i+1][j+1]: " << Im[i+1][j+1] << " Im[i+1][j]: " <<
        Im[i+1][j]
             << " Im[i+1][j-1]: " << Im[i+1][j-1] << " Im[i][j-1]: " <<
        Im[i][j-1] << endl; cout << " A[i-1][j-1]: " << A[i-1][j-1] << "
        A[i-1][j]: " << A[i-1][j]
             << " A[i-1][j+1]: " << A[i-1][j+1] << " A[i][j+1]: " << A[i][j+1]
             << " A[i+1][j+1]: " << A[i+1][j+1] << " A[i+1][j]: " << A[i+1][j]
             << " A[i+1][j-1]: " << A[i+1][j-1] << " A[i][j-1]: " << A[i][j-1]
        << endl;
        */

        // cout << "( " << i << ", " << j << " )  " << "Im[i][j]: " << Im[i][j]
        // << endl;
        Im[i][j] = not((double)rand() / RAND_MAX < w);

        if (Im[i][j] == 0) {
          // cout << "ok" << endl;
          // cout << "Im[i][j] " <<  Im[i][j] << " A[i][j]: " << A[i][j] << "
          // I[i][j]: " << I[i][j]
          //      << " F[i][j]: " << F[i][j] << ' ';
          A_cor_sum -= A[i][j];
          A[i][j] = 0;
          I_cor_sum -= I[i][j];
          I[i][j] = 0;
          F[i][j] = 0;

          // cout << A[i][j] << ' ' << I[i][j] << ' ' << F[i][j] << endl;

          // DTmod start
          AC_cor_sum -= AC[i][j];
          AC[i][j] = 0;
          IC_cor_sum -= IC[i][j];
          IC[i][j] = 0;
          FC[i][j] = 0;
          // DTmod end
        }
      }
    }
  }
  update_volume();
  update_outline();
  update_inner_outline();
}

void Cell::retract_nuc() {
  int rows_rand_idx[fr_rows_num - 2];
  int cols_rand_idx[fr_cols_num - 2];
  for (int i = 1; i != fr_rows_num - 1; i++)
    rows_rand_idx[i - 1] = i + fr_rows_pos;
  for (int i = 1; i != fr_cols_num - 1; i++)
    cols_rand_idx[i - 1] = i + fr_cols_pos;
  if constexpr (!LIB_CELL_NUC_DEBUG_CPP) {
    randomize(rows_rand_idx, fr_rows_num - 2);
    randomize(cols_rand_idx, fr_cols_num - 2);
  } else {
    TRACE_MSG("Initialized random values.")
  }

  int i = 0;
  int j = 0;

  // DTmod start
  double AC_average = 0;
  double IC_average = 0;
  double FC_average = 0;
  double Im_nuc_local_sum = 0;
  // DTmod end

  double V_cor = 1 / (1 + exp(-(V_nuc - V0_nuc) / T_nuc));
  // In addition to vol constraint, add a roundness constraint. Calculated that
  // 4-neighbor outline of circle gives roundness of 10.
  int perim_nuc = outline_4(Im_nuc, fr_rows_num, fr_cols_num, fr_rows_pos,
                            fr_cols_pos, env_rows_num, env_cols_num);
  double R = (perim_nuc * perim_nuc) / V_nuc;
  double R_cor = 1 / (1 + exp((R - R0) / R_nuc)); // took out the negative sign

  // Note: for retraction I am projecting onto inner outline of nuc
  //  DTmod start
  // double ** dyn_f =
  // generate_dyn_field_retr(Im,Im_nuc,outline,inner_outline_nuc,
  //                                                     fr_rows_num,
  //                                                     fr_cols_num,
  //                                                     fr_rows_pos,
  //                                                     fr_cols_pos,
  //                                                     env_rows_num,
  //                                                     env_cols_num,AC);
  double **dyn_f = generate_dyn_field_retr(
      Im, Im_nuc, inner_outline, inner_outline_nuc, fr_rows_num, fr_cols_num,
      fr_rows_pos, fr_cols_pos, env_rows_num, env_cols_num, AC);
  if constexpr (LIB_CELL_NUC_DEBUG_CPP) {
    test_dyn_f = dyn_f;
  }
  // DTmod end

  // cout << "retr: made dmap" << endl;
  double w = 0;
  for (int jj = 0; jj < (fr_cols_num - 2); jj++) {
    j = cols_rand_idx[jj];
    for (int ii = 0; ii < (fr_rows_num - 2); ii++) {
      i = rows_rand_idx[ii];
      if constexpr (LIB_CELL_NUC_DEBUG_CPP) {
        i = ii + fr_rows_pos;
        j = jj + fr_cols_pos;
      }

      if (inner_outline_nuc[i][j] == 1 and

          not((Im_nuc[i - 1][j - 1] == 0 and Im_nuc[i][j - 1] == 1 and
               Im_nuc[i - 1][j] == 1) or
              (Im_nuc[i - 1][j + 1] == 0 and Im_nuc[i - 1][j] == 1 and
               Im_nuc[i][j + 1] == 1) or
              (Im_nuc[i + 1][j + 1] == 0 and Im_nuc[i][j + 1] == 1 and
               Im_nuc[i + 1][j] == 1) or
              (Im_nuc[i + 1][j - 1] == 0 and Im_nuc[i + 1][j] == 1 and
               Im_nuc[i][j - 1] == 1)) and

          not((Im_nuc[i - 1][j] == 0 and Im_nuc[i + 1][j] == 0 and
               Im_nuc[i][j - 1] == 1 and Im_nuc[i][j + 1] == 1) or
              (Im_nuc[i - 1][j] == 1 and Im_nuc[i + 1][j] == 1 and
               Im_nuc[i][j - 1] == 0 and Im_nuc[i][j + 1] == 0))) {

        // DTmod start: finds pixels outside of nucleus in the neighborhood of
        // the retracted pixel
        Im_nuc_local_sum = 8 - Im_nuc[i - 1][j - 1] - Im_nuc[i - 1][j] -
                           Im_nuc[i - 1][j + 1] - Im_nuc[i][j + 1] -
                           Im_nuc[i + 1][j + 1] - Im_nuc[i + 1][j] -
                           Im_nuc[i + 1][j - 1] - Im_nuc[i][j - 1];

        AC_average = (AC[i - 1][j - 1] + AC[i - 1][j] + AC[i - 1][j + 1] +
                      AC[i][j + 1] + AC[i + 1][j + 1] + AC[i + 1][j] +
                      AC[i + 1][j - 1] + AC[i][j - 1]) /
                     Im_nuc_local_sum;
        // DTmod end

        double n = (not Im_nuc[i - 1][j] + not Im_nuc[i][j + 1] +
                    not Im_nuc[i + 1][j] + not Im_nuc[i][j - 1] +
                    (not Im_nuc[i - 1][j - 1] + not Im_nuc[i - 1][j + 1] +
                     not Im_nuc[i + 1][j + 1] + not Im_nuc[i + 1][j - 1]) /
                        pow(sqrt(2), g));
        w = pow(n / (4 + 4 / pow(sqrt(2), g)), k_nuc) * R_cor * V_cor *
            (d_basal + (1 - d_basal) * dyn_f[i][j]);

        // reference from retract
        //  If A_average is high (more A in cell), then w (prob of protrusion
        //  from outside) is lower aka less retraction ( 1 - act_slope *
        //  A_average / A_max )

        Im_nuc[i][j] = not((double)rand() / RAND_MAX < w);

        // DTmod start: assign average value to the new pixel due the nucleus
        // retraction
        if (Im_nuc[i][j] == 0) {
          // cout << "ok" << endl;
          IC_average = (IC[i - 1][j - 1] + IC[i - 1][j] + IC[i - 1][j + 1] +
                        IC[i][j + 1] + IC[i + 1][j + 1] + IC[i + 1][j] +
                        IC[i + 1][j - 1] + IC[i][j - 1]) /
                       Im_nuc_local_sum;
          FC_average = (FC[i - 1][j - 1] + FC[i - 1][j] + FC[i - 1][j + 1] +
                        FC[i][j + 1] + FC[i + 1][j + 1] + FC[i + 1][j] +
                        FC[i + 1][j - 1] + FC[i][j - 1]) /
                       Im_nuc_local_sum;
          AC_cor_sum += AC_average;
          IC_cor_sum += IC_average;

          AC[i][j] = AC_average;
          IC[i][j] = IC_average;
          FC[i][j] = FC_average;
        }

        // DTmod end
      }
    }
  }
  if constexpr (!LIB_CELL_NUC_DEBUG_CPP) {
    free_array2d(dyn_f);
  }
  update_volume_nuc();
  update_outline_nuc();
  update_inner_outline_nuc();
}

void Cell::diffuse_k0_adh() {
  double f = 0;
  double h = 0;
  // DTmod start
  double fC = 0;
  double hC = 0;

  double s2C = 0.05;
  // DTmod end
  for (int k = 0; k < diff_t; k++) {
#pragma omp parallel for collapse(2) private(f, h)
    for (int i = (fr_rows_pos + 1); i < (fr_rows_pos + fr_rows_num - 1); i++) {
      for (int j = (fr_cols_pos + 1); j < (fr_cols_pos + fr_cols_num - 1);
           j++) {
        if (Im[i][j] == 1) {
          f = (k0_adh[i][j] +
               gamma * pow(A[i][j], 3) / (pow(A0, 3) + pow(A[i][j], 3))) *
                  I[i][j] -
              delta * (s1 + s2 * F[i][j] / (F0 + F[i][j])) * A[i][j];
          h = eps * (kn * A[i][j] - ks * F[i][j]);

          A_new[i][j] =
              A[i][j] + (f + DA / pow(dx, 2) *
                                 (Im[i + 1][j] * (A[i + 1][j] - A[i][j]) -
                                  Im[i - 1][j] * (A[i][j] - A[i - 1][j]) +
                                  Im[i][j + 1] * (A[i][j + 1] - A[i][j]) -
                                  Im[i][j - 1] * (A[i][j] - A[i][j - 1]))) *
                            dt;

          I_new[i][j] =
              I[i][j] + (-f + DI / pow(dx, 2) *
                                  (Im[i + 1][j] * (I[i + 1][j] - I[i][j]) -
                                   Im[i - 1][j] * (I[i][j] - I[i - 1][j]) +
                                   Im[i][j + 1] * (I[i][j + 1] - I[i][j]) -
                                   Im[i][j - 1] * (I[i][j] - I[i][j - 1]))) *
                            dt;

          F_new[i][j] = F[i][j] + h * dt;
        }
        if (Im[i][j] == 1 && Im_nuc[i][j] == 0) {
          // DTmod start
          // AC_new[i][j] = A_new[i][j];
          // IC_new[i][j] = I_new[i][j];
          // FC_new[i][j] = F_new[i][j];

          // fC = ( k0_adh[i][j] + gamma * pow(AC[i][j],3) / ( pow(A0,3) +
          // pow(AC[i][j],3) ) ) * IC[i][j] -
          //     delta * ( s1 + s2 * FC[i][j] / ( F0 + FC[i][j] )) * AC[i][j];
          fC = (k0 +
                gamma * pow(AC[i][j], 3) / (pow(A0, 3) + pow(AC[i][j], 3))) *
                   IC[i][j] -
               delta * (s1 + s2C * FC[i][j] / (F0 + FC[i][j])) * AC[i][j];
          // need to try k0 = 0.05 (k0/2)
          hC = eps * (kn * AC[i][j] - ks * FC[i][j]);

          AC_new[i][j] =
              AC[i][j] + (fC + DA / pow(dx, 2) *
                                   ((Im[i + 1][j] - Im_nuc[i + 1][j]) *
                                        (AC[i + 1][j] - AC[i][j]) -
                                    (Im[i - 1][j] - Im_nuc[i - 1][j]) *
                                        (AC[i][j] - AC[i - 1][j]) +
                                    (Im[i][j + 1] - Im_nuc[i][j + 1]) *
                                        (AC[i][j + 1] - AC[i][j]) -
                                    (Im[i][j - 1] - Im_nuc[i][j - 1]) *
                                        (AC[i][j] - AC[i][j - 1]))) *
                             dt;

          IC_new[i][j] =
              IC[i][j] + (-fC + DI / pow(dx, 2) *
                                    ((Im[i + 1][j] - Im_nuc[i + 1][j]) *
                                         (IC[i + 1][j] - IC[i][j]) -
                                     (Im[i - 1][j] - Im_nuc[i - 1][j]) *
                                         (IC[i][j] - IC[i - 1][j]) +
                                     (Im[i][j + 1] - Im_nuc[i][j + 1]) *
                                         (IC[i][j + 1] - IC[i][j]) -
                                     (Im[i][j - 1] - Im_nuc[i][j - 1]) *
                                         (IC[i][j] - IC[i][j - 1]))) *
                             dt;

          FC_new[i][j] = FC[i][j] + hC * dt;
          // DTmod end
        }
      }
    }
    swap_array2d(A, A_new, env_rows_num, env_cols_num);
    swap_array2d(I, I_new, env_rows_num, env_cols_num);
    swap_array2d(F, F_new, env_rows_num, env_cols_num);
    // DTmod start
    swap_array2d(AC, AC_new, env_rows_num, env_cols_num);
    swap_array2d(IC, IC_new, env_rows_num, env_cols_num);
    swap_array2d(FC, FC_new, env_rows_num, env_cols_num);
    // DTmod end
  }
}

void Cell::correct_concentrations() {
  double A_local_cor = A_cor_sum / V;
  double I_local_cor = I_cor_sum / V;
  // DTmod start
  double AC_local_cor = AC_cor_sum / (V - V_nuc);
  double IC_local_cor = IC_cor_sum / (V - V_nuc);
  // DTmod end

  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      if (Im[i][j] == 1) {
        A[i][j] -= A_local_cor;
        I[i][j] -= I_local_cor;
      }
      // DTmod start
      if (Im[i][j] == 1 && Im_nuc[i][j] == 0) {
        AC[i][j] -= AC_local_cor;
        IC[i][j] -= IC_local_cor;
      }
      // DTmod end
    }
  }
  A_cor_sum = 0;
  I_cor_sum = 0;
  // DTmod start
  AC_cor_sum = 0;
  IC_cor_sum = 0;
  // DTmod end
}

void Cell::generate_adhesion_distribution() {
  int cur_num = 0;
  int i = 0;  // row index of the random point
  int j = 0;  // column index of the random point
  int rn = 0; // random number

  while (cur_num < adh_num) {
    rn = (int)((double)rand() / RAND_MAX * fr_rows_num * fr_cols_num);
    // random integer number in the range [0, rows_num*cols_num]
    // convert it to two indexes of the matrix
    i = rn / fr_cols_num + fr_rows_pos;
    j = rn % fr_cols_num + fr_cols_pos;
    // cout << "cur_num: " << cur_num << endl;
    // cout << "i: " << i << endl;
    // cout << "j: " << j << endl;
    // cout << "Im[i][j]: " << Im[i][j] << endl;
    // cout << "env[i][j]: " << env[i][j] << endl;
    if (Im[i][j] == 1 and env[i][j] == 1 and adh[i][j] != 1) {
      adh[i][j] = 1;
      adh_r_pos[cur_num] = i;
      adh_c_pos[cur_num] = j;
      cur_num += 1;
    }
  }
}

void Cell::generate_adhesion_distribution_polarized() {
  int cur_num = 0;
  int i = 0;  // row index of the random point
  int j = 0;  // column index of the random point
  int rn = 0; // random number
  double A_max = max_element_in_frame(A, fr_rows_num, fr_cols_num, fr_rows_pos,
                                      fr_cols_pos);
  while (cur_num < adh_num) {
    rn = (int)((double)rand() / RAND_MAX * fr_rows_num * fr_cols_num);
    // random integer number in the range [0, rows_num*cols_num]
    // convert it to two indexes of the matrix
    i = rn / fr_cols_num + fr_rows_pos;
    j = rn % fr_cols_num + fr_cols_pos;
    if (Im[i][j] == 1 and env[i][j] == 1 and adh[i][j] != 1 and
        (double) rand() / RAND_MAX * A_max < A[i][j]) {
      adh[i][j] = 1;
      adh_r_pos[cur_num] = i;
      adh_c_pos[cur_num] = j;
      cur_num += 1;
    }
  }
}

void Cell::update_adhesion_field() {
  double f_value = 0;
  double sigma_sq = pow(adh_sigma, 2);
  double ampl = 1 / (2 * M_PI * sigma_sq);
#pragma omp parallel for collapse(2) private(f_value)
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      f_value = 0;
      for (int k = 0; k < adh_num; k++) {
        f_value += exp(-(pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2)) /
                       (2 * sigma_sq));
      }
      adh_g[i][j] = ampl * f_value;
    }
  }
  double max_f = max_element_in_frame(adh_g, fr_rows_num, fr_cols_num,
                                      fr_rows_pos, fr_cols_pos);
#pragma omp parallel for collapse(2)
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      adh_f[i][j] = 1 - adh_g[i][j] / max_f;
      if (adh[i][j] == 1)
        adh_f[i][j] = 0;
    }
  }
}

// TODO: Send this fragment
void Cell::update_adhesion_field_normalized() {
  double f_value = 0;
  double norm_numer = 0;
  double norm_denom = 0;
  double sigma_sq = pow(adh_sigma, 2);
  double ampl = 1 / (2 * M_PI * sigma_sq);

// in this function I want within one parallel for loop calculate adh_g
// (Gaussian smoothing of adhesions) and adh_f_norm (normalized adhesion field),
// which I do in the next loop. to do it, I need to know values of ahd_f in
// points of adhesions (to calculate normalizing factors) that is why I have the
// first parallel for loop (next one), where I preliminary calculate values that
// are needed
#pragma omp parallel for private(f_value)
  for (int k_cur = 0; k_cur < adh_num; k_cur++) {
    f_value = 0;
    for (int k = 0; k < adh_num; k++) {
      f_value += exp(-(pow(adh_r_pos[k_cur] - adh_r_pos[k], 2) +
                       pow(adh_c_pos[k_cur] - adh_c_pos[k], 2)) /
                     (2 * sigma_sq));
    }
    adh_g[adh_r_pos[k_cur]][adh_c_pos[k_cur]] = ampl * f_value;
  }

// parallel calculation of adh_g and adh_f_norm
#pragma omp parallel for collapse(2) private(f_value, norm_numer, norm_denom)
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      f_value = 0;
      norm_numer = 0;
      norm_denom = 0;
      if (adh[i][j] != 1) {
        for (int k = 0; k < adh_num; k++) {
          f_value +=
              exp(-(pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2)) /
                  (2 * sigma_sq));
          norm_numer += adh_g[adh_r_pos[k]][adh_c_pos[k]] /
                        (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          norm_denom +=
              1 / (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
        }
        adh_g[i][j] = ampl * f_value;
        adh_f[i][j] = adh_g[i][j] / norm_numer * norm_denom;

      } /* else {
            for (int k = 0; k < adh_num; k++){
               f_value += exp( -(pow(i-adh_r_pos[k],2) + pow(j-adh_c_pos[k],2))
       / (2*sigma_sq) );
           }
           adh_g[i][j] = ampl * f_value;
           adh_f[i][j] = 1;
       }*/
    }
  }

  double max_f = max_element_in_frame(adh_f, fr_rows_num, fr_cols_num,
                                      fr_rows_pos, fr_cols_pos);
#pragma omp parallel for collapse(2)
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      adh_f[i][j] = 1 - adh_f[i][j] / max_f;
      if (adh[i][j] == 1)
        adh_f[i][j] = 0;
    }
  }
}

/**
 * Rearrange the adhesion points around the cell to simulate evolution of cell
 * adhesions Randomly picks adh_frac of the adhesions and finds other valid
 * positions to move them
 */
void Cell::rearrange_adhesions() {

  int N = round(adh_num * adh_frac); // number of adhesions to rearrange
  int ind = 0; // indext of adhesion in adh_r_num and adh_c_num to rearrange

  bool check_set = false; // check if adhesion was set on propper place (inside
                          // cell and good environment)
  int rn = 0;
  int i = 0;
  int j = 0;
  // logic of permutation is the following:
  // to avoid selecting the same index and do it efficiently,
  // I chose a random integer number in the range [0, adh_num-1-k]
  // then the last not updated adhesion in array (with index adh_num-1-k)
  // changes it index to ind the new adhesion goes to position with index
  // (adh_num-1-k) cout << "start rearrange adhesions" << endl;
  for (int k = 0; k < N; k++) {
    // I shoudl avoid permutation of adhesion with itself to prevet bugs
    ind = round((double)rand() / RAND_MAX * (adh_num - 1 - k));
    while (ind == (adh_num - 1 - k)) {
      ind = round((double)rand() / RAND_MAX * (adh_num - 1 - k));
    }
    // cout << "ind: " << ind << endl;
    // cout << "pos: " << adh_r_pos[ind] << " " << adh_c_pos[ind] << endl;
    check_set = false;
    while (not check_set) {
      rn = (int)((double)rand() / RAND_MAX * fr_rows_num * fr_cols_num);
      // random integer number in the range [0, rows_num*cols_num]
      // convert it to two indexes of the matrix
      i = rn / fr_cols_num + fr_rows_pos;
      j = rn % fr_cols_num + fr_cols_pos;
      if (Im[i][j] == 1 and env[i][j] == 1 and adh[i][j] != 1) {
        adh[adh_r_pos[ind]][adh_c_pos[ind]] = 0;
        adh[i][j] = 1;
        adh_r_pos[ind] = adh_r_pos[adh_num - 1 - k];
        adh_c_pos[ind] = adh_c_pos[adh_num - 1 - k];
        adh_r_pos[adh_num - 1 - k] = i;
        adh_c_pos[adh_num - 1 - k] = j;
        // cout << "adh_num-1-k: " << adh_num-1-k << " " <<
        // adh_r_pos[adh_num-1-k] << " " <<
        //                                           adh_c_pos[adh_num-1-k] <<
        //                                           endl;
        // cout << "new adhesion: " << i << " " << j << " " << adh[i][j] <<
        // endl; cout << "old adhesion: " << adh_r_pos[ind] << " " <<
        // adh_c_pos[ind] << " " <<
        //                             adh[adh_r_pos[ind]][adh_c_pos[ind]] <<
        //                             endl;
        check_set = true;
      }
    }
  }
  // cout << "adh positions after rearrangement: " << endl;
  // for (int k = 0; k < adh_num; k++) {
  //     cout << "   " << adh_r_pos[k] << " " << adh_c_pos[k] << " " <<
  //             adh[adh_r_pos[k]][adh_c_pos[k]] << endl;
  // }
  update_adhesion_field_normalized();
}

void Cell::rearrange_adhesions_polarized() {
  // logic of rearrangerent is the following:
  // any adhesion is removed randomly (removal rate is not polarized)
  // but formation rate of new adhesion is polarized (it follows distribution of
  // A)

  int N = round(adh_num * adh_frac); // number of adhesions to rearrange
  int ind = 0; // indext of adhesion in adh_r_num and adh_c_num to rearrange

  bool check_set = false; // check if adhesion was set on propper place (inside
                          // cell and good environment)
  int rn = 0;
  int i = 0;
  int j = 0;
  double A_max = max_element_in_frame(A, fr_rows_num, fr_cols_num, fr_rows_pos,
                                      fr_cols_pos);
  // logic of permutation is the following:
  // to avoid selecting the same index and do it efficiently,
  // I chose a random integer number in the range [0, adh_num-1-k]
  // then the last not updated adhesion in array (with index adh_num-1-k)
  // changes it index to ind the new adhesion goes to position with index
  // (adh_num-1-k) cout << "start rearrange adhesions" << endl;
  for (int k = 0; k < N; k++) {
    // I shoudl avoid permutation of adhesion with itself to prevet bugs
    ind = round((double)rand() / RAND_MAX * (adh_num - 1 - k));
    while (ind == (adh_num - 1 - k)) {
      ind = round((double)rand() / RAND_MAX * (adh_num - 1 - k));
    }
    // cout << "ind: " << ind << endl;
    // cout << "pos: " << adh_r_pos[ind] << " " << adh_c_pos[ind] << endl;
    check_set = false;
    while (not check_set) {
      rn = (int)((double)rand() / RAND_MAX * fr_rows_num * fr_cols_num);
      // random integer number in the range [0, rows_num*cols_num]
      // convert it to two indexes of the matrix
      i = rn / fr_cols_num + fr_rows_pos;
      j = rn % fr_cols_num + fr_cols_pos;
      if (Im[i][j] == 1 and env[i][j] == 1 and adh[i][j] != 1 and
          (double) rand() / RAND_MAX * A_max < A[i][j]) {
        adh[adh_r_pos[ind]][adh_c_pos[ind]] = 0;
        adh[i][j] = 1;
        adh_r_pos[ind] = adh_r_pos[adh_num - 1 - k];
        adh_c_pos[ind] = adh_c_pos[adh_num - 1 - k];
        adh_r_pos[adh_num - 1 - k] = i;
        adh_c_pos[adh_num - 1 - k] = j;
        // cout << "adh_num-1-k: " << adh_num-1-k << " " <<
        // adh_r_pos[adh_num-1-k] << " " <<
        //                                           adh_c_pos[adh_num-1-k] <<
        //                                           endl;
        // cout << "new adhesion: " << i << " " << j << " " << adh[i][j] <<
        // endl; cout << "old adhesion: " << adh_r_pos[ind] << " " <<
        // adh_c_pos[ind] << " " <<
        //                             adh[adh_r_pos[ind]][adh_c_pos[ind]] <<
        //                             endl;
        check_set = true;
      }
    }
  }
  // cout << "adh positions after rearrangement: " << endl;
  // for (int k = 0; k < adh_num; k++) {
  //     cout << "   " << adh_r_pos[k] << " " << adh_c_pos[k] << " " <<
  //             adh[adh_r_pos[k]][adh_c_pos[k]] << endl;
  // }
  update_adhesion_field_normalized();
}

void Cell::update_adh_positions() {
  int adh_ind = 0;
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      if (adh[i][j] == 1) {
        adh_r_pos[adh_ind] = i;
        adh_c_pos[adh_ind] = j;
        adh_ind++;
      }
    }
  }
  adh_num = adh_ind;
}

void Cell::excite_random_points() {
  // method sets specific value of A (A_excite) in specified number of points
  // (excite_num) in order to satisfy mass concervation law (int(A+I) = const),
  // difference is substracted from I component and distributed evenly over all
  // domain used parameters: excite_num - number of points to excite A_excite -
  // concentration of A to set in excited points

  bool check_set =
      false; // check if excite point was set on propper place (inside cell)
  int rn = 0;
  int i = 0;
  int j = 0;
  double A_corr = 0;

  for (int k = 1; k <= excite_num; k++) {
    check_set = false;
    while (not check_set) {
      rn = (int)((double)rand() / RAND_MAX * fr_rows_num * fr_cols_num);
      i = rn / fr_cols_num + fr_rows_pos;
      j = rn % fr_cols_num + fr_cols_pos;
      if (Im[i][j] == 1 and A[i][j] != A_excite) {
        A_corr += A_excite - A[i][j];
        A[i][j] = A_excite;
        check_set = true;
      }
    }
  }

  A_corr = A_corr / (V - excite_num);
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      if (Im[i][j] == 1 and A[i][j] != A_excite) {
        A[i][j] -= A_corr;
      }
    }
  }
}

void Cell::adjust_frame() {
  int fr_r_min, fr_r_max, fr_c_min, fr_c_max;
  max_min_position_in_frame(Im, fr_rows_num, fr_cols_num, fr_rows_pos,
                            fr_cols_pos, env_rows_num, env_cols_num, fr_r_min,
                            fr_r_max, fr_c_min, fr_c_max);
  fr_rows_pos = fr_rows_pos + fr_r_min - fr_dist;
  if (fr_rows_pos < 0)
    fr_rows_pos = 0;

  fr_rows_num = fr_r_max - fr_r_min + 1 * 2 * fr_dist;
  if (fr_rows_pos + fr_rows_num > env_rows_num)
    fr_rows_num = env_rows_num - fr_rows_pos;

  fr_cols_pos = fr_cols_pos + fr_c_min - fr_dist;
  if (fr_cols_pos < 0)
    fr_cols_pos = 0;

  fr_cols_num = fr_c_max - fr_c_min + 1 * 2 * fr_dist;
  if (fr_cols_pos + fr_cols_num > env_cols_num)
    fr_cols_num = env_cols_num - fr_cols_pos;
}

// updated version, include normalization for height of peaks
// normalization function is calculated according to the following formula:
//     sum( w_i / ( (x-x_i)^2 + (y-y_i)^2 ) ) / ( 1 / (x-x_i)^2 + (y-y_i)^2 )
//     summation is done by all adhesions, w_i - value of adhesion field after
//     Gaussian smoothing
void Cell::update_k0_adh() {
  double norm_numer = 0;
  double norm_denom = 0;

#pragma omp parallel for collapse(2) private(norm_numer, norm_denom)
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      if (adh[i][j] != 1) {
        // cout << i << ' ' << j << endl;
        norm_numer = 0;
        norm_denom = 0;
        for (int k = 0; k < adh_num; k++) {
          // cout << "adh pos: " << adh_r_pos[k] << ' ' << adh_c_pos[k] << endl;
          // cout << "adh_g: " << adh_g[adh_r_pos[k]][adh_c_pos[k]] << endl;
          norm_numer += adh_g[adh_r_pos[k]][adh_c_pos[k]] /
                        (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          norm_denom +=
              1 / (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          // cout << "numer denom: " << norm_numer << ' ' << norm_denom << endl
          // << endl;
        }
        // cout << norm_numer << ' ' << norm_denom << endl;
        k0_adh[i][j] =
            (k0 - k0_min) * adh_g[i][j] / norm_numer * norm_denom + k0_min;
      } else {
        k0_adh[i][j] = k0;
      }
      if (k0_adh[i][j] != k0_adh[i][j]) {
        // cout << "nan: " << i << " " << j << " " << k0_adh[i][j] << endl;
        // cout << "position: " << i << " " << j << endl;
        // cout << "adh positions" << endl;
        // for (int ii = 0; ii < adh_num; ii++){
        //     cout << " " << adh_r_pos[ii] << " " << adh_c_pos[ii] << " " <<
        //             adh[adh_r_pos[ii]][adh_c_pos[ii]] << endl;
        // }
        norm_numer = 0;
        norm_denom = 0;
        for (int k = 0; k < adh_num; k++) {
          // cout << "k: " << k << endl;
          // cout << "adh pos: " << adh_r_pos[k] << ' ' << adh_c_pos[k] << endl;
          // cout << "adh[i][j]: " << adh[i][j] <<
          // "adh[adh_r_pos[k]][adh_c_pos[k]]: " <<
          // adh[adh_r_pos[k]][adh_c_pos[k]] << endl; cout << "adh_g: " <<
          // adh_g[adh_r_pos[k]][adh_c_pos[k]] << endl; cout << "numer denom
          // before: " << norm_numer << ' ' << norm_denom << endl;
          norm_numer += adh_g[adh_r_pos[k]][adh_c_pos[k]] /
                        (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          norm_denom +=
              1 / (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          // cout << "numer denom after: " << norm_numer << ' ' << norm_denom <<
          // endl; cout << adh_g[adh_r_pos[k]][adh_c_pos[k]] << endl; cout <<
          // (pow(i-adh_r_pos[k], 2) + pow(j-adh_c_pos[k], 2)) << endl; cout <<
          // 1/(pow(i-adh_r_pos[k], 2) + pow(j-adh_c_pos[k], 2)) << endl; cout
          // <<  adh_g[adh_r_pos[k]][adh_c_pos[k]] /
          //               (pow(i-adh_r_pos[k], 2) + pow(j-adh_c_pos[k], 2)) <<
          //               endl << endl;
        }
      }
    }
  }
}

void Cell::update_k0_adh_new(double scalar) {
  double norm_numer = 0;
  double norm_denom = 0;

#pragma omp parallel for collapse(2) private(norm_numer, norm_denom)
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      if (adh[i][j] != 1) {
        // cout << i << ' ' << j << endl;
        norm_numer = 0;
        norm_denom = 0;
        for (int k = 0; k < adh_num; k++) {
          // cout << "adh pos: " << adh_r_pos[k] << ' ' << adh_c_pos[k] << endl;
          // cout << "adh_g: " << adh_g[adh_r_pos[k]][adh_c_pos[k]] << endl;
          norm_numer += adh_g[adh_r_pos[k]][adh_c_pos[k]] /
                        (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          norm_denom +=
              1 / (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          // cout << "numer denom: " << norm_numer << ' ' << norm_denom << endl
          // << endl;
        }
        // cout << norm_numer << ' ' << norm_denom << endl;
        k0_adh[i][j] =
            (k0 - k0_min) * scalar * adh_g[i][j] / norm_numer * norm_denom +
            k0_min; //********ADDED SCALAR TO INCREASE EFFECT OF ADHESION ON
                    // LOCAL K0
      } else {
        k0_adh[i][j] = k0;
      }
      if (k0_adh[i][j] != k0_adh[i][j]) {
        // cout << "nan: " << i << " " << j << " " << k0_adh[i][j] << endl;
        // cout << "position: " << i << " " << j << endl;
        // cout << "adh positions" << endl;
        // for (int ii = 0; ii < adh_num; ii++){
        //     cout << " " << adh_r_pos[ii] << " " << adh_c_pos[ii] << " " <<
        //             adh[adh_r_pos[ii]][adh_c_pos[ii]] << endl;
        // }
        norm_numer = 0;
        norm_denom = 0;
        for (int k = 0; k < adh_num; k++) {
          // cout << "k: " << k << endl;
          // cout << "adh pos: " << adh_r_pos[k] << ' ' << adh_c_pos[k] << endl;
          // cout << "adh[i][j]: " << adh[i][j] <<
          // "adh[adh_r_pos[k]][adh_c_pos[k]]: " <<
          // adh[adh_r_pos[k]][adh_c_pos[k]] << endl; cout << "adh_g: " <<
          // adh_g[adh_r_pos[k]][adh_c_pos[k]] << endl; cout << "numer denom
          // before: " << norm_numer << ' ' << norm_denom << endl;
          norm_numer += adh_g[adh_r_pos[k]][adh_c_pos[k]] /
                        (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          norm_denom +=
              1 / (pow(i - adh_r_pos[k], 2) + pow(j - adh_c_pos[k], 2));
          // cout << "numer denom after: " << norm_numer << ' ' << norm_denom <<
          // endl; cout << adh_g[adh_r_pos[k]][adh_c_pos[k]] << endl; cout <<
          // (pow(i-adh_r_pos[k], 2) + pow(j-adh_c_pos[k], 2)) << endl; cout <<
          // 1/(pow(i-adh_r_pos[k], 2) + pow(j-adh_c_pos[k], 2)) << endl; cout
          // <<  adh_g[adh_r_pos[k]][adh_c_pos[k]] /
          //               (pow(i-adh_r_pos[k], 2) + pow(j-adh_c_pos[k], 2)) <<
          //               endl << endl;
        }
      }
    }
  }
}

void Cell::add_CoM_to_track(int value) {
  // methods calculates center of mass of the current position of the cell in
  // environment matrix and add calculated point to CoM_track matrix

  int r_CoM = 0; // position of the row
  int c_CoM = 0; // poeition of the column
  for (int i = fr_rows_pos; i < (fr_rows_pos + fr_rows_num); i++) {
    for (int j = fr_cols_pos; j < (fr_cols_pos + fr_cols_num); j++) {
      if (Im[i][j] == 1) {
        r_CoM += i;
        c_CoM += j;
      }
    }
  }
  r_CoM = round((double)r_CoM / V);
  c_CoM = round((double)c_CoM / V);
  CoM_track[r_CoM][c_CoM] = value;
}

void Cell::save_parameters_configuration(string file_name) {
  ofstream output(file_name.c_str());
  output << "scalar for k_adh: " << scalar << endl;
  output << "diff_t: " << diff_t << endl;
  output << "fr_rows_num: " << fr_rows_num << endl;
  output << "fr_cols_num: " << fr_cols_num << endl;
  output << "fr_rows_pos: " << fr_rows_pos << endl;
  output << "fr_cols_pos: " << fr_cols_pos << endl;
  output << "env_rows_num: " << env_rows_num << endl;
  output << "env_cols_num: " << env_cols_num << endl;
  output << "V0: " << V0 << endl;
  output << "V: " << V << endl;
  output << "T: " << T << endl;
  output << "k: " << k << endl;
  output << "V0_nuc: " << V0_nuc << endl;
  output << "V_nuc: " << V_nuc << endl;
  output << "T_nuc: " << T_nuc << endl;
  output << "k_nuc: " << k_nuc << endl;
  output << "R_nuc: " << R_nuc << endl;
  output << "R0: " << R0 << endl;
  output << "d_basal: " << d_basal << endl;
  output << "g: " << g << endl;
  output << "act_slope: " << act_slope << endl;
  output << "DA: " << DA << endl;
  output << "DI: " << DI << endl;
  output << "k0: " << k0 << endl;
  output << "s1: " << s1 << endl;
  output << "s2: " << s2 << endl;
  output << "A0: " << A0 << endl;
  output << "F0: " << F0 << endl;
  output << "gamma: " << gamma << endl;
  output << "delta: " << delta << endl;
  output << "kn: " << kn << endl;
  output << "ks: " << ks << endl;
  output << "eps: " << eps << endl;
  output << "dt: " << dt << endl;
  output << "dx: " << dx << endl;
  output << "adh_num: " << adh_num << endl;
  output << "adh_frac: " << adh_frac << endl;
  output << "adh_sigma: " << adh_sigma << endl;
  output << "fr_dist: " << fr_dist << endl;
  output.close();
}
