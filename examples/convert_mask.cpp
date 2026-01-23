#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include <metric_utils/metric_utils.hpp>

void text_to_image(const std::string &file, const std::string &output_path);

int main(int argc, char *argv[]) {
  // timer for metrics
  metrics::ScopedTimer timer("text_to_image");

  if (argc != 3) {
    std::cerr << "Expected 2 arguments, found " << argc - 1 << std::endl;
    return 1;
  }

  std::string text_mask = argv[1];
  std::string output_img = argv[2];

  text_to_image(text_mask, output_img);
}

/**
 * @brief Read a binary (0/1) matrix from a text file and output a black–white
 * image.
 *
 * File format:
 *   First number: number of rows
 *   Second number: number of columns
 *   Then: row-major order matrix entries (0 or 1)
 */
void text_to_image(const std::string &file, const std::string &output_path) {
  int rows_num = 0, cols_num = 0;
  std::ifstream m_file(file);
  if (!m_file.is_open()) {
    std::cerr << "Error: Cannot open file " << file << std::endl;
    return;
  }

  // Read dimensions
  m_file >> rows_num >> cols_num;
  if (rows_num <= 0 || cols_num <= 0) {
    std::cerr << "Invalid matrix dimensions!" << std::endl;
    return;
  }

  // Create 8-bit grayscale image
  cv::Mat image(rows_num, cols_num, CV_8UC1);

  int val;
  for (int r = 0; r < rows_num; ++r) {
    for (int c = 0; c < cols_num; ++c) {
      if (!(m_file >> val)) {
        std::cerr << "Error: Not enough data in file!" << std::endl;
        return;
      }

      image.at<uchar>(r, c) = (val == 1) ? 255 : (val == 2) ? 127 : 0;
    }
  }

  m_file.close();

  // Save image
  if (!cv::imwrite(output_path, image)) {
    std::cerr << "Error: Could not write image to " << output_path << std::endl;
  } else {
    std::cout << "Saved binary image to " << output_path << std::endl;
  }
}
