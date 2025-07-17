add_library(dynein_cell_model dynein_cell_model.cpp)

find_package (Eigen3 3.4 REQUIRED)

target_include_directories(dynein_cell_model PUBLIC include)
target_link_libraries(dynein_cell_model Eigen3::Eigen)
