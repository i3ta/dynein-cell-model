add_library(dynein_cell_model dynein_cell_model.cpp)

find_package (Eigen3 3.4 REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dynein_cell_model PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(dynein_cell_model PUBLIC include)
target_link_libraries(dynein_cell_model Eigen3::Eigen)
