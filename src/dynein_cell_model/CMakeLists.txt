add_library(dynein_cell_model dynein_cell_model.cpp)

# Include directories
target_include_directories(dynein_cell_model PUBLIC include)

# -------- OpenMP --------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dynein_cell_model PUBLIC OpenMP::OpenMP_CXX)
endif()

# -------- Eigen3 --------
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found, fetching with FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)
  target_link_libraries(dynein_cell_model PUBLIC eigen)
else()
  target_link_libraries(dynein_cell_model PUBLIC Eigen3::Eigen)
endif()

# -------- OpenCV --------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
target_include_directories(dynein_cell_model PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(dynein_cell_model PRIVATE ${OpenCV_LIBS})

# -------- yaml-cpp --------
if(NOT TARGET yaml-cpp)
    find_package(yaml-cpp QUIET)
    if(NOT yaml-cpp_FOUND)
        message(STATUS "yaml-cpp not found, fetching with FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
            GIT_TAG 0.8.0
        )
        FetchContent_MakeAvailable(yaml-cpp)
    endif()
endif()

# Link the target. FetchContent defines the 'yaml-cpp' target.
target_link_libraries(dynein_cell_model PUBLIC yaml-cpp)

# -------- HighFive --------
if(NOT TARGET HighFive::HighFive)
    message(STATUS "HighFive not found, fetching with FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      HighFive
      GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
      GIT_TAG        v3.3.0
    )
    FetchContent_MakeAvailable(HighFive)
endif()
target_link_libraries(dynein_cell_model PUBLIC HighFive)

# -------- Local Libraries --------
target_link_libraries(dynein_cell_model PRIVATE metric_utils test_utils)
