add_library(dynein_cell_model dynein_cell_model.cpp)

# Include directories
target_include_directories(dynein_cell_model PUBLIC include)

# -------- CURL (for HDF5 system package) --------
find_package(CURL REQUIRED)
target_link_libraries(dynein_cell_model PRIVATE CURL::libcurl)

# -------- OpenMP (if available) --------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dynein_cell_model PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(dynein_cell_model PUBLIC USE_OPENMP)
    message(STATUS "OpenMP found - parallelization enabled")
else()
    message(STATUS "OpenMP not found - running in single-threaded mode")
endif()

# -------- HDF5 (system package) --------
find_package(HDF5 REQUIRED COMPONENTS CXX)
message(STATUS "HDF5 include path: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")
target_include_directories(dynein_cell_model PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(dynein_cell_model PRIVATE ${HDF5_LIBRARIES})

# -------- Eigen3 (FetchContent) --------
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

# -------- OpenCV (system package) --------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
target_include_directories(dynein_cell_model PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(dynein_cell_model PRIVATE ${OpenCV_LIBS})

# -------- yaml-cpp (FetchContent) --------
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
target_link_libraries(dynein_cell_model PUBLIC yaml-cpp::yaml-cpp)

# -------- HighFive (FetchContent) --------
if(NOT TARGET HighFive::HighFive)
    message(STATUS "HighFive not found, fetching with FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        HighFive
        GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
        GIT_TAG v3.3.0
    )
    FetchContent_MakeAvailable(HighFive)
endif()
target_link_libraries(dynein_cell_model PUBLIC HighFive)

# -------- Local Libraries --------
target_link_libraries(dynein_cell_model PRIVATE metric_utils test_utils)
