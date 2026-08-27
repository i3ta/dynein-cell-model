add_library(dynein_cell_model
    diffusion.cpp
    kokkos_diffusion.cpp
    morphology.cpp
    config.cpp
    state.cpp
    dynein_field.cpp
    io.cpp
    simulate.cpp)

# Include directories
target_include_directories(dynein_cell_model PUBLIC include)

option(DCM_ENABLE_KOKKOS "Build the portable Kokkos diffusion backend" ON)
if(DCM_ENABLE_KOKKOS)
    find_package(Kokkos 4.5...<5.0 CONFIG QUIET)
    if(NOT Kokkos_FOUND)
        message(STATUS "Kokkos not found, fetching Kokkos 4.5.01")
        include(FetchContent)
        # AppleClang's C++23 mdspan headers are not compatible with Kokkos
        # 4.5.x's experimental mdspan integration; this model does not use it.
        set(Kokkos_ENABLE_IMPL_MDSPAN OFF CACHE BOOL
            "Disable Kokkos experimental mdspan support" FORCE)
        FetchContent_Declare(
            Kokkos
            URL https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
            URL_HASH SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c)
        FetchContent_MakeAvailable(Kokkos)
    endif()
    target_link_libraries(dynein_cell_model PUBLIC Kokkos::kokkos)
    target_compile_definitions(dynein_cell_model PUBLIC DCM_ENABLE_KOKKOS=1)
endif()

# -------- CURL (for HDF5 system package) --------
find_package(CURL REQUIRED)
target_link_libraries(dynein_cell_model PRIVATE CURL::libcurl)

# -------- OpenMP (if available) --------
# Apple Clang needs help finding Homebrew's libomp on Apple Silicon
# if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
#     set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
#     set(OpenMP_CXX_LIB_NAMES "omp")
#     set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
#     set(OpenMP_omp_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
# endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dynein_cell_model PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(dynein_cell_model PUBLIC USE_OPENMP)
    # if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    #     target_include_directories(dynein_cell_model PUBLIC /opt/homebrew/opt/libomp/include)
    # endif()
    message(STATUS "OpenMP found - parallelization enabled")
else()
    message(STATUS "OpenMP not found - running in single-threaded mode")
endif()

# -------- HDF5 (system package) --------
find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_include_directories(dynein_cell_model PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(dynein_cell_model PUBLIC HDF5::HDF5)

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
        GIT_TAG 0.9.0
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
