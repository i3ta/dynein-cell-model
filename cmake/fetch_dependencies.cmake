# fetch_dependencies.cmake
# Centralized FetchContent for all third-party dependencies
# System packages (HDF5, OpenCV) should be installed separately

include(FetchContent)

# -------- Eigen3 --------
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

# -------- yaml-cpp --------
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.9.0
)

# -------- HighFive --------
FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG v3.3.0
)

# -------- nlohmann_json --------
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# Fetch all dependencies
# Note: System packages (HDF5, OpenCV) should be installed separately
FetchContent_MakeAvailable(eigen yaml-cpp HighFive nlohmann_json)
