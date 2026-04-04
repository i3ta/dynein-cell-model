cmake_minimum_required(VERSION 3.15...3.31)

project(DyneinCellModel
        VERSION 0.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Define flags for high-performance math and Eigen
set(PERF_FLAGS 
    -O3 
    -ffast-math 
    -flto 
    -DEIGEN_NO_DEBUG 
    -DNDEBUG
)

# Detect Apple Silicon to use the correct architecture flag
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    list(APPEND PERF_FLAGS -mcpu=apple-m1)
else()
    list(APPEND PERF_FLAGS -march=native)
endif()

# Apply these flags only when building in Release or RelWithDebInfo modes
add_compile_options("$<$<CONFIG:Release>:${PERF_FLAGS}>")
