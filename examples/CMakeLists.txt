file(GLOB TEST_SOURCES "*.cpp")

find_package(nlohmann_json REQUIRED)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})

    target_link_libraries(${test_name}
        PRIVATE
            dynein_cell_model
            metric_utils
            test_utils
            tqdm
            nlohmann_json::nlohmann_json
            yaml-cpp
    )
endforeach()

file(COPY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION ${CMAKE_BINARY_DIR}/examples)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/examples/output)
