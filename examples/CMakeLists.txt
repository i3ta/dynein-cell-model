file(GLOB TEST_SOURCES "*.cpp")

find_package(nlohmann_json 3.12.0 REQUIRED)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/dynein_cell_model/include
            ${CMAKE_SOURCE_DIR}/src/metric_utils/include
    )

    target_link_libraries(${test_name}
        PRIVATE
            dynein_cell_model
            metric_utils
            tqdm
            nlohmann_json::nlohmann_json
    )
endforeach()

file(COPY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION ${CMAKE_BINARY_DIR}/examples)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/examples/output)
