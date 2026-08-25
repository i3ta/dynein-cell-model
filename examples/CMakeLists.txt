file(GLOB TEST_SOURCES "*.cpp")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})

    target_link_libraries(${test_name}
        PRIVATE
            CURL::libcurl
            dynein_cell_model
            metric_utils
            test_utils
            tqdm
    )
endforeach()

file(COPY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION ${CMAKE_BINARY_DIR}/examples)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/examples/output)
