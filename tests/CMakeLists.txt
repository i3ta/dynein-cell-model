file(GLOB TEST_SOURCES "*.cpp")

find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} 
        ${test_src}
        ${CMAKE_SOURCE_DIR}/src/cell_nuc/cell_nuc.cpp
        ${CMAKE_SOURCE_DIR}/src/dynein_cell_model/dynein_cell_model.cpp
    )

    target_compile_definitions(${test_name}
        PRIVATE
            LIB_CELL_NUC_DEBUG
            LIB_DYNEIN_CELL_MODEL_DEBUG
    )

    target_link_libraries(${test_name}
        PRIVATE
            dynein_cell_model
            cell_nuc
            metric_utils
            test_utils
            tqdm
            nlohmann_json::nlohmann_json
            GTest::gtest
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(${test_name})
endforeach()
