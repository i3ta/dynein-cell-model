add_library(test_utils test_utils.cpp)
find_package(GTest REQUIRED)

target_include_directories(test_utils PUBLIC include)
target_link_libraries(test_utils PRIVATE 
    dynein_cell_model
    GTest::gtest
    GTest::gtest_main
)
