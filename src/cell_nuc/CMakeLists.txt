add_library(cell_nuc cell_nuc.cpp)

target_include_directories(cell_nuc PUBLIC include)
target_link_libraries(cell_nuc PRIVATE test_utils dynein_cell_model)
