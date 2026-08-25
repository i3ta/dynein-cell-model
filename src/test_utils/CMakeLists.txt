add_library(test_utils INTERFACE)

target_include_directories(test_utils INTERFACE include)
target_link_libraries(test_utils INTERFACE
    dynein_cell_model
    cell_nuc
)
