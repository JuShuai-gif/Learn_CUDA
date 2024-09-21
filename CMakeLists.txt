cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(./include)

project(hellocuda LANGUAGES CXX CUDA)
add_subdirectory(0_hello_world)
add_subdirectory(1_check_dimension)
add_subdirectory(2_grid_block)
add_subdirectory(3_sum_arrays)

add_subdirectory(4_sum_arrays_timer)

add_subdirectory(5_thread_index)

add_subdirectory(6_sum_matrix)

add_subdirectory(7_device_information)

add_subdirectory(8_divergence)

add_subdirectory(9_sum_matrix2D)

add_subdirectory(maolei_cuda)

add_subdirectory(PMPP)