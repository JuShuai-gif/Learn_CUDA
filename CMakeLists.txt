cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(./include)

project(hellocuda LANGUAGES CXX CUDA)

add_subdirectory(maolei_cuda)

add_subdirectory(PMPP)

add_subdirectory(cuda_training_series)

add_subdirectory(ECE408)

add_subdirectory(cuda_freshman)

add_subdirectory(ops)