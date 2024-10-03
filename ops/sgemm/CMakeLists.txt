
set( CUDA_CUBLAS_LIBRARY "/usr/local/cuda-12.4/lib64/libcublas.so" )

add_executable(ops_sgemm_sgemm_v1 sgemm_v1.cu)

target_link_libraries(ops_sgemm_sgemm_v1 PRIVATE ${CUDA_CUBLAS_LIBRARY})