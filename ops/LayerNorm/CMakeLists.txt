#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
#SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
#SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})

set( CUDA_CUBLAS_LIBRARY "/usr/local/cuda-12.4/lib64/libcublas.so" )
set( CUDA_cusparse_LIBRARY "/usr/local/cuda-12.4/lib64/libcusparse.so" )
set( CUDA_cublasLt_LIBRARY "/usr/local/cuda-12.4/lib64/libcublasLt.so" )

message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")

add_executable(ops_LayerNorm_layernorm_forward layernorm_forward.cu)

target_link_libraries(ops_LayerNorm_layernorm_forward PRIVATE 
    ${CUDA_LIBRARIES} 
    ${CUDA_cusparse_LIBRARY} 
    ${CUDA_CUBLAS_LIBRARY}
    ${CUDA_cublasLt_LIBRARY}
)