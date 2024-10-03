#include <stdio.h>

#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FATLED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

const int DSIZE = 4096;
const int block_size = 256;

__global__ void vadd(float *A)
{
    int i_1{0},i_2{0};
    i_1 = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    i_2 = i_1 + blockDim.x;
    printf("i_1:%d i_2:%d\n",i_1,i_2);

}

__global__ void if_div(){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i % 2 == 0)
    {
        printf("idx: %d",i);
    }else{
        printf("idx: %d",i);
    }
    
}

int main()
{
    float *h_A, *d_A;
    h_A = new float[1];

    h_A[0] = 1;

    cudaMalloc(&d_A, sizeof(float));

    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy H2D failure");
    //vadd<<<2, 32>>>(d_A);
    if_div<<<2,32>>>();
    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete
    //  copy vector C from device to host:
    cudaMemcpy(h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    printf("A[0] = %f\n", h_A[0]);
    return 0;
}
