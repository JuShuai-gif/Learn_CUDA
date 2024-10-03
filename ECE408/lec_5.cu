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
    //A[0]++;
    atomicAdd(A,1.0);
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
    vadd<<<8, 8>>>(d_A);

    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete
    //  copy vector C from device to host:
    cudaMemcpy(h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    printf("A[0] = %f\n", h_A[0]);
    return 0;
}
