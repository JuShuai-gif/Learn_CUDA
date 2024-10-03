#include <stdio.h>

// 检查错误
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

const int DSIZE = 32 * 1048576;
// 向量相加
__global__ void vadd(const float *A, const float *B, float *C, int ds)
{

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds; idx += gridDim.x * blockDim.x)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * sizeof(float));

    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudamemcpy H2D failure");

    int blocks = 1;
    int threads = 1;
    vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);

    cudaCheckErrors("kernel launch failure");
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}
