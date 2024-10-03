#include <stdio.h>
#include <time.h>

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
const int block_size = 16;

const float A_val = 1.0f;
const float B_val = 2.0f;

// 朴素矩阵相乘
__global__ void mmul(const float *A, const float *B, float *C, int ds)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    //printf("idx %d,idy: %d\n", idx,idy);
    if ((idx < ds) && (idy < ds))
    {
        float temp = 0;
        // ds是宽度
        for (int i = 0; i < ds; ++i)
        {
            temp += A[idy * ds + i] * B[i * ds + idx];
        }
        if (idx == 0 && idy == 0)
        {
            printf("temp %f\n", temp);
        }

        C[idy * ds + idx] = temp;
    }
}

int main()
{
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // 时间记录
    clock_t t0, t1, t2;
    double t1sum = 0.0f;
    double t2sum = 0.0f;

    t0 = clock();

    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    for (size_t i = 0; i < DSIZE * DSIZE; ++i)
    {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Cuda processing sequence step 1 is complete

    // block_size: 16
    dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions
    // block.x - 1 确保任何不足一个线程块大小的数据区域也会分配到一个额外的线程块
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Cuda processing sequence step 2 is complete

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", t2sum);

    // Cuda processing sequence step 3 is complete

    // Verify results
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    for (int i = 0; i < DSIZE * DSIZE; i++)
        if (h_C[i] != A_val * B_val * DSIZE)
        {
            printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
            return -1;
        }
    printf("Success!\n");

    return 0;
}