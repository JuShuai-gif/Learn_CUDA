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
const int tile_width = 16;
const float A_val = 1.0f;
const float B_val = 2.0f;

__global__ void native_gemm(float *matA, float *matB, float *matC, int width)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    float temp = 0;
    if (idx < width && idy < width)
    {
        for (size_t i = 0; i < width; ++i)
        {
            temp += matA[idy * width + idx] * matB[idx * width + idy];
        }
        matC[idy * width + idx] = temp;
    }
}

__global__ void shared_gemm(float *matA, float *matB, float *matC, int width)
{

    __shared__ float subTileM[tile_width][tile_width];
    __shared__ float subTileN[tile_width][tile_width];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    float temp = 0;

    for (int m = 0; m < (width - 1) / tile_width + 1; ++m)
    {
        subTileM[threadIdx.x][threadIdx.y] = matA[(m * tile_width + threadIdx.y) * width + idx];
        subTileN[threadIdx.x][threadIdx.y] = matB[(m * tile_width + threadIdx.x) * width + idy];

        __syncthreads();

        for (int k = 0; k < tile_width; ++k)
        {
            temp += subTileM[threadIdx.x][k] * subTileN[k][threadIdx.y];
            __syncthreads();
        }
    }
    matC[idy * width + idx] = temp;
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

    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // block_size: 16
    dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions
    // block.x - 1 确保任何不足一个线程块大小的数据区域也会分配到一个额外的线程块
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
    shared_gemm<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", t2sum);

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