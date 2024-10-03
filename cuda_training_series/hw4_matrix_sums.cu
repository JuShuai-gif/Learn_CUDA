#include <stdio.h>

// error checking macro
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

const size_t DSIZE = 16384;
const int block_size = 256;

// 行求和
__global__ void row_sums(const float *A, float *sums, size_t ds)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    printf("idx: %d,idy: %d\n",idx);
    if (idx < ds)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < ds; ++i)
        {
            sum += A[i + idx * DSIZE];
        }
        sums[idx] = sum;
    }
}

// 列求和
__global__ void column_sums(const float *A, float *sums, size_t ds)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < ds)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < ds; ++i)
        {
            sum += A[idx + i * ds];
        }
        sums[idx] = sum;
    }
}

bool validate(float *data, size_t sz)
{
    for (size_t i = 0; i < sz; i++)
        if (data[i] != (float)sz)
        {
            printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz);
            return false;
        }
    return true;
}

int main()
{
    float *h_A, *h_sums, *d_A, *d_sums;
    // 分配数组，并没有初始化这些元素，数组中的值是未定义的
    h_A = new float[DSIZE * DSIZE];
    // 分配空间，并将数组中的所有元素初始化为0
    h_sums = new float[DSIZE]();

    for (int i = 0; i < DSIZE * DSIZE; ++i)
    {
        h_A[i] = 1.0f;
    }

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_sums, DSIZE * sizeof(float));

    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // 每个block分256个线程，总共64个block
    // printf("xxx %d\n",(DSIZE + block_size - 1) / block_size);
    row_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE))
        return -1;
    printf("row sums correct!\n");

    cudaMemset(d_sums, 0, DSIZE * sizeof(float));

    column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete

    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    if (!validate(h_sums, DSIZE))
        return -1;
    printf("column sums correct!\n");
    return 0;
}
