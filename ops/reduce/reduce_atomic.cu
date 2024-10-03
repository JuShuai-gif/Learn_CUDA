#include "error.cuh"
#include <stdio.h>

const int NUM_REPEATS = 100;
const int N = 10000000;
const int M = sizeof(float) * N;
const int BLOCK_SIZE = 128;

void __global__ reduce(const float *d_x, float *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    // 本来应该这样写
    /*
    但是这样写会造成一个问题，就是线程冲突了，每个块的第一个线程都会访问d_y[0]
    且它们的执行次序是不确定的。在每一个线程中，该语句可以分解为两个操作：首先从
    d_y[0]中取数据并与s_y[0]相加，然后将结果写入d_y[0].不管次序如何，只有当一个
    线程的“读-写”操作不被其它线程干扰时，才能得到正确的结果。如果一个线程还未将结果
    写入d_y[0]，另一个线程就读取了d_y[0],那么这两个线程读取的d_y[0]就是一样的，
    这必将导致错误的结果。
    */
    //if (tid == 0)
    //{
    //    d_y[0] += s_y[0];
    //}

    if (tid == 0)
    {
        // 原子操作
        atomicAdd(d_y, s_y[0]);
    }
}

float reduce(const float *d_x)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(float) * BLOCK_SIZE;

    float h_y[1] = {0};
    float *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(float)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice));

    reduce<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const float *d_x)
{
    float sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}

int main(void)
{
    float *h_x = (float *)malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    float *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    printf("\nusing atomicAdd:\n");
    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}
