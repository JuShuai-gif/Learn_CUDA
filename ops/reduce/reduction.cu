#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

int reduce(const int *x, const int N)
{
    int sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += x[i];
    }
    printf("sum: %d\n", sum);
    return sum;
}

int reduceRec(int *x, const int N)
{
    if (N == 1)
        return x[0];
    int stride = N / 2;
    // 奇数情况
    if (N % 2 == 1)
    {
        for (size_t i = 0; i < stride; ++i)
        {
            x[i] += x[i + stride];
        }
        x[0] += x[N - 1];
    }
    else
    { // 偶数情况
        for (size_t i = 0; i < stride; ++i)
        {
            x[i] += x[i + stride];
        }
    }
    return reduceRec(x, stride);
}

int reduceRecUnroll4(int *x, const int N)
{
    if (N == 1)
        return x[0];
    int stride = N / 4;
    // 奇数情况
    if (N % 2 == 1)
    {
        for (size_t i = 0; i < stride; ++i)
        {
            x[i] += x[i + stride];
            x[i] += x[i + stride * 2];
            x[i] += x[i + stride * 3];
        }
        x[0] += x[N - 1];
    }
    else
    { // 偶数情况
        for (size_t i = 0; i < stride; ++i)
        {
            x[i] += x[i + stride];
            x[i] += x[i + stride * 2];
            x[i] += x[i + stride * 3];
        }
    }
    return reduceRec(x, stride);
}

void __global__ reduce_global(int *dx, int *dy)
{
    const int tid = threadIdx.x;
    int *x = dx + blockDim.x * blockIdx.x;
    // offset初始设置为blockDim.x的一半
    // 每次offset往右平移一位，也就是除以2
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        // 同步块内线程
        __syncthreads();
    }

    if (tid == 0)
    {
        dy[blockIdx.x] = x[0];
    }
}

// GPU--归约  静态共享内存
void __global__ reduce_shared(int *dx, int *dy, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 分配共享内存就是128
    __shared__ int s_y[128];

    s_y[tid] = (tid < N) ? dx[idx] : 0.0;

    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        dy[blockIdx.x] = s_y[0];
    }
}

// GPU -- 归约  动态共享内存
void __global__ reduce_dynamic(int *dx, int *dy, int N)
{
    // 块内线程的索引
    const int tid = threadIdx.x;
    // 每个块的索引
    const int bid = blockIdx.x;

    const int n = bid * blockDim.x + tid;

    extern __shared__ int s_y[];
    s_y[tid] = (n < N) ? dx[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        dy[bid] = s_y[0];
}

int main(int argc, char **argv)
{
    // 初始化设备
    initDevice(0);

    bool bResult = false;

    // 初始化长度
    int size = 1 << 24;
    printf("	with array size %d  \n", size);

    // 执行配置
    int blocksize = 128;
    if (argc > 1)
    {
        blocksize = atoi(argv[1]);
    }
    // 线程数
    dim3 block(blocksize, 1);
    // 线程块个数
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("grid %d block %d \n", grid.x, block.x);

    // 分配主机端内存
    size_t bytes = size * sizeof(int);
    // 分配所有数据内存
    int *idata_host = (int *)malloc(bytes);
    // 只分配块个数内存
    int *odata_host = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // 初始化数组
    initialData_int(idata_host, size);

    memcpy(tmp, idata_host, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // 分配设备内存
    int *idata_dev = NULL;
    int *odata_dev = NULL;
    CHECK(cudaMalloc((void **)&idata_dev, bytes));
    CHECK(cudaMalloc((void **)&odata_dev, grid.x * sizeof(int)));

    // CPU 端归约 -- 朴素
    int cpu_sum = 0;
    iStart = cpuSecond();
    // cpu_sum = reduce(tmp, size);
    for (int i = 0; i < size; i++)
        cpu_sum += tmp[i];
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce           elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // CPU 端归约 -- 递归
    cpu_sum = 0;
    iStart = cpuSecond();
    cpu_sum = reduceRec(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce           elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // CPU 端归约 -- 递归,4层
    cpu_sum = 0;
    iStart = cpuSecond();
    cpu_sum = reduceRecUnroll4(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce           elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // GPU实现归约
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    //reduce_global<<<grid, block>>>(idata_dev, odata_dev);

    reduce_shared<<<grid, block>>>(idata_dev, odata_dev, size);

    // reduce_dynamic<<<grid, block>>>(idata_dev, odata_dev,size);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    cpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        cpu_sum += odata_host[i];
    printf("gpu reduce           elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    free(idata_host);
    free(odata_host);
    CHECK(cudaFree(idata_dev));
    CHECK(cudaFree(odata_dev));
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
