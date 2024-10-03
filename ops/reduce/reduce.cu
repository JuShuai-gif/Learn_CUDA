#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#include <algorithm>
#include <iostream>
__global__ void reduceNeighbored(float *g_idata, float *g_odata, unsigned int n)
{

    unsigned int tid = threadIdx.x;
    if (tid >= n)
    {
        return;
    }

    float *idata = g_idata + blockDim.x * blockIdx.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void test()
{
    printf("test\n");
}

__global__ void reduce_shared(float *g_idata, float *g_odata, unsigned int n)
{
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_y[128];
    s_y[tid] = idx < n ? g_idata[idx] : 0.0;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            s_y[tid] += s_y[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[blockIdx.x] = s_y[0];
    }
}

// 优化技巧一：解决warp divergence
__global__ void reduce1(float *g_idata, float *g_odata, unsigned int n)
{
    __shared__ float sdata[128];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = idx < n ? g_idata[idx] : 0.0f;
    __syncthreads();

    for (size_t s = 1; s < blockDim.x; s << 1)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[tid];
    }
}

// 优化技巧2：解决bank冲突
__global__ void reduce2(float *g_idata, float *g_odata, unsigned int n)
{
    __shared__ float sdata[128];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = idx < n ? g_idata[idx] : 0.0f;

    __syncthreads();
    // 做这一处改变即可避免bank conflict
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[blockDim.x] = sdata[tid];
    }
}

// 优化技巧3：解决idle线程
__global__ void reduce3(float *g_idata, float *g_odata, unsigned int n)
{
    __shared__ float sdata[128];

    int idx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = g_idata[idx] + g_idata[idx + blockDim.x];

    __syncthreads();
    // 做这一处改变即可避免bank conflict
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_odata[blockDim.x] = sdata[tid];
    }
}

// 优化技巧4：展开最后一步减少同步
__device__ void warpReduce(volatile float *cache, int tid)
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}
__global__ void reduce4(float *g_idata, float *g_odata, unsigned int n)
{
    __shared__ float sdata[128];

    int idx = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = g_idata[idx] + g_idata[idx + blockDim.x];

    __syncthreads();
    // 做这一处改变即可避免bank conflict
    for (int s = blockDim.x >> 1; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[tid];
}

// 优化技巧5：完全展开减少计算
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *cache, int tid)
{
    if (blockSize >= 64)
        cache[tid] += cache[tid + 32];
    if (blockSize >= 32)
        cache[tid] += cache[tid + 16];
    if (blockSize >= 16)
        cache[tid] += cache[tid + 8];
    if (blockSize >= 8)
        cache[tid] += cache[tid + 4];
    if (blockSize >= 4)
        cache[tid] += cache[tid + 2];
    if (blockSize >= 2)
        cache[tid] += cache[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce5(float *d_in, float *d_out)
{
    __shared__ float sdata[128];

    // each thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    if (tid == 0)
        d_out[blockIdx.x] = sdata[tid];
}

// 优化6：使用shuffle指令
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum)
{
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template <unsigned int blockSize>
__global__ void reduce6(float *d_in, float *d_out, unsigned int n)
{
    float sum = 0;

    // each thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    while (i < n)
    {
        sum += d_in[i] + d_in[i + blockSize];
        i += gridSize;
    }

    // shared mem for partial sums(one per warp in the block
    static __shared__ float warpLevelSums[32];
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;

    sum = warpReduceSum<blockSize>(sum);

    if (laneId == 0)
        warpLevelSums[warpId] = sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / 32) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0)
        sum = warpReduceSum<blockSize / 32>(sum);
    // write result for this block to global mem
    if (tid == 0)
        d_out[blockIdx.x] = sum;
}

// PyTorch BlockReduce + Pack + 选择更更合理的 GridSize
#define PackSize 4
#define kWarpSize 32
#define N 32 * 1024 * 1024
constexpr int BLOCK_SIZE = 256;

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 1;

int64_t GetNumBlocks(int64_t n)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int64_t num_blocks = std::max<int64_t>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                                sm_count * tpm / kBlockSize * kNumWaves));
    return num_blocks;
}

template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed(T val)
    {
#pragma unroll
        for (int i = 0; i < pack_size; i++)
        {
            elem[i] = val;
        }
    }
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        T elem[pack_size];
    };
    __device__ void operator+=(Packed<T, pack_size> packA)
    {
#pragma unroll
        for (int i = 0; i < pack_size; i++)
        {
            elem[i] += packA.elem[i];
        }
    }
};

template <typename T, int pack_size>
__device__ T PackReduce(Packed<T, pack_size> pack)
{
    T res = 0.0;
#pragma unroll
    for (int i = 0; i < pack_size; i++)
    {
        res += pack.elem[i];
    }
    return res;
}

template <typename T>
__device__ T warpReduceSum(T val)
{
    for (int lane_mask = 16; lane_mask > 0; lane_mask /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, lane_mask);
    }
    return val;
}

__global__ void reduce_v8(float *g_idata, float *g_odata, unsigned int n)
{

    // each thread loads one element from global to shared mem

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    Packed<float, PackSize> sum_pack(0.0);
    Packed<float, PackSize> load_pack(0.0);
    const auto *pack_ptr = reinterpret_cast<const Packed<float, PackSize> *>(g_idata);

    for (int32_t linear_index = i; linear_index < n / PackSize; linear_index += blockDim.x * gridDim.x)
    {
        Packed<float, PackSize> g_idata_load = pack_ptr[linear_index];
        sum_pack += g_idata_load;
    }
    float PackReduceVal = PackReduce<float, PackSize>(sum_pack);
    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[kWarpSize];
    const int laneId = threadIdx.x % kWarpSize;
    const int warpId = threadIdx.x / kWarpSize;

    float sum = warpReduceSum<float>(PackReduceVal);
    __syncthreads();

    if (laneId == 0)
        warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / kWarpSize) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0)
        sum = warpReduceSum<float>(sum);
    // write result for this block to global mem
    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = sum;
}

int main(int argc, char **argv)
{
    // 初始化设备
    initDevice(0);

    bool bResult = false;

    int size = 1 << 20;
    printf("	with array size %d  \n", size);

    // 执行配置
    int blocksize = 128;

    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("grid: %d block: %d \n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *idata_host = (float *)malloc(bytes);
    float *odata_host = (float *)malloc(grid.x * sizeof(float));
    float *tmp = (float *)malloc(bytes);

    // 初始化数组
    for (int n = 0; n < size; ++n)
    {
        idata_host[n] = 1.23f;
    }

    // 将idata_host 复制到tmp
    memcpy(tmp, idata_host, bytes);
    double iStart, iElaps;
    double gpu_sum = 0.0f;

    // 设备内存
    float *idata_dev = NULL;
    float *odata_dev = NULL;
    CHECK(cudaMalloc((void **)&idata_dev, bytes));
    CHECK(cudaMalloc((void **)&odata_dev, grid.x * sizeof(float)));

    // cpu归约
    double cpu_sum = 0.0f;
    iStart = cpuSecond();
    for (int i = 0; i < size; i++)
        cpu_sum += idata_host[i];
    printf("cpu sum:%f \n", cpu_sum);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce   elapsed %lf ms cpu_sum: %f\n", iElaps, cpu_sum);

    printf("bytes: %d\n", bytes);
    // 热启动GPU
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighbored  elapsed %lf ms gpu_sum: %f<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // 热启动GPU
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduce_shared<<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduce_shared  elapsed %lf ms gpu_sum: %f<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // 释放主机端内存
    free(idata_host);
    free(odata_host);
    // 释放设备端内存
    CHECK(cudaFree(idata_dev));
    CHECK(cudaFree(odata_dev));

    // 重置设备
    cudaDeviceReset();

    // 检查结果
    if (gpu_sum == cpu_sum)
    {
        printf("Test success!\n");
    }
    return EXIT_SUCCESS;
}