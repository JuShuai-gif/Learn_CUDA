#include <stdio.h>

__global__ void hello()
{
    printf("gridDim: %u,blockDim: %u\n", gridDim.x, blockIdx.x);
    printf("block: %u,thread: %u\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // 第一个参数表示每个维度多少个块  gridDim，第二个参数表示每个维度多少个线程 blockDim
    // 可以是三维的dim
    dim3 grid(2, 1, 1);  // x y z
    dim3 block(2, 2, 2); // x y z
    hello<<<grid, block>>>();
    cudaDeviceSynchronize();
}
