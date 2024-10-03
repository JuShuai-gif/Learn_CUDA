#include <stdio.h>
#include <cuda.h>

const int N = 1024; // 数组大小
__shared__ float sharedMem[512]; // 共享内存，大小为512个float

__global__ void bankConflictTest() {
    int idx = threadIdx.x;

    // 测试 bank 冲突：每个线程访问共享内存的相同位置
    // 所有线程都访问 sharedMem[0]
    sharedMem[0] = idx; // 每个线程写入自己的索引
    __syncthreads(); // 等待所有线程完成写入

    // 读取共享内存中的值
    float value = sharedMem[0]; // 读取同一个位置
    //printf("Thread %d read value: %f\n", idx, value);
}

int main() {
    // 启动内核，使用512个线程
    bankConflictTest<<<1, 512>>>();
    cudaDeviceSynchronize(); // 等待内核完成

    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
