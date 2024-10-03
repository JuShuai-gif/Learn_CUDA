#include <cuda_runtime.h>
#include <stdio.h>

/*
变量dev_data只是作为一个标识符存在，并不是device端的全局内存变量地址，
所以不能直接使用cudaMemcpy函数把host上的数据拷贝到device端。
不能直接在host端的代码中使用运算符&对device端的变量进行取地址操作，
因为它只是一个表示device端物理位置的符号。但是在device端可以使用&对它进行取地址
*/
__device__ float devData;

__global__ void checkGlobalVariable()
{
    printf("Device: The value of the global variable is %f\n", devData);
    devData += 2.0;
}

int main()
{
    float value = 3.14f;
    // 将主机变量（value）的数据拷贝到设备符号（devData）
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host: copy %f to the global variable\n", value);
    checkGlobalVariable<<<1, 1>>>();
    // 从设备符号（devData）拷贝数据到主机变量（value）
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f \n", value);
    // 此函数会释放所有CUDA设备的资源，清除当前上下文，并将设备状态重置为初始化状态。
    // 通常在程序结束时使用，以确保资源被正确释放
    cudaDeviceReset();
    return EXIT_SUCCESS;
}