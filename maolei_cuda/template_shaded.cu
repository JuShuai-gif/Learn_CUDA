#include <stdio.h>

__global__ void kernel()
{
    int n = (blockDim.x * blockIdx.x) * 2 + threadIdx.x;
    int m = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    printf("n: %d\n",n);
    printf("m: %d\n",m);
}

int main()
{
    kernel<<<4,2>>>();
    cudaDeviceReset();
    return 0;
}










