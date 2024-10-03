#include <stdio.h>

__global__ void hello_world(void)
{
    printf("GPU: Hello world!\n");
    printf("threadIdx.x: %d\n",threadIdx.x);
}

int main(int argc, char **argv)
{
    printf("CPU: hello world!\n");
    hello_world<<<1,10>>>();
    //cudaDeviceReset();//如果没有这一句，直接host执行完程序就返回了，不能GPU执行完就直接执行return 0
    return 0;
}