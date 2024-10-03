#include <stdio.h>

// 检查错误
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

const int DSIZE = 30768;

// 向量相加
__global__ void vadd(const float *A, const float *B, float *C, int ds)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // 定义共享内存
    __shared__ float submemA[512];
    __shared__ float submemB[512];

    // 计算每个线程块要处理的数据大小
    int numElementsPerBlock = (ds + gridDim.x - 1) / gridDim.x; // 向上取整
    //printf("numElementsPerBlock %d\n", numElementsPerBlock);
    // 
    int startIdx = blockIdx.x * numElementsPerBlock;
    int endIdx = min(startIdx + numElementsPerBlock, ds);

    for (int i = startIdx + threadIdx.x; i < endIdx; i += blockDim.x)
    {
        submemA[threadIdx.x] = A[i];
        submemB[threadIdx.x] = B[i];
    }
    __syncthreads(); // 确保所有数据加载完成

    // 计算和
    for (int i = startIdx + threadIdx.x; i < endIdx; i += blockDim.x)
    {
        C[i] = submemA[threadIdx.x] + submemB[threadIdx.x];
    }
}

int main()
{
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * sizeof(float));

    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy H2D failure");

    int threads = 512;                            // 每个线程块中的线程数
    int blocks = (DSIZE + threads - 1) / threads; // 计算块数

    vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    for (size_t i = 0; i < DSIZE; i++)
    {
        if (h_C[i] != (h_A[i] + h_B[i]))
        {
            printf("Failure !!! \n");
            printf("i = %d\n", i);
            printf("A[i] = %f\n", h_A[i]);
            printf("B[i] = %f\n", h_B[i]);
            printf("C[i] = %f\n", h_C[i]);
            return 1; // 返回错误码
        }
    }

    printf("Success !!! \n");
    return 0;
}
