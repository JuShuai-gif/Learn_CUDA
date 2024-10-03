#include <cuda_runtime.h>
#include <stdio.h>
#include "../../include/freshman.h"

#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16

#define IPAD 1

__global__ void warmup(int *out){
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.x;
}


