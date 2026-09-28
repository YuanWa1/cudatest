#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

__global__ void helloKernel()
{
    printf(
        "Block %d, Thread %d: Hello from GPU!\n",
        blockIdx.x,
        threadIdx.x
    );
}

int main()
{
    helloKernel<<<2, 2>>>();
    cudaDeviceSynchronize();

    return 0;
}