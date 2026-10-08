#include <cstdio>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "01-vector-add/VectorAdd.cu"
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
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    helloKernel<<<2, 2>>>();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel time: "
              << milliseconds
              << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}