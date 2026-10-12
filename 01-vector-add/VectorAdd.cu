//
// Created by jxmwa on 10/7/2026.
//
#include <cuda_runtime.h>
#include <iostream>

#include "timer.cuh"

#include <random>
#include <vector>
constexpr int N = 100000000;

void random_gen(std::vector<float> &A, std::vector<float> &B) {
    std::mt19937 rng(42);

    std::uniform_real_distribution<float> dist(-10000.0f, 10000.0f);

    for (int i = 0; i < N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }
}

__global__ void vector_add_kernel(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vector_add_grid_stride(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

void vector_add_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

void vector_add() {
    std::vector<float> A(N), B(N), C(N);

    random_gen(A, B);

    std::vector<float> A1(N), B1(N), C1(N);

    float *d_A, *d_B, *d_C;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);


    //
    BenchResult result_gpu = benchmark_gpu([&]() {

        vector_add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    });

    std::cout << "GPU1 Benchmark:\n";
    std::cout << "Min: " << result_gpu.min_ms << " ms, "
              << "Median: " << result_gpu.median_ms << " ms, "
              << "Mean: " << result_gpu.mean_ms << " ms\n";

    size_t total_bytes = 3ULL * N * sizeof(float);
    double bw = bandwidth_GBps(total_bytes, result_gpu.median_ms);
    std::cout << "Achieved Bandwidth: " << bw << " GB/s ("
              << bandwidth_utilization(bw) << "% of theoretical peak)\n";


    BenchResult result_gpu_2 = benchmark_gpu([&]() {
       vector_add_grid_stride<<<216, blockSize>>>(d_A, d_B, d_C, N);
    });

    std::cout << "GPU2 Benchmark:\n";
    std::cout << "Min: " << result_gpu_2.min_ms << " ms, "
              << "Median: " << result_gpu_2.median_ms << " ms, "
              << "Mean: " << result_gpu_2.mean_ms << " ms\n";


    total_bytes = 3ULL * N * sizeof(float);
    bw = bandwidth_GBps(total_bytes, result_gpu_2.median_ms);
    std::cout << "Achieved Bandwidth: " << bw << " GB/s ("
              << bandwidth_utilization(bw) << "% of theoretical peak)\n";


    cudaMemcpy(C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}