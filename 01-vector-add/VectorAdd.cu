//
// Created by jxmwa on 10/7/2026.
//
#include <cuda_runtime.h>
#include "timer.cuh"

#include <random>
#include <vector>
constexpr int N = 1024;

void random_gen(std::vector<float> &A, std::vector<float> &B) {
    std::mt19937 rng(42);

    std::uniform_real_distribution<float> dist(-10000.0f, 10000.0f);

    for (int i = 0; i < N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }
}

__global__ void vector_add_kernel(float *A, float *B, float *C, int n) {
}

void vector_add() {
    std::vector<float> A(N), B(N), C(N);

    random_gen(A, B);

    benchmark_gpu([&]() {
        float *d_A, *d_B, *d_C;

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;

        vector_add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    });
}