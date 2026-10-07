//
// Created by jxmwa on 10/7/2026.
//

#ifndef CUDAGPU_TIMER_CUH
#define CUDAGPU_TIMER_CUH

#include <cuda_runtime.h>

struct GpuTimer
{
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GpuTimer()
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start()
    {
        cudaEventRecord(start_event);
    }

    float stop_ms()
    {
        cudaEventRecord(stop_event);

        // CPU 等待 GPU 真正执行到 stop_event
        cudaEventSynchronize(stop_event);

        float ms = 0.0f;

        cudaEventElapsedTime(
            &ms,
            start_event,
            stop_event
        );

        return ms;
    }
};

struct BenchResult
{
    float min_ms;
    float median_ms;
    float mean_ms;
};


template <class F>
BenchResult benchmark_gpu(F&& launch,
                          int warmup = 5,
                          int reps = 30)
{
    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        launch();
    }

    cudaDeviceSynchronize();

    // Actual benchmark
    std::vector<float> times;
    times.reserve(reps);

    GpuTimer timer;

    for (int i = 0; i < reps; ++i)
    {
        timer.start();

        launch();

        times.push_back(timer.stop_ms());
    }

    // Statistics
    std::sort(times.begin(), times.end());

    const float min_ms = times.front();

    const float median_ms =
        times[times.size() / 2];

    const float mean_ms =
        std::accumulate(times.begin(), times.end(), 0.0f)
        / static_cast<float>(times.size());

    return {
        min_ms,
        median_ms,
        mean_ms
    };
}

double bandwidth_GBps(std::size_t bytes, float ms)
{
    return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

double theoretical_bandwidth_GBps()
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const double memory_clock_hz =
        static_cast<double>(prop.memoryClockRate) * 1e3;

    const double bus_width_bytes =
        static_cast<double>(prop.memoryBusWidth) / 8.0;

    return 2.0 * memory_clock_hz * bus_width_bytes / 1e9;
}

double bandwidth_utilization(double measured_GBps)
{
    return measured_GBps / theoretical_bandwidth_GBps() * 100.0;
}

#endif //CUDAGPU_TIMER_CUH