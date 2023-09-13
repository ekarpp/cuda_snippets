#include "radix_sort.h"

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <chrono>
#include <vector>
#include <iostream>
#include <parallel/algorithm>


std::vector<u32> random_vec(u32 len, u32 seed)
{
    std::mt19937 rng(seed);
    std::vector<u32> data(len);
    std::uniform_int_distribution<u32> d(0, 0xFFFFFFFF);

    for (u32 i = 0; i < len; i++)
        data[i] = d(rng);

    return data;
}

static void benchmark_gpu(u32 len, u32 iters, u32 seed)
{
    for (u32 iter = 0; iter < iters; iter++) {
        std::vector<u32> data = random_vec(len, seed + iter);

        auto start = std::chrono::high_resolution_clock::now();
        radix_sort(len, data.data());
        auto end = std::chrono::high_resolution_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sorted " << len << " in " << delta.count() << " ms"
                  << " (" << iter + 1 << "/" << iters << ") with GPU (our)" << std::endl;
    }
}

static void benchmark_thrust(u32 len, u32 iters, u32 seed)
{
    for (u32 iter = 0; iter < iters; iter++) {
        std::vector<u32> data = random_vec(len, seed + iter);

        auto start = std::chrono::high_resolution_clock::now();
        thrust::device_vector<u32> d_data(data.data(), data.data() + len);
        thrust::sort(d_data.begin(), d_data.end());
        thrust::host_vector<u32> h_data = d_data;
        auto end = std::chrono::high_resolution_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sorted " << len << " in " << delta.count() << " ms"
                  << " (" << iter + 1 << "/" << iters << ") with GPU (Thrust)" << std::endl;
    }
}

static void benchmark_cpu(u32 len, u32 iters, u32 seed)
{
    for (u32 iter = 0; iter < iters; iter++) {
        std::vector<u32> data = random_vec(len, seed + iter);

        auto start = std::chrono::high_resolution_clock::now();
        __gnu_parallel::sort(data.begin(), data.end());
        auto end = std::chrono::high_resolution_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sorted " << len << " in " << delta.count() << " ms"
                  << " (" << iter + 1 << "/" << iters << ") with CPU (GCC)" << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cout << "./radix_sort_benchmark <testsize> [iters]" << std::endl;
        return 1;
    }
    const u32 len = std::stoi(argv[1]);
    const u32 iters = (argc > 2) ? std::stoi(argv[2]) : 1;
    const u32 seed = time(NULL);

    benchmark_gpu(len, iters, seed);
    std::cout << std::endl;
    benchmark_thrust(len, iters, seed);
    std::cout << std::endl;
    benchmark_cpu(len, iters, seed);

    return 0;
}
