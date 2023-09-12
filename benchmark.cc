#include "radix_sort.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <parallel/algorithm>

u32 random_u32()
{
    return rand();
}

static void benchmark_gpu(u32 len, int iters)
{
    std::vector<u32> data(len);

    for (int iter = 0; iter < iters; iter++) {
        for (uint i = 0; i < len; i++)
            data[i] = random_u32();

        auto start = std::chrono::high_resolution_clock::now();
        radix_sort(len, data.data());
        auto end = std::chrono::high_resolution_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sorted " << len << " in " << delta.count() << " ms"
                  << " (" << iter + 1 << "/" << iters << ") with GPU" << std::endl;
    }
}

static void benchmark_cpu(u32 len, int iters)
{
    std::vector<u32> data(len);

    for (int iter = 0; iter < iters; iter++) {
        for (uint i = 0; i < len; i++)
            data[i] = random_u32();

        auto start = std::chrono::high_resolution_clock::now();
        __gnu_parallel::sort(data.begin(), data.end());
        auto end = std::chrono::high_resolution_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sorted " << len << " in " << delta.count() << " ms"
                  << " (" << iter + 1 << "/" << iters << ") with CPU" << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cout << "./radix_sort_benchmark <testsize> [iters]" << std::endl;
        return 1;
    }
    const u32 len = std::stol(argv[1]);
    const int iters = (argc > 2) ? std::stoi(argv[2]) : 1;

    benchmark_gpu(len, iters);
    benchmark_cpu(len, iters);
    return 0;
}
