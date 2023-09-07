#include "radix_sort.h"

#include <chrono>
#include <vector>
#include <random>
#include <iostream>

u64 random_u64()
{
    return ((u64) rand() << 32) | rand();
}

static void benchmark(u64 len, int iters)
{
    std::vector<u64> data(len);

    for (int iter = 0; iter < iters; iter++) {
        for (uint i = 0; i < len; i++)
            data[i] = random_u64();

        auto start = std::chrono::high_resolution_clock::now();
        radix_sort(len, data.data());
        auto end = std::chrono::high_resolution_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sorted " << len << " in " << delta.count() << " ms"
                  << " (" << iter + 1 << "/" << iters << ")" << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cout << "./radix_sort_benchmark <testsize> [iters]" << std::endl;
        return 1;
    }
    const u64 len = std::stol(argv[1]);
    const int iters = (argc > 2) ? std::stoi(argv[2]) : 1;

    benchmark(len, iters);
    return 0;
}
