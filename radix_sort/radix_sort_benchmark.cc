#include "radix_sort.h"

#include <vector>
#include <random>
#include <iostream>

u64 random_data()
{
    return ((u64) rand() << 32) | rand();
}

static void benchmark(u64 len, int iters)
{
    std::vector<u64> input(len);

    for (int iter = 0; iter < iters; ++iter) {
        for (int i = 0; i < len; i++)
        {
            input[i] = random_data();
        }

        radix_sort(len, input.data());
    }
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cout << "./radix_sort_benchmark <testsize> [iters]" << std::endl;
        return 1;
    }
    const u64 len = std::stol(argv[1]);
    const int iters = argc >= 4 ? std::stoi(argv[2]) : 1;

    benchmark(len, iters);
    return 0;
}
