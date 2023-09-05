#include "radix_sort.h"

#include <vector>
#include <random>
#include <iostream>

u64 random_data()
{
    return 1;//((u64) rand() << 32) | rand();
}

static void benchmark(u64 len, int iters)
{
    std::vector<u64> input(len);

    for (int iter = 0; iter < iters; ++iter) {
        for (uint i = 0; i < len; i++)
        {
            input[i] = random_data() % 32;
        }

        if (radix_sort(len, input.data()) == 0)
        {
            for (uint i = 0; i < len; i++)
            {
                printf("%.4lld ", input[i]);
                if ((i+1)%4 == 0)
                    std::cout << "| ";
                if ((i+1)%32 == 0)
                    std::cout << std::endl;
            }
        }
        std::cout << std::endl;
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
