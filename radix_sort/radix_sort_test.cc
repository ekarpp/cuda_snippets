#include "radix_sort.h"

#include <vector>
#include <random>
#include <iostream>

u64 random_data()
{
    return ((u64) rand() << 32) | rand();
}

bool is_sorted(std::vector<u64> vec)
{
    u64 prev = vec[0];

    for (uint i = 1; i < vec.size(); i++)
    {
        if (vec[i] > prev)
        {
            return false;
        }
        else
        {
            prev = vec[i];
        }
    }
    return true;
}

static void test(u64 len)
{
    std::vector<u64> input(len);

    for (uint i = 0; i < len; i++)
    {
        input[i] = random_data();
    }

    radix_sort(len, input.data());
    if (!is_sorted(input))
    {
        std::cout << "FAIL: ";
    }
    else
    {
        std::cout << "OK: ";
    }

    std::cout << len << std::endl;
}

int main()
{
    test(1024);
    test(1024 * 1024);
    return 0;
}
