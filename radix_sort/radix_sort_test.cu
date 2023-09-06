#include "radix_sort.cu"

#include <vector>
#include <iostream>

std::vector<u64> random_u64(uint len)
{
    std::vector<u64> data(len);

    for (uint i = 0; i < len; i++)
        data[i] = ((u64) rand() << 32) | rand();

    return data;
}

std::vector<u32> random_u32(uint len)
{
    std::vector<u32> data(len);

    for (uint i = 0; i < len; i++)
        data[i] = rand() % 0xFFFF;

    return data;
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

static void test_sort(uint len)
{
    std::cout << "Testing sort for " << len << " elements... ";

    std::vector<u64> input = random_u64(len);

    radix_sort(len, input.data());

    if (!is_sorted(input))
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "OK";
    }

    std::cout << std::endl;
}

static void test_local_scan()
{
    std::cout << "Testing local scan... ";
    std::vector<u32> data = random_u32(ELEM_PER_BLOCK);
    u32* gpu = NULL;
    cudaMalloc((void **) &gpu, data.size() * sizeof(u32));
    cudaMemcpy(gpu, data.data(), data.size() * sizeof(u32), cudaMemcpyHostToDevice);

    scan_histograms<false>
        <<<1, THREADS>>>
        (gpu, NULL);
    check_gpu_error("scan_histograms");

    std::vector<u32> out(data.size());
    cudaMemcpy(out.data(), gpu, data.size() * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = 0;

    for (uint i = 0; i < data.size(); i++)
    {
        if (sum != out[i])
        {
            std::cout << "FAIL at " << i << "/" << data.size() << std::endl;
            return;
        }
        sum += data[i];
    }

    std::cout << "OK" << std::endl;
}

static void test_global_scan()
{
    std::cout << "Testing global scan... ";
    const int blocks = ELEM_PER_BLOCK;
    const int scan_depth = 2;
    std::vector<u32> data = random_u32(blocks * blocks);
    int scan_sizes[2];
    scan_sizes[0] = blocks;
    scan_sizes[1] = 1;

    u32 *gpu = NULL;
    cudaMalloc((void **) &gpu, blocks * blocks * sizeof(u32));
    cudaMemcpy(gpu, data.data(), blocks * blocks * sizeof(u32), cudaMemcpyHostToDevice);

    u32 *scan_sums[2];
    cudaMalloc((void **) &scan_sums[0], blocks * sizeof(u32));
    cudaMalloc((void **) &scan_sums[1], sizeof(u32));

    global_scan(gpu, scan_sums, scan_sizes, scan_depth, blocks);

    std::vector<u32> out(blocks * blocks);
    cudaMemcpy(out.data(), gpu, blocks * blocks * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = 0;

    for (uint i = 0; i < data.size(); i++)
    {
        if (sum != out[i])
        {
            std::cout << "FAIL at " << i << "/" << data.size() << std::endl;
            return;
        }
        sum += data[i];
    }

    std::cout << "OK" << std::endl;
}

int main()
{
    test_local_scan();
    test_global_scan();
    test_sort(1024);
    return 0;
}
