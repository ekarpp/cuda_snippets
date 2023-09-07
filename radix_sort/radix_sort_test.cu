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
        data[i] = rand() % 0xFF;

    return data;
}


bool is_sorted(u64 *vec, uint len)
{
    u64 prev = vec[0];

    for (uint i = 1; i < len; i++)
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

    if (!is_sorted(input.data(), len))
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

    scan_histograms<false, false>
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
    const int scan_depth = 1;
    const int blocks = ELEM_PER_BLOCK;
    std::vector<u32> data = random_u32(blocks * blocks);
    int scan_sizes[1];
    scan_sizes[0] = blocks;

    u32 *gpu = NULL;
    cudaMalloc((void **) &gpu, blocks * blocks * sizeof(u32));
    cudaMemcpy(gpu, data.data(), blocks * blocks * sizeof(u32), cudaMemcpyHostToDevice);

    u32 *scan_sums[1];
    cudaMalloc((void **) &scan_sums[0], blocks * sizeof(u32));

    global_scan(gpu, scan_sums, scan_sizes, scan_depth, blocks);

    std::vector<u32> out(blocks * blocks);
    cudaMemcpy(out.data(), gpu, blocks * blocks * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = 0;
    for (uint i = 0; i < data.size(); i++)
    {
        sum += data[i];
        if (sum != out[i])
        {
            std::cout << "FAIL at " << i << "/" << data.size() << std::endl;
            return;
        }

    }

    std::cout << "OK" << std::endl;
}

static void test_sort_block()
{
    std::cout << "Testing sort block...";
    const int blocks = ELEM_PER_BLOCK;
    std::vector<u64> data = random_u64(blocks * blocks);
    for (int i = 0; i < data.size(); i++)
        data[i] &= 0xF;

    u64 *gpu = NULL;
    cudaMalloc((void **) &gpu, blocks * blocks * sizeof(u64));
    cudaMemcpy(gpu, data.data(), blocks * blocks * sizeof(u64), cudaMemcpyHostToDevice);

    u64 *out = NULL;
    cudaMalloc((void **) &out, blocks * blocks * sizeof(u64));

    sort_block
        <<<blocks, THREADS>>>
        ((u64_vec *) gpu, (u64_vec *) out, 0);

    u64 *sorted = (u64 *) std::malloc(blocks * blocks * sizeof(u64));
    cudaMemcpy(sorted, out, blocks * blocks * sizeof(u64), cudaMemcpyDeviceToHost);

    u64 offset = 0;
    while (offset < blocks)
    {
        if (!is_sorted(sorted + offset * blocks, blocks))
        {
            std::cout << "FAIL" << std::endl;
            return;
        }
        offset += blocks;
    }

    std::cout << "OK" << std::endl;
}

static void test_create_histogram()
{
    std::cout << "Testing create histogram...";
    const int blocks = ELEM_PER_BLOCK;

    std::vector<u64> data = random_u64(blocks * blocks);
    for (int i = 0; i < data.size(); i++)
        data[i] &= 0xF;

    u64 *gpu = NULL;
    cudaMalloc((void **) &gpu, blocks * blocks * sizeof(u64));
    cudaMemcpy(gpu, data.data(), blocks * blocks * sizeof(u64), cudaMemcpyHostToDevice);

    u32 *grams = NULL;
    cudaMalloc((void **) &grams, blocks * RADIX_SIZE * sizeof(u32));

    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks * RADIX_SIZE * sizeof(u32));

    compute_histograms
        <<<blocks, THREADS>>>
        ((u64_vec *) gpu, grams, start_ptrs, blocks, 0);

    std::vector<u32> out(blocks * RADIX_SIZE);
    cudaMemcpy(out.data(), grams, blocks * RADIX_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks; i++)
    {
        std::vector<int> local(RADIX_SIZE, 0);

        for (int j = 0; j < ELEM_PER_BLOCK; j++)
            local[data[i * blocks + j]]++;

        for (int j = 0; j < RADIX_SIZE; j++)
        {
            if (local[j] != out[j * blocks + i])
            {
                std::cout << "FAIL" << std::endl;
                return;
            }
        }
    }

    std::cout << "OK" << std::endl;
}

int main()
{
    test_sort_block();
    test_local_scan();
    test_global_scan();
    test_create_histogram();
    test_sort(1024);
    test_sort(1024 * 1024);
    return 0;
}
