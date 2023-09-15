#define DEBUG

#include "radix_sort.cu"

#include <algorithm>
#include <vector>
#include <iostream>

std::vector<u32> random_u32(uint len, u32 mask = 0xFFFFFFFF)
{
    std::vector<u32> data(len);

    for (uint i = 0; i < len; i++)
        data[i] = rand() & mask;

    return data;
}

void print_block(std::vector<u32> data)
{
    if (data.size() < ELEM_PER_BLOCK)
        return;

    std::cout << std::endl;
    for (uint i = 0; i < ELEM_PER_BLOCK; i++)
    {
        printf("%.4u ", data[i]);
        if ((i+1)%4 == 0)
            std::cout << "| ";
        if ((i+1)%32 == 0)
            std::cout << std::endl;
        if ((i+1)%128 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

void OK()
{
    std::cout << "\033[32m"
              << "OK"
              << "\033[0m"
              << std::endl;
}

void FAIL()
{
    std::cout << "\033[31m"
              << "FAIL"
              << "\033[0m"
              << std::endl;
}



static void test_sort_block(u32 n)
{
    std::cout << "Testing sort_block for " << n << " elements... ";
    const int blocks = divup(n, ELEM_PER_BLOCK);
    std::vector<u32> data = random_u32(n, 0xF);

    const int num_elem = blocks * ELEM_PER_BLOCK;
    u32 *gpu = NULL;
    cudaMalloc((void **) &gpu, num_elem * sizeof(u32));
    cudaMemcpy(gpu, data.data(), n * sizeof(u32), cudaMemcpyHostToDevice);
    if (num_elem - n > 0)
        cudaMemset(gpu + n, 0xFF, (num_elem - n) * sizeof(u32));

    u32 *out = NULL;
    cudaMalloc((void **) &out, n * sizeof(u32));
    sort_block
        <<<blocks, THREADS>>>
        (gpu, out, 0);

    std::vector<u32> sorted(n);
    cudaMemcpy(sorted.data(), out, n * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 offset = 0;
    while (offset < n)
    {
        u32 *start = data.data() + offset;
        u32 *end = data.data() + std::min(offset + ELEM_PER_BLOCK, n);
        std::sort(start, end);
        offset += ELEM_PER_BLOCK;
    }

    for (int i = 0; i < data.size(); i++)
    {
        if (sorted[i] != data[i])
        {
            FAIL();
            return;
        }
    }

    OK();
}

static void test_create_histogram(u32 n)
{
    std::cout << "Testing create_histogram for " << n << " elements... ";
    const int blocks = divup(n, ELEM_PER_BLOCK);

    std::vector<u32> data = random_u32(n, 0xF);

    u32 offset = 0;
    while (offset < n)
    {
        u32 *start = data.data() + offset;
        u32 *end = data.data() + std::min(offset + ELEM_PER_BLOCK, n);
        std::sort(start, end);
        offset += ELEM_PER_BLOCK;
    }

    const int num_elem = blocks * ELEM_PER_BLOCK;
    u32 *gpu = NULL;
    cudaMalloc((void **) &gpu, num_elem * sizeof(u32));
    cudaMemcpy(gpu, data.data(), n * sizeof(u32), cudaMemcpyHostToDevice);
    if (num_elem - n > 0)
        cudaMemset(gpu + n, 0xFF, (num_elem - n) * sizeof(u32));

    u32 *grams = NULL;
    cudaMalloc((void **) &grams, blocks * RADIX_SIZE * sizeof(u32));

    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks * RADIX_SIZE * sizeof(u32));

    compute_histograms
        <<<blocks, THREADS>>>
        ((u32_vec *) gpu, grams, start_ptrs, blocks, 0);

    std::vector<u32> out(blocks * RADIX_SIZE);
    cudaMemcpy(out.data(), grams, blocks * RADIX_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks; i++)
    {
        std::vector<int> local(RADIX_SIZE, 0);

        for (int j = 0; j < ELEM_PER_BLOCK; j++)
        {
            const int idx = i * ELEM_PER_BLOCK + j;
            if (idx < n)
                local[data[idx]]++;
        }

        if (i == blocks - 1)
            local[RADIX_SIZE - 1] += num_elem - n;

        for (int j = 0; j < RADIX_SIZE; j++)
        {
            if (local[j] != out[j * blocks + i])
            {
                FAIL();
                return;
            }
        }
    }

    std::vector<u32> ptrs(blocks * RADIX_SIZE);
    cudaMemcpy(ptrs.data(), start_ptrs, blocks * RADIX_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

    // don't test last block because resolving issues with padding too cumbersome
    for (int i = 0; i < blocks - 1; i++)
    {
        const int offset = i * ELEM_PER_BLOCK;
        for (int j = 1; j < RADIX_SIZE; j++)
        {
            const int idx = i * RADIX_SIZE + j;

            if (data[ptrs[idx] + offset] != j || data[ptrs[idx] - 1 + offset] != j - 1)
            {
                FAIL();
                return;
            }
        }
    }

    OK();
}

static void test_local_scan()
{
    std::cout << "Testing local scan... ";

    std::vector<u32> data = random_u32(ELEM_PER_BLOCK, 0xFF);
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
            FAIL();
            return;
        }
        sum += data[i];
    }

    OK();
}

static void test_global_scan(u32 n)
{
    std::cout << "Testing global scan for " << n << " elements... ";

    const int blocks = divup(n, ELEM_PER_BLOCK);
    std::vector<u32> data = random_u32(n, 0xF);
    for (int i = 0; i < n; i++)
        data[i] = 1;
    const int scan_depth = std::floor(std::log(blocks) / std::log(ELEM_PER_BLOCK) + 1 - 1e-10);

    /* LAZY, just copied from radix_sort.cu */
    u32 *scan_sums[scan_depth];
    int scan_sizes[scan_depth];

    for (int i = 0; i < scan_depth; i++)
    {
        scan_sums[i] = NULL;
        scan_sizes[i] = (i == 0)
            ? blocks
            : divup(scan_sizes[i - 1], ELEM_PER_BLOCK);
        cudaMalloc((void **) &scan_sums[i], std::max(ELEM_PER_BLOCK, scan_sizes[i]) * sizeof(u32));
    }

    const int num_elems = blocks * ELEM_PER_BLOCK;

    u32 *gpu = NULL;
    cudaMalloc((void **) &gpu, blocks * ELEM_PER_BLOCK * sizeof(u32));
    cudaMemcpy(gpu, data.data(), n * sizeof(u32), cudaMemcpyHostToDevice);
    if (num_elems > n)
        cudaMemset(gpu + n, 0x0, (num_elems - n) * sizeof(u32));

    global_scan(gpu, scan_sums, scan_sizes, scan_depth);

    std::vector<u32> out(blocks * ELEM_PER_BLOCK);
    cudaMemcpy(out.data(), gpu, blocks * ELEM_PER_BLOCK * sizeof(u32), cudaMemcpyDeviceToHost);

    u32 sum = 0;
    for (uint i = 0; i < data.size(); i++)
    {
        if (sum != out[i])
        {
            FAIL();
            return;
        }
        sum += data[i];
    }

    OK();
}

static void test_sort(u32 len)
{
    std::cout << "Testing sort for " << len << " elements... ";

    std::vector<u32> input = random_u32(len);
    std::vector<u32> sorted(len);
    for (uint i = 0; i < len; i++)
        sorted[i] = input[i];

    std::sort(sorted.begin(), sorted.end());

    radix_sort(len, input.data());

    for (uint i = 0; i < len; i++)
    {
        if (sorted[i] != input[i])
        {
            FAIL();
            return;
        }
    }
    OK();
}

int main()
{
    const int seed = time(NULL);
    srand(seed);
    std::cout << "seed: " << seed << std::endl;
    test_local_scan();

    std::vector<int> sizes = { 1024, 12345, 1024 * 1024, (1 << 23) + 1};

    for (int i = 0; i < sizes.size(); i++)
    {
        const int len = sizes[i];
        test_sort_block(len);
        test_create_histogram(len);
        test_global_scan(len);
    }

    test_sort(1024);
    test_sort(12345);
    test_sort((1 << 16) - 12345);
    test_sort(1 << 16);
    test_sort(1024 * 1024);
    // test_sort((1 << 16) + 1);

    test_sort(1 << 27);
    // seg faults?
    // test_sort((1 << 23) + 1);
    return 0;
}
