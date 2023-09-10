#include "radix_sort.cu"

#include <algorithm>
#include <vector>
#include <iostream>

std::vector<u64> random_u64(uint len)
{
    std::vector<u64> data(len);

    for (uint i = 0; i < len; i++)
        data[i] = ((u64) rand() << 32) | rand();

    return data;
}

std::vector<u32> random_u32_masked_8bit(uint len)
{
    std::vector<u32> data(len);

    for (uint i = 0; i < len; i++)
        data[i] = rand() & 0xFF;

    return data;
}

void print_block(std::vector<u64> data)
{
    if (data.size() < ELEM_PER_BLOCK)
        return;

    std::cout << std::endl;
    for (uint i = 0; i < ELEM_PER_BLOCK; i++)
    {
        printf("%.4llu ", data[i]);
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



static void test_sort_block(u64 n)
{
    std::cout << "Testing sort_block for " << n << " elements... ";
    const int blocks = divup(n, ELEM_PER_BLOCK);
    std::vector<u64> data = random_u64(n);
    for (int i = 0; i < data.size(); i++)
        data[i] &= 0xF;

    const int num_elem = blocks * ELEM_PER_BLOCK;
    u64 *gpu = NULL;
    cudaMalloc((void **) &gpu, num_elem * sizeof(u64));
    cudaMemcpy(gpu, data.data(), n * sizeof(u64), cudaMemcpyHostToDevice);
    if (num_elem - n > 0)
        cudaMemset(gpu + n, 0xFF, (num_elem - n) * sizeof(u64));

    u64 *out = NULL;
    cudaMalloc((void **) &out, n * sizeof(u64));

    sort_block
        <<<blocks, THREADS>>>
        ((u64_vec *) gpu, (u64_vec *) out, 0);

    std::vector<u64> sorted(n);
    cudaMemcpy(sorted.data(), out, n * sizeof(u64), cudaMemcpyDeviceToHost);

    u64 offset = 0;
    while (offset < n)
    {
        u64 *start = data.data() + offset;
        u64 *end = data.data() + std::min(offset + ELEM_PER_BLOCK, n);
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

static void test_create_histogram(u64 n)
{
    std::cout << "Testing create_histogram for " << n << " elements... ";
    const int blocks = divup(n, ELEM_PER_BLOCK);

    std::vector<u64> data = random_u64(n);
    for (int i = 0; i < data.size(); i++)
        data[i] &= 0xF;

    u64 offset = 0;
    while (offset < n)
    {
        u64 *start = data.data() + offset;
        u64 *end = data.data() + std::min(offset + ELEM_PER_BLOCK, n);
        std::sort(start, end);
        offset += ELEM_PER_BLOCK;
    }

    const int num_elem = blocks * ELEM_PER_BLOCK;
    u64 *gpu = NULL;
    cudaMalloc((void **) &gpu, num_elem * sizeof(u64));
    cudaMemcpy(gpu, data.data(), n * sizeof(u64), cudaMemcpyHostToDevice);
    if (num_elem - n > 0)
        cudaMemset(gpu + n, 0xFF, (num_elem - n) * sizeof(u64));

    u32 *grams = NULL;
    cudaMalloc((void **) &grams, blocks * RADIX_SIZE * sizeof(u32));

    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks * RADIX_SIZE * sizeof(u32));

    compute_histograms
        <<<blocks, THREADS>>>
        ((u64_vec *) gpu, grams, start_ptrs, blocks, 0);

    std::vector<u32> out(blocks * RADIX_SIZE);
    cudaMemcpy(out.data(), grams, blocks * RADIX_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);
    std::vector<u32> ptrs(blocks*RADIX_SIZE);
    cudaMemcpy(ptrs.data(), start_ptrs, blocks * RADIX_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

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

    /* TODO: check start_ptrs */

    OK();
}

static void test_local_scan()
{
    std::cout << "Testing local scan... ";

    std::vector<u32> data = random_u32_masked_8bit(ELEM_PER_BLOCK);
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

static void test_global_scan(u64 n)
{
    std::cout << "Testing global scan for " << n << " elements... ";

    const int blocks = divup(n, ELEM_PER_BLOCK);
    std::vector<u32> data = random_u32_masked_8bit(n);
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
        sum += data[i];
        if (sum != out[i])
        {
            FAIL();
            return;
        }

    }

    OK();
}

static void test_sort(u64 len)
{
    std::cout << "Testing sort for " << len << " elements... ";

    std::vector<u64> input = random_u64(len);
    std::vector<u64> sorted(len);
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
    srand(time(NULL));
    test_sort_block(1024);
    test_sort_block(1024 * 1024);
    test_sort_block(12345);
    test_create_histogram(1024);
    test_create_histogram(1024 * 1024);
    test_create_histogram(12345);
    test_local_scan();
    test_global_scan(1024);
    test_global_scan(1024 * 1024);
    test_global_scan(12345);
    test_sort(1024);
    test_sort(12345);
    test_sort(1 << 16);
    test_sort(1024 * 1024);
    return 0;
}
