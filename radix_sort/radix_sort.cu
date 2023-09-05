#include "radix_sort.h"
#include "constants.h"
#include "scan.h"
#include <cuda_runtime.h>
#include <iostream>

void check_gpu_error(const char *fn)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in \"" << fn << "\": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__device__ int4 split(int *shared, const int4 bits)
{
    const int idx = threadIdx.x * ELEM_PER_THREAD;
    shared[idx + 0] = bits.x;
    shared[idx + 1] = bits.y;
    shared[idx + 2] = bits.z;
    shared[idx + 3] = bits.w;
    __syncthreads();
    scan::scan_block<int>(shared);
    __syncthreads();

    int4 ptr;
    ptr.w = shared[idx + 3] - bits.w;
    ptr.z = shared[idx + 2] - bits.w - bits.z;
    ptr.y = shared[idx + 1] - bits.w - bits.z - bits.y;
    ptr.x = shared[idx + 0] - bits.w - bits.z - bits.y - bits.x;

    __shared__ uint trues;
    if (threadIdx.x == THREADS - 1)
    {
        trues = ptr.w + bits.w;
    }
    __syncthreads();

    ptr.x = (bits.x) ? ptr.x : trues + idx + 0 - ptr.x;
    ptr.y = (bits.y) ? ptr.y : trues + idx + 1 - ptr.y;
    ptr.z = (bits.z) ? ptr.z : trues + idx + 2 - ptr.z;
    ptr.w = (bits.w) ? ptr.w : trues + idx + 3 - ptr.w;

    return ptr;
}


__global__ void sort_block(const u64_vec* data_in, u64_vec* data_out, const int start_bit)
{
    __shared__ u64 shared[ELEM_PER_BLOCK];
    // how to get rid of ptrs?
    __shared__ int ptrs[ELEM_PER_BLOCK];

    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;
    const int idx = lidx * ELEM_PER_THREAD;

    u64_vec my_data = data_in[gidx];

    for (int bit = start_bit; bit < start_bit + RADIX; bit++)
    {
        /* TODO: adjust for other vector lengths */
        int4 bits;
        bits.x = !((my_data.x >> bit) & 1);
        bits.y = !((my_data.y >> bit) & 1);
        bits.z = !((my_data.z >> bit) & 1);
        bits.w = !((my_data.w >> bit) & 1);

        int4 ptr = split(ptrs, bits);

        /* bank issues? */
        ptrs[ptr.x] = my_data.x;
        ptrs[ptr.y] = my_data.y;
        ptrs[ptr.z] = my_data.z;
        ptrs[ptr.w] = my_data.w;
        __syncthreads();

        my_data.x = shared[idx + 0];
        my_data.y = shared[idx + 1];
        my_data.z = shared[idx + 2];
        my_data.w = shared[idx + 3];
        __syncthreads();
    }

    data_out[gidx] = my_data;
}

__global__ void compute_histograms(
    const u64_vec2 *data,
    u32 *block_histograms,
    u32 *start_ptrs,
    const int num_blocks,
    const int start_bit
)
{
    // latter value of pair from each thread
    __shared__ int radix[THREADS];
    // start indices for each radix
    __shared__ int ptrs[RADIX_SIZE];
    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    /*
     * fill histogram by finding the start points of each radix in the sorted data block
     */

    u64_vec2 my_data = data[gidx];
    int2 my_radix;
    my_radix.x = (my_data.x >> start_bit) & RADIX_MASK;
    my_radix.y = (my_data.y >> start_bit) & RADIX_MASK;
    radix[lidx] = my_radix.y;

    if (lidx < RADIX_SIZE)
    {
        ptrs[lidx] = -1;
    }
    __syncthreads();

    if (lidx > 0 && my_radix.x != radix[lidx - 1])
    {
        ptrs[my_radix.x] = 2 * lidx;
    }
    if (my_radix.x != my_radix.y)
    {
        ptrs[my_radix.y] = 2 * lidx + 1;
    }
    __syncthreads();

    /* store in column major order */
    if (lidx < RADIX_SIZE)
    {
        const int hist_idx = num_blocks * lidx + blockIdx.x;
        start_ptrs[blockIdx.x * RADIX_SIZE + lidx] = ptrs[lidx];
        if (lidx == RADIX_SIZE - 1)
        {
            block_histograms[hist_idx] = (ptrs[lidx] != -1)
                ? 2 * THREADS - ptrs[lidx]
                : 0;
        }
        else if (lidx == 0)
        {
            block_histograms[hist_idx] = ptrs[lidx] + 1;
        }
        else
        {
            block_histograms[hist_idx] = (ptrs[lidx] != -1)
                ? ptrs[lidx] - ptrs[lidx - 1]
                : 0;
        }
    }
}

template <bool add_total>
__global__ void scan_histograms(u32 *block_histograms, u32 *scan_sums)
{
    __shared__ u32 result[ELEM_PER_BLOCK];
    const int lidx = threadIdx.x * ELEM_PER_THREAD;
    const int gidx = blockIdx.x * ELEM_PER_BLOCK + lidx;

    result[lidx + 0] = block_histograms[gidx + 0];
    result[lidx + 1] = block_histograms[gidx + 1];
    result[lidx + 2] = block_histograms[gidx + 2];
    result[lidx + 3] = block_histograms[gidx + 3];

    __syncthreads();
    scan::scan_block<u32>(result);
    __syncthreads();

    block_histograms[gidx + 0] = result[lidx + 0];
    block_histograms[gidx + 1] = result[lidx + 1];
    block_histograms[gidx + 2] = result[lidx + 2];
    block_histograms[gidx + 3] = result[lidx + 3];

    if (add_total && lidx == THREADS - 1)
    {
        scan_sums[blockIdx.x] = result[lidx + 3];
    }
}

__global__ void add_sums(const u32 *from, u32 *to)
{
    /* lazy... scan not exclusive.. */
    if (blockIdx.x == 0)
        return;
    __shared__ u32 sum;
    const int lidx = threadIdx.x * ELEM_PER_THREAD;
    const int gidx = blockIdx.x * ELEM_PER_BLOCK + lidx;

    if (lidx == 0)
    {
        sum = from[blockIdx.x - 1];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++)
        to[gidx + i] += sum;
}

__global__ void reorder_data(const u64_vec2 *data_in,
                             u64 *data_out,
                             const u32 *block_histograms,
                             const u32 *start_ptrs,
                             const int bucket_size,
                             const int start_bit)
{
    __shared__ u32 global_ptrs[RADIX_SIZE];
    __shared__ u32 local_ptrs[RADIX_SIZE];

    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    u64_vec2 my_data = data_in[gidx];
    int2 my_radix;
    my_radix.x = (my_data.x >> start_bit) & RADIX_MASK;
    my_radix.y = (my_data.y >> start_bit) & RADIX_MASK;

    if (lidx < 16)
    {
        global_ptrs[lidx] = block_histograms[lidx * bucket_size + blockIdx.x];
        local_ptrs[lidx] = start_ptrs[blockIdx.x * RADIX_SIZE + lidx];
    }
    __syncthreads();

    u32 my_offsets[2];
    my_offsets[0] = global_ptrs[my_radix.x] + lidx - local_ptrs[my_radix.x];
    my_offsets[1] = global_ptrs[my_radix.y] + lidx - local_ptrs[my_radix.y];

    data_out[my_offsets[0]] = my_data.x;
    data_out[my_offsets[1]] = my_data.y;
}


inline int static divup(int a, int b) { return (a + b - 1) / b; }

// https://www.cs.umd.edu/class/spring2021/cmsc714/readings/Satish-sorting.pdf
int radix_sort(int n, u64* input) {
    // partial blocks not implemented and overflow in scan if too large (can increase to u64)
    if (n % ELEM_PER_BLOCK != 0 || n >= 1 << 30)
    {
        return -1;
    }

    const int blocks = divup(n, ELEM_PER_BLOCK);

    const int scan_depth = std::max(1, (int) std::ceil(std::log(n) / (10.0 * std::log(2.0)) - 1.0));

    // in reorder and histogram creation each thread takes two elements
    const int blocks2 = divup(n, 2 * THREADS);

    /* main data array */
    u64 *data = NULL;
    cudaMalloc((void **) &data, n * sizeof(u64));
    cudaMemcpy(data, input, n * sizeof(u64), cudaMemcpyHostToDevice);

    /* stores data where each block is sorted */
    u64 *data_tmp = NULL;
    cudaMalloc((void **) &data_tmp, n * sizeof(u64));

    /* start index for each radix in each block */
    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks2 * RADIX_SIZE * sizeof(u32));

    /* histogram of radixes from blocks in column major order */
    u32 *block_histograms = NULL;
    cudaMalloc((void **) &block_histograms, blocks2 * RADIX_SIZE * sizeof(u32));

    /* sum of each block during histogram scan */
    u32 *scan_sums[scan_depth];
    int scan_sizes[scan_depth];

    for (int i = 0; i < scan_depth; i++)
    {
        scan_sizes[i] = (i == 0) ? blocks : scan_sizes[i - 1];
        scan_sizes[i] = divup(scan_sizes[i], ELEM_PER_BLOCK);
        scan_sums[i] = NULL;
        cudaMalloc((void **) &scan_sums[i], scan_sizes[i] * sizeof(u32));
    }

/*
    scan_histograms
        <<<blocks, THREADS>>>
        ((u32 *) data, status, blocks);
    check_gpu_error("scan_histograms");
    cudaMemcpy(input, data, n * sizeof(u64), cudaMemcpyDeviceToHost);
    return 0;
*/
    int start_bit = 0;
    while (start_bit < BITS)
    {
        /*
         * (1) sort blocks by iterating 1-bit split.
         */
        sort_block
            <<<blocks, THREADS>>>
            ((u64_vec *) data, (u64_vec *) data_tmp, start_bit);
        check_gpu_error("sort_block");

        /* (2) write histogram for each block to global memory */
        compute_histograms
            <<<blocks2, THREADS>>>
            ((u64_vec2 *) data_tmp, block_histograms, start_ptrs, blocks2, start_bit);
        check_gpu_error("compute_histograms");

        /* (3) prefix sum across blocks over the histograms */
        {
            scan_histograms<true>
                <<<blocks, THREADS>>>
                (block_histograms, scan_sums[scan_depth - 1]);
            check_gpu_error("scan_histograms<true>");

            for (int i = 0; i < scan_depth - 1; i++)
            {
                scan_histograms<true>
                    <<<scan_sizes[i], THREADS>>>
                    (scan_sums[i], scan_sums[i+1]);
                check_gpu_error("scan_histograms<true> in loop");
            }

            if (scan_sizes[scan_depth - 1] != 1)
            {
                std::cout << "something is wrong" << std::endl;
            }

            scan_histograms<false>
                <<<1, THREADS>>>
                (scan_sums[scan_depth - 1], NULL);
            check_gpu_error("scan_histograms<false>");

            for (int i = scan_depth - 1; i > 0; i--)
            {
                add_sums
                    <<<scan_sizes[i - 1], THREADS>>>
                    (scan_sums[i], scan_sums[i - 1]);
            }

            add_sums
                <<<blocks, THREADS>>>
                (scan_sums[0], block_histograms);
        }

        /* (4) using histogram scan each block moves their elements to correct position */
        reorder_data
            <<<blocks2, THREADS>>>
            ((u64_vec2 *) data_tmp, data, block_histograms, start_ptrs, blocks2, start_bit);
        check_gpu_error("reorder_data");

        start_bit += RADIX;
    }

    cudaMemcpy(input, data, n * sizeof(u64), cudaMemcpyDeviceToHost);

    cudaFree(data);
    cudaFree(data_tmp);
    cudaFree(block_histograms);
    cudaFree(start_ptrs);

    for (int i = 0; i < scan_depth; i++)
        cudaFree(scan_sums[i]);

    return 0;
}
