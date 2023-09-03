#include "radix_sort.h"
#include <cuda_runtime.h>

/* tunable */
constexpr int THREADS = 256;
// this should be 2 or 4, needs changes to code if we want 2
constexpr int ELEM_PER_THREAD = 4;

typedef ulonglong2 u64_vec2;
/* TODO: adjust for other vector lengths */
typedef ulonglong4 u64_vec;

constexpr int ELEM_PER_BLOCK = THREADS * ELEM_PER_THREAD;
// constexpr int WARP_SIZE = 32;

/* tunable */
constexpr int BITS = 64;
constexpr int RADIX = 4;

constexpr int RADIX_SIZE = 1 << RADIX;
constexpr int RADIX_MASK = RADIX_SIZE - 1;

__device__ void scan_bits(int depth, u64 *result)
{
    int offset = 1 << depth;
    if (offset >= THREADS)
    {
        return;
    }

    int idx = threadIdx.x * ELEM_PER_THREAD;
    if (offset + idx < THREADS)
    {
        result[idx + offset + 0] += result[idx + 0];
        result[idx + offset + 1] += result[idx + 1];
        result[idx + offset + 2] += result[idx + 2];
        result[idx + offset + 3] += result[idx + 3];
    }
    __syncthreads();
    scan_bits(depth + 1, result);
}


__global__ void sort_block(u64_vec* data_in, u64_vec* data_out, const int length, const int start_bit)
{
    extern __shared__ u64 shared[];

    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;
    const int idx = lidx * ELEM_PER_THREAD;

    u64_vec my_data = data_in[gidx];

    for (int bit = start_bit; bit > start_bit - RADIX; bit++)
    {
        /* TODO: adjust for other vector lengths */
        short4 bits;
        bits.x = !((my_data.x >> bit) & 1);
        bits.y = !((my_data.y >> bit) & 1);
        bits.z = !((my_data.z >> bit) & 1);
        bits.w = !((my_data.w >> bit) & 1);


        shared[idx + 0] = bits.x;
        shared[idx + 1] = bits.y;
        shared[idx + 2] = bits.z;
        shared[idx + 3] = bits.w;
        __syncthreads();
        scan_bits(1, shared);
        __syncthreads();

        short4 ptr;
        ptr.w = shared[idx + 3] - bits.w;
        ptr.z = shared[idx + 2] - bits.w - bits.z;
        ptr.y = shared[idx + 1] - bits.w - bits.z - bits.y;
        ptr.x = shared[idx + 0] - bits.w - bits.z - bits.y - bits.x;

        __shared__ uint trues;
        if (lidx == THREADS - 1)
        {
            trues = ptr.w + bits.w;
        }
        __syncthreads();

        int idx = lidx * ELEM_PER_THREAD;
        ptr.x = (bits.x) ? ptr.x : trues + idx + 0 - ptr.x;
        ptr.y = (bits.y) ? ptr.y : trues + idx + 1 - ptr.y;
        ptr.z = (bits.z) ? ptr.z : trues + idx + 2 - ptr.z;
        ptr.w = (bits.w) ? ptr.w : trues + idx + 3 - ptr.w;

        shared[ptr.x] = my_data.x;
        shared[ptr.y] = my_data.y;
        shared[ptr.z] = my_data.z;
        shared[ptr.w] = my_data.w;
        __syncthreads();

        my_data.x = shared[idx + 0];
        my_data.y = shared[idx + 1];
        my_data.z = shared[idx + 2];
        my_data.w = shared[idx + 3];
        __syncthreads();
    }

    data_out[gidx] = my_data;
}

__global__ void compute_histograms(u64_vec2 *data, u32 *histograms, const int length, const int start_bit)
{
    // latter value of pair from each thread
    __shared__ short radix[THREADS];
    // start indices for each radix
    __shared__ short ptrs[RADIX_SIZE];
    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    u64_vec2 my_data = data[gidx];
    short2 my_radix;
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

    const int hist_idx = blockIdx.x * RADIX_SIZE + lidx;
    if (lidx == RADIX_SIZE - 1)
    {
        histograms[hist_idx] = (ptrs[lidx] != -1)
            ? 2 * THREADS - ptrs[lidx]
            : 0;
    }
    else if (lidx == 0)
    {
        histograms[hist_idx] = ptrs[lidx] + 1;
    }
    else if (lidx < RADIX_SIZE)
    {
        histograms[hist_idx] = (ptrs[lidx] != -1)
            ? ptrs[lidx] - ptrs[lidx - 1]
            : 0;
    }
}

inline int static divup(int a, int b) { return (a + b - 1) / b; }

// https://www.cs.umd.edu/class/spring2021/cmsc714/readings/Satish-sorting.pdf
void radix_sort(int n, u64* input) {
    const int blocks = divup(n, ELEM_PER_BLOCK);
    // for histogram each thread takes two elements
    const int blocks_histogram = divup(n, 2 * THREADS);

    u64* data = NULL;
    cudaMalloc((void**) &data, n * sizeof(u64));
    cudaMemcpy(data, input, n * sizeof(u64), cudaMemcpyHostToDevice);

    u64* data_tmp = NULL;
    cudaMalloc((void**) &data_tmp, n * sizeof(u64));

    u32* histograms = NULL;
    cudaMalloc((void**) &histograms, blocks * RADIX_SIZE * sizeof(u32));

    int start_bit = 0;
    while (start_bit < BITS)
    {
        /*
         * (1) sort blocks by iterating 1-bit split.
         * uses `ELEM_PER_BLOCK * sizeof(int)` of shared memory.
         */
        sort_block
            <<<blocks, THREADS, ELEM_PER_BLOCK * sizeof(u64)>>>
            ((u64_vec *) data, (u64_vec *) data_tmp, n, start_bit);
        /* (2) write histogram for each block to global memory */
        compute_histograms
            <<<blocks_histogram, THREADS>>>
            ((u64_vec2 *) data_tmp, histograms, n, start_bit);
        /* (3) prefix sum across blocks over the histograms */
        /* (4) using prefix sum each block moves their elements to correct position */
        start_bit += RADIX;
    }


    cudaFree(data);
    cudaFree(data_tmp);
    cudaFree(histograms);
}
