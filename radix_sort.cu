#include "radix_sort.h"
#include <cuda_runtime.h>

/* tunable */
constexpr int THREADS = 256;
constexpr int ELEM_PER_THREAD = 4;

/* TODO: adjust for other vector lengths */
typedef ulonglong4 u64_vec;

constexpr int ELEM_PER_BLOCK = THREADS * ELEM_PER_THREAD;
// constexpr int WARP_SIZE = 32;

/* tunable */
constexpr int BITS = 64;
constexpr int RADIX = 4;

constexpr int RADIX_SIZE = 1 << RADIX;

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

    /* local thread index */
    const int lidx = threadIdx.x;
    const int idx = lidx * ELEM_PER_THREAD;
    /* global thread index */
    const int gidx = blockIdx.x * THREADS + lidx;
    u64_vec my_vec = data_in[gidx];

    for (int bit = start_bit; bit > start_bit - RADIX; bit++)
    {
        /* TODO: adjust for other vector lengths */
        short4 bits;
        bits.x = !((my_vec.x >> bit) & 1);
        bits.y = !((my_vec.y >> bit) & 1);
        bits.z = !((my_vec.z >> bit) & 1);
        bits.w = !((my_vec.w >> bit) & 1);


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

        shared[ptr.x] = my_vec.x;
        shared[ptr.y] = my_vec.y;
        shared[ptr.z] = my_vec.z;
        shared[ptr.w] = my_vec.w;
        __syncthreads();

        my_vec.x = shared[idx + 0];
        my_vec.y = shared[idx + 1];
        my_vec.z = shared[idx + 2];
        my_vec.w = shared[idx + 3];
        __syncthreads();
    }

    data_out[gidx] = my_vec;
}

inline int static divup(int a, int b) { return (a + b - 1) / b; }

void radix_sort(int n, u64* input) {
    const int blocks = divup(n, ELEM_PER_BLOCK);

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
        /* (3) prefix sum across blocks over the histograms */
        /* (4) using prefix sum each block moves their elements to correct position */
        start_bit += RADIX;
    }


    cudaFree(data);
    cudaFree(data_tmp);
    cudaFree(histograms);
}
