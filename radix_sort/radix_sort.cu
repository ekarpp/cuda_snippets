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

typedef struct
{
    u32 sum;
    int blocks;
    volatile u32 finished_blocks;
} scan_status;

/*
 * simple scan with O(n log n) work, can be optimized...
 * each thread handles 4 consecutive elements.
 */
template <typename T>
__device__ void scan_block(int depth, T *result)
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
    scan_block<T>(depth + 1, result);
}

__device__ short4 split(u64 *shared, short4 bits)
{
    const int idx = threadIdx.x * ELEM_PER_THREAD;
    shared[idx + 0] = bits.x;
    shared[idx + 1] = bits.y;
    shared[idx + 2] = bits.z;
    shared[idx + 3] = bits.w;
    __syncthreads();
    scan_block<u64>(1, shared);
    __syncthreads();

    short4 ptr;
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

        short4 ptr = split(shared, bits);

        /* bank issues? */
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

__global__ void compute_histograms(
    const u64_vec2 *data,
    u32 *block_histograms,
    const int num_blocks,
    const int start_bit
)
{
    // latter value of pair from each thread
    __shared__ short radix[THREADS];
    // start indices for each radix
    __shared__ short ptrs[RADIX_SIZE];
    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    /*
     * fill histogram by finding the start points of each radix in the sorted data block
     */

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

    /* store in column major order */
    const int hist_idx = num_blocks * lidx + blockIdx.x;
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
    else if (lidx < RADIX_SIZE)
    {
        block_histograms[hist_idx] = (ptrs[lidx] != -1)
            ? ptrs[lidx] - ptrs[lidx - 1]
            : 0;
    }
}

__global__ void scan_histograms(
    u32 *block_histograms,
    u32 * histogram,
    scan_status *status,
    const int num_blocks,
    const int bucket_size
)
{
    __shared__ u32 result[ELEM_PER_BLOCK];
    __shared__ uint block_id;
    const int lidx = threadIdx.x * ELEM_PER_THREAD;

    if (lidx == 0)
    {
        block_id = atomicAdd(&status->blocks, 1);
    }
    __syncthreads();

    const int gidx = block_id * ELEM_PER_BLOCK + lidx;
    result[lidx + 0] = block_histograms[gidx + 0];
    result[lidx + 1] = block_histograms[gidx + 1];
    result[lidx + 2] = block_histograms[gidx + 2];
    result[lidx + 3] = block_histograms[gidx + 3];

    __syncthreads();
    scan_block<u32>(1, result);
    __syncthreads();

    __shared__ u32 previous_sum;
    if (lidx == THREADS - 1)
    {
        while (status->finished_blocks < block_id);
        u32 local_sum = result[lidx + 3];
        previous_sum = status->sum;
        status->sum += local_sum;
        __threadfence();
        u32 finished_blocks = status->finished_blocks + 1;
        status->finished_blocks = finished_blocks;
    }
    __syncthreads();

    block_histograms[gidx + 0] = result[lidx + 0] + previous_sum;
    block_histograms[gidx + 1] = result[lidx + 1] + previous_sum;
    block_histograms[gidx + 2] = result[lidx + 2] + previous_sum;
    block_histograms[gidx + 3] = result[lidx + 3] + previous_sum;

    __syncthreads();

    /*
     * reset status for next iterations of radix sort
     * and write final histogram
     */
    if (lidx == 0 && block_id == num_blocks - 1)
    {
        status->sum = 0;
        status->blocks = 0;
        status->finished_blocks = 0;

        for (int i = 0; i < RADIX_SIZE; i++)
        {
            int bucket_idx = (i + 1) * bucket_size - 1;
            histogram[i] = block_histograms[bucket_idx];
        }
    }
}

__global__ void reorder_data(const u64_vec *data_in, u64_vec *data_out)
{

}


inline int static divup(int a, int b) { return (a + b - 1) / b; }

// https://www.cs.umd.edu/class/spring2021/cmsc714/readings/Satish-sorting.pdf
int radix_sort(int n, u64* input) {
    if (n % ELEM_PER_BLOCK != 0)
    {
        return -1;
    }

    const int blocks = divup(n, ELEM_PER_BLOCK);
    // for histogram each thread takes two elements
    const int blocks2 = divup(n, 2 * THREADS);
    // for histogram scan each thread takes one element
    const int blocks1 = divup(n, THREADS);

    u64 *data = NULL;
    cudaMalloc((void **) &data, n * sizeof(u64));
    cudaMemcpy(data, input, n * sizeof(u64), cudaMemcpyHostToDevice);

    u64 *data_tmp = NULL;
    cudaMalloc((void **) &data_tmp, n * sizeof(u64));

    u32 *block_histograms = NULL;
    cudaMalloc((void **) &block_histograms, blocks2 * RADIX_SIZE * sizeof(u32));

    u32 *histogram_scan = NULL;
    cudaMalloc((void **) &histogram_scan, blocks2 * RADIX_SIZE * sizeof(u32));

    scan_status status_host;
    status_host.sum = 0;
    status_host.blocks = 0;
    status_host.finished_blocks = 0;

    scan_status *status = NULL;
    cudaMalloc((void **) &status_host, sizeof(scan_status));
    cudaMemcpy(status, &status_host, sizeof(scan_status), cudaMemcpyHostToDevice);

    int start_bit = 0;
    while (start_bit < BITS)
    {
        /*
         * (1) sort blocks by iterating 1-bit split.
         * uses `ELEM_PER_BLOCK * sizeof(int)` of shared memory.
         */
        sort_block
            <<<blocks, THREADS, ELEM_PER_BLOCK * sizeof(u64)>>>
            ((u64_vec *) data, (u64_vec *) data_tmp, start_bit);
        /* (2) write histogram for each block to global memory */
        compute_histograms
            <<<blocks2, THREADS>>>
            ((u64_vec2 *) data_tmp, block_histograms, blocks2, start_bit);
        /* (3) prefix sum across blocks over the histograms */
        scan_histograms
            <<<blocks, THREADS>>>
            (block_histograms, histogram_scan, status, blocks, blocks2);
        /* (4) using histogram scan each block moves their elements to correct position */
        reorder_data
            <<<blocks, THREADS>>>
            ((u64_vec *) data_tmp, (u64_vec *) data);
        start_bit += RADIX;
    }

    cudaMemcpy(input, data, n * sizeof(u64), cudaMemcpyDeviceToHost);

    cudaFree(data);
    cudaFree(data_tmp);
    cudaFree(block_histograms);
    cudaFree(histogram_scan);

    return 0;
}
