#include "radix_sort.h"
#include <cuda_runtime.h>
#include <iostream>

/* tunable */
constexpr int THREADS = 256;
// this should be 2 or 4, needs changes to code if we want 2
constexpr int ELEM_PER_THREAD = 4;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS / WARP_SIZE;

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

void check_gpu_error(const char *fn)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in \"" << fn << "\": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

template <typename T>
__device__ T scan_warp(T my_val)
{
    __shared__ T warp_sums[WARPS_PER_BLOCK];

    const int idx = threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;

    for (int i = 1; i < WARP_SIZE; i*=2)
    {
        T neighbor_val = __shfl_up_sync(-1, my_val, i);
        if (lane_id >= i)
            my_val += neighbor_val;
    }

    if (lane_id == WARP_SIZE - 1)
    {
        warp_sums[idx / WARP_SIZE] = my_val;
    }
    __syncthreads();

    if (idx < WARP_SIZE)
    {
        T sum = 0;
        if (idx < WARPS_PER_BLOCK)
            sum = warp_sums[idx];

        for (int i = 1; i < WARP_SIZE; i *= 2)
        {
            T neighbor_val = __shfl_up_sync(-1, sum, i);
            if (lane_id >= i)
                sum += neighbor_val;
        }

        if (idx < WARPS_PER_BLOCK)
            warp_sums[idx] = sum;
    }

    __syncthreads();

    if (idx >= WARP_SIZE)
        my_val += warp_sums[idx / WARP_SIZE - 1];

    return my_val;
}

/*
 * warp-scan algorithm. adds all elements of a thread together and performs a single warp-scan.
 * HAVE TO DOUBLE CHECK FOR OVERFLOW. (histogram+split)
 */
template <typename T>
__device__ void scan_block(T *data)
{
    const int idx = threadIdx.x * ELEM_PER_THREAD;
    T my_data[ELEM_PER_THREAD];
    for (int i = 0; i < ELEM_PER_THREAD; i++)
        my_data[i] = data[idx + i];

    T my_sum = 0;
    for (int i = 0; i < ELEM_PER_THREAD; i++)
        my_sum += my_data[i];

    my_sum = scan_warp<T>(my_sum);

    data[idx + ELEM_PER_THREAD - 1] = my_sum - my_data[ELEM_PER_THREAD - 1];
    for (int i = ELEM_PER_THREAD - 2; i >= 0; i--)
    {
        data[idx + i] = my_sum - my_data[i];
        my_sum -= my_data[i];
    }
}

__device__ int4 split(int *shared, int4 bits)
{
    const int idx = threadIdx.x * ELEM_PER_THREAD;
    shared[idx + 0] = bits.x;
    shared[idx + 1] = bits.y;
    shared[idx + 2] = bits.z;
    shared[idx + 3] = bits.w;
    __syncthreads();
    scan_block<int>(shared);
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

    for (int bit = start_bit; bit > start_bit - RADIX; bit++)
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

__global__ void scan_histograms(u32 *block_histograms, scan_status *status, const int num_blocks)
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
    scan_block<u32>(result);
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
    }
}

__global__ void reorder_data(const u64_vec2 *data_in,
                             u64 *data_out,
                             u32 *block_histograms,
                             u32 *start_ptrs,
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
    // partial blocks not implemented and overflow if too large
    if (n % ELEM_PER_BLOCK != 0 || n >= 1 << 30)
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

    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks2 * RADIX_SIZE * sizeof(u32));

    scan_status status_host;
    status_host.sum = 0;
    status_host.blocks = 0;
    status_host.finished_blocks = 0;

    scan_status *status = NULL;
    cudaMalloc((void **) &status, sizeof(scan_status));
    cudaMemcpy(status, &status_host, sizeof(scan_status), cudaMemcpyHostToDevice);
    check_gpu_error("init");

    int start_bit = 0;
    while (start_bit < BITS)
    {
        /*
         * (1) sort blocks by iterating 1-bit split.
         * uses `ELEM_PER_BLOCK * sizeof(int)` of shared memory.
         */
        sort_block
            <<<blocks, THREADS, WARPS_PER_BLOCK * sizeof(u64)>>>
            ((u64_vec *) data, (u64_vec *) data_tmp, start_bit);
        check_gpu_error("sort_block");
        /* (2) write histogram for each block to global memory */
        compute_histograms
            <<<blocks2, THREADS>>>
            ((u64_vec2 *) data_tmp, block_histograms, start_ptrs, blocks2, start_bit);
        check_gpu_error("compute_histograms");
        /* (3) prefix sum across blocks over the histograms */
        scan_histograms
          <<<blocks, THREADS>>>
          (block_histograms, status, blocks);
        check_gpu_error("scan_histograms");
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
    cudaFree(status);
    cudaFree(start_ptrs);

    return 0;
}
