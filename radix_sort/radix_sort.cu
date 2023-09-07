#include "radix_sort.h"
#include "constants.h"
#include "scan.h"
#include "reduce.h"
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

/* 1-bit split */
__device__ int4 split(int *shared, const int4 bits)
{
    const int idx = threadIdx.x * ELEM_PER_THREAD;
    shared[idx + 0] = bits.x;
    shared[idx + 1] = bits.y;
    shared[idx + 2] = bits.z;
    shared[idx + 3] = bits.w;
    __syncthreads();
    scan::scan_block<int, false>(shared);
    __syncthreads();

    int4 ptr;
    ptr.x = shared[idx + 0];
    ptr.y = shared[idx + 1];
    ptr.z = shared[idx + 2];
    ptr.w = shared[idx + 3];

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

/*
 * sort block-wise in data out according to RADIX-bits starting from start_bit (LSB).
 * uses 1-bit splits RADIX times.
 */
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

/*
 * fill histogram by finding the start points of each radix in the sorted data block.
 * store start indices for use later on.
 */
__global__ void compute_histograms(
    const u64_vec *data,
    u32 *block_histograms,
    u32 *start_ptrs,
    const int num_blocks,
    const int start_bit
)
{
    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    u64_vec my_data = data[gidx];
    int4 my_radix;
    my_radix.x = (my_data.x >> start_bit) & RADIX_MASK;
    my_radix.y = (my_data.y >> start_bit) & RADIX_MASK;
    my_radix.z = (my_data.z >> start_bit) & RADIX_MASK;
    my_radix.w = (my_data.w >> start_bit) & RADIX_MASK;

    int histogram[RADIX_SIZE];

    #pragma unroll
    for (int i = 0; i < RADIX_SIZE; i++)
        histogram[i] = 0;

    histogram[my_radix.x]++;
    histogram[my_radix.y]++;
    histogram[my_radix.z]++;
    histogram[my_radix.w]++;

    #pragma unroll
    for (int i = 0; i < RADIX_SIZE; i++)
        histogram[i] = reduce::reduce_block<int>(histogram[i]);

    /* store in column major order */
    if (lidx < RADIX_SIZE)
    {
        const int hist_idx = num_blocks * lidx + blockIdx.x;
        block_histograms[hist_idx] = histogram[lidx];
    }

    if (lidx == 0)
    {
        int sum = 0;
        for (int i = 0; i < RADIX_SIZE; i++)
        {
            start_ptrs[blockIdx.x * RADIX_SIZE + i] = sum;
            sum += histogram[i];
        }
    }
}

/*
 * scan the block_histograms and add sum of each block to scan_sums
 */
template <bool add_total, bool inclusive>
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
    scan::scan_block<u32, inclusive>(result);
    __syncthreads();

    block_histograms[gidx + 0] = result[lidx + 0];
    block_histograms[gidx + 1] = result[lidx + 1];
    block_histograms[gidx + 2] = result[lidx + 2];
    block_histograms[gidx + 3] = result[lidx + 3];

    if (add_total && threadIdx.x == THREADS - 1)
    {
        scan_sums[blockIdx.x] = result[lidx + 3];
    }
}

/* add rolling sum from previous blocks */
__global__ void add_sums(const u32 *from, u32 *to)
{
    if (blockIdx.x == 0)
        return;

    __shared__ u32 sum;
    const int lidx = threadIdx.x * ELEM_PER_THREAD;
    const int gidx = blockIdx.x * ELEM_PER_BLOCK + lidx;

    if (lidx == 0)
    {
        sum = from[blockIdx.x];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++)
        to[gidx + i] += sum;
}


/*
 * (inclusive) scan-then-propagate: first scan histogram blocks and gather total sum for each.
 * iteratively scan over sum arrays and finally add offset sum to each histogram block.
 */
void global_scan(u32 *block_histograms,
                 u32 **scan_sums,
                 const int *scan_sizes,
                 const int scan_depth,
                 const int blocks)
{

    if (scan_depth == 0)
    {
        scan_histograms<false, true>
            <<<1, THREADS>>>
            (block_histograms, NULL);
        check_gpu_error("scan_histograms<false>");
        return;
    }

    /* first scan the histograms and gather sum for each block */
    scan_histograms<true, true>
        <<<blocks, THREADS>>>
        (block_histograms, scan_sums[scan_depth - 1]);
    check_gpu_error("scan_histograms<true>");

    /* iteratively scan the sum arrays if they are too large */
    for (int i = 0; i < scan_depth - 1; i++)
    {
        scan_histograms<true, false>
            <<<scan_sizes[i], THREADS>>>
            (scan_sums[i], scan_sums[i+1]);
        check_gpu_error("scan_histograms<true> in loop");
    }


    /* the final sum array is just a scan of one block */
    scan_histograms<false, false>
        <<<1, THREADS>>>
        (scan_sums[scan_depth - 1], NULL);
    check_gpu_error("scan_histograms<false>");

    /* iteratively in reverse add the scan totals back */
    for (int i = scan_depth - 1; i > 0; i--)
    {
        add_sums
            <<<scan_sizes[i - 1], THREADS>>>
            (scan_sums[i], scan_sums[i - 1]);
    }

    /* and finally to the histogram array */
    add_sums
        <<<blocks, THREADS>>>
        (scan_sums[0], block_histograms);
}

/*
 * given block sorted data, histogram and start index for each radix in the sorted data
 * we reorder across blocks.
 */
__global__ void reorder_data(const u64_vec *data_in,
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

    u64_vec my_data = data_in[gidx];
    int4 my_radix;
    my_radix.x = (my_data.x >> start_bit) & RADIX_MASK;
    my_radix.y = (my_data.y >> start_bit) & RADIX_MASK;
    my_radix.z = (my_data.z >> start_bit) & RADIX_MASK;
    my_radix.w = (my_data.w >> start_bit) & RADIX_MASK;

    if (lidx < 16)
    {
        global_ptrs[lidx] = block_histograms[lidx * bucket_size + blockIdx.x];
        local_ptrs[lidx] = start_ptrs[blockIdx.x * RADIX_SIZE + lidx];
    }
    __syncthreads();

    u32_vec my_offsets;
    my_offsets.x = global_ptrs[my_radix.x] + lidx - local_ptrs[my_radix.x];
    my_offsets.y = global_ptrs[my_radix.y] + lidx - local_ptrs[my_radix.y];
    my_offsets.z = global_ptrs[my_radix.z] + lidx - local_ptrs[my_radix.z];
    my_offsets.w = global_ptrs[my_radix.w] + lidx - local_ptrs[my_radix.w];

    data_out[my_offsets.x] = my_data.x;
    data_out[my_offsets.y] = my_data.y;
    data_out[my_offsets.z] = my_data.z;
    data_out[my_offsets.w] = my_data.w;
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
    const int scan_depth = std::floor(std::log(n) / std::log(ELEM_PER_BLOCK) - 1.0);

    /* main data array */
    u64 *data = NULL;
    cudaMalloc((void **) &data, n * sizeof(u64));
    cudaMemcpy(data, input, n * sizeof(u64), cudaMemcpyHostToDevice);

    /* stores data where each block is sorted */
    u64 *data_tmp = NULL;
    cudaMalloc((void **) &data_tmp, n * sizeof(u64));

    /* start index for each radix in each block */
    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks * RADIX_SIZE * sizeof(u32));

    /* histogram of radixes from blocks in column major order */
    u32 *block_histograms = NULL;
    cudaMalloc((void **) &block_histograms, blocks * RADIX_SIZE * sizeof(u32));

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
            <<<blocks, THREADS>>>
            ((u64_vec *) data_tmp, block_histograms, start_ptrs, blocks, start_bit);
        check_gpu_error("compute_histograms");

        /* (3) prefix sum across blocks over the histograms. */
        global_scan(block_histograms, scan_sums, scan_sizes, scan_depth, blocks);

        /* (4) using histogram scan each block moves their elements to correct position */
        reorder_data
            <<<blocks, THREADS>>>
            ((u64_vec *) data_tmp, data, block_histograms, start_ptrs, blocks, start_bit);
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
