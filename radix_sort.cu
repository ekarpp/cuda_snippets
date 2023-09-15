#include "radix_sort.h"
#include "constants.h"
#include "scan.h"
#include <cuda_runtime.h>
#include <iostream>

typedef uint4 u32_vec;

void check_gpu_error(const char *fn)
{
#ifdef DEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cout << "CUDA error in \"" << fn << "\": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
#endif
}

/* 1-bit split */
__device__ int4 split(const int4 bits)
{
    __shared__ int ptrs[ELEM_PER_BLOCK];
    const int lidx = threadIdx.x;

    ptrs[lidx + 0 * THREADS] = bits.x;
    ptrs[lidx + 1 * THREADS] = bits.y;
    ptrs[lidx + 2 * THREADS] = bits.z;
    ptrs[lidx + 3 * THREADS] = bits.w;

    __syncthreads();
    scan::scan_block<int, false>(ptrs);
    __syncthreads();

    int4 ptr;
    ptr.x = ptrs[lidx + 0 * THREADS];
    ptr.y = ptrs[lidx + 1 * THREADS];
    ptr.z = ptrs[lidx + 2 * THREADS];
    ptr.w = ptrs[lidx + 3 * THREADS];

    __shared__ uint trues;
    if (threadIdx.x == THREADS - 1)
        trues = ptr.w + bits.w;
    __syncthreads();

    ptr.x = (bits.x) ? ptr.x : trues + lidx + 0 * THREADS - ptr.x;
    ptr.y = (bits.y) ? ptr.y : trues + lidx + 1 * THREADS - ptr.y;
    ptr.z = (bits.z) ? ptr.z : trues + lidx + 2 * THREADS - ptr.z;
    ptr.w = (bits.w) ? ptr.w : trues + lidx + 3 * THREADS - ptr.w;

    return ptr;
}

/*
 * sort block-wise in data out according to RADIX-bits starting from start_bit (LSB).
 * uses 1-bit splits RADIX times.
 */
__global__ void sort_block(const u32 *data_in, u32 *data_out, const int start_bit)
{
    __shared__ u32 shared[ELEM_PER_BLOCK];

    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * ELEM_PER_BLOCK + lidx;

    u32_vec my_data;
    my_data.x = data_in[gidx + 0 * THREADS];
    my_data.y = data_in[gidx + 1 * THREADS];
    my_data.z = data_in[gidx + 2 * THREADS];
    my_data.w = data_in[gidx + 3 * THREADS];

    #pragma unroll
    for (int bit = start_bit; bit < start_bit + RADIX; bit++)
    {
        /* TODO: adjust for other vector lengths */
        int4 bits;
        bits.x = !((my_data.x >> bit) & 1);
        bits.y = !((my_data.y >> bit) & 1);
        bits.z = !((my_data.z >> bit) & 1);
        bits.w = !((my_data.w >> bit) & 1);

        int4 ptr = split(bits);

        /* bank issues? */
        shared[ptr.x] = my_data.x;
        shared[ptr.y] = my_data.y;
        shared[ptr.z] = my_data.z;
        shared[ptr.w] = my_data.w;
        __syncthreads();

        my_data.x = shared[lidx + 0 * THREADS];
        my_data.y = shared[lidx + 1 * THREADS];
        my_data.z = shared[lidx + 2 * THREADS];
        my_data.w = shared[lidx + 3 * THREADS];
    }

    data_out[gidx + 0 * THREADS] = my_data.x;
    data_out[gidx + 1 * THREADS] = my_data.y;
    data_out[gidx + 2 * THREADS] = my_data.z;
    data_out[gidx + 3 * THREADS] = my_data.w;
}

/*
 * fill histogram by finding the start points of each radix in the sorted data block.
 * store start indices for use later on.
 */
__global__ void compute_histograms(
    const u32_vec *data,
    u32 *block_histograms,
    u32 *start_ptrs,
    const int num_blocks,
    const int start_bit
)
{
    // radix of final element for each thread
    __shared__ int radix[THREADS];
    // start index for each radix
    __shared__ int ptrs[RADIX_SIZE];

    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    u32_vec my_data = data[gidx];
    int4 my_radix;
    my_radix.x = (my_data.x >> start_bit) & RADIX_MASK;
    my_radix.y = (my_data.y >> start_bit) & RADIX_MASK;
    my_radix.z = (my_data.z >> start_bit) & RADIX_MASK;
    my_radix.w = (my_data.w >> start_bit) & RADIX_MASK;

    radix[lidx] = my_radix.w;

    if (lidx < RADIX_SIZE)
        ptrs[lidx] = 0;
    __syncthreads();

    if (lidx > 0 && my_radix.x != radix[lidx - 1])
        ptrs[my_radix.x] = ELEM_PER_THREAD * lidx;
    if (my_radix.x != my_radix.y)
        ptrs[my_radix.y] = ELEM_PER_THREAD * lidx + 1;
    if (my_radix.y != my_radix.z)
        ptrs[my_radix.z] = ELEM_PER_THREAD * lidx + 2;
    if (my_radix.z != my_radix.w)
        ptrs[my_radix.w] = ELEM_PER_THREAD * lidx + 3;
    __syncthreads();


    if (lidx < RADIX_SIZE)
        start_ptrs[blockIdx.x * RADIX_SIZE + lidx] = ptrs[lidx];
    __syncthreads();


    if (lidx > 0 && my_radix.x != radix[lidx - 1])
        ptrs[radix[lidx - 1]] = ELEM_PER_THREAD * lidx + 0 - ptrs[radix[lidx - 1]];
    if (my_radix.x != my_radix.y)
        ptrs[my_radix.x] = ELEM_PER_THREAD * lidx + 1 - ptrs[my_radix.x];
    if (my_radix.y != my_radix.z)
        ptrs[my_radix.y] = ELEM_PER_THREAD * lidx + 2 - ptrs[my_radix.y];
    if (my_radix.z != my_radix.w)
        ptrs[my_radix.z] = ELEM_PER_THREAD * lidx + 3 - ptrs[my_radix.z];
    if (lidx == THREADS - 1)
        ptrs[radix[lidx]] = ELEM_PER_BLOCK - ptrs[radix[lidx]];
    __syncthreads();

    if (lidx < RADIX_SIZE)
        block_histograms[num_blocks * lidx + blockIdx.x] = ptrs[lidx];
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

    if (add_total && threadIdx.x == THREADS - 1)
        scan_sums[blockIdx.x] = result[lidx + 3] + block_histograms[gidx + 3];

    block_histograms[gidx + 0] = result[lidx + 0];
    block_histograms[gidx + 1] = result[lidx + 1];
    block_histograms[gidx + 2] = result[lidx + 2];
    block_histograms[gidx + 3] = result[lidx + 3];
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
        sum = from[blockIdx.x];
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++)
        to[gidx + i] += sum;
}

/*
 * (exclusive) scan-then-propagate: first scan histogram blocks and gather total sum for each.
 * iteratively scan over sum arrays and finally add offset sum to each histogram block.
 */
void global_scan(u32 *block_histograms,
                 u32 **scan_sums,
                 const int *scan_sizes,
                 const int scan_depth)
{
    if (scan_depth == 0)
    {
        scan_histograms<false, false>
            <<<1, THREADS>>>
            (block_histograms, NULL);
        check_gpu_error("scan_histograms<false, false>");
        return;
    }

    /* first scan the histograms and gather sum for each block */
    scan_histograms<true, false>
        <<<scan_sizes[0], THREADS>>>
        (block_histograms, scan_sums[0]);
    check_gpu_error("scan_histograms<true, false>");

    /* iteratively scan the sum arrays if they are too large */
    for (int i = 0; i < scan_depth - 1; i++)
    {
        scan_histograms<true, false>
            <<<scan_sizes[i + 1], THREADS>>>
            (scan_sums[i], scan_sums[i + 1]);
        check_gpu_error("scan_histograms<true, false>, loop");
    }

    /* the final sum array is just a scan of one block */
    scan_histograms<false, false>
        <<<1, THREADS>>>
        (scan_sums[scan_depth - 1], NULL);
    check_gpu_error("scan_histograms<false, false>");

    /* iteratively in reverse add the scan totals back */
    for (int i = scan_depth - 1; i > 0; i--)
    {
        add_sums
            <<<scan_sizes[i], THREADS>>>
            (scan_sums[i], scan_sums[i - 1]);
        check_gpu_error("add_sums, loop");
    }

    /* and finally to the histogram array */
    add_sums
        <<<scan_sizes[0], THREADS>>>
        (scan_sums[0], block_histograms);
    check_gpu_error("add_sums");
}

/*
 * given block sorted data, histogram and start index for each radix in the sorted data
 * we reorder across blocks.
 */
__global__ void reorder_data(const u32_vec *data_in,
                             u32 *data_out,
                             const u32 *block_histograms,
                             const u32 *start_ptrs,
                             const int bucket_size,
                             const int start_bit)
{
    __shared__ u32 global_ptrs[RADIX_SIZE];
    __shared__ u32 local_ptrs[RADIX_SIZE];

    const int lidx = threadIdx.x;
    const int gidx = blockIdx.x * THREADS + lidx;

    u32_vec my_data = data_in[gidx];
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

    u32_vec my_indices;
    my_indices.x = global_ptrs[my_radix.x] + (ELEM_PER_THREAD * lidx + 0) - local_ptrs[my_radix.x];
    my_indices.y = global_ptrs[my_radix.y] + (ELEM_PER_THREAD * lidx + 1) - local_ptrs[my_radix.y];
    my_indices.z = global_ptrs[my_radix.z] + (ELEM_PER_THREAD * lidx + 2) - local_ptrs[my_radix.z];
    my_indices.w = global_ptrs[my_radix.w] + (ELEM_PER_THREAD * lidx + 3) - local_ptrs[my_radix.w];

    data_out[my_indices.x] = my_data.x;
    data_out[my_indices.y] = my_data.y;
    data_out[my_indices.z] = my_data.z;
    data_out[my_indices.w] = my_data.w;
}


inline int static divup(int a, int b) { return (a + b - 1) / b; }

// https://www.cs.umd.edu/class/spring2021/cmsc714/readings/Satish-sorting.pdf
int radix_sort(int n, u32* input) {
    // overflow in scan if too large (can increase to u64)
    if (n >= 1 << 30)
        return -1;

    const int blocks = divup(n, ELEM_PER_BLOCK);
    const int num_elems = ELEM_PER_BLOCK * blocks;

    /* main data array */
    u32 *data = NULL;
    cudaMalloc((void **) &data, blocks * ELEM_PER_BLOCK * sizeof(u32));
    cudaMemcpy(data, input, blocks * ELEM_PER_BLOCK * sizeof(u32), cudaMemcpyHostToDevice);
    if (num_elems > n)
        cudaMemset(data + n, 0xFF, (num_elems - n) * sizeof(u32));

    /* stores data where each block is sorted */
    u32 *data_tmp = NULL;
    cudaMalloc((void **) &data_tmp, blocks * ELEM_PER_BLOCK * sizeof(u32));

    /* start index for each radix in each block */
    u32 *start_ptrs = NULL;
    cudaMalloc((void **) &start_ptrs, blocks * RADIX_SIZE * sizeof(u32));

    /* histogram of radixes from blocks in column major order */
    u32 *block_histograms = NULL;
    cudaMalloc((void **) &block_histograms, blocks * RADIX_SIZE * sizeof(u32));
    const int scan_depth = std::floor(std::log(RADIX_SIZE * blocks) / std::log(ELEM_PER_BLOCK) - 1e-10);

    /* sum of each block during histogram scan */
    u32 *scan_sums[scan_depth];
    int scan_sizes[scan_depth];

    for (int i = 0; i < scan_depth; i++)
    {
        scan_sums[i] = NULL;
        scan_sizes[i] = (i == 0)
            ? divup(RADIX_SIZE * blocks, ELEM_PER_BLOCK)
            : divup(scan_sizes[i - 1], ELEM_PER_BLOCK);
        cudaMalloc((void **) &scan_sums[i], std::max(ELEM_PER_BLOCK, scan_sizes[i]) * sizeof(u32));
    }

    int start_bit = 0;
    #pragma unroll
    while (start_bit < BITS)
    {
        /*
         * (1) sort blocks by iterating 1-bit split.
         */
        sort_block
            <<<blocks, THREADS>>>
            (data, data_tmp, start_bit);
        check_gpu_error("sort_block");

        /* (2) write histogram for each block to global memory */
        compute_histograms
            <<<blocks, THREADS>>>
            ((u32_vec *) data_tmp, block_histograms, start_ptrs, blocks, start_bit);
        check_gpu_error("compute_histograms");

        /* (3) prefix sum across blocks over the histograms. */
        global_scan(block_histograms, scan_sums, scan_sizes, scan_depth);

        /* (4) using histogram scan each block moves their elements to correct position */
        reorder_data
            <<<blocks, THREADS>>>
            ((u32_vec *) data_tmp, data, block_histograms, start_ptrs, blocks, start_bit);
        check_gpu_error("reorder_data");

        start_bit += RADIX;
    }

    cudaMemcpy(input, data, n * sizeof(u32), cudaMemcpyDeviceToHost);

    cudaFree(data);
    cudaFree(data_tmp);
    cudaFree(block_histograms);
    cudaFree(start_ptrs);

    for (int i = 0; i < scan_depth; i++)
        cudaFree(scan_sums[i]);

    return 0;
}
