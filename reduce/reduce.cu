#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "errchk.h"
#include "reduce.cuh"

#define THREADS 256
#define MAX(x,y) ((x > y) ? x : y)
#define MIN(x,y) ((x > y) ? y : x)

// reduce single warp
__device__ void warp_reduce(volatile int *mxs, int id)
{
    mxs[id] = MAX(mxs[id], mxs[id + 32]);
    mxs[id] = MAX(mxs[id], mxs[id + 16]);
    mxs[id] = MAX(mxs[id], mxs[id + 8]);
    mxs[id] = MAX(mxs[id], mxs[id + 4]);
    mxs[id] = MAX(mxs[id], mxs[id + 2]);
    mxs[id] = MAX(mxs[id], mxs[id + 1]);
}


__global__ void reduce_kernel(int *arr, const size_t count)
{
    __shared__ int mxs[THREADS];
    int id = threadIdx.x;
    int first = MIN(blockIdx.x * THREADS * 2 + id, count - 1);
    int second = MIN(first + THREADS, count - 1);

    mxs[id] = MAX(arr[first], arr[second]);
    __syncthreads();
    for (int i = THREADS / 2; i > 32; i >>= 1)
    {
        if (id < i)
            mxs[id] = MAX(mxs[id], mxs[id + i]);
        __syncthreads();
    }

    if (id < 32)
        warp_reduce(mxs, id);

    if (!id)
        // should be free of race cond
        arr[blockIdx.x] = MAX(arr[blockIdx.x], mxs[0]);
}

int reduce(const int* arr, const size_t initial_count)
{
    int *arr_gpu = NULL;
    cudaMalloc((void **) &arr_gpu, initial_count * sizeof(int));
    cudaMemcpy(arr_gpu, arr, initial_count * sizeof(int), cudaMemcpyHostToDevice);

    int count = initial_count;
    while (count >= THREADS)
    {
        cudaDeviceSynchronize();
        // we might get one extra to count but negligible
        int xtra = (count % (2*THREADS)) ? 1 : 0;
        reduce_kernel<<<(count / (2*THREADS)) + xtra, THREADS>>>(arr_gpu, count);
        count = (count / THREADS) + xtra;
    }
    cudaDeviceSynchronize();
    if (count != 1)
        reduce_kernel<<<1, THREADS>>>(arr_gpu, count);

    int sol;
    cudaMemcpy(&sol, arr_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(arr_gpu);

    return sol;
}
