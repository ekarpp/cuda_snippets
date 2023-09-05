#ifndef SCAN_H
#define SCAN_H

#include "constants.h"
#include <cuda_runtime.h>

namespace scan
{
    constexpr uint FULL_MASK = 0xFFFFFFFF;

    /*
     * scans at warp level then combines the results
     */
    template <typename T>
    __device__ T scan_warp(T my_val)
    {
        __shared__ T warp_sums[WARPS_PER_BLOCK];

        const int idx = threadIdx.x;
        const int lane_id = threadIdx.x % WARP_SIZE;

        #pragma unroll
        for (int i = 1; i < WARP_SIZE; i*=2)
        {
            T neighbor_val = __shfl_up_sync(FULL_MASK, my_val, i);
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

            #pragma unroll
            for (int i = 1; i < WARP_SIZE; i *= 2)
            {
                T neighbor_val = __shfl_up_sync(FULL_MASK, sum, i);
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
     * have to be aware of possible overflow for T.
     */
    template <typename T>
    __device__ void scan_block(T *data)
    {
        const int idx = threadIdx.x * ELEM_PER_THREAD;

        T my_data[ELEM_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEM_PER_THREAD; i++)
            my_data[i] = data[idx + i];

        T my_sum = 0;
        #pragma unroll
        for (int i = 0; i < ELEM_PER_THREAD; i++)
            my_sum += my_data[i];

        my_sum = scan_warp<T>(my_sum);

        data[idx + ELEM_PER_THREAD - 1] = my_sum - my_data[ELEM_PER_THREAD - 1];
        #pragma unroll
        for (int i = ELEM_PER_THREAD - 2; i >= 0; i--)
        {
            data[idx + i] = my_sum - my_data[i];
            my_sum -= my_data[i];
        }
    }
}
#endif
