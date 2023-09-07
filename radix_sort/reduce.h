#ifndef REDUCE_H
#define REDUCE_H

#include "constants.h"
#include <cuda_runtime.h>


namespace reduce
{
    constexpr uint FULL_MASK = 0xFFFFFFFF;

    template <typename T>
    __device__ T reduce_block(T my_val)
    {
        __shared__ T warp_reduces[WARPS_PER_BLOCK];

        const int idx = threadIdx.x;
        const int lane_id = idx % WARP_SIZE;

        #pragma unroll
        for (int i = WARP_SIZE / 2; i > 0; i /= 2)
            my_val += __shfl_down_sync(FULL_MASK, my_val, i);

        if (lane_id == 0)
            warp_reduces[idx / WARP_SIZE] = my_val;
        __syncthreads();

        if (idx < WARP_SIZE)
        {
            T sum = 0;
            if (idx < WARPS_PER_BLOCK)
                sum = warp_reduces[idx];

            #pragma unroll
            for (int i = WARP_SIZE / 2; i > 0; i /= 2)
                sum += __shfl_down_sync(FULL_MASK, sum, i);

            if (idx == 0)
                warp_reduces[0] = sum;
        }

        __syncthreads();

        return warp_reduces[0];
    }
}
#endif
