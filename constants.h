#ifndef CONSTANT_H
#define CONSTANT_H

/* tunable */
constexpr int THREADS = 256;
// this should be 2 or 4, needs changes to code if we want 2
constexpr int ELEM_PER_THREAD = 4;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS / WARP_SIZE;

constexpr int ELEM_PER_BLOCK = THREADS * ELEM_PER_THREAD;
// constexpr int WARP_SIZE = 32;

/* tunable */
constexpr int BITS = 32;
constexpr int RADIX = 4;

constexpr int RADIX_SIZE = 1 << RADIX;
constexpr int RADIX_MASK = RADIX_SIZE - 1;

#endif
