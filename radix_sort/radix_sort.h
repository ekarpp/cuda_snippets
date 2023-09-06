#ifndef RADIX_SORT_H
#define RADIX_SORT_H

typedef unsigned long long u64;
typedef unsigned long u32;

typedef ulonglong2 u64_vec2;
/* TODO: adjust for other vector lengths */
typedef ulonglong4 u64_vec;


int radix_sort(int n, u64* input);

#endif
