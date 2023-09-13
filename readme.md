CUDA implementation of the GPU radix sort algorithm by Satish, Harris, and Garland [[1]](https://www.cs.umd.edu/class/spring2021/cmsc714/readings/Satish-sorting.pdf). The algorithm consists of four phases that are iterated:
1. Blocks are sorted by iterating 1-bit split
2. Histogram of radixes is written for each block
3. Exclusive scan over the histograms
4. Using histogram scan the elements are reordered

This implementation is for 32 bit unsigned integers and follows much of what is described by Satish et al.; we use blocks of 256 threads and iterate 4 bits at a time. The scan operation uses a scan-then-propagate scheme (scan blocks locally and gather total sums, add total sums to local scans) with local scans using a warp-scan algorithm.


![Scalability](https://i.imgur.com/TXTZ4Up.png)

GPU sorting times include copying data to and from the device.

###### Bibliography
<sup><sub>
[1] Satish, N., Harris, M. and Garland, M. Designing efficient sorting algorithms for manycore GPUs. In *2009 IEEE International Symposium on Parallel & Distributed Processing, IPDPS 2009, Rome, Italy, May 23-29, 2009*, IEEE, pages 1-10.
</sub></sup>
