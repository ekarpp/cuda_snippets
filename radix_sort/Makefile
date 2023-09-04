NVCC=nvcc
CXX=g++
CXXFLAGS=-g --std=c++20 -O3

all: radix_sort_benchmark radix_sort_test

clean:
	rm -f *.o radix_sort_benchmark

radix_sort.o: radix_sort.cu
	$(NVCC) $(CXXFLAGS) -o $@ -c $<

radix_sort_benchmark: radix_sort.o radix_sort_benchmark.o
	$(NVCC) $^ -o $@

radix_sort_test: radix_sort.o radix_sort_test.o
	$(NVCC) $^ -o $@
