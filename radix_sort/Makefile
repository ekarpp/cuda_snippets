NVCC=nvcc
CXX=g++
NVCCFLAGS=-G -g --std=c++14 -O3 -arch=sm_61
CXXFLAGS=-g -Wall -Wextra --std=c++14 -O3 -march=native

all: radix_sort_benchmark radix_sort_test

clean:
	rm -f *.o radix_sort_benchmark

radix_sort_benchmark: radix_sort.o radix_sort_benchmark.o
	$(NVCC) $^ -o $@

radix_sort_test: radix_sort.o radix_sort_test.o
	$(NVCC) $^ -o $@


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<
