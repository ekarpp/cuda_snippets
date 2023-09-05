NVCC=nvcc
CXX=g++
NVCCFLAGS=-G -g --std=c++20 -O3
CXXFLAGS=-g -Wall -Wextra --std=c++20 -O3 -march=native

all: radix_sort_benchmark radix_sort_test

clean:
	rm -f *.o radix_sort_benchmark

radix_sort.o: radix_sort.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

radix_sort_benchmark.o: radix_sort_benchmark.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<

radix_sort_test.o: radix_sort_test.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<

radix_sort_benchmark: radix_sort.o radix_sort_benchmark.o
	$(NVCC) $^ -o $@

radix_sort_test: radix_sort.o radix_sort_test.o
	$(NVCC) $^ -o $@
