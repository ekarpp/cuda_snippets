BIN=radix_sort_benchmark radix_sort_test
NVCC=nvcc
CXX=g++
NVCCFLAGS=-G -g --std=c++14 -O3 -arch=sm_61
CXXFLAGS=-g -Wall -Wextra --std=c++14 -O3 -march=native

all: $(BIN)

clean:
	rm -f *.o $(BIN)

radix_sort_benchmark: radix_sort.o radix_sort_benchmark.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

radix_sort_test: radix_sort_test.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

radix_sort.o: radix_sort.cu scan.h reduce.h
radix_sort_test.o: radix_sort_test.cu radix_sort.o

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<
