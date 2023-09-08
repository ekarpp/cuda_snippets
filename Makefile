BIN=benchmark test
NVCC=nvcc
CXX=g++
NVCCFLAGS=-G -g --std=c++14 -O3 -arch=sm_61
CXXFLAGS=-g -Wall -Wextra --std=c++14 -O3 -march=native

all: $(BIN)

clean:
	rm -f *.o $(BIN)

benchmark: radix_sort.o benchmark.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test: test.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

radix_sort.o: radix_sort.cu scan.h reduce.h
test.o: test.cu radix_sort.o

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<
