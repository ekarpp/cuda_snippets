#!/usr/bin/bash

make clean
make

for exp in 16 18 20 22 24 26 28
do
	./benchmark $((2**$exp)) 6 > $exp.result
done
