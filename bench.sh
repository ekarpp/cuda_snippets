#!/usr/bin/bash

make clean
make

for exp in 24 26 27 28
do
	./benchmark $((2**$exp)) 6 > $exp.result
done
