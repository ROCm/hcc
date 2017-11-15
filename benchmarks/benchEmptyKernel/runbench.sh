#!/bin/bash 

./bench --dispatch_count 50000 --burst_count 1  $@

# burst of kernels:
./bench --dispatch_count 5000 --burst_count 100  $@

# Just the code mode dispatches
./bench --dispatch_count 5000 --burst_count 100   --tests 0x30 $@


