# run kernel # of times
N := 500000

# TODO - should enable OPT=-O3 for benchmarking.
OPT=-O3

ROCM_PATH=/opt/rocm

bench: bench.cpp
	hcc `hcc-config --build --cxxflags --ldflags` $(OPT) -I$(ROCM_PATH)/include bench.cpp -o bench  -L$(ROCM_PATH)/lib -lhsa-runtime64

plot: bench
	./bench ${N}
	gnuplot plot.plt
	@echo
	@echo "Done, please check the svg files."

clean:
	rm -f bench


.PHONY: clean plot
