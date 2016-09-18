# run kernel # of times
N := 500000

# TODO - should enable OPT=-O3 for benchmarking.
OPT=

bench: bench.cpp
	hcc `hcc-config --build --cxxflags --ldflags` $(OPT) bench.cpp -o bench

plot: bench
	./bench ${N}
	gnuplot plot.plt
	@echo
	@echo "Done, please check the svg files."

clean:
	rm -f bench


.PHONY: clean plot
