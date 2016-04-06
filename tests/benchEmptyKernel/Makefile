# run kernel # of times
N := 500000

bench: bench.cpp
	hcc `hcc-config --build --cxxflags --ldflags` bench.cpp -o bench

plot: bench
	./bench ${N}
	gnuplot plot.plt
	@echo
	@echo "Done, please check the svg files."

clean:
	rm -f bench


.PHONY: clean plot
