# run kernel # of times
N := 500000


plot: run
	./bench ${N}
	gnuplot plot.plt
	@echo
	@echo "Done, please check pfe.svg"


bench: bench.cpp
	hcc `hcc-config --build --cxxflags --ldflags` bench.cpp -o bench


clean:
	rm bench


.PHONY: clean plot
