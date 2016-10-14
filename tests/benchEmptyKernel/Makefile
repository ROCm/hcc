# run kernel # of times
N := 500000

# TODO - should enable OPT=-O3 for benchmarking.
OPT=-O3

ROCM_PATH=/opt/rocm

SOURCES=bench.cpp
EX_SOURCES=statutils.CPP hsacodelib.CPP

OBJECTS=$(SOURCES:.cpp=.o) $(EX_SOURCES:.CPP=.o)


bench: $(OBJECTS)
	hcc `hcc-config --build --ldflags` $(OPT) $(OBJECTS) -o bench  -L$(ROCM_PATH)/lib -lhsa-runtime64

%.o: %.CPP
	hcc `hcc-config --build --cxxflags` $(OPT) -I$(ROCM_PATH)/include $< -c -o $@

%.o: %.cpp
	hcc `hcc-config --build --cxxflags` $(OPT) -I$(ROCM_PATH)/include $< -c -o $@

plot: bench
	./bench ${N}
	gnuplot plot.plt
	@echo
	@echo "Done, please check the svg files."

clean:
	rm -f bench *.o


.PHONY: clean plot
