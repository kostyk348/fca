# libfca
CXX      ?= g++
CXXFLAGS := -O2 -march=native -std=gnu++20 -Wall -Wextra -Iinclude -fopenmp
CFLAGS    = $(CXXFLAGS)
OMPFLAGS := -fopenmp

all: fca_upscale fca_bench fca_video

fca_upscale: src/fca_upscale.cpp include/fca/rules.hpp include/fca/grid.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/fca_upscale.cpp

fca_bench: bench/bench.cpp include/fca/rules.hpp include/fca/grid.hpp
	$(CXX) $(CXXFLAGS) -o $@ bench/bench.cpp

fca_video: src/fca_video.cpp src/postfx.cpp src/vsr.cpp include/fca/rules.hpp include/fca/grid.hpp include/fca/denoise.hpp include/fca/temporal.hpp include/fca/postfx.hpp include/fca/vsr.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/fca_video.cpp src/postfx.cpp src/vsr.cpp

clean:
	rm -f fca_upscale fca_bench fca_video

.PHONY: all clean
