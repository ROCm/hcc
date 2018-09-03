// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace hc;

#define T float
#define INIT 0.5f
#define TOLERANCE 1e-5

int main(void) {
  #if defined(FLOAT_ATOMICS)
    const int vecSize = 100;
    const int tile_size = 10;

    // Alloc & init input data
    extent<2> e_a(vecSize, vecSize);
    std::vector<T> va(vecSize * vecSize, INIT);
    array_view<T, 2> av_a(e_a, va); 

    extent<2> compute_domain(e_a);
    parallel_for_each(
      compute_domain.tile(tile_size, tile_size), [=](tiled_index<2> tidx) [[hc]] {
      tile_static T localA[tile_size][tile_size];
      localA[tidx.local[0]][tidx.local[1]] = 0;
      tidx.barrier.wait();

      for(int i = 0; i < tile_size; i++) {
        for(int j = 0; j < tile_size; j++) {
          atomic_fetch_add(&(localA[i][j]), INIT);
        }
      }
    tidx.barrier.wait();
    av_a[tidx.global] = localA[tidx.local[0]][tidx.local[1]];
    });

    // accumlate tile_size * tile_size times
    float sum = 0.0f;
    for (int i = 0; i < tile_size * tile_size; ++i)
      sum += INIT;
    for(unsigned i = 0; i < vecSize; i++) {
      for(unsigned j = 0; j < vecSize; j++) {
        if(fabs(av_a(i, j) - sum) > TOLERANCE) {
          return 1;
        }
      }
    }

    return 0;
  #else
    return EXIT_SUCCESS;
  #endif
}
