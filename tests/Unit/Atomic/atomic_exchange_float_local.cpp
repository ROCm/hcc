// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace concurrency;

#define T float
#define INIT 0.5f
#define NEW_VALUE 99.5f

int main(void) {
  const int vecSize = 100;
  const int tile_size = 10;

  // Alloc & init input data
  extent<2> e_a(vecSize, vecSize);
  std::vector<T> va(vecSize * vecSize, INIT);
  array_view<T, 2> av_a(e_a, va); 

  extent<2> compute_domain(e_a);
  parallel_for_each(compute_domain.tile<tile_size, tile_size>(), [=] (tiled_index<tile_size, tile_size> tidx) restrict(amp,cpu) {
    index<2> localIdx = tidx.local;
    index<2> globalIdx = tidx.global;

    tile_static T localA[tile_size][tile_size];
    localA[localIdx[0]][localIdx[1]] = 0;
    tidx.barrier.wait();

    for(int i = 0; i < tile_size; i++) {
      for(int j = 0; j < tile_size; j++) {
        atomic_exchange(&(localA[i][j]), NEW_VALUE);
      }
    }
    tidx.barrier.wait();
    av_a[globalIdx[0]][globalIdx[1]] = localA[localIdx[0]][localIdx[1]];
  });

  for(int i = 0; i < vecSize; ++i) {
    for(int j = 0; j < vecSize; ++j) {
      if(av_a(i, j) != NEW_VALUE) {
        return 1;
      }
    }
  }

  return 0;
}
