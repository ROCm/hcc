// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>

using namespace concurrency;

int main(void) {
  const int vecSize = 100;
  const int tile_size = 10;

  // Alloc & init input data
  extent<2> e_a(vecSize, vecSize);
  std::vector<int> va(vecSize * vecSize);
  for(int i = 0; i < vecSize * vecSize; i++) {
    va[i] = 0;
  }
  array_view<int, 2> av_a(e_a, va); 

  extent<2> compute_domain(e_a);
  parallel_for_each(compute_domain.tile<tile_size, tile_size>(), [=] (tiled_index<tile_size, tile_size> tidx) restrict(amp,cpu) {
    index<2> localIdx = tidx.local;
    index<2> globalIdx = tidx.global;
    int v = 0;

    tile_static int localA[tile_size][tile_size];
    localA[localIdx[0]][localIdx[1]] = 0;
    tidx.barrier.wait();

    for(int i = 0; i < tile_size; i++) {
      for(int j = 0; j < tile_size; j++) {
        atomic_compare_exchange(&(localA[i][j]), &v, 2);
      }
    }
    tidx.barrier.wait();
    av_a[globalIdx[0]][globalIdx[1]] = localA[localIdx[0]][localIdx[1]];
  });

  for(int i = 0; i < vecSize; ++i) {
    for(int j = 0; j < vecSize; ++j) {
      if(av_a(i, j) != 2) {
        return 1;
      }
    }
  }

  return 0;
}
