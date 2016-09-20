
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>

using namespace hc;

template<typename T>
bool test() {
  const int vecSize = 100;
  const int tile_size = 10;

  // Alloc & init input data
  extent<2> e_a(vecSize, vecSize);
  std::vector<T> va(vecSize * vecSize);
  for(int i = 0; i < vecSize * vecSize; ++i) {
    va[i] = 0;
  }
  array_view<T, 2> av_a(e_a, va); 

  extent<2> compute_domain(e_a);
  parallel_for_each(compute_domain.tile(tile_size, tile_size), [=] (tiled_index<2> tidx) [[hc]] {
    index<2> localIdx = tidx.local;
    index<2> globalIdx = tidx.global;

    tile_static T localA[tile_size][tile_size];
    localA[localIdx[0]][localIdx[1]] = T(tile_size * tile_size);
    tidx.barrier.wait();

    for(int i = 0; i < tile_size; ++i) {
      for(int j = 0; j < tile_size; ++j) {
        atomic_fetch_sub(&(localA[i][j]), T(1));
      }
    }
    tidx.barrier.wait();
    av_a[globalIdx[0]][globalIdx[1]] = localA[localIdx[0]][localIdx[1]];
  });

  bool ret = true;
  for(int i = 0; i < vecSize; ++i) {
    for(int j = 0; j < vecSize; ++j) {
      if(av_a(i, j) != T(0)) {
        ret = false;
      }
    }
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<unsigned int>();
  ret &= test<int>();

  return !(ret == true);
}

