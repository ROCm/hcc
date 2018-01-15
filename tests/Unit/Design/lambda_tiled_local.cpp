// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
int main(void){
  const int vecSize = 1280;
#define TILE 128
  // Alloc & init input data
  Concurrency::extent<1> e(vecSize);
  Concurrency::tiled_extent<TILE> et(e);
  Concurrency::tiled_extent<TILE> et2 = e.tile<TILE>();
  assert(et.tile_dim0 == TILE);
  assert(et2.tile_dim0 == TILE);
  Concurrency::array<int, 1> a(vecSize);
  Concurrency::array<int, 1> b(vecSize);
  Concurrency::array<int, 1> c(vecSize);
  int sum = 0;
  Concurrency::array_view<int> ga(a);
  Concurrency::array_view<int> gb(b);
  Concurrency::array_view<int> gc(c);
  for (Concurrency::index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = 100.0f * rand() / RAND_MAX;
    gb[i] = 100.0f * rand() / RAND_MAX;
    sum += a[i] + b[i];
  }

  Concurrency::parallel_for_each(
    et,
    [=](Concurrency::tiled_index<TILE> idx) restrict(amp) {
    tile_static int shm[TILE];
    shm[idx.local[0]] = ga[idx];
    idx.barrier.wait();
    gc[idx] = shm[(TILE-1)-idx.local[0]]+gb[idx];
  });

  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += gc[i] - (ga[i] + gb[i]);
  }
  std::cout << "Error = " << error << "\n";
  return error != 0;
}
