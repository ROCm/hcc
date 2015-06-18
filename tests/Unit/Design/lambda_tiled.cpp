// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
int main(void){
  const int vecSize = 128;

  // Alloc & init input data
  Concurrency::extent<1> e(vecSize);
  Concurrency::tiled_extent<16> et(e);
  Concurrency::tiled_extent<16> et2 = e.tile<16>();
  assert(et.tile_dim0 == 16);
  assert(et2.tile_dim0 == 16);
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
    [=](Concurrency::tiled_index<16> idx) restrict(amp) {
    gc[idx] = ga[idx]+gb[idx];
  });

  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += gc[i] - (ga[i] + gb[i]);
  }
  return error != 0;
}
