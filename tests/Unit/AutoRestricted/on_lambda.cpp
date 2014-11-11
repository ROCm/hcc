// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
int main(void){
  const int vecSize = 100;

  // Alloc & init input data
  Concurrency::extent<1> e(vecSize);
  Concurrency::array<int, 1> a(vecSize);
  Concurrency::array<int, 1> b(vecSize);
  Concurrency::array<int, 1> c(vecSize);
  int sum = 0;
  for (Concurrency::index<1> i(0); i[0] < vecSize; i++) {
    a[i] = 100.0f * rand() / RAND_MAX;
    b[i] = 100.0f * rand() / RAND_MAX;
    sum += a[i] + b[i];
  }

  Concurrency::array_view<int> ga(a);
  Concurrency::array_view<int> gb(b);
  Concurrency::array_view<int> gc(c);
  Concurrency::parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp,auto) {
    gc[idx] = ga[idx]+gb[idx];
  });

  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += gc[i] - (ga[i] + gb[i]);
  }
  return (error != 0);
}
// SPIR code generation test
// CHECK: metadata !{metadata !"kernel_arg_addr_space", i32 0, 
