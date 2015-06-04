// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <amp_math.h>

using namespace concurrency;

int main(void) {
  const int vecSize = 1000;

  // Alloc & init input data
  extent<1> e(vecSize);
  array<float, 1> a(vecSize);
  array<float, 1> b(vecSize);
  array<float, 1> c(vecSize);
  array<int, 1>   exp(vecSize);
   
  array_view<float> ga(a);
  array_view<float> gb(b);
  array_view<float> gc(c);
  array_view<int> gexp(exp);
  for (index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = rand() / 1000.0f;
    gexp[i] = rand();
  }

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    gb[idx] = precise_math::scalbnf(ga[idx], gexp[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gc[i] = precise_math::scalbnf(ga[i], gexp[i]);
  }

  float sum = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    sum += fast_math::fabs(fast_math::fabs(gc[i]) - fast_math::fabs(gb[i]));
  }
  return (sum > 0.1f);
}
