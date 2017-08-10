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
  array_view<float> ga(a);
  array_view<float> gb(b);
  array_view<float> gc(c);
  for (index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = rand() + 1/ 1000.0f;
  }

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    gc[idx] = fast_math::rsqrtf(ga[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gb[i] = 1.f / fast_math::sqrtf(ga[i]);
  }

  float sum = 0;
  float a1 = 0, b1 = -1.0f;
  float c1 = fmax(a1, b1);
  for(unsigned i = 0; i < vecSize; i++) {
    sum += fast_math::fabs(fast_math::fabs(gc[i]) - fast_math::fabs(gb[i]));
  }
  return ((sum + c1) > 0.1f);
}
