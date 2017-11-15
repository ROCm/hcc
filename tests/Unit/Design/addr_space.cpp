// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <amp_math.h>
#include <random>

using namespace concurrency;

float x(float *p) restrict(amp) {
    return fast_math::sin(*p);
}

int main(void) {
  const int vecSize = 1000;

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(-4.0f * M_PI, 4.0f * M_PI);

  // Alloc & init input data
  extent<1> e(vecSize);
  array_view<float> ga(vecSize);
  array_view<float> gb(vecSize);
  array_view<float> gc(vecSize);
  for (index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = dis(gen);
  }

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    gc[idx] = x(&ga[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gb[i] = fast_math::sin(ga[i]);
  }

  float sum = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    sum = sum + fast_math::fabs(gc[i] - gb[i]);
  }
  return (sum > 0.1f);
}
