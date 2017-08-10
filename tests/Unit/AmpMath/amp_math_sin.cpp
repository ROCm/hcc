// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_math.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

using namespace concurrency;

int main()
{
  constexpr int vec_size = 1000;

  // Alloc & init input data
  extent<1> e(vec_size);
  array<float, 1> a(vec_size);
  array<float, 1> b(vec_size);
  array<float, 1> c(vec_size);
  array_view<float> ga(a);
  array_view<float> gb(b);
  array_view<float> gc(c);

  std::mt19937_64 g;
  std::uniform_real_distribution<float> d{-2 * M_PI, 2 * M_PI};
  std::generate_n(ga.data(), vec_size, [&]() { return d(g); });

  parallel_for_each(e, [=](index<1> idx) restrict(amp) {
    gc[idx] = fast_math::sin(ga[idx]);
  });

  for(unsigned i = 0; i < vec_size; i++) {
    gb[i] = fast_math::sin(ga[i]);
  }

  float sum = 0;
  for(unsigned i = 0; i < vec_size; i++) {
    sum += fast_math::fabs(fast_math::fabs(gc[i]) - fast_math::fabs(gb[i]));
  }
  return (sum > 0.1f);
}
