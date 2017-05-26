// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_math.h>

#include <iostream>
#include <random>
#include <cmath>

using namespace concurrency;

#define ERROR_THRESHOLD (1e-4)

template<typename _Tp>
bool test() {
  const int vecSize = 1024;

  // Alloc & init input data
  extent<1> e(vecSize);
  array<_Tp, 1> a(vecSize);
  array<_Tp, 1> b(vecSize);
  array<_Tp, 1> c(vecSize);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<_Tp> dis(-100, 100);
  array_view<_Tp> ga(a);
  array_view<_Tp> gb(b);
  array_view<_Tp> gc(c);
  for (index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = dis(gen);
  }

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    gc[idx] = precise_math::tanpi(ga[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gb[i] = precise_math::tan(static_cast<_Tp>(M_PI) * ga[i]);
  }

  _Tp sum = 0.0;
  for(unsigned i = 0; i < vecSize; i++) {
    sum += precise_math::fabs(
        precise_math::fabs(gc[i]) - precise_math::fabs(gb[i]));
  }
  return (sum < ERROR_THRESHOLD);
}

int main(void) {
  bool ret = true;

  //ret &= test<float>();
  ret &= test<double>();

  return !(ret == true);
}

