// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_math.hpp>

#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

using namespace hc;

#define ERROR_THRESHOLD (1e-4)

template<typename _Tp>
bool test() {
  const int vecSize = 1024;

  // Alloc & init input data
  extent<1> e(vecSize);
  array<_Tp, 1> a(vecSize);

  array<_Tp, 1> b(vecSize);
  array<_Tp, 1> c(vecSize);
  array<int, 1>   exp(vecSize);
   
  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<_Tp> dis(1, 100);
  std::uniform_int_distribution<int> dis_int(1, 10);
  array_view<_Tp> ga(a);
  array_view<_Tp> gb(b);
  array_view<_Tp> gc(c);
  array_view<int> gexp(exp);
  for (hc::index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = dis(gen);
    gexp[i] = dis_int(gen);
  }

  parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
    gb[idx] = precise_math::ldexp(ga[idx], gexp[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gc[i] = precise_math::ldexp(ga[i], gexp[i]);
  }

  _Tp sum = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    if (std::isnan(gc[i])) {
      printf("gc[%d] is NaN!\n", i);
      assert(false);
    }
    _Tp diff = precise_math::fabs(gc[i] - gb[i]);
    sum += diff;
  }
  return (sum < ERROR_THRESHOLD);
}

int main(void) {
  bool ret = true;

  ret &= test<float>();
  ret &= test<double>();

  return !(ret == true);
}

