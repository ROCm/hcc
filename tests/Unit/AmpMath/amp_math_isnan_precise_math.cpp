// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <hc_math.hpp>
#include <cmath>
#include <cassert>

using namespace hc;

template<typename _Tp>
bool test() {
  const int vecSize = 3;

  // Alloc & init input data
  extent<1> e(vecSize);
  array_view<_Tp, 1> in(vecSize);
  array_view<int, 1> out(vecSize);

  in[0] = std::numeric_limits<_Tp>::quiet_NaN();
  in[1] = 413.612;
  in[2] = std::numeric_limits<_Tp>::signaling_NaN();

  parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
    out[idx] = precise_math::isnan(in[idx]);
  });

  //check accelerator results
  for (int i=0; i<vecSize; ++i) {
    if (std::isnan(in[i]) != (out[i] ? true : false))
      return false;
  }

  //check on cpu
  for (int i=0; i<vecSize; ++i) {
    if (std::isnan(in[i]) != (precise_math::isnan(in[i]) ? true : false))
      return false;
  }

  return true;
}

int main(void) {
  bool ret = true;

  ret &= test<float>();
  ret &= test<double>();

  return !(ret == true);
}

