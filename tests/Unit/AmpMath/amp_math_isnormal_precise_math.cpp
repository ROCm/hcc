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
  const int vecSize = 2;

  // Alloc & init input data
  extent<1> e(vecSize);
  array<_Tp> a(e);
  array<int> b(e);
  array_view<_Tp, 1> in(a);
  array_view<int, 1> out(b);

  in[0] = 1.0;
  in[1] = 0.0;

  parallel_for_each(
    e,
    [=](hc::index<1> idx) [[hc]] {
    out[idx] = precise_math::isnormal(in[idx]);
  });

  //check accelerator results
  for (int i=0; i<vecSize; ++i) {
    if (std::isnormal(in[i]) != (out[i] ? true : false))
      return false;
  }

  //check on cpu
  for (int i=0; i<vecSize; ++i) {
    if (std::isnormal(in[i]) != (precise_math::isnormal(in[i]) ? true : false))
      return false;
  }

  return true;
}

int main(void) {
  bool ret = true;

  //ret &= test<float>();
  ret &= test<double>();

  return !(ret == true);
}

