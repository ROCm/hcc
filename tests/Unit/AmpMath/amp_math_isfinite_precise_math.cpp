// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <amp_math.h>

using namespace concurrency;

template<typename _Tp>
bool test() {
  const int vecSize = 2;

  // Alloc & init input data
  extent<1> e(vecSize);
  array<_Tp> a(vecSize);
  array<int> b(vecSize);
  array_view<_Tp, 1> in(a);
  array_view<int, 1> out(b);

  in[0] = 1.0;
  in[1] = 0.0;

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    out[idx] = precise_math::isfinite(6.5/in[idx]);
  });

  //check accelerator results
  for (int i=0; i<vecSize; ++i) {
    if (std::isfinite(6.5/in[i]) != out[i] ? true : false)
      return false;
  }

  //check on cpu
  for (int i=0; i<vecSize; ++i) {
    if (std::isfinite(in[i]) != (precise_math::isfinite(in[i]) ? true : false))
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

