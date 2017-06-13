// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <amp_math.h>

using namespace concurrency;

int main(void) {
  const int vecSize = 3;

  // Alloc & init input data
  extent<1> e(vecSize);
  array_view<float, 1> in(vecSize);
  array_view<int, 1> out(vecSize);

  in[0] = 1.0f;
  in[1] = 0.0f;
  in[2] = -1.0f;

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    out[idx] = fast_math::signbitf(in[idx]);
  });

  //check accelerator results
  for (int i=0; i<vecSize; ++i) {
    if (std::signbit(in[i]) != (out[i] ? true : false))
      return 1;
  }

  //check on cpu
  for (int i=0; i<vecSize; ++i) {
    if (std::signbit(in[i]) != (fast_math::signbit(in[i]) ? true : false))
      return 1;
  }


  return 0;
}
