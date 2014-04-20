// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
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

  in[0] = std::numeric_limits<float>::quiet_NaN();
  in[1] = 413.612;
  in[2] = std::numeric_limits<float>::signaling_NaN();

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    out[idx] = precise_math::isnan(in[idx]);
  });

  //check accelerator results
  for (int i=0; i<vecSize; ++i) {
    if (std::isnan(in[i]) != (out[i] ? true : false))
      return 1;
  }

  //check on cpu
  for (int i=0; i<vecSize; ++i) {
    if (std::isnan(in[i]) != (precise_math::isnan(in[i]) ? true : false))
      return 1;
  }


  return 0;
}
