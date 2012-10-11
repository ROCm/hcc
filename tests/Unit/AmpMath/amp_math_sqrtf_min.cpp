// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O3 -o %t.ll && mkdir -p %t
// RUN: %llc -march=c -o %t/kernel_.cl < %t.ll
// RUN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// RUN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
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
  for (index<1> i(0); i[0] < vecSize; i++) {
    a[i] = rand() / 1000.0f;
  }

  array_view<float> ga(a);
  array_view<float> gb(b);
  array_view<float> gc(c);
  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    gc[idx] = fast_math::sqrtf(ga[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gb[i] = fast_math::sqrtf(ga[i]);
  }

  int a1 = 0, b1 = 3;
  int c1 = min(a1, b1);

  float sum = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    sum += fast_math::fabs(fast_math::fabs(gc[i]) - fast_math::fabs(gb[i]));
  }
  return ((sum + (float)c1) > 0.1f);
}
