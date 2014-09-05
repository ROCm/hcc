// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int fooAMP() restrict(amp)
{
  auto a_lambda = []() restrict(cpu) {}; // OK
  return 1;
}


int main(void)
{
   parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
  {
    auto a_lambda = []() restrict(cpu) {};// OK
  });
  return 0;
}

