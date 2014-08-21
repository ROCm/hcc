// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

int foo() restrict(,)  // OK
{
  return 0;
}


int foo1() restrict(amp,)  // OK
{
  return 0;
}
int fooAMP() restrict(,amp)  // OK
{
  foo1();  // OK
  return 0;
}


int foo2() restrict(,   ,,,   ,cpu,,,,)  // OK
{
  return 0;
}
int fooCPU() restrict(cpu)  // OK
{
  foo2();  // OK
  return 0;
}


int main(void)
{
  parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
    {
        fooAMP();
    });
}

