// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int fooCPU() restrict(cpu)
{
  return 1;
}

int foo()
{
  return 2;
}

int main(void)
{
  fooCPU();
  foo();
  auto a_lambda = [] () restrict(cpu) {};
  auto another_lambda = [] () {};

  return 0;
}

