//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,auto1)  // expected-error{{'auto1': unrecognized restriction sepcifier}}
{
  return 1;
}

int f2() restrict(auto2,,,,,)  // expected-error{{'auto2': unrecognized restriction sepcifier}}
{
  return 2;
}

int f3() restrict(,,auto2,,,)  // expected-error{{'auto2': unrecognized restriction sepcifier}}
{
  return 2;
}

int f4() restrict(,,,,,auto3)  // expected-error{{'auto2': unrecognized restriction sepcifier}}
{
  return 2;
}

int main(void)
{
  return 0;
}

