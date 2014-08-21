//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(amp:,)  // expected-error{{':': unrecognized restriction sepcifier}}
{
  return 1;
}

// 'amp' should not be attached to f1()
int f2() restrict(amp)
{
  f1();  // expected-error{{'f1': no overload...}}
  return 0;
}

int main(void)
{
  return 0;
}

