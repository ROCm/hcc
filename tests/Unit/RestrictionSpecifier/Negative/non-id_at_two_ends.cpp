//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int foo() restrict(!,,,,)  // expected-error{{'!': unrecognized restriction sepcifier}}
{
  return 1;
}

// consecutive
int foo1() restrict(!!,,,,)  // expected-error{{'!': unrecognized restriction sepcifier}} // expected-error{{'!': unrecognized restriction sepcifier}}
{
  return 1;
}


int foo2() restrict(,,,,*)  // expected-error{{'*': unrecognized restriction sepcifier}} // expected-error{{'*': unrecognized restriction sepcifier}}
{
  return 1;
}


int foo3() restrict(,,,,**)  // expected-error{{'*': unrecognized restriction sepcifier}} // expected-error{{'*': unrecognized restriction sepcifier}}
{
  return 1;
}

// both
int foo4() restrict(!,,,,*)  // expected-error{{'!': unrecognized restriction sepcifier}} // expected-error{{'*': unrecognized restriction sepcifier}}
{
  return 1;
}


int main(void)
{
  return 0;
}

