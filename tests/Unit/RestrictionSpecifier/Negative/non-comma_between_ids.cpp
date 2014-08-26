//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int 
foo() restrict(xx:auto1) // expected-error{{'xx': unrecognized restriction sepcifier}} // expected-error{{':': unrecognized restriction sepcifier}} // expected-error{{'auto1': unrecognized restriction sepcifier}}
{
  return 1;
}

// Left end
int 
fooxx() restrict(:auto2,,,) // expected-error{{':': unrecognized restriction sepcifier}} // expected-error{{'auto1': unrecognized restriction sepcifier}}
{
  return 1;
}

// Right end
int 
fooyy() restrict(,,,::auto3)  // expected-error{{'::': unrecognized restriction sepcifier}}  // expected-error{{'auto3': unrecognized restriction sepcifier}}
{
  return 1;
}

// At both ends
int 
foozz() restrict(!X,,,a)  // expected-error{{'!': unrecognized restriction sepcifier}}  // expected-error{{'X': unrecognized restriction sepcifier}} // expected-error{{'a': unrecognized restriction sepcifier}}
{
  return 1;
}

int foo1() restrict(cpu:auto1)  // expected-error{{expected ':' in restriction sepcifier}} // expected-error{{'auto1': unrecognized restriction sepcifier}}
{
  return 1;
}

int foo2() restrict(auto1&cpu)  // expected-error{{expected ':' in restriction sepcifier}} // expected-error{{'auto1': unrecognized restriction sepcifier}}
{
  return 1;
}

int main(void)
{
  return 0;
}

