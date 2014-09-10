//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2() restrict(cpu);        // expected-note{{previous declaration is here}}
int f2() restrict(cpu,auto) {  // expected-error{{'f2':  expected no other declaration since it is auto restricted}}
  return f1();
}
int main(void)
{
  f2();
  return 0;  // should not compile
}

