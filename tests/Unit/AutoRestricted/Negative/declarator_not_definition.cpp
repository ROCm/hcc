//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2xx() restrict(cpu,auto);  // expected-error{{'auto' restriction specifier is only allowed on function definition}}
int f2xx() restrict(cpu)
{
  return f1();
}
int main(void)
{
  f2xx();
  return 0;  // should not compile
}

