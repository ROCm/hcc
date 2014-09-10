// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2() restrict(auto) {
  static int i;
  return f1();
}

int AMP_AND_CPU_Func() restrict(cpu,amp) 
{
  f2();  // OK. 'auto' is inferred to (cpu,amp)
  return 1;
}

int main(void)
{
  return 0;  // expected: success
}

