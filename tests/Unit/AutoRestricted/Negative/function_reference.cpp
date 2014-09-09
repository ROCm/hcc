// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll
// RUN: mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out 2>&1 | FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

//initialize function reference with a function with incompatible restriction specifier</summary>
int glorp(int x) restrict(amp) {
  return 668 + x;
}

int f_func_ref() restrict(auto) {
  typedef int FT(int);
  FT& p = glorp;
  return 1;
}

void CPU_Func() restrict(cpu)
{
  f_func_ref();
}

int main(void)
{
  return 0;
}
// CHECK: In function `f_func_ref()':
// CHECK: undefined reference to `glorp(int)'
