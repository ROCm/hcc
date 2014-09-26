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

int f2() restrict(amp) {return 2;}

int AMP_AND_CPU_Func() restrict(cpu,amp)
{
  // Link error: undefined reference to `f2()'
  // clang-3.3: error: linker command failed with exit code 1 (use -v to see invocation)
  // Since in CPU path, there is no any cpu restricted 'f2'
  return f2();
}
// CHECK: In function `AMP_AND_CPU_Func()':
// CHECK-NEXT: undefined reference to `f2()'


int main()
{
  return 1; // Should not compile
}
