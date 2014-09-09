// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll
// mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out 2>&1 | FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>

int f1() restrict(amp) {return 1;} 
int f2() restrict(cpu,auto) {
  return f1();
}
// CHECK: diagnose_before_perform_inferring_AMP.cpp:[[@LINE-2]]:12: error:  'f1':  no overloaded function has restriction specifiers that are compatible with the ambient context 'f2'
// CHECK-NEXT: return f1();
// CHECK-NEXT:        ^


int main(void)
{
  f2();
  return 0;
}

