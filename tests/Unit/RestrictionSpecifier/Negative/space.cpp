// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////

#include <amp.h>

int f1() restrict(cpu,   ,auto1)  // expected-error{{'auto1': unrecognized restriction sepcifier}}
{
  return 1;
}
// CHECK: space.cpp:[[@LINE-4]]:27: error: 'auto1' : unrecognized restriction specifier
// CHECK-NEXT:int f1() restrict(cpu,   ,auto1)
// CHECK-NEXT:                          ^

int main(void)
{
  return 0;
}

