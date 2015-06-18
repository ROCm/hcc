// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>

int foo() restrict()
{
  return 1;
}
// CHECK: empty_restriction.cpp:[[@LINE-4]]:20: error: empty restriction sepcifier is not allowed
// CHECK-NEXT:int foo() restrict()
// CHECK-NEXT:                   ^

int main(void)
{
  return 2;
}

