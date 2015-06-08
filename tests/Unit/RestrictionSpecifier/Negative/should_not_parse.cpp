// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>

int f1() restrict(amp:,)
{
  return 1;
}
// CHECK: should_not_parse.cpp:[[@LINE-4]]:22: error: ':' : unrecognized restriction specifier
// CHECK-NEXT:int f1() restrict(amp:,)
// CHECK-NEXT:                     ^

// 'amp' should not be attached to f1()
int f2() restrict(amp)
{
  f1();  // expected-error{{'f1': no overload...}}
  return 0;
}

int main(void)
{
  return 0;
}

