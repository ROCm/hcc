// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////

#include <amp.h>

int f1() restrict(cpu,auto1)
{
  return 1;
}
// CHECK: id_is_unrecognized.cpp:[[@LINE-4]]:23: error: 'auto1' : unrecognized restriction specifier
// CHECK-NEXT:int f1() restrict(cpu,auto1)
// CHECK-NEXT:                      ^

int f2() restrict(auto2,,,,,)
{
  return 2;
}
// CHECK: id_is_unrecognized.cpp:[[@LINE-4]]:19: error: 'auto2' : unrecognized restriction specifier
// CHECK-NEXT:int f2() restrict(auto2,,,,,)
// CHECK-NEXT:                  ^

int f3() restrict(,,auto2,,,)
{
  return 2;
}
// CHECK: id_is_unrecognized.cpp:[[@LINE-4]]:21: error: 'auto2' : unrecognized restriction specifier
// CHECK-NEXT:int f3() restrict(,,auto2,,,)
// CHECK-NEXT:                    ^

int f4() restrict(,,,,,auto3)
{
  return 2;
}
// CHECK: id_is_unrecognized.cpp:[[@LINE-4]]:24: error: 'auto3' : unrecognized restriction specifier
// CHECK-NEXT:int f4() restrict(,,,,,auto3)
// CHECK-NEXT:                       ^

int main(void)
{
  return 0;
}

