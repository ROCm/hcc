// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////

#include <amp.h>

int foo() restrict(!,,,,)
{
  return 1;
}
// CHECK: non-id_at_two_ends.cpp:[[@LINE-4]]:20: error: '!' : unrecognized restriction specifier
// CHECK-NEXT:int foo() restrict(!,,,,)
// CHECK-NEXT:                   ^

// consecutive
int foo1() restrict(!!,,,,)
{
  return 1;
}
// CHECK: non-id_at_two_ends.cpp:[[@LINE-4]]:21: error: '!' : unrecognized restriction specifier
// CHECK-NEXT:int foo1() restrict(!!,,,,)
// CHECK-NEXT:                    ^
// CHECK: non-id_at_two_ends.cpp:[[@LINE-7]]:22: error: '!' : unrecognized restriction specifier
// CHECK-NEXT:int foo1() restrict(!!,,,,)
// CHECK-NEXT:                     ^


int foo2() restrict(,,,,*)
{
  return 1;
}
// CHECK: non-id_at_two_ends.cpp:[[@LINE-4]]:25: error: '*' : unrecognized restriction specifier
// CHECK-NEXT:int foo2() restrict(,,,,*)
// CHECK-NEXT:                        ^


int foo3() restrict(,,,,**)
{
  return 1;
}
// CHECK: non-id_at_two_ends.cpp:[[@LINE-4]]:25: error: '*' : unrecognized restriction specifier
// CHECK-NEXT:int foo3() restrict(,,,,**)
// CHECK-NEXT:                        ^
// CHECK: non-id_at_two_ends.cpp:[[@LINE-7]]:26: error: '*' : unrecognized restriction specifier
// CHECK-NEXT:int foo3() restrict(,,,,**)
// CHECK-NEXT:                         ^

// both
int foo4() restrict(!,,,,*)
{
  return 1;
}
// CHECK: non-id_at_two_ends.cpp:[[@LINE-4]]:21: error: '!' : unrecognized restriction specifier
// CHECK-NEXT:int foo4() restrict(!,,,,*)
// CHECK-NEXT:                    ^
// CHECK: non-id_at_two_ends.cpp:[[@LINE-7]]:26: error: '*' : unrecognized restriction specifier
// CHECK-NEXT:int foo4() restrict(!,,,,*)
// CHECK-NEXT:                         ^


int main(void)
{
  return 0;
}

