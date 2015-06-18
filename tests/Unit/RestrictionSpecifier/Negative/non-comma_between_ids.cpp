// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////

#include <amp.h>

int foo() restrict(xx:auto1)
{
  return 1;
}
// CHECK: non-comma_between_ids.cpp:[[@LINE-4]]:20: error: 'xx' : unrecognized restriction specifier
// CHECK-NEXT:int foo() restrict(xx:auto1)
// CHECK-NEXT:                   ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-7]]:22: error: ':' : unrecognized restriction specifier
// CHECK-NEXT:int foo() restrict(xx:auto1)
// CHECK-NEXT:                     ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-10]]:23: error: 'auto1' : unrecognized restriction specifier
// CHECK-NEXT:int foo() restrict(xx:auto1)
// CHECK-NEXT:                      ^

// Left end
int fooxx() restrict(:auto2,,,)
{
  return 1;
}
// CHECK: non-comma_between_ids.cpp:[[@LINE-4]]:22: error: ':' : unrecognized restriction specifier
// CHECK-NEXT:int fooxx() restrict(:auto2,,,)
// CHECK-NEXT:                     ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-7]]:23: error: 'auto2' : unrecognized restriction specifier
// CHECK-NEXT:int fooxx() restrict(:auto2,,,)
// CHECK-NEXT:                      ^


// Right end
int fooyy() restrict(,,,::auto3)
{
  return 1;
}
// CHECK: non-comma_between_ids.cpp:[[@LINE-4]]:25: error: '::' : unrecognized restriction specifier
// CHECK-NEXT:int fooyy() restrict(,,,::auto3)
// CHECK-NEXT:                        ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-7]]:27: error: 'auto3' : unrecognized restriction specifier
// CHECK-NEXT:int fooyy() restrict(,,,::auto3)
// CHECK-NEXT:                         ^

// At both ends
int foozz() restrict(!X,,,a)
{
  return 1;
}
// CHECK: non-comma_between_ids.cpp:[[@LINE-4]]:22: error: '!' : unrecognized restriction specifier
// CHECK-NEXT:int foozz() restrict(!X,,,a)
// CHECK-NEXT:                     ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-7]]:23: error: 'X' : unrecognized restriction specifier
// CHECK-NEXT:int foozz() restrict(!X,,,a)
// CHECK-NEXT:                      ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-10]]:27: error: 'a' : unrecognized restriction specifier
// CHECK-NEXT:int foozz() restrict(!X,,,a)
// CHECK-NEXT:                          ^

int foo1() restrict(cpu:auto1)
{
  return 1;
}
// CHECK: non-comma_between_ids.cpp:[[@LINE-4]]:24: error: ':' : unrecognized restriction specifier
// CHECK-NEXT:int foo1() restrict(cpu:auto1)
// CHECK-NEXT:                       ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-7]]:25: error: 'auto1' : unrecognized restriction specifier
// CHECK-NEXT:int foo1() restrict(cpu:auto1)
// CHECK-NEXT:                        ^


int foo2() restrict(auto1&cpu)
{
  return 1;
}
// CHECK: non-comma_between_ids.cpp:[[@LINE-4]]:21: error: 'auto1' : unrecognized restriction specifier
// CHECK-NEXT:int foo2() restrict(auto1&cpu)
// CHECK-NEXT:                    ^
// CHECK: non-comma_between_ids.cpp:[[@LINE-7]]:26: error: '&' : unrecognized restriction specifier
// CHECK-NEXT:int foo2() restrict(auto1&cpu)
// CHECK-NEXT:                         ^

int main(void)
{
  return 0;
}

