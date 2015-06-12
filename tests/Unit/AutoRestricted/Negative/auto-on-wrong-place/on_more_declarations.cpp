// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

// different restriction specifier on function declaration and definition
struct S
{
  int test() restrict(amp);
};

int S::test() restrict(auto) {
    return 1;
}
// CHECK: on_more_declarations.cpp:[[@LINE-3]]:28: error: 'test':  expected no other declaration since it is auto restricted
// CHECK-NEXT:int S::test() restrict(auto)
// CHECK-NEXT:                            ^
// CHECK-NEXT:note: previous declaration is here
// CHECK-NEXT:  int test() restrict(amp);
// CHECK-NEXT:      ^
// CHECK-NEXT:on_more_declarations.cpp:[[@LINE-9]]:8: error: out-of-line definition of 'test' does not match any declaration in 'S'
// CHECK-NEXT:int S::test() restrict(auto)
// CHECK-NEXT        ^~~~

int main(void)
{
  return 0;
}

