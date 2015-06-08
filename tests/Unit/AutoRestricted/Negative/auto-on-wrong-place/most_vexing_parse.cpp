// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

//Most vexing parse
struct S_vex {
    S_vex(int) {};
};

int f_most_vexing_parse() {
    int a = 1;
    S_vex foo((int) restrict(auto) a);
// CHECK: most_vexing_parse.cpp:[[@LINE-1]]:30: error: expected expression
// CHECK-NEXT: S_vex foo((int) restrict(auto) a);
// CHECK-NEXT:                          ^
    S_vex foo1((int)a) restrict(auto); // expected_error{{expected ';' at end of declaration}}
// CHECK: most_vexing_parse.cpp:[[@LINE-1]]:23: error: expected ';' at end of declaration
// CHECK-NEXT: S_vex foo1((int)a) restrict(auto);
// CHECK-NEXT:                   ^
// CHECK-NEXT:                   ;
    return 1;
}

int main(void)
{
  return 0;
}

