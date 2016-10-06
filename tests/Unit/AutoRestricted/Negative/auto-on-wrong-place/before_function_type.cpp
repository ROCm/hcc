// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

// before function type
restrict(auto) int f_before_function_type() restrict(amp) {return 1;}
// CHECK: before_function_type.cpp:[[@LINE-1]]:10: error: 'auto' not allowed in function prototype
// CHECK-NEXT:restrict(auto) int f_before_function_type() restrict(amp) {return 1;}
// CHECK-NEXT:         ^~~~
// CHECK-NEXT:before_function_type.cpp:[[@LINE-4]]:1: error: C++ requires a type specifier for all declarations
// CHECK-NEXT:restrict(auto) int f_before_function_type() restrict(amp) {return 1;}

int main(void)
{
  return 0;
}

