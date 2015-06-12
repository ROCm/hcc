// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

// before function name
int restrict(auto) f_before_function_name() {
  return 1;
}
// CHECK: before_function_name.cpp:[[@LINE-3]]:14: error: 'auto' not allowed in function prototype
// CHECK-NEXT:int restrict(auto) f_before_function_name() {
// CHECK-NEXT:             ^~~~
// CHECK-NEXT:before_function_name.cpp:[[@LINE-6]]:20: error: expected 'restrict' specifier
// CHECK-NEXT:int restrict(auto) f_before_function_name() {
// CHECK-NEXT:                   ^
// CHECK-NEXT:before_function_name.cpp:[[@LINE-9]]:13: error: function cannot return function type 'int ()'
// CHECK-NEXT:int restrict(auto) f_before_function_name() {

int main(void)
{
  return 0;
}

