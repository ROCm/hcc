// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>

using namespace concurrency;

int test(int (*p)(int, int) restrict(auto)) // expected-error{{'auto' not allowed in function prototype}}
{
  return 1;
}
// CHECK: auto_in_function_prototype.cpp:[[@LINE-4]]:42: error: 'auto' restriction specifier is only allowed on function definition
// CHECK-NEXT:int test(int (*p)(int, int) restrict(auto))
// CHECK-NEXT:                                         ^
int main(void)
{
  return 0;
}

