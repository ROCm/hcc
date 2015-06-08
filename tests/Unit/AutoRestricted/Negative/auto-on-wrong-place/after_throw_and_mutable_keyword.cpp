// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

void f_wrong_order_of_mutable_throw() {
  // error: inner lambda has incorrect lamda-declarator clause
  parallel_for_each(extent<1>(1), [&](index<1> idx) restrict(amp) { 
   []() mutable throw() -> void restrict(auto) {}();
   });
}
// CHECK: after_throw_and_mutable_keyword.cpp:[[@LINE-3]]:17: error: exception specifier is not allowed in C++AMP context
// CHECK-NEXT:   []() mutable throw() -> void restrict(auto) {}();
// CHECK-NEXT:                        ^

int main(void)
{
  return 0;
}

