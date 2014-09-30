// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

void f_wrong_order_of_trailing_return() {
  // error: inner lambda has incorrect lamda-declarator clause
  parallel_for_each(extent<1>(1), [&](index<1> idx) restrict(amp) { 
   []() mutable -> void restrict(auto) {}(); // expected_error{{expected body of lambda expression}}
   });
}
// CHECK: after_mutable_keyword.cpp:[[@LINE-3]]:25: error: expected body of lambda expression
// CHECK-NEXT:   []() mutable -> void restrict(auto) {}();
// CHECK-NEXT:                        ^
// CHECK-NEXT:[DeclareAMPTrampoline] Deserialization constructor is invalid!
// CHECK-NEXT:[DefineAMPTrampoline] Deserialization constructor is invalid!

int main(void)
{
  return 0;
}

