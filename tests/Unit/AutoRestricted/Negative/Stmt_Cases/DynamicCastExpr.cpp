// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

// DynamicCast
class A {};
class B : public A {};

void f_dynamiccast() restrict(auto)
{
    B * b = 0;
    A * a = dynamic_cast<A*>(b);
}

void AMP_AND_CPU_Func() [[cpu, hc]] {
  f_dynamiccast();
}
// CHECK: DynamicCastExpr.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: f_dynamiccast();
// CHECK-NEXT: ^
int main(void)
{
  return 0;
}

