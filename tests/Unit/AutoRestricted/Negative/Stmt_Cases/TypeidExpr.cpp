// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

//typeid
void f_typeid() restrict(auto)
{
    int n;
    int & r = n;
    typeid(r);
}
void AMP_AND_CPU_Func() [[cpu, hc]]
{
  f_typeid();
}
// CHECK: TypeidExpr.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: f_typeid();
// CHECK-NEXT: ^

int main(void)
{
  return 0;
}

