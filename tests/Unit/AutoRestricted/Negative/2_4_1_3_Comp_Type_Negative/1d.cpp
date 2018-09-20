// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>>
#include "common.h"

using std::vector;
using namespace hc;

// From 2_Cxx_Lang_Exte/2_4_amp_Rest_Modi/2_4_1_Rest_on_Type/2_4_1_3_Comp_Type/Negative/1d/Test01/test.cpp
void f_1d() restrict(auto) { 
  struct s1
  {
    s1(array_view<int> a) [[cpu, hc]] : m(a) {}
    ~s1() [[cpu, hc]] {}

    array_view<int> &m;
  };
}
// CHECK: 1d.cpp:[[@LINE-3]]:22: error: pointer or reference is not allowed as pointed to type, array element type or data member type (except reference to hc::array/texture)
// CHECK-NEXT: array_view<int> &m;
// CHECK-NEXT: ^

void AMP_AND_CPU_Func() [[cpu, hc]] {
  f_1d();
}
// CHECK: 1d.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: f_1d();
// CHECK-NEXT: ^

int main(void)
{
  exit(1);
  return 0;
}

