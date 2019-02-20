// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

int f1() [[cpu]] {return 1;} 

int AMP_AND_CPU_Func_1() [[cpu, hc]]
{
  return f1();
}
// CHECK: call_distinct_from_dual_context.cpp:[[@LINE-2]]:10: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT:  return f1();
// CHECK-NEXT:         ^


int foo() {}

int main()
{
  auto a_lambda_func = []() [[cpu, hc]] { 
    foo();
  };
// CHECK: call_distinct_from_dual_context.cpp:[[@LINE-2]]:8: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::(anonymous class)::operator()'
// CHECK-NEXT:    foo();
// CHECK-NEXT:       ^


  parallel_for_each(extent<1>(1), [](hc::index<1>) [[cpu, hc]] {
    foo();
  });
// CHECK: call_distinct_from_dual_context.cpp:[[@LINE-2]]:8: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::(anonymous class)::operator()'
// CHECK-NEXT:    foo();
// CHECK-NEXT:       ^


  return 1; // Should not compile
}
