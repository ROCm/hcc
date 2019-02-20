// RUN: %cxxamp %s -o %t.out 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

void foo() [[hc]]
{
}


int main()
{
  auto a_lambda_func = []() [[cpu]] { 
    foo();
  };
// CHECK: call_amp_function_in_cpu_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:8: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::(anonymous class)::operator()'
// CHECK-NEXT:    foo();
// CHECK-NEXT:       ^

  parallel_for_each(extent<1>(1), [](hc::index<1>) [[cpu]] {
    foo();
  });
// CHECK: call_amp_function_in_cpu_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:8: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::(anonymous class)::operator()'
// CHECK-NEXT:        foo();
// CHECK-NEXT:           ^

  return 1; // Should not compile
}
