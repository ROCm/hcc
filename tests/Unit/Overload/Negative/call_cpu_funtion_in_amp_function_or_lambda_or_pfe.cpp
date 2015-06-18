// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

void foo()
{
}

int f1() restrict(cpu) {return 1;} 
int f2() restrict(cpu) {
  return f1();
}

int AMP_Func() restrict(amp)
{
  return f2();
}
// CHECK: call_cpu_funtion_in_amp_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:10: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: return f2();
// CHECK-NEXT:        ^

int main()
{
  auto a_lambda_func = []() restrict(amp) { 
    foo();
  };
// CHECK: call_cpu_funtion_in_amp_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:8: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::(anonymous class)::operator()'
// CHECK-NEXT:    foo();
// CHECK-NEXT:       ^

  parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
  {
    foo();
  });
// CHECK: call_cpu_funtion_in_amp_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:8: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::(anonymous class)::operator()'
// CHECK-NEXT:    foo();
// CHECK-NEXT:       ^
   
    return 1; // Should not compile
}
