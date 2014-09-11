// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll
// RUN: mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out 2>&1 | FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

void foo() restrict(amp)
{
}


int main()
{
  auto a_lambda_func = []() restrict(cpu) { 
    foo();
  };
// CHECK: call_amp_function_in_cpu_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:8: error:  'main()::<anonymous class>::operator()':  no overloaded function has restriction specifiers that are compatible with the ambient context ''std::__1::__tree_node<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, void *>''
// CHECK-NEXT:    foo();
// CHECK-NEXT:       ^

  parallel_for_each(extent<1>(1), [](index<1>) restrict(cpu) {
    foo();
  });
// CHECK: call_amp_function_in_cpu_function_or_lambda_or_pfe.cpp:[[@LINE-2]]:8: error:  'main()::<anonymous class>::operator()':  no overloaded function has restriction specifiers that are compatible with the ambient context ''std::__1::__tree_node<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, void *>''
// CHECK-NEXT:        foo();
// CHECK-NEXT:           ^

  return 1; // Should not compile
}
