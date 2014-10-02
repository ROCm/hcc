// RUN: %cxxamp %s -o %t.out 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

int foo() restrict(amp)
{
  return 1;
}

int main()
{
  foo();
  return 1; // Should not compile
}
// CHECK: call_amp_function_in_main.cpp:[[@LINE-3]]:6: error:  'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main'
// CHECK-NEXT:  foo();
// CHECK-NEXT:     ^



