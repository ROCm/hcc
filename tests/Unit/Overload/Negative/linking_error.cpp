// RUN: %cxxamp %s -o %t.out 2>&1 | %FileCheck --strict-whitespace %s
// XFAIL: *

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

int f2() [[hc]] {return 2;}

int AMP_AND_CPU_Func() [[cpu, hc]]
{
  // Link error: undefined reference to `f2()'
  // clang-3.3: error: linker command failed with exit code 1 (use -v to see invocation)
  // Since in CPU path, there is no any cpu restricted 'f2'
  return f2();
}
// CHECK: In function `AMP_AND_CPU_Func()':
// CHECK-NEXT: undefined reference to `f2()'


int main()
{
  return 1; // Should not compile
}
