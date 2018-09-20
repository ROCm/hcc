// RUN: %cxxamp %s -o %t.out 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

//initialize function reference with a function with incompatible restriction specifier</summary>
int glorp(int x) [[hc]] {
  return 668 + x;
}

int f_func_ref() restrict(auto) {
  typedef int FT(int);
  FT& p = glorp;
  return 1;
}

void CPU_Func() [[cpu]]
{
  f_func_ref();
}

int main(void)
{
  return 0;
}
// CHECK: In function `f_func_ref()':
// CHECK: undefined reference to `glorp(int)'
