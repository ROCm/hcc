// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

short foo_short(unsigned short s) [[hc]] {
    return (s + 2);
}
// CHECK: char_short_wchar_longlong_longdouble.cpp:[[@LINE-3]]:1: error: short type can't be used as function return type in AMP-restricted functions
// CHECK-NEXT: short foo_short(unsigned short s) [[hc]] {
// CHECK-NEXT: ^

int f_char_short_wchar_longlong_longdouble() restrict(auto)
{
  char c = 65;
  long double ld = 6LL;
  long long ll = 6LL; 
  foo_short(2);
  wchar_t c1 = 65;
  return 0;
}

void AMP_AND_CPU_Func() [[cpu, hc]] {
  f_char_short_wchar_longlong_longdouble();
}
// CHECK: char_short_wchar_longlong_longdouble.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: f_char_short_wchar_longlong_longdouble();
// CHECK-NEXT:        ^

int main(void)
{
  return 0;
}

