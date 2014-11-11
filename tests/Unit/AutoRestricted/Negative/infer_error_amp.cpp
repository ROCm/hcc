// RUN: %cxxamp %s -o %t.out 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>

int f1() restrict(amp) {return 1;} 
int f2() restrict(auto) {
  return f1();
}
// CHECK: infer_error_amp.cpp:[[@LINE-2]]:12: error:  'f1':  no overloaded function has restriction specifiers that are compatible with the ambient context 'f2'
// CHECK-NEXT: return f1();
// CHECK-NEXT:        ^

int CPU_Func() restrict(cpu) {
  return f2();
}
// CHECK: infer_error_amp.cpp:[[@LINE-2]]:12: error:  'f2':  no overloaded function has restriction specifiers that are compatible with the ambient context 'CPU_Func'
// CHECK-NEXT: return f2();
// CHECK-NEXT:        ^

int main(void)
{
  return 0;
}

